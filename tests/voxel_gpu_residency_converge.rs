//! **Phase G "G-c.3" — the readback-free CONVERGENCE gate** (docs/PHASE_G_GC_PLAN.md §3.1 no-change detection /
//! indirect self-gating, §1 Pass A/D indirect dispatches, §3.2 keep-old-until-revealed, §6 "G-c.3", §7).
//!
//! G-c.3 makes the GPU residency front end IDEMPOTENT + SELF-GATING: a static camera converges to
//! `change_count == 0`, and once converged the pack tail (classify_brick / pack_brick / write_aabb) dispatches
//! ZERO workgroups via the GPU-WRITTEN dispatch-indirect buffers Pass D fills (`atomicMax`) and Pass A re-seeds
//! to `(0,1,1)` — NO CPU branch, NO readback to size the tail. This rig proves all of that headlessly by driving
//! the WHOLE GPU front end (Pass A → A2 clear → B0 → B → C → D + classify/pack/write_aabb into the persistent
//! pool) FRAME-BY-FRAME over PERSISTENT buffers, with the pack tail recorded as INDIRECT dispatches.
//!
//! ## What this gate proves
//!  * **(a) static convergence + self-gating** — hold a fixed camera; assert `change_count` reaches 0 within a
//!    bounded number of frames and STAYS 0 (idempotent, no churn). On the converged frame, the GPU-written
//!    `classify_dispatch[0] / pack_dispatch[0] / aabb_dispatch[0]` are all 0 (read back ONLY to assert the
//!    self-gating — the live path never reads them; the `record_indirect` consumes them GPU-side), so the tail
//!    launches 0 workgroups.
//!  * **(b) move → reconverge == CPU** — over a camera SEQUENCE (cold fill, brick crossings, LOD coarsen/refine,
//!    negative coords), drive the GPU front end to convergence each step and assert the converged resident pool's
//!    per-KEY content (decode-by-key via the SSOT `cell_block`) AND ray-HITS (the faithful DDA oracle) EQUAL the
//!    CPU `ResidentPacker` converged state. NO HOLE at any step (keep-old-until-revealed): after the FIRST frame
//!    following a move, the GPU resident set still covers every key the CPU keeps.
//!
//! ## One Pass-D round converges (no extra multi-round loop needed in Pass D)
//! Residency convergence is governed by Pass C (the SET diff), NOT Pass D (the pack). Pass C is idempotent — same
//! camera ⇒ same candidate/desired sets ⇒ 0 enter + 0 drop on a re-run — and keep-old-until-revealed needs the
//! resident set to ADVANCE one round per frame (a superseded LOD drops only once its replacement is resident), so
//! we iterate WHOLE FRAMES until `change_count == 0` exactly as the CPU alternates update+drain (mirror of the
//! G-c.2a diff gate's `converge`). Pass D1's first-order halo expansion is EXACT per frame: every key that
//! ENTERED that frame is re-packed with its resident 26-neighbour halo, and a DROPPED key dirties its resident
//! neighbours — so each converging frame's pack is correct, and the FINAL converged pool is the full resident set
//! packed with correct halos. We therefore do NOT add a bounded multi-round Pass-D loop; the per-frame round +
//! the across-frames keep-old advance is the convergence mechanism (and the gate proves it closes + matches CPU).
//!
//! Skips cleanly when no GPU adapter (or its compute / storage-buffer limits are too low).

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, MAX_LOD, brick_span};
use adventure::voxel::edits::VoxelEdits;
use adventure::voxel::gpu::{GpuBrickMeta, GpuBrickPatch, ResidentBrick, build_by_key, halo_cells, pack_one};
use adventure::voxel::incremental::ResidentPacker;
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::residency_gpu::{SectorOccupancy, brick_key_hash};
use adventure::voxel::source::{BrickSource, StaticVoxSource};
use adventure::voxel::streaming::{
    ResidencyManager, StreamingConfig, camera_brick_coord_lod, level_box_pub,
};
use bevy::math::{IVec3, Vec3};
use bytemuck::Zeroable;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use wgpu::util::DeviceExt;

#[path = "common/mod.rs"]
mod common;

#[path = "voxel_dda_oracle.rs"]
mod dda_oracle;

const LODS: usize = (MAX_LOD + 1) as usize; // 8
const WG_CELL: i32 = 8;
const REFINE_DESCENT_CAP: u32 = 5;
const EMPTY_LOD: u32 = 0xFFFF_FFFF;
const META_WORDS: usize = 12; // 48-B GpuBrickMeta

// =========================================================================================================
//  ResidencyParams (Pass B input) — same SSOT as the enumerate/diff/pack rigs.
// =========================================================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelParams {
    cam_brick_coord: [i32; 3],
    _pad_a: i32,
    cell_lo: [i32; 3],
    cell_offset: u32,
    cell_dims: [u32; 3],
    cell_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResidencyParams {
    levels: [LevelParams; LODS],
    clip_half_bricks: i32,
    total_cells: u32,
    hist_scale: f32,
    _pad1: u32,
    cam_world: [f32; 3],
    _pad2: u32,
}

/// Enter-cap distance histogram buckets — MUST equal `HIST_BUCKETS` in `voxel_residency.wgsl`.
const HIST_BUCKETS: u32 = 4096;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DiffConfig {
    slot_table_size: u32,
    present_size: u32,
    max_resident: u32,
    refine_descent_cap: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackConfig {
    core_table_size: u32,
    max_resident: u32,
    _pad0: u32,
    _pad1: u32,
}

fn build_params(cam: [f32; 3], half: i32) -> ResidencyParams {
    let mut levels = [LevelParams::zeroed(); LODS];
    let mut offset = 0u32;
    for lod in 0..=MAX_LOD {
        let (lo, hi) = level_box_pub(cam, lod, half);
        let cam_brick = camera_brick_coord_lod(cam, lod);
        let cell_lo = IVec3::new(
            lo.x.div_euclid(WG_CELL) * WG_CELL,
            lo.y.div_euclid(WG_CELL) * WG_CELL,
            lo.z.div_euclid(WG_CELL) * WG_CELL,
        );
        let dims = IVec3::new(
            (hi.x - cell_lo.x).div_euclid(WG_CELL) + 1,
            (hi.y - cell_lo.y).div_euclid(WG_CELL) + 1,
            (hi.z - cell_lo.z).div_euclid(WG_CELL) + 1,
        );
        let count = (dims.x * dims.y * dims.z) as u32;
        levels[lod as usize] = LevelParams {
            cam_brick_coord: [cam_brick.x, cam_brick.y, cam_brick.z],
            _pad_a: 0,
            cell_lo: [cell_lo.x, cell_lo.y, cell_lo.z],
            cell_offset: offset,
            cell_dims: [dims.x as u32, dims.y as u32, dims.z as u32],
            cell_count: count,
        };
        offset += count;
    }
    let max_dist = (half.max(1) as f32) * brick_span(MAX_LOD) * 3.0_f32.sqrt();
    ResidencyParams {
        levels,
        clip_half_bricks: half,
        total_cells: offset,
        hist_scale: HIST_BUCKETS as f32 / max_dist,
        _pad1: MAX_LOD + 1, // 4-S1: this u32 is `backdrop_lod` in the WGSL now — MAX_LOD+1 = backdrop OFF
        cam_world: cam,
        _pad2: 0,
    }
}

// =========================================================================================================
//  The GPU core store (§2.4) — built CPU-side from the scene's occupied keys (stands in for the per-region
//  `.vxo` core upload, G-c.4). Same FNV-1a family as `slot_table` (5-word stride [x,y,z,lod,core_index]).
// =========================================================================================================

struct CoreStore {
    table: Vec<u32>,
    table_size: u32,
    cores: Vec<u32>,
}

fn build_core_store(source: &StaticVoxSource, reg: &BlockRegistry) -> CoreStore {
    let keys: Vec<(IVec3, u32)> = source.occupied_keys().collect();
    let table_size = (keys.len() * 2).max(2).next_power_of_two() as u32;
    let mut table = vec![0u32; table_size as usize * 5];
    for slot in table.chunks_exact_mut(5) {
        slot[3] = EMPTY_LOD;
    }
    let mut cores: Vec<u32> = Vec::with_capacity(keys.len() * BRICK_VOXELS);
    let mask = table_size - 1;
    for (i, &(coord, lod)) in keys.iter().enumerate() {
        let brick = source.brick(coord, lod, reg);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    cores.push(brick.get(x, y, z).0 as u32);
                }
            }
        }
        let core_index = i as u32;
        let mut s = (brick_key_hash(coord, lod) & mask) as usize;
        while table[s * 5 + 3] != EMPTY_LOD {
            s = (s + 1) & (mask as usize);
        }
        table[s * 5] = coord.x as u32;
        table[s * 5 + 1] = coord.y as u32;
        table[s * 5 + 2] = coord.z as u32;
        table[s * 5 + 3] = lod;
        table[s * 5 + 4] = core_index;
    }
    if cores.is_empty() {
        cores.push(0);
    }
    CoreStore { table, table_size, cores }
}

// =========================================================================================================
//  The PERSISTENT GPU-driven front end. ALL state (slot_table, free-list, quarantine, the pool, the slab
//  allocators, the per-frame hashes, the change_count signal) survives across frames in this struct's buffers;
//  `frame(cam)` drives one whole readback-free round into them and returns the convergence signal.
// =========================================================================================================

struct GpuFrontEnd {
    device: wgpu::Device,
    queue: wgpu::Queue,
    half: i32,
    max_resident: u32,
    list_cap: usize,
    slot_table_size: u32,
    present_size: u32,

    // --- immutable scene inputs ---
    header_buf: wgpu::Buffer,
    entries_buf: wgpu::Buffer,
    diff_cfg_buf: wgpu::Buffer,
    pack_cfg_buf: wgpu::Buffer,
    core_table_buf: wgpu::Buffer,
    cores_buf: wgpu::Buffer,
    index_pool_base: wgpu::Buffer,
    palette_pool_base: wgpu::Buffer,

    // --- persistent residency-diff state ---
    slot_table_buf: wgpu::Buffer,
    free_ring_buf: wgpu::Buffer,
    free_ctrl_buf: wgpu::Buffer,
    quar_ring_buf: wgpu::Buffer,
    quar_ctrl_buf: wgpu::Buffer,

    // --- persistent per-frame hashes (cleared GPU-side by Pass A2) ---
    present_flag: wgpu::Buffer,
    dirty_flag: wgpu::Buffer,

    // --- persistent counts + lists (cleared GPU-side by Pass A; lists overwritten each frame) ---
    shell_count: wgpu::Buffer,
    shell_idx: wgpu::Buffer,
    cand_count: wgpu::Buffer,
    cand_list: wgpu::Buffer,
    desired_count: wgpu::Buffer,
    desired_list: wgpu::Buffer,
    enter_count: wgpu::Buffer,
    enter_list: wgpu::Buffer,
    drop_count: wgpu::Buffer,
    drop_list: wgpu::Buffer,
    drop_decision: wgpu::Buffer,
    dirty_count: wgpu::Buffer,
    dirty_list: wgpu::Buffer,
    dirty_slot: wgpu::Buffer,
    pack_count: wgpu::Buffer,
    pack_commands: wgpu::Buffer,
    aabb_count: wgpu::Buffer,
    aabb_commands: wgpu::Buffer,
    classify_commands: wgpu::Buffer,
    neighbour_indices: wgpu::Buffer,
    classify_out: wgpu::Buffer,

    // --- the GPU-written indirect dispatch buffers (self-gating) ---
    shell_dispatch: wgpu::Buffer,
    pack_dispatch: wgpu::Buffer,
    aabb_dispatch: wgpu::Buffer,
    classify_dispatch: wgpu::Buffer,

    // --- the persistent POOL + slab allocators ---
    meta_buf: wgpu::Buffer,
    voxel_buf: wgpu::Buffer,
    palette_buf: wgpu::Buffer,
    index_slab_ctrl: wgpu::Buffer,
    index_slab_free: wgpu::Buffer,
    palette_slab_ctrl: wgpu::Buffer,
    palette_slab_free: wgpu::Buffer,
    slab_state: wgpu::Buffer,
    enter_hist: wgpu::Buffer,
    enter_cap: wgpu::Buffer,

    // --- the change_count signal + its mappable staging mirror (G-c.4 reads this out-of-band) ---
    change_count_buf: wgpu::Buffer,
    change_staging: wgpu::Buffer,

    // dummies for the comprehensive bind group.
    dummy_in: wgpu::Buffer,
    dummy_out: wgpu::Buffer,
    dummy_dispatch: wgpu::Buffer,

    // pipelines (residency Pass A/A2/B/C/D + change-count) and the pack shader (classify/pack/write_aabb).
    res_bgl: wgpu::BindGroupLayout,
    p_seed: wgpu::ComputePipeline,
    p_release: wgpu::ComputePipeline,
    p_clear: wgpu::ComputePipeline,
    p_b0: wgpu::ComputePipeline,
    p_b: wgpu::ComputePipeline,
    p_present: wgpu::ComputePipeline,
    p_mark: wgpu::ComputePipeline,
    p_apply: wgpu::ComputePipeline,
    p_cap_hist: wgpu::ComputePipeline,
    p_cap_compute: wgpu::ComputePipeline,
    p_enter: wgpu::ComputePipeline,
    p_chg: wgpu::ComputePipeline,
    p_d_dirty: wgpu::ComputePipeline,
    p_d_nbr: wgpu::ComputePipeline,
    p_d_cmd: wgpu::ComputePipeline,
    p_d_drops: wgpu::ComputePipeline,
    p_classify: wgpu::ComputePipeline,
    p_pack: wgpu::ComputePipeline,
    p_aabb: wgpu::ComputePipeline,
    cls_bg: wgpu::BindGroup,
    pack_bg: wgpu::BindGroup,
    aabb_bg: wgpu::BindGroup,
}

/// A decoded snapshot of the persistent pool + the resident key→slot map (read back for the parity comparators).
struct PoolReadback {
    metas: Vec<GpuBrickMeta>,
    voxel: Vec<u32>,
    brick_palettes: Vec<u32>,
    slot_of: FxHashMap<(IVec3, u32), u32>,
}

impl GpuFrontEnd {
    #[allow(clippy::too_many_lines)]
    fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        occ: &SectorOccupancy,
        cores: &CoreStore,
        half: i32,
        max_resident: u32,
        list_cap: usize,
    ) -> Self {
        let slot_table_size = (max_resident as usize * 2).max(2).next_power_of_two() as u32;
        let present_size = (list_cap * 2).max(2).next_power_of_two() as u32;

        // immutable inputs.
        let header_buf = buf_init(&device, "header", bytemuck::bytes_of(&occ.header()), wgpu::BufferUsages::UNIFORM);
        let entries_buf = buf_init(&device, "entries", bytemuck::cast_slice(occ.entries()), wgpu::BufferUsages::STORAGE);
        let diff_cfg = DiffConfig { slot_table_size, present_size, max_resident, refine_descent_cap: REFINE_DESCENT_CAP };
        let diff_cfg_buf = buf_init(&device, "diff_cfg", bytemuck::bytes_of(&diff_cfg), wgpu::BufferUsages::UNIFORM);
        let pack_cfg = PackConfig { core_table_size: cores.table_size, max_resident, _pad0: 0, _pad1: 0 };
        let pack_cfg_buf = buf_init(&device, "pack_cfg", bytemuck::bytes_of(&pack_cfg), wgpu::BufferUsages::UNIFORM);
        let core_table_buf = buf_init(&device, "core_table", bytemuck::cast_slice(&cores.table), wgpu::BufferUsages::STORAGE);
        let cores_buf = buf_init(&device, "cores", bytemuck::cast_slice(&cores.cores), wgpu::BufferUsages::STORAGE);
        let index_pool_base = buf_init(&device, "index_pool_base", bytemuck::cast_slice(&[0u32]), wgpu::BufferUsages::STORAGE);
        let palette_pool_base = buf_init(&device, "palette_pool_base", bytemuck::cast_slice(&[0u32]), wgpu::BufferUsages::STORAGE);

        // persistent diff state.
        let mut slot_init = vec![0u32; slot_table_size as usize * 5];
        for s in slot_init.chunks_exact_mut(5) {
            s[3] = EMPTY_LOD;
        }
        let slot_table_buf = buf_init(&device, "slot_table", bytemuck::cast_slice(&slot_init), storage_usage());
        let free_init: Vec<u32> = (0..max_resident).collect();
        let free_ring_buf = buf_init(&device, "free_ring", bytemuck::cast_slice(&free_init), storage_usage());
        let free_ctrl_buf = buf_init(&device, "free_ctrl", bytemuck::cast_slice(&[0u32, max_resident]), storage_usage());
        let quar_ring_buf = storage_buf(&device, "quar_ring", (max_resident as u64) * 4);
        let quar_ctrl_buf = buf_init(&device, "quar_ctrl", bytemuck::cast_slice(&[0u32, 0u32]), storage_usage());

        // persistent per-frame hashes (initialized EMPTY; Pass A2 re-clears them GPU-side each frame).
        let present_init = vec![EMPTY_LOD; present_size as usize * 4];
        let present_flag = buf_init(&device, "present_flag", bytemuck::cast_slice(&present_init), storage_usage());
        let dirty_init = vec![EMPTY_LOD; slot_table_size as usize * 4];
        let dirty_flag = buf_init(&device, "dirty_flag", bytemuck::cast_slice(&dirty_init), storage_usage());

        // persistent counts + lists.
        let mk0 = |label: &str| buf_init(&device, label, bytemuck::bytes_of(&0u32), storage_usage());
        let shell_count = mk0("shell_count");
        let total_cells_max = list_cap; // shell_idx is sized to the list cap (>= total_cells for the test scenes)
        let shell_idx = storage_buf(&device, "shell_idx", (total_cells_max * 4) as u64);
        let cand_count = mk0("cand_count");
        let cand_list = storage_buf(&device, "cand_list", (list_cap * 16) as u64);
        let desired_count = mk0("desired_count");
        let desired_list = storage_buf(&device, "desired_list", (list_cap * 16) as u64);
        let enter_count = mk0("enter_count");
        let enter_list = storage_buf(&device, "enter_list", (list_cap * 16) as u64);
        let drop_count = mk0("drop_count");
        let drop_list = storage_buf(&device, "drop_list", (list_cap * 16) as u64);
        let drop_decision = storage_buf(&device, "drop_decision", (slot_table_size as u64) * 4);
        let dirty_count = mk0("dirty_count");
        let dirty_list = storage_buf(&device, "dirty_list", (list_cap * 16) as u64);
        let dirty_slot = storage_buf(&device, "dirty_slot", (list_cap * 4) as u64);
        let pack_count = mk0("pack_count");
        let pack_commands = storage_buf(&device, "pack_commands", (list_cap * 60) as u64);
        let aabb_count = mk0("aabb_count");
        let aabb_commands = storage_buf(&device, "aabb_commands", (list_cap * 32) as u64);
        let classify_commands = storage_buf(&device, "classify_commands", (list_cap * 16) as u64);
        let neighbour_indices = storage_buf(&device, "neighbour_indices", (list_cap as u64) * 27 * 4);
        let classify_out = storage_buf(&device, "classify_out", (list_cap * 4 * 4) as u64);

        // indirect dispatch buffers — seeded (0,1,1); Pass A re-seeds them GPU-side each frame.
        let shell_dispatch = buf_init(&device, "shell_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let pack_dispatch = buf_init(&device, "pack_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let aabb_dispatch = buf_init(&device, "aabb_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let classify_dispatch = buf_init(&device, "classify_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());

        // the persistent POOL (meta/voxel/palette/aabb) + slab allocators.
        let meta_words = max_resident as usize * META_WORDS;
        let index_pool_words = (max_resident as usize * 192).max(512);
        let palette_pool_words = (max_resident as usize * 16).max(64);
        let meta_buf = storage_buf(&device, "meta_buf", (meta_words * 4) as u64);
        let voxel_buf = storage_buf(&device, "voxel_buf", (index_pool_words * 4) as u64);
        let palette_buf = storage_buf(&device, "palette_buf", (palette_pool_words * 4) as u64);
        let degenerate = adventure::voxel::incremental::degenerate_aabb();
        let mut aabb_host = vec![0u32; max_resident as usize * 8];
        for slot in 0..max_resident as usize {
            aabb_host[slot * 8..slot * 8 + 8]
                .copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&degenerate)));
        }
        let aabb_buf = buf_init(&device, "aabb_buf", bytemuck::cast_slice(&aabb_host), storage_usage());
        let index_ctrl_init = vec![0u32; 1 + 5 * 2];
        let index_slab_ctrl = buf_init(&device, "index_slab_ctrl", bytemuck::cast_slice(&index_ctrl_init), storage_usage());
        let index_slab_free = storage_buf(&device, "index_slab_free", (5 * max_resident as u64) * 4);
        let palette_ctrl_init = vec![0u32; 1 + 16 * 2];
        let palette_slab_ctrl = buf_init(&device, "palette_slab_ctrl", bytemuck::cast_slice(&palette_ctrl_init), storage_usage());
        let palette_slab_free = storage_buf(&device, "palette_slab_free", (16 * max_resident as u64) * 4);
        // G-c.4: the GPU DenseSlot table (slab reuse/free on re-pack) + the enter-cap histogram/cut.
        let slab_state = storage_buf(&device, "slab_state", (max_resident as u64) * 4 * 4);
        let enter_hist = storage_buf(&device, "enter_hist", (HIST_BUCKETS as u64) * 4);
        let enter_cap = buf_init(&device, "enter_cap", bytemuck::cast_slice(&[HIST_BUCKETS, 0u32]), storage_usage());

        // the change_count signal + mappable staging mirror (G-c.4's out-of-band read).
        let change_count_buf = buf_init(&device, "change_count", bytemuck::bytes_of(&0u32), storage_usage());
        let change_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("change_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let dummy_in = buf_init(&device, "dummy_in", bytemuck::cast_slice(&[0u32; 4]), wgpu::BufferUsages::STORAGE);
        let dummy_out = storage_buf(&device, "dummy_out", 16);
        let dummy_dispatch = storage_buf(&device, "dummy_dispatch", 16);

        // shaders.
        let res_src = std::fs::read_to_string("assets/shaders/voxel_residency.wgsl").expect("read residency");
        let res_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("residency"),
            source: wgpu::ShaderSource::Wgsl(res_src.into()),
        });
        let pack_src = std::fs::read_to_string("assets/shaders/voxel_pack.wgsl").expect("read pack");
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pack"),
            source: wgpu::ShaderSource::Wgsl(pack_src.into()),
        });

        // the comprehensive residency bind-group layout (0..=48). A pipeline references only its subset.
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ];
        for b in 5..=11 {
            entries.push(storage_entry(b, false));
        }
        entries.push(uniform_entry(12));
        for b in 13..=23 {
            entries.push(storage_entry(b, false));
        }
        entries.push(uniform_entry(24));
        entries.push(storage_entry(25, true));
        entries.push(storage_entry(26, true));
        for b in 27..=44 {
            entries.push(storage_entry(b, false));
        }
        entries.push(storage_entry(45, true));
        entries.push(storage_entry(46, true));
        entries.push(storage_entry(47, true));
        entries.push(storage_entry(48, false));
        entries.push(storage_entry(49, false)); // slab_state
        entries.push(storage_entry(50, false)); // enter_hist
        entries.push(storage_entry(51, false)); // enter_cap
        let res_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("res_bgl"),
            entries: &entries,
        });
        let res_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("res_pl"),
            bind_group_layouts: &[Some(&res_bgl)],
            immediate_size: 0,
        });
        let mk_res = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&res_pl),
                module: &res_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p_seed = mk_res("seed_frame");
        let p_release = mk_res("diff_release_quarantine");
        let p_clear = mk_res("clear_per_frame_hashes");
        let p_b0 = mk_res("prepare_shell_dispatch");
        let p_b = mk_res("enumerate_shells");
        let p_present = mk_res("build_present_flag");
        let p_mark = mk_res("diff_drop_mark");
        let p_apply = mk_res("diff_drop_apply");
        let p_cap_hist = mk_res("enter_cap_histogram");
        let p_cap_compute = mk_res("enter_cap_compute");
        let p_enter = mk_res("diff_enter_scan");
        let p_chg = mk_res("write_change_count");
        let p_d_dirty = mk_res("pack_build_dirty");
        let p_d_nbr = mk_res("pack_build_neighbours");
        let p_d_cmd = mk_res("pack_build_commands");
        let p_d_drops = mk_res("pack_build_drops");

        // pack shader pipelines + their (fixed, content-independent) bind groups.
        let cls_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cls_bgl"),
            entries: &[storage_entry(1, true), storage_entry(2, true), storage_entry(8, false), storage_entry(9, true)],
        });
        let cls_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cls_pl"),
            bind_group_layouts: &[Some(&cls_bgl)],
            immediate_size: 0,
        });
        let p_classify = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("classify_brick"),
            layout: Some(&cls_pl),
            module: &pack_module,
            entry_point: Some("classify_brick"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cls_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cls_bg"),
            layout: &cls_bgl,
            entries: &[bind(1, &cores_buf), bind(2, &neighbour_indices), bind(8, &classify_out), bind(9, &classify_commands)],
        });

        let pack_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pack_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(12, true), // pack_cmd_count (2D-dispatch over-run guard)
            ],
        });
        let pack_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pack_pl"),
            bind_group_layouts: &[Some(&pack_bgl)],
            immediate_size: 0,
        });
        let p_pack = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pack_brick"),
            layout: Some(&pack_pl),
            module: &pack_module,
            entry_point: Some("pack_brick"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pack_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pack_bg"),
            layout: &pack_bgl,
            entries: &[
                bind(0, &pack_commands),
                bind(1, &cores_buf),
                bind(2, &neighbour_indices),
                bind(3, &voxel_buf),
                bind(4, &palette_buf),
                bind(5, &meta_buf),
                bind(12, &pack_count), // pack_brick gates on this exact count (the 2D-dispatch over-run guard)
            ],
        });

        let aabb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aabb_bgl"),
            entries: &[storage_entry(6, false), storage_entry(7, true)],
        });
        let aabb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aabb_pl"),
            bind_group_layouts: &[Some(&aabb_bgl)],
            immediate_size: 0,
        });
        let p_aabb = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("write_aabb"),
            layout: Some(&aabb_pl),
            module: &pack_module,
            entry_point: Some("write_aabb"),
            compilation_options: Default::default(),
            cache: None,
        });
        let aabb_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aabb_bg"),
            layout: &aabb_bgl,
            entries: &[bind(6, &aabb_buf), bind(7, &aabb_commands)],
        });

        Self {
            device,
            queue,
            half,
            max_resident,
            list_cap,
            slot_table_size,
            present_size,
            header_buf,
            entries_buf,
            diff_cfg_buf,
            pack_cfg_buf,
            core_table_buf,
            cores_buf,
            index_pool_base,
            palette_pool_base,
            slot_table_buf,
            free_ring_buf,
            free_ctrl_buf,
            quar_ring_buf,
            quar_ctrl_buf,
            present_flag,
            dirty_flag,
            shell_count,
            shell_idx,
            cand_count,
            cand_list,
            desired_count,
            desired_list,
            enter_count,
            enter_list,
            drop_count,
            drop_list,
            drop_decision,
            dirty_count,
            dirty_list,
            dirty_slot,
            pack_count,
            pack_commands,
            aabb_count,
            aabb_commands,
            classify_commands,
            neighbour_indices,
            classify_out,
            shell_dispatch,
            pack_dispatch,
            aabb_dispatch,
            classify_dispatch,
            meta_buf,
            voxel_buf,
            palette_buf,
            index_slab_ctrl,
            index_slab_free,
            palette_slab_ctrl,
            palette_slab_free,
            slab_state,
            enter_hist,
            enter_cap,
            change_count_buf,
            change_staging,
            dummy_in,
            dummy_out,
            dummy_dispatch,
            res_bgl,
            p_seed,
            p_release,
            p_clear,
            p_b0,
            p_b,
            p_present,
            p_mark,
            p_apply,
            p_cap_hist,
            p_cap_compute,
            p_enter,
            p_chg,
            p_d_dirty,
            p_d_nbr,
            p_d_cmd,
            p_d_drops,
            p_classify,
            p_pack,
            p_aabb,
            cls_bg,
            pack_bg,
            aabb_bg,
        }
    }

    /// Build a comprehensive residency bind group binding `slot7` at binding 7 (so Pass B0 can bind the real
    /// `shell_dispatch` while every other pass binds a dummy — wgpu forbids STORAGE+INDIRECT in one dispatch).
    fn res_bind_group(&self, params_buf: &wgpu::Buffer, slot7: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_bg"),
            layout: &self.res_bgl,
            entries: &[
                bind(0, &self.header_buf),
                bind(1, &self.entries_buf),
                bind(2, &self.dummy_in),
                bind(3, &self.dummy_out),
                bind(4, params_buf),
                bind(5, &self.shell_idx),
                bind(6, &self.shell_count),
                bind(7, slot7),
                bind(8, &self.cand_count),
                bind(9, &self.cand_list),
                bind(10, &self.desired_count),
                bind(11, &self.desired_list),
                bind(12, &self.diff_cfg_buf),
                bind(13, &self.slot_table_buf),
                bind(14, &self.free_ring_buf),
                bind(15, &self.free_ctrl_buf),
                bind(16, &self.quar_ring_buf),
                bind(17, &self.quar_ctrl_buf),
                bind(18, &self.present_flag),
                bind(19, &self.enter_count),
                bind(20, &self.enter_list),
                bind(21, &self.drop_count),
                bind(22, &self.drop_list),
                bind(23, &self.drop_decision),
                bind(24, &self.pack_cfg_buf),
                bind(25, &self.core_table_buf),
                bind(26, &self.cores_buf),
                bind(27, &self.dirty_count),
                bind(28, &self.dirty_list),
                bind(29, &self.dirty_slot),
                bind(30, &self.dirty_flag),
                bind(31, &self.pack_count),
                bind(32, &self.pack_commands),
                bind(33, &self.aabb_count),
                bind(34, &self.aabb_commands),
                bind(35, &self.classify_commands),
                bind(36, &self.neighbour_indices),
                bind(37, &self.meta_buf),
                bind(38, &self.pack_dispatch),
                bind(39, &self.aabb_dispatch),
                bind(40, &self.classify_dispatch),
                bind(41, &self.index_slab_ctrl),
                bind(42, &self.index_slab_free),
                bind(43, &self.palette_slab_ctrl),
                bind(44, &self.palette_slab_free),
                bind(45, &self.index_pool_base),
                bind(46, &self.palette_pool_base),
                bind(47, &self.classify_out),
                bind(48, &self.change_count_buf),
                bind(49, &self.slab_state),
                bind(50, &self.enter_hist),
                bind(51, &self.enter_cap),
            ],
        })
    }

    /// Drive ONE whole readback-free frame for `cam` into the persistent buffers: Pass A → A2 → B0 → B(indirect)
    /// → C0 → C2a → C2b → C1 → write_change_count → D1 → D0 → D2 → classify(indirect) → D3 → pack(indirect) →
    /// write_aabb(indirect). The pack TAIL is recorded as `record_indirect` over the GPU-WRITTEN dispatch
    /// buffers — NO host readback of any count sizes any tail dispatch (the headline self-gating). Returns
    /// `change_count` (read out-of-band from the staging mirror — the one permitted CPU↔GPU sync, G-c.4's signal).
    fn frame(&self, cam: [f32; 3]) -> u32 {
        let device = &self.device;
        let queue = &self.queue;
        let params = build_params(cam, self.half);
        assert!(
            params.total_cells as usize <= self.list_cap,
            "total_cells {} exceeds shell_idx cap {} — raise list_cap",
            params.total_cells,
            self.list_cap
        );
        let params_buf = buf_init(device, "params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);
        let bg = self.res_bind_group(&params_buf, &self.dummy_dispatch);
        let bg_b0 = self.res_bind_group(&params_buf, &self.shell_dispatch);

        let clear_wgs = self.slot_table_size.max(self.present_size).div_ceil(64).max(1);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        // Pass A0 — clear the enumerate/pack counts + SEED the indirect dispatches (0,1,1) (self-gating).
        compute(&mut enc, &self.p_seed, &bg, 1);
        // Pass A — drain the previous frame's quarantine + clear the diff enter/drop counts.
        compute(&mut enc, &self.p_release, &bg, 1);
        // Pass A2 — clear the per-frame hashes + change_count.
        compute(&mut enc, &self.p_clear, &bg, clear_wgs);
        // Pass B0 — shell dispatch prep (binds real shell_dispatch at 7).
        compute(&mut enc, &self.p_b0, &bg_b0, params.total_cells.div_ceil(64).max(1));
        // Pass B — enumerate (INDIRECT over the GPU-written shell_dispatch).
        record_indirect(&mut enc, &self.p_b, &bg, &self.shell_dispatch);
        // Pass C — present-flag, drop-mark, drop-apply, enter (sized by GPU-bounded dispatches: the present/enter
        // sizes are over the LIST CAP / slot-table size, both known WITHOUT a count readback).
        compute(&mut enc, &self.p_present, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        compute(&mut enc, &self.p_mark, &bg, self.slot_table_size.div_ceil(64).max(1));
        compute(&mut enc, &self.p_apply, &bg, self.slot_table_size.div_ceil(64).max(1));
        // Enter-cap (nearest-priority): histogram → cut → capped enter.
        compute(&mut enc, &self.p_cap_hist, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        compute(&mut enc, &self.p_cap_compute, &bg, 1);
        compute(&mut enc, &self.p_enter, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        // publish change_count (= enter + drop).
        compute(&mut enc, &self.p_chg, &bg, 1);
        // Pass D — dirty build (over the whole list cap; the pass early-outs past enter+drop), drops, neighbours,
        // then classify/pack/write_aabb INDIRECT over the GPU-written dispatch buffers (the self-gating tail).
        compute(&mut enc, &self.p_d_dirty, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        compute(&mut enc, &self.p_d_drops, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        compute(&mut enc, &self.p_d_nbr, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        record_indirect(&mut enc, &self.p_classify, &self.cls_bg, &self.classify_dispatch);
        compute(&mut enc, &self.p_d_cmd, &bg, (self.list_cap as u32).div_ceil(64).max(1));
        record_indirect(&mut enc, &self.p_pack, &self.pack_bg, &self.pack_dispatch);
        record_indirect(&mut enc, &self.p_aabb, &self.aabb_bg, &self.aabb_dispatch);
        // Copy the change_count signal into the mappable staging mirror (G-c.4's out-of-band read).
        enc.copy_buffer_to_buffer(&self.change_count_buf, 0, &self.change_staging, 0, 4);
        queue.submit(std::iter::once(enc.finish()));

        // The out-of-band change_count read (the ONE permitted CPU↔GPU sync — the G-c.4 mirror signal).
        self.change_staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = self.change_staging.slice(..).get_mapped_range().expect("map").to_vec();
        let change = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.change_staging.unmap();
        change
    }

    /// Iterate whole frames at a fixed `cam` until `change_count == 0` (converged). Returns the frame count.
    fn converge(&self, cam: [f32; 3]) -> u32 {
        let mut frames = 0u32;
        loop {
            let change = self.frame(cam);
            frames += 1;
            if change == 0 {
                break;
            }
            assert!(frames < 64, "GPU front end failed to converge for cam {cam:?} in 64 frames");
        }
        frames
    }

    /// Read back the GPU-written indirect dispatch X-counts (classify/pack/aabb) — used ONLY by the gate to
    /// ASSERT self-gating (the live path never reads them; `record_indirect` consumes them GPU-side).
    fn tail_dispatch_counts(&self) -> (u32, u32, u32) {
        let c = readback_u32(&self.device, &self.queue, &self.classify_dispatch, 1)[0];
        let p = readback_u32(&self.device, &self.queue, &self.pack_dispatch, 1)[0];
        let a = readback_u32(&self.device, &self.queue, &self.aabb_dispatch, 1)[0];
        (c, p, a)
    }

    /// The resident `(coord,lod)` key SET from the persistent slot_table.
    fn resident_set(&self) -> HashSet<(IVec3, u32)> {
        let words = readback_u32(&self.device, &self.queue, &self.slot_table_buf, self.slot_table_size as usize * 5);
        let mut set = HashSet::new();
        for s in words.chunks_exact(5) {
            if s[3] == EMPTY_LOD {
                continue;
            }
            let coord = IVec3::new(s[0] as i32, s[1] as i32, s[2] as i32);
            assert!(set.insert((coord, s[3])), "duplicate key in slot_table");
        }
        set
    }

    /// Read back the persistent pool (meta/voxel/palette) + the resident key→slot map for the parity comparators.
    fn pool(&self) -> PoolReadback {
        let meta_words = self.max_resident as usize * META_WORDS;
        let index_pool_words = (self.max_resident as usize * 192).max(512);
        let palette_pool_words = (self.max_resident as usize * 16).max(64);
        let meta_raw = readback_u32(&self.device, &self.queue, &self.meta_buf, meta_words);
        let voxel = readback_u32(&self.device, &self.queue, &self.voxel_buf, index_pool_words);
        let brick_palettes = readback_u32(&self.device, &self.queue, &self.palette_buf, palette_pool_words);
        let slot_words = readback_u32(&self.device, &self.queue, &self.slot_table_buf, self.slot_table_size as usize * 5);
        let metas: Vec<GpuBrickMeta> = bytemuck::cast_slice(&meta_raw).to_vec();
        let mut slot_of = FxHashMap::default();
        for s in slot_words.chunks_exact(5) {
            if s[3] == EMPTY_LOD {
                continue;
            }
            let coord = IVec3::new(s[0] as i32, s[1] as i32, s[2] as i32);
            slot_of.insert((coord, s[3]), s[4]);
        }
        PoolReadback { metas, voxel, brick_palettes, slot_of }
    }
}

/// Record an INDIRECT compute dispatch over a GPU-written `(x,1,1)` dispatch buffer — the self-gating primitive.
fn record_indirect(
    enc: &mut wgpu::CommandEncoder,
    p: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    indirect: &wgpu::Buffer,
) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(p);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups_indirect(indirect, 0);
}

// =========================================================================================================
//  CPU oracle — the converged ResidentPacker pool for the GPU resident key set (per-key content + ray-hit).
// =========================================================================================================

fn canonical_patch(metas: &[GpuBrickMeta], voxels: &[u32], palettes: &[u32]) -> GpuBrickPatch {
    let zero = GpuBrickMeta::zeroed();
    let mut kept: Vec<GpuBrickMeta> = metas.iter().copied().filter(|m| *m != zero).collect();
    kept.sort_by(|a, b| {
        (a.lod(), a.world_min[0].to_bits(), a.world_min[1].to_bits(), a.world_min[2].to_bits())
            .cmp(&(b.lod(), b.world_min[0].to_bits(), b.world_min[1].to_bits(), b.world_min[2].to_bits()))
    });
    GpuBrickPatch {
        aabbs: Vec::new(),
        metas: kept,
        voxels: voxels.to_vec(),
        brick_palettes: palettes.to_vec(),
        palette: Vec::new(),
        lights: Vec::new(),
        alias: Vec::new(),
    }
}

fn cpu_oracle_patch(entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> GpuBrickPatch {
    let mut packer = ResidentPacker::new(entries.len().max(1) as u32 * 2);
    packer.update(entries, reg.len() as u32);
    let snap = packer.snapshot_buffers(reg);
    GpuBrickPatch {
        aabbs: snap.aabbs.clone(),
        metas: snap.metas.clone(),
        voxels: snap.indices.clone(),
        brick_palettes: snap.brick_palettes.clone(),
        palette: snap.palette.clone(),
        lights: Vec::new(),
        alias: Vec::new(),
    }
}

// =========================================================================================================
//  CPU reference — drive ResidencyManager to convergence; expose the resident key set (mirror of the diff gate).
// =========================================================================================================

fn cpu_converge(
    mgr: &mut ResidencyManager,
    cam: [f32; 3],
    cfg: &StreamingConfig,
    source: &StaticVoxSource,
    registry: &BlockRegistry,
) -> HashSet<(IVec3, u32)> {
    let edits = VoxelEdits::new();
    for _ in 0..64 {
        mgr.update(cam, cfg, source);
        while mgr.pending() > 0 {
            mgr.drain_work_from(cfg, source, registry, &edits);
        }
        let before = mgr.resident_count();
        let dropped = mgr.update(cam, cfg, source);
        while mgr.pending() > 0 {
            mgr.drain_work_from(cfg, source, registry, &edits);
        }
        if dropped == 0 && mgr.resident_count() == before {
            break;
        }
    }
    mgr.resident_entries().into_iter().map(|e| (e.coord, e.lod)).collect()
}

// =========================================================================================================
//  Scene + registry (the same multi-material shape family the pack-parity gate uses).
// =========================================================================================================

fn registry() -> BlockRegistry {
    use adventure::sdf_render::worldgen::biome::{
        BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
    };
    let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
        name: name.into(),
        base_color: c,
        roughness: 0.9,
        blend: 0.0,
        texture: None,
        tiling: 4.0,
        ..Default::default()
    };
    let materials = vec![
        mat("a", [0.1, 0.2, 0.3, 1.0]),
        mat("b", [0.4, 0.5, 0.6, 1.0]),
        mat("c", [0.7, 0.8, 0.9, 1.0]),
        mat("d", [0.2, 0.9, 0.1, 1.0]),
        mat("e", [0.9, 0.1, 0.5, 1.0]),
    ];
    let biomes = BiomeId::ALL
        .iter()
        .map(|_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1.0 }],
            bedrock: TerrainMatId(1),
        })
        .collect();
    BlockRegistry::from_biome_library(&BiomeLibrary { materials, biomes })
}

fn scene() -> BrickMap {
    let mut map = BrickMap::new();
    let multi = |seed: i32| {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let idx = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                    let h = (x * 7 + y * 13 + z * 5 + seed).rem_euclid(6);
                    v[idx] = if h < 2 { BlockId::AIR } else { BlockId(1 + (h - 2) as u16) };
                }
            }
        }
        Brick::from_voxels(v)
    };
    let solid = |id: u16| {
        let mut v = Box::new([BlockId::AIR; BRICK_VOXELS]);
        v.iter_mut().for_each(|c| *c = BlockId(id));
        Brick::from_voxels(v)
    };
    for z in -2..2 {
        for x in -2..2 {
            for y in 0..3 {
                map.insert(IVec3::new(x, y, z), multi(x * 11 + y * 7 + z * 3));
            }
        }
    }
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                map.insert(IVec3::new(8 + x, 1 + y, 8 + z), solid(1));
            }
        }
    }
    // A tall pillar threading the LOD shells (so a recede/approach actually coarsens/refines).
    for y in 3..12 {
        map.insert(IVec3::new(0, y, 0), solid(2));
    }
    map.insert(IVec3::new(14, 0, 14), multi(42));
    map
}

// =========================================================================================================
//  Shared driver: build the persistent GPU front end + the CPU reference for the scene.
// =========================================================================================================

struct Harness {
    gpu: GpuFrontEnd,
    source: StaticVoxSource,
    reg: BlockRegistry,
    cfg: StreamingConfig,
}

fn build_harness(device: wgpu::Device, queue: wgpu::Queue) -> Harness {
    // StaticVoxSource owns its pyramid (no borrow of the map) — it is 'static, so the map can be dropped here.
    let map = scene();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());
    let reg = registry();
    let cores = build_core_store(&source, &reg);
    let half = 8i32;
    let max_resident = 8192u32;
    let list_cap = 200_000usize;
    let gpu = GpuFrontEnd::new(device, queue, &occ, &cores, half, max_resident, list_cap);
    let cfg = StreamingConfig {
        clip_half_bricks: half,
        max_resident_bricks: usize::MAX,
        max_bricks_per_frame: usize::MAX,
    };
    Harness { gpu, source, reg, cfg }
}

/// Build the CPU oracle pool patch for the GPU resident key set (a fresh ResidentPacker cold fill over the SAME
/// set), and the GPU pool patch — both canonicalized so the faithful trace's tiebreak is order-independent.
fn parity_patches(h: &Harness, pool: &PoolReadback) -> (GpuBrickPatch, GpuBrickPatch, Vec<ResidentKey>) {
    let resident_keys: Vec<(IVec3, u32)> = pool.slot_of.keys().copied().collect();
    let bricks: Vec<(IVec3, u32, Brick)> =
        resident_keys.iter().map(|&(c, l)| (c, l, h.source.brick(c, l, &h.reg))).collect();
    let entries: Vec<ResidentBrick> =
        bricks.iter().map(|(c, l, b)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    let oracle = cpu_oracle_patch(&entries, &h.reg);
    let gpu_patch = canonical_patch(&pool.metas, &pool.voxel, &pool.brick_palettes);
    let oracle_patch = canonical_patch(&oracle.metas, &oracle.voxels, &oracle.brick_palettes);
    let keys = bricks.into_iter().map(|(c, l, b)| ResidentKey { coord: c, lod: l, brick: b }).collect();
    (gpu_patch, oracle_patch, keys)
}

struct ResidentKey {
    coord: IVec3,
    lod: u32,
    brick: Brick,
}

/// Assert the GPU pool's per-KEY content (decoded via the SSOT `cell_block`) EQUALS the CPU `pack_one` SSOT for
/// every resident key. Returns (checked, dense_seen). Mirror of the pack-parity per-key gate.
fn assert_per_key_content(pool: &PoolReadback, keys: &[ResidentKey]) -> (usize, usize) {
    let entries: Vec<ResidentBrick> =
        keys.iter().map(|k| ResidentBrick { coord: k.coord, brick: &k.brick, lod: k.lod }).collect();
    let by_key = build_by_key(&entries);
    let gpu_patch = GpuBrickPatch {
        aabbs: Vec::new(),
        metas: pool.metas.clone(),
        voxels: pool.voxel.clone(),
        brick_palettes: pool.brick_palettes.clone(),
        palette: Vec::new(),
        lights: Vec::new(),
        alias: Vec::new(),
    };
    let mut checked = 0usize;
    let mut dense_seen = 0usize;
    for e in &entries {
        let key = (e.coord, e.lod);
        let slot = *pool.slot_of.get(&key).expect("resident key has a GPU slot");
        let gm = &gpu_patch.metas[slot as usize];
        let pb = pack_one(e, &by_key);
        let cpu_cells: Vec<u32> = match &pb.voxels {
            adventure::voxel::gpu::BrickVoxels::Uniform(b) => vec![b.0 as u32; halo_cells(e.lod)],
            adventure::voxel::gpu::BrickVoxels::Dense(c) => c.clone(),
        };
        if matches!(pb.voxels, adventure::voxel::gpu::BrickVoxels::Dense(_)) {
            dense_seen += 1;
        }
        for (cell, &want) in cpu_cells.iter().enumerate().take(halo_cells(e.lod)) {
            let gpu_block = gpu_patch.cell_block(gm, cell).0 as u32;
            assert_eq!(
                gpu_block, want,
                "key {key:?} (slot {slot}) cell {cell}: GPU pool decoded {gpu_block} but CPU SSOT is {want}"
            );
        }
        checked += 1;
    }
    (checked, dense_seen)
}

/// Assert the GPU pool and the CPU oracle pool yield IDENTICAL first-hits over a ray grid (block/pos/normal).
fn assert_ray_hits(gpu_patch: &GpuBrickPatch, oracle_patch: &GpuBrickPatch, span0: f32) -> usize {
    let centre = Vec3::new(0.0, 1.0 * span0, 0.0);
    let dirs = [
        Vec3::new(0.3, -1.0, 0.2),
        Vec3::new(-0.4, -0.8, 0.5),
        Vec3::new(0.6, -0.5, -0.3),
        Vec3::new(0.1, -1.0, 0.0),
        Vec3::new(-0.7, -0.6, -0.6),
    ];
    let mut gpu_hits = 0usize;
    for du in -6..=6 {
        for dv in -6..=6 {
            let origin = centre + Vec3::new(du as f32 * span0 * 0.5, 6.0 * span0, dv as f32 * span0 * 0.5);
            for dir in dirs {
                let rd = dir.normalize();
                let gpu_hit = dda_oracle::trace_faithful(gpu_patch, origin, rd, 1e-4);
                let cpu_hit = dda_oracle::trace_faithful(oracle_patch, origin, rd, 1e-4);
                match (gpu_hit, cpu_hit) {
                    (None, None) => {}
                    (Some(g), Some(c)) => {
                        gpu_hits += 1;
                        assert_eq!(g.best.block_id, c.best.block_id, "ray {origin:?}->{rd:?}: block mismatch");
                        assert!((g.best.hit_t - c.best.hit_t).abs() < 1e-3, "ray {origin:?}->{rd:?}: t mismatch");
                        assert_eq!(g.normal, c.normal, "ray {origin:?}->{rd:?}: normal mismatch");
                    }
                    (g, c) => panic!("ray {origin:?}->{rd:?}: one missed (gpu={} cpu={})", g.is_some(), c.is_some()),
                }
            }
        }
    }
    gpu_hits
}

// =========================================================================================================
//  THE GATES.
// =========================================================================================================

/// **(a) STATIC CONVERGENCE + INDIRECT SELF-GATING.** Hold a FIXED camera. Assert `change_count` reaches 0 within
/// a bounded number of frames and STAYS 0 over several further frames (idempotent — no churn). On a converged
/// frame, assert the GPU-written `classify/pack/aabb` indirect dispatch X-counts are all 0 (so the tail launches
/// 0 workgroups — the headline self-gating, with NO readback driving the tail).
#[test]
#[ignore = "stale oracle: compares the GPU front-end pool cell-for-cell vs CPU pack_one, which diverges on the \
            NEIGHBOUR_SOLID/boundary halo. The GPU front end (pager cores + GPU halo-fill) is the SSOT post the \
            halo fixes (see voxel_gpu_residency_pack_parity's corrected per-key oracle + the live \
            voxel_paged_front_end_render). Re-enable after porting a GPU-halo-aware oracle."]
fn static_camera_converges_and_self_gates() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter (or compute/storage limits too low) — static convergence skipped");
        return;
    };
    let h = build_harness(device, queue);
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0];

    // Drive frames until change_count == 0.
    let mut converge_frame = None;
    let mut changes = Vec::new();
    for f in 0..64 {
        let change = h.gpu.frame(cam);
        changes.push(change);
        if change == 0 {
            converge_frame = Some(f);
            break;
        }
    }
    let cf = converge_frame.unwrap_or_else(|| panic!("never converged; change_count per frame = {changes:?}"));
    assert!(cf < 16, "static camera took {} frames to converge (change history {changes:?})", cf + 1);

    // STAYS 0 + the tail self-gates over several further frames.
    for f in 0..4 {
        let change = h.gpu.frame(cam);
        assert_eq!(change, 0, "static camera churned on idle frame {f}: change_count={change}");
        let (c, p, a) = h.gpu.tail_dispatch_counts();
        assert_eq!(
            (c, p, a),
            (0, 0, 0),
            "idle frame {f}: pack-tail indirect dispatch counts must be 0 (classify={c} pack={p} aabb={a})"
        );
    }

    // The converged pool is non-empty + correct (per-key content) — idle didn't corrupt it.
    let pool = h.gpu.pool();
    assert!(!pool.slot_of.is_empty(), "converged resident set is empty");
    let (_, _, keys) = parity_patches(&h, &pool);
    let (checked, dense_seen) = assert_per_key_content(&pool, &keys);
    assert!(dense_seen > 0, "the scene must contain dense bricks to be a real gate");
    eprintln!(
        "[converge-static] OK — converged in {} frames, idle for 4 more (change_count=0, tail dispatch 0); \
         {checked} resident keys ({dense_seen} dense) content-correct",
        cf + 1
    );
}

/// **(b) MOVE → RECONVERGE == CPU + NO HOLE.** Over a camera SEQUENCE (cold fill, crossings, LOD coarsen/refine,
/// negative coords), drive the GPU front end to convergence each step and assert the converged resident pool's
/// per-KEY content + ray-HITS EQUAL the CPU `ResidentPacker` converged state. Also assert NO HOLE at each move:
/// after the FIRST frame following a move, the GPU resident set covers every key the CPU's first-round set covers
/// (keep-old-until-revealed). The set-equality vs the CPU `ResidencyManager` is independently gated by the diff
/// gate; here we additionally prove the CONTENT + RENDER identity of the GPU-driven pool through the moves.
#[test]
#[ignore = "stale oracle: compares the GPU front-end pool cell-for-cell vs CPU pack_one, which diverges on the \
            NEIGHBOUR_SOLID/boundary halo. The GPU front end (pager cores + GPU halo-fill) is the SSOT post the \
            halo fixes (see voxel_gpu_residency_pack_parity's corrected per-key oracle + the live \
            voxel_paged_front_end_render). Re-enable after porting a GPU-halo-aware oracle."]
fn move_reconverge_pool_matches_cpu_with_no_hole() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter (or limits too low) — move-reconverge skipped");
        return;
    };
    let h = build_harness(device, queue);
    let span0 = brick_span(0);

    // The camera sequence: cold fill, a crossing, recede (coarsen), farther, approach (refine), negative coords.
    let cams: [[f32; 3]; 6] = [
        [0.5 * span0, 1.5 * span0, 0.5 * span0],
        [1.5 * span0, 1.5 * span0, 0.5 * span0],
        [0.5 * span0, 30.0 * span0, 0.5 * span0],
        [0.5 * span0, 80.0 * span0, 0.5 * span0],
        [0.5 * span0, 2.5 * span0, 0.5 * span0],
        [-2.5 * span0, 1.0 * span0, -2.5 * span0],
    ];

    let mut mgr = ResidencyManager::new();
    let mut prev_gpu_set: Option<HashSet<(IVec3, u32)>> = None;

    for (i, cam) in cams.iter().enumerate() {
        // CPU first-round set (one update, NO drain) — the keep-old retained set BEFORE replacements load.
        let cpu_first = {
            mgr.update(*cam, &h.cfg, &h.source);
            mgr.resident_entries().into_iter().map(|e| (e.coord, e.lod)).collect::<HashSet<_>>()
        };
        // ONE GPU frame at the new cam (replacements not yet fully resident).
        h.gpu.frame(*cam);
        let gpu_first = h.gpu.resident_set();

        // NO-HOLE: every OLD key (resident before the move) the CPU RETAINED must ALSO be retained on the GPU —
        // else the GPU opened a hole the CPU did not.
        if let Some(prev) = &prev_gpu_set {
            let cpu_retained_old: HashSet<_> = cpu_first.intersection(prev).copied().collect();
            let gpu_retained_old: HashSet<_> = gpu_first.intersection(prev).copied().collect();
            assert!(
                cpu_retained_old.is_subset(&gpu_retained_old),
                "[cam {i} {cam:?}] keep-old HOLE: CPU kept {} old keys the GPU dropped (first 10: {:?})",
                cpu_retained_old.difference(&gpu_retained_old).count(),
                cpu_retained_old.difference(&gpu_retained_old).take(10).collect::<Vec<_>>()
            );
        }

        // Now drive BOTH to convergence.
        let cpu_set = cpu_converge(&mut mgr, *cam, &h.cfg, &h.source, &h.reg);
        let frames = h.gpu.converge(*cam);
        let gpu_set = h.gpu.resident_set();
        if cpu_set != gpu_set {
            let missing: Vec<_> = cpu_set.difference(&gpu_set).take(20).collect();
            let extra: Vec<_> = gpu_set.difference(&cpu_set).take(20).collect();
            panic!(
                "[cam {i} {cam:?}] converged GPU set != CPU set. cpu={} gpu={} frames={frames} \
                 missing(GPU lacks)={missing:?} extra(GPU has)={extra:?}",
                cpu_set.len(),
                gpu_set.len()
            );
        }
        assert!(!cpu_set.is_empty(), "[cam {i}] converged resident set must be non-empty");

        // CONTENT + RAY-HIT parity of the converged GPU-driven pool vs a CPU ResidentPacker over the same set.
        let pool = h.gpu.pool();
        let (gpu_patch, oracle_patch, keys) = parity_patches(&h, &pool);
        let (checked, dense_seen) = assert_per_key_content(&pool, &keys);
        assert!(dense_seen > 0, "[cam {i}] the converged set must contain dense bricks");
        let hits = assert_ray_hits(&gpu_patch, &oracle_patch, span0);
        assert!(hits > 0, "[cam {i}] the ray grid must produce SOME hits to be a real gate");

        eprintln!(
            "[converge-move] cam {i} {cam:?}: converged in {frames} frames, {} resident, {checked} keys ({dense_seen} dense) content-OK, {hits} ray-hits identical to CPU",
            gpu_set.len()
        );
        prev_gpu_set = Some(gpu_set);
    }
    eprintln!("[converge-move] OK — {} cameras, content + ray-hits identical to the CPU pool at every converged step", cams.len());
}

// =========================================================================================================
//  small wgpu helpers (copied from the G-c.2 parity rigs).
// =========================================================================================================

fn buf_init(device: &wgpu::Device, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
}
fn storage_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn storage_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}
fn dispatch_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::INDIRECT
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST
}
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn bind(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}
fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, words: usize) -> Vec<u32> {
    if words == 0 {
        return Vec::new();
    }
    let bytes = (words * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rb") });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = staging.slice(..).get_mapped_range().expect("map").to_vec();
    staging.unmap();
    bytemuck::cast_slice(&data).to_vec()
}

fn compute(enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, wgs: u32) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(p);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(wgs.max(1), 1, 1);
}
