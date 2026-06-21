//! **Phase G "G-c.2b" — the GPU-DRIVEN PACK parity gate** (docs/PHASE_G_GC_PLAN.md §1 Pass D, §2.3, §2.4, §6
//! "G-c.2", §7).
//!
//! Pass D (in `assets/shaders/voxel_residency.wgsl`) GPU-builds the SAME `PackCommand`/`AabbCommand`/
//! `ClassifyCommand` + uniform/freed metas the LANDED `voxel_pack.wgsl` (`classify_brick`/`pack_brick`/
//! `write_aabb`) consumes — from the Pass-C `slot_table`/`enter_list`/`drop_list` + a GPU slab allocator —
//! so the GPU-built commands REPLACE the CPU `ResidentPacker::update_gpu` driver. This rig drives the WHOLE
//! GPU front end (Pass A → B0 → B → C → D1 → D2 → `classify_brick` → D3 → D0 → `pack_brick` → `write_aabb`)
//! over a known scene into the persistent pool buffers, reads them back, and proves the resident pool is
//! correct two ways:
//!
//!   (a) **per-KEY content parity** — for each resident `(coord,lod)` key, every haloed cell decoded from the
//!       GPU pool via the SSOT `cell_block` EQUALS the CPU `ResidentPacker` snapshot's cell for that key.
//!       Decode-BY-KEY (match by `voxel_origin`): the GPU free-list + GPU slab allocator assign the slot AND
//!       the slab offsets in a DIFFERENT order than the CPU, so per-slot byte-identity is neither expected nor
//!       required — the slot/slab MAPPING differs but each key's CONTENT is identical.
//!   (b) **ray-HIT parity** — run the FAITHFUL CPU DDA trace (`voxel_normal_swap`'s `trace_faithful`, a
//!       line-for-line port of `voxel_raytrace.wgsl`'s `dda_brick`+`trace`) over a ray GRID against BOTH the
//!       GPU-driven pool and the CPU oracle pool, and assert the first-hit (block id + world position +
//!       normal) is identical. This is render-identity at the pool level (the GPU `ray_query` layer over an
//!       identical pool is independently gated by `voxel_raytrace_gpu.rs`).
//!
//! SCOPE (G-c.2b): a COLD FILL — drive the front end to convergence over a static camera. Every resident key
//! is ENTERED, so Pass D1's first-order dirty expansion = the whole resident set (exact). Multi-round dynamic
//! convergence / indirect self-gating / the `change_count` mirror is the NEXT stage G-c.3.
//!
//! Skips cleanly when no GPU adapter (or its compute / storage-buffer limits are too low).

use adventure::voxel::brickmap::{BRICK_EDGE, BRICK_VOXELS, Brick, BrickMap, MAX_LOD, brick_span};
use adventure::voxel::gpu::{GpuBrickMeta, GpuBrickPatch, ResidentBrick, build_by_key, halo_cells, pack_one};
use adventure::voxel::incremental::{ResidentPacker, index_class_words};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::residency_gpu::{SectorOccupancy, brick_key_hash};
use adventure::voxel::source::{BrickSource, StaticVoxSource};
use adventure::voxel::streaming::{camera_brick_coord_lod, level_box_pub};
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
const NEIGHBOUR_ABSENT: u32 = 0xFFFF_FFFF;
const META_WORDS: usize = 12; // 48-B GpuBrickMeta

// =========================================================================================================
//  ResidencyParams (Pass B input) — same SSOT as the enumerate/diff rigs.
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
    index_stride: u32,   // WORDS per slot in the index pool (fixed per-slot slab; = RESERVE_INDEX_WORDS_PER_BRICK)
    palette_stride: u32, // WORDS per slot in the palette pool
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
//  The GPU core store (§2.4): a (coord,lod) -> deduped-core-index hash + the deduped 8³ cores. Built CPU-side
//  from the scene's occupied keys (the test stands in for the per-region `.vxo` core upload, G-c.4). The hash
//  is the SAME FNV-1a family as `slot_table` (5-word stride [x,y,z,lod,core_index]) so the WGSL `core_lookup`
//  probes the identical chain.
// =========================================================================================================

struct CoreStore {
    table: Vec<u32>, // 5 u32/slot
    table_size: u32,
    cores: Vec<u32>, // 512 u32/core
}

fn build_core_store(source: &StaticVoxSource, reg: &BlockRegistry) -> CoreStore {
    let keys: Vec<(IVec3, u32)> = source.occupied_keys().collect();
    // Open-addressing hash sized ~0.5 load factor.
    let table_size = (keys.len() * 2).max(2).next_power_of_two() as u32;
    let mut table = vec![0u32; table_size as usize * 5];
    for slot in table.chunks_exact_mut(5) {
        slot[3] = EMPTY_LOD;
    }
    let mut cores: Vec<u32> = Vec::with_capacity(keys.len() * BRICK_VOXELS);
    let mask = table_size - 1;
    for (i, &(coord, lod)) in keys.iter().enumerate() {
        // Extract the 8³ core in voxel_index order (mirror of incremental::extract_core).
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
//  The full GPU-driven front end (Pass A → B0 → B → C → D + classify/pack/write_aabb into the pool).
// =========================================================================================================

struct GpuDrive {
    pools: PoolReadback,
}

/// The GPU-driven pool, read back for decoding.
struct PoolReadback {
    metas: Vec<GpuBrickMeta>,
    voxel: Vec<u32>,
    brick_palettes: Vec<u32>,
    /// The resident slot_table read back as a `(coord,lod) -> slot` map.
    slot_of: FxHashMap<(IVec3, u32), u32>,
}

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

impl GpuDrive {
    /// Drive the WHOLE GPU front end for a static `cam` (a cold fill) and read back the pool. `max_resident`
    /// bounds the slot ring + the slab pools; `pool_slots` is the pool buffer capacity (= max_resident).
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn run(
        device: wgpu::Device,
        queue: wgpu::Queue,
        occ: &SectorOccupancy,
        cores: &CoreStore,
        cam: [f32; 3],
        half: i32,
        max_resident: u32,
        list_cap: usize,
    ) -> Self {
        let params = build_params(cam, half);
        let slot_table_size = (max_resident as usize * 2).max(2).next_power_of_two() as u32;
        let present_size = (list_cap * 2).max(2).next_power_of_two() as u32;

        // --- occupancy + params + configs ---
        let header_buf = buf_init(&device, "header", bytemuck::bytes_of(&occ.header()), wgpu::BufferUsages::UNIFORM);
        let entries_buf =
            buf_init(&device, "entries", bytemuck::cast_slice(occ.entries()), wgpu::BufferUsages::STORAGE);
        let params_buf = buf_init(&device, "params", bytemuck::bytes_of(&params), wgpu::BufferUsages::UNIFORM);
        let diff_cfg = DiffConfig { slot_table_size, present_size, max_resident, refine_descent_cap: REFINE_DESCENT_CAP };
        let diff_cfg_buf = buf_init(&device, "diff_cfg", bytemuck::bytes_of(&diff_cfg), wgpu::BufferUsages::UNIFORM);
        // index/palette per-slot strides MUST match the pool sizing below (256/256 words/slot, fixed per-slot slabs).
        let pack_cfg = PackConfig { core_table_size: cores.table_size, max_resident, index_stride: 256, palette_stride: 256 };
        let pack_cfg_buf = buf_init(&device, "pack_cfg", bytemuck::bytes_of(&pack_cfg), wgpu::BufferUsages::UNIFORM);

        // --- Pass C persistent state ---
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

        // --- Pass B/C per-round transients ---
        let total_cells = params.total_cells.max(1) as usize;
        let shell_idx = storage_buf(&device, "shell_idx", (total_cells * 4) as u64);
        let shell_count = buf_init(&device, "shell_count", bytemuck::bytes_of(&0u32), storage_usage());
        let shell_dispatch =
            buf_init(&device, "shell_dispatch", bytemuck::cast_slice(&[0u32, 1u32, 1u32]), dispatch_usage());
        let cand_count = buf_init(&device, "cand_count", bytemuck::bytes_of(&0u32), storage_usage());
        let cand_list = storage_buf(&device, "cand_list", (list_cap * 16) as u64);
        let desired_count = buf_init(&device, "desired_count", bytemuck::bytes_of(&0u32), storage_usage());
        let desired_list = storage_buf(&device, "desired_list", (list_cap * 16) as u64);
        let present_init = vec![EMPTY_LOD; present_size as usize * 4];
        let present_flag = buf_init(&device, "present_flag", bytemuck::cast_slice(&present_init), storage_usage());
        let enter_count = buf_init(&device, "enter_count", bytemuck::bytes_of(&0u32), storage_usage());
        let enter_list = storage_buf(&device, "enter_list", (list_cap * 16) as u64);
        let drop_count = buf_init(&device, "drop_count", bytemuck::bytes_of(&0u32), storage_usage());
        let drop_list = storage_buf(&device, "drop_list", (list_cap * 16) as u64);
        let drop_decision = storage_buf(&device, "drop_decision", (slot_table_size as u64) * 4);

        // --- Pass D state ---
        let core_table_buf = buf_init(&device, "core_table", bytemuck::cast_slice(&cores.table), wgpu::BufferUsages::STORAGE);
        let cores_buf = buf_init(&device, "cores", bytemuck::cast_slice(&cores.cores), wgpu::BufferUsages::STORAGE);
        let dirty_count = buf_init(&device, "dirty_count", bytemuck::bytes_of(&0u32), storage_usage());
        let dirty_list = storage_buf(&device, "dirty_list", (list_cap * 16) as u64);
        let dirty_slot = storage_buf(&device, "dirty_slot", (list_cap * 4) as u64);
        let dirty_flag_init = vec![EMPTY_LOD; slot_table_size as usize * 4];
        let dirty_flag = buf_init(&device, "dirty_flag", bytemuck::cast_slice(&dirty_flag_init), storage_usage());
        let pack_count = buf_init(&device, "pack_count", bytemuck::bytes_of(&0u32), storage_usage());
        let pack_commands = storage_buf(&device, "pack_commands", (list_cap * 60) as u64);
        let aabb_count = buf_init(&device, "aabb_count", bytemuck::bytes_of(&0u32), storage_usage());
        let aabb_commands = storage_buf(&device, "aabb_commands", (list_cap * 32) as u64);
        let classify_commands = storage_buf(&device, "classify_commands", (list_cap * 16) as u64);
        let neighbour_indices = storage_buf(&device, "neighbour_indices", (list_cap as u64) * 27 * 4);

        // The persistent POOL (meta/voxel/palette). The GPU front end uses FIXED PER-SLOT slabs (slot·stride), so
        // the pools are sized per-slot to the MAX SUPPORTED width (mirror `RESERVE_INDEX_WORDS_PER_BRICK`=256 /
        // index_bits=8's 250w, `RESERVE_PALETTE_WORDS_PER_BRICK`=256 / index_bits=8's max). index_bits=16 bricks
        // degenerate (D3 `fits` guard), so 256 holds every packable brick — each slot owns its own region.
        const INDEX_STRIDE: usize = 256;
        const PALETTE_STRIDE: usize = 256;
        let meta_words = max_resident as usize * META_WORDS;
        let index_pool_words = (max_resident as usize * INDEX_STRIDE).max(INDEX_STRIDE);
        let palette_pool_words = (max_resident as usize * PALETTE_STRIDE).max(PALETTE_STRIDE);
        let meta_buf = storage_buf(&device, "meta_buf", (meta_words * 4) as u64);
        let voxel_buf = storage_buf(&device, "voxel_buf", (index_pool_words * 4) as u64);
        let palette_buf = storage_buf(&device, "palette_buf", (palette_pool_words * 4) as u64);
        // The AABB pool (8 u32/slot), seeded degenerate.
        let degenerate = adventure::voxel::incremental::degenerate_aabb();
        let mut aabb_host = vec![0u32; max_resident as usize * 8];
        for slot in 0..max_resident as usize {
            aabb_host[slot * 8..slot * 8 + 8]
                .copy_from_slice(bytemuck::cast_slice(std::slice::from_ref(&degenerate)));
        }
        let aabb_buf = buf_init(&device, "aabb_buf", bytemuck::cast_slice(&aabb_host), storage_usage());

        let pack_dispatch = buf_init(&device, "pack_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let aabb_dispatch = buf_init(&device, "aabb_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let classify_dispatch =
            buf_init(&device, "classify_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());

        // --- the GPU slab allocators (mirror of the CPU `SlabArena`): ONE shared bump high-water per pool +
        //     a per-class free-list. ctrl layout = [shared_hw, (head,tail)×N_classes]. The pools are a SINGLE
        //     shared bump region (classes interleave by alloc order, exactly as the CPU), sized to the
        //     aggregate reserve above; the pool base is 0 (the pools start at word 0).
        // FIXED per-slot slabs: the pools start at word 0; Pass D3 writes `slot·stride` directly (no allocator
        // buffers, no DenseSlot table). `index_pool_base[0]`/`palette_pool_base[0]` are the pool word bases (0).
        let index_pool_base =
            buf_init(&device, "index_pool_base", bytemuck::cast_slice(&[0u32]), wgpu::BufferUsages::STORAGE);
        let palette_pool_base =
            buf_init(&device, "palette_pool_base", bytemuck::cast_slice(&[0u32]), wgpu::BufferUsages::STORAGE);
        let _ = index_class_words; // (kept imported for the pool-sizing comment; per-class sizing no longer used)
        // +1: the trailing slot is the 4-S1 backdrop-reserve counter (clear_per_frame_hashes always clears it).
        let enter_hist = storage_buf(&device, "enter_hist", (HIST_BUCKETS as u64 + 1) * 4);
        let enter_cap = buf_init(&device, "enter_cap", bytemuck::cast_slice(&[HIST_BUCKETS, 0u32]), storage_usage());
        // 4-S2/S3: per-slot last_used (binding 52). Zeroed dummy — with the mirror's frame=0, `demand_on()` is false
        // so the residency never reads/writes it (behaviour = the distance cut). Sized to the pool slot count.
        let last_used = storage_buf(&device, "last_used", (max_resident as u64) * 4);

        // --- the residency shader (Pass A–D) + the pack shader (classify/pack/write_aabb) ---
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

        // ONE comprehensive bind-group layout (0..=47) for the residency passes. A pipeline only references its
        // subset; wgpu validates against the layout superset.
        let res_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("res_bgl"),
            entries: &[
                uniform_entry(0),         // header
                storage_entry(1, true),   // entries
                storage_entry(2, true),   // query_keys (dummy)
                storage_entry(3, false),  // query_out (dummy)
                uniform_entry(4),         // params
                storage_entry(5, false),  // shell_wg_indices
                storage_entry(6, false),  // shell_count
                storage_entry(7, false),  // shell_dispatch
                storage_entry(8, false),  // candidate_count
                storage_entry(9, false),  // candidate_list
                storage_entry(10, false), // desired_count
                storage_entry(11, false), // desired_list
                uniform_entry(12),        // diff_cfg
                storage_entry(13, false), // slot_table
                storage_entry(14, false), // free_ring
                storage_entry(15, false), // free_ctrl
                storage_entry(16, false), // quarantine_ring
                storage_entry(17, false), // quarantine_ctrl
                storage_entry(18, false), // present_flag
                storage_entry(19, false), // enter_count
                storage_entry(20, false), // enter_list
                storage_entry(21, false), // drop_count
                storage_entry(22, false), // drop_list
                storage_entry(23, false), // drop_decision
                uniform_entry(24),        // pack_cfg
                storage_entry(25, true),  // core_table
                storage_entry(26, true),  // cores
                storage_entry(27, false), // dirty_count
                storage_entry(28, false), // dirty_list
                storage_entry(29, false), // dirty_slot
                storage_entry(30, false), // dirty_flag
                storage_entry(31, false), // pack_count
                storage_entry(32, false), // pack_commands
                storage_entry(33, false), // aabb_count
                storage_entry(34, false), // aabb_commands
                storage_entry(35, false), // classify_commands
                storage_entry(36, false), // neighbour_indices
                storage_entry(37, false), // meta_buf
                storage_entry(38, false), // pack_dispatch
                storage_entry(39, false), // aabb_dispatch
                storage_entry(40, false), // classify_dispatch
                storage_entry(45, true),  // index_pool_base
                storage_entry(46, true),  // palette_pool_base
                storage_entry(47, true),  // classify_out
                storage_entry(50, false), // enter_hist
                storage_entry(51, false), // enter_cap
                storage_entry(52, false), // 4-S2/S3 last_used
            ],
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
        let p_release = mk_res("diff_release_quarantine");
        let p_b0 = mk_res("prepare_shell_dispatch");
        let p_b = mk_res("enumerate_shells");
        let p_present = mk_res("build_present_flag");
        let p_mark = mk_res("diff_drop_mark");
        let p_apply = mk_res("diff_drop_apply");
        let p_enter = mk_res("diff_enter_scan");
        let p_d_dirty = mk_res("pack_build_dirty");
        let p_d_nbr = mk_res("pack_build_neighbours");
        let p_d_cmd = mk_res("pack_build_commands");
        let p_d_drops = mk_res("pack_build_drops");

        // classify_out buffer (one per dirty key, 4 u32) — shared by classify_brick (writes) + D3 (reads).
        let classify_out = storage_buf(&device, "classify_out", (list_cap * 4 * 4) as u64);

        let dummy_in = buf_init(&device, "dummy_in", bytemuck::cast_slice(&[0u32; 4]), wgpu::BufferUsages::STORAGE);
        let dummy_out = storage_buf(&device, "dummy_out", 16);
        // A dummy storage buffer for binding 7 in every pass EXCEPT B0 — so `shell_dispatch` is NOT bound as a
        // STORAGE resource during Pass B's indirect dispatch (wgpu forbids STORAGE+INDIRECT in one dispatch).
        let dummy_dispatch = storage_buf(&device, "dummy_dispatch", 16);

        // Two residency bind groups (all 48 bindings): `res_bg` binds a DUMMY at 7 (used by every pass but B0),
        // `res_bg_b0` binds the real `shell_dispatch` at 7 (Pass B0 atomicMaxes it). Pass B uses `res_bg` (dummy
        // at 7) AND `shell_dispatch` as its indirect-dispatch arg — no STORAGE+INDIRECT usage conflict.
        macro_rules! res_entries {
            ($slot7:expr) => {
                [
                    bind(0, &header_buf),
                    bind(1, &entries_buf),
                    bind(2, &dummy_in),
                    bind(3, &dummy_out),
                    bind(4, &params_buf),
                    bind(5, &shell_idx),
                    bind(6, &shell_count),
                    bind(7, $slot7),
                    bind(8, &cand_count),
                    bind(9, &cand_list),
                    bind(10, &desired_count),
                    bind(11, &desired_list),
                    bind(12, &diff_cfg_buf),
                    bind(13, &slot_table_buf),
                    bind(14, &free_ring_buf),
                    bind(15, &free_ctrl_buf),
                    bind(16, &quar_ring_buf),
                    bind(17, &quar_ctrl_buf),
                    bind(18, &present_flag),
                    bind(19, &enter_count),
                    bind(20, &enter_list),
                    bind(21, &drop_count),
                    bind(22, &drop_list),
                    bind(23, &drop_decision),
                    bind(24, &pack_cfg_buf),
                    bind(25, &core_table_buf),
                    bind(26, &cores_buf),
                    bind(27, &dirty_count),
                    bind(28, &dirty_list),
                    bind(29, &dirty_slot),
                    bind(30, &dirty_flag),
                    bind(31, &pack_count),
                    bind(32, &pack_commands),
                    bind(33, &aabb_count),
                    bind(34, &aabb_commands),
                    bind(35, &classify_commands),
                    bind(36, &neighbour_indices),
                    bind(37, &meta_buf),
                    bind(38, &pack_dispatch),
                    bind(39, &aabb_dispatch),
                    bind(40, &classify_dispatch),
                    bind(45, &index_pool_base),
                    bind(46, &palette_pool_base),
                    bind(47, &classify_out),
                    bind(50, &enter_hist),
                    bind(51, &enter_cap),
                    bind(52, &last_used),
                ]
            };
        }
        let res_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_bg"),
            layout: &res_bgl,
            entries: &res_entries!(&dummy_dispatch),
        });
        let res_bg_b0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_bg_b0"),
            layout: &res_bgl,
            entries: &res_entries!(&shell_dispatch),
        });

        // --- classify_brick + pack_brick + write_aabb pipelines (voxel_pack.wgsl) ---
        // classify_brick: cores@1, neighbour_indices@2, classify_out@8, classify_commands@9.
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

        // pack_brick: commands@0, cores@1, neighbour_indices@2, voxel_buf@3, brick_palettes@4, meta_buf@5.
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

        // write_aabb: aabb_buf@6, aabb_commands@7.
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

        // ============================== ENCODE THE WHOLE FRONT END ==============================
        // Round 1: A → B0 → B(indirect) → C0 → C2a → C2b → C1. Then read counts for sizing the D dispatches.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("front_enc") });
        compute(&mut enc, &p_release, &res_bg, 1);
        compute(&mut enc, &p_b0, &res_bg_b0, params.total_cells.div_ceil(64).max(1));
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("b"), timestamp_writes: None });
            pass.set_pipeline(&p_b);
            pass.set_bind_group(0, &res_bg, &[]);
            pass.dispatch_workgroups_indirect(&shell_dispatch, 0);
        }
        queue.submit(std::iter::once(enc.finish()));
        let d_cnt = readback_u32(&device, &queue, &desired_count, 1)[0];
        let c_cnt = readback_u32(&device, &queue, &cand_count, 1)[0];

        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("c_enc") });
        compute(&mut enc2, &p_present, &res_bg, d_cnt.div_ceil(64).max(1));
        compute(&mut enc2, &p_mark, &res_bg, slot_table_size.div_ceil(64).max(1));
        compute(&mut enc2, &p_apply, &res_bg, slot_table_size.div_ceil(64).max(1));
        compute(&mut enc2, &p_enter, &res_bg, c_cnt.div_ceil(64).max(1));
        queue.submit(std::iter::once(enc2.finish()));
        let n_enter = readback_u32(&device, &queue, &enter_count, 1)[0];
        let n_drop = readback_u32(&device, &queue, &drop_count, 1)[0];

        // Pass D1 (build dirty) over enter+drop; read dirty_count; D2 (neighbours) + classify (indirect) + D3
        // (commands) + D0 (drops). Then pack_brick (indirect) + write_aabb (indirect).
        let mut enc3 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("d1_enc") });
        compute(&mut enc3, &p_d_dirty, &res_bg, (n_enter + n_drop).div_ceil(64).max(1));
        compute(&mut enc3, &p_d_drops, &res_bg, n_drop.div_ceil(64).max(1));
        queue.submit(std::iter::once(enc3.finish()));
        let n_dirty = readback_u32(&device, &queue, &dirty_count, 1)[0];

        let mut enc4 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("d_enc") });
        compute(&mut enc4, &p_d_nbr, &res_bg, n_dirty.div_ceil(64).max(1));
        // classify_brick — one workgroup per dirty key (record over classify_dispatch's count == n_dirty).
        {
            let mut pass = enc4.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cls"), timestamp_writes: None });
            pass.set_pipeline(&p_classify);
            pass.set_bind_group(0, &cls_bg, &[]);
            pass.dispatch_workgroups(n_dirty.max(1), 1, 1);
        }
        compute(&mut enc4, &p_d_cmd, &res_bg, n_dirty.div_ceil(64).max(1));
        queue.submit(std::iter::once(enc4.finish()));
        let n_pack = readback_u32(&device, &queue, &pack_count, 1)[0];
        let n_aabb = readback_u32(&device, &queue, &aabb_count, 1)[0];

        let mut enc5 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pack_enc") });
        {
            let mut pass = enc5.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pk"), timestamp_writes: None });
            pass.set_pipeline(&p_pack);
            pass.set_bind_group(0, &pack_bg, &[]);
            pass.dispatch_workgroups(n_pack.max(1), 1, 1);
        }
        {
            let mut pass = enc5.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ab"), timestamp_writes: None });
            pass.set_pipeline(&p_aabb);
            pass.set_bind_group(0, &aabb_bg, &[]);
            pass.dispatch_workgroups(n_aabb.div_ceil(64).max(1), 1, 1);
        }
        queue.submit(std::iter::once(enc5.finish()));

        // Read back the pool + slot table.
        let meta_raw = readback_u32(&device, &queue, &meta_buf, meta_words);
        let voxel = readback_u32(&device, &queue, &voxel_buf, index_pool_words);
        let brick_palettes = readback_u32(&device, &queue, &palette_buf, palette_pool_words);
        let slot_words = readback_u32(&device, &queue, &slot_table_buf, slot_table_size as usize * 5);

        let metas: Vec<GpuBrickMeta> = bytemuck::cast_slice(&meta_raw).to_vec();
        let mut slot_of = FxHashMap::default();
        for s in slot_words.chunks_exact(5) {
            if s[3] == EMPTY_LOD {
                continue;
            }
            let coord = IVec3::new(s[0] as i32, s[1] as i32, s[2] as i32);
            slot_of.insert((coord, s[3]), s[4]);
        }

        eprintln!(
            "[gpu-pack] cam {cam:?}: enter={n_enter} drop={n_drop} dirty={n_dirty} pack(dense)={n_pack} aabb={n_aabb} resident={}",
            slot_of.len()
        );
        Self { pools: PoolReadback { metas, voxel, brick_palettes, slot_of } }
    }
}

// =========================================================================================================
//  CPU oracle — drive a ResidentPacker cold fill, decode each resident key's content from the snapshot.
// =========================================================================================================

/// The CPU oracle pool as a `GpuBrickPatch` (slot-indexed) + the key→slot map (recovered by `voxel_origin`).
struct CpuOracle {
    patch: GpuBrickPatch,
    slot_of: FxHashMap<(IVec3, u32), u32>,
    resident: Vec<(IVec3, u32)>,
}

/// Build a `GpuBrickPatch` whose META order is CANONICAL (sorted by lod, then world_min) so the faithful
/// trace's candidate iteration — and thus its equidistant-coplanar `best_axis` tiebreak — is order-independent
/// across two pools that hold the SAME bricks in DIFFERENT slot orders. Drops freed/degenerate slots (a meta
/// with all-zero world_min AND not the origin brick is an unused slot; we keep only metas a resident key wrote
/// — here every non-degenerate meta). The voxel/palette buffers are shared verbatim (indexed by the meta's own
/// offsets, unaffected by reordering).
fn canonical_patch(metas: &[GpuBrickMeta], voxels: &[u32], palettes: &[u32]) -> GpuBrickPatch {
    // Keep metas of RESIDENT slots: a slot is resident iff its meta is uniform OR its lod_and_bits/voxel_offset
    // describe a real brick. A freed slot is GpuBrickMeta::zeroed (all zero). The origin brick (0,0,0) lod0 is
    // also all-zero world_min but is dense/uniform with a non-zero flags/lod — distinguish by "not fully zero".
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

fn cpu_oracle(entries: &[ResidentBrick<'_>], reg: &BlockRegistry) -> CpuOracle {
    let mut packer = ResidentPacker::new(entries.len().max(1) as u32 * 2);
    packer.update(entries, reg.len() as u32);
    let snap = packer.snapshot_buffers(reg);
    let patch = GpuBrickPatch {
        aabbs: snap.aabbs.clone(),
        metas: snap.metas.clone(),
        voxels: snap.indices.clone(),
        brick_palettes: snap.brick_palettes.clone(),
        palette: snap.palette.clone(),
        lights: Vec::new(),
        alias: Vec::new(),
    };
    let degenerate = adventure::voxel::incremental::degenerate_aabb();
    let mut slot_of = FxHashMap::default();
    let mut resident = Vec::new();
    for (slot, m) in snap.metas.iter().enumerate() {
        if snap.aabbs[slot] == degenerate {
            continue; // freed/unused slot
        }
        let coord = IVec3::new(
            m.voxel_origin[0].div_euclid(BRICK_EDGE),
            m.voxel_origin[1].div_euclid(BRICK_EDGE),
            m.voxel_origin[2].div_euclid(BRICK_EDGE),
        );
        let key = (coord, m.lod());
        slot_of.insert(key, slot as u32);
        resident.push(key);
    }
    CpuOracle { patch, slot_of, resident }
}

// =========================================================================================================
//  Scene + helpers.
// =========================================================================================================

/// A registry with several materials so dense bricks carry multi-id palettes (k ≥ 2) AND the
/// `ResidentPacker`'s `k <= palette_stride` (registry length) invariant holds for the scene's ids 1..=5.
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

/// A multi-material dense + uniform-core scene at LOD0, with negative coords + an isolated cluster — the same
/// shape family the diff/enumerate gates use, but with non-trivial brick CONTENT so the pack is a real gate.
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
    // A 4×3×4 dense slab (multi-material) straddling the origin into negative coords.
    for z in -2..2 {
        for x in -2..2 {
            for y in 0..3 {
                map.insert(IVec3::new(x, y, z), multi(x * 11 + y * 7 + z * 3));
            }
        }
    }
    // A solid 3³ block (its centre collapses uniform-incl-halo) offset away.
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                map.insert(IVec3::new(8 + x, 1 + y, 8 + z), solid(1));
            }
        }
    }
    // An isolated brick (all-AIR halo).
    map.insert(IVec3::new(14, 0, 14), multi(42));
    map
}

// =========================================================================================================
//  THE GATES.
// =========================================================================================================

/// **(a) PER-KEY CONTENT PARITY.** Drive the full GPU front end (B/C/D + pack) over a cold fill; for each
/// resident `(coord,lod)` key, decode every haloed cell from the GPU pool via the SSOT `cell_block` and assert
/// it EQUALS the CPU `ResidentPacker` snapshot's cell for that same key. Decode-BY-KEY (the slot/slab MAPPING
/// differs between the GPU parallel allocator and the CPU serial one — only the CONTENT must match).
#[test]
fn gpu_driven_pack_per_key_content_matches_cpu() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter (or compute/storage limits too low) — GPU-driven pack parity skipped");
        return;
    };

    let map = scene();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());
    let reg = registry();
    let cores = build_core_store(&source, &reg);

    let half = 8i32;
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0];

    // Drive the GPU front end.
    let max_resident = 8192u32;
    let list_cap = 200_000usize;
    let gpu = GpuDrive::run(device, queue, &occ, &cores, cam, half, max_resident, list_cap);

    // The CPU oracle resident set = the GPU resident set must agree (sanity — proven exactly by the diff gate;
    // here we use the GPU resident set as the key universe and oracle each key's content via pack_one).
    assert!(!gpu.pools.slot_of.is_empty(), "the GPU front end produced an empty resident set");

    // The CPU SSOT content per key: pack_one over the resident set (the canonical haloed cells). The entries we
    // ITERATE for the comparison are the GPU resident keys (which the diff gate proves == the CPU manager set).
    let resident_keys: Vec<(IVec3, u32)> = gpu.pools.slot_of.keys().copied().collect();
    let bricks: Vec<(IVec3, u32, Brick)> =
        resident_keys.iter().map(|&(c, l)| (c, l, source.brick(c, l, &reg))).collect();
    let entries: Vec<ResidentBrick> =
        bricks.iter().map(|(c, l, b)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    // `by_key` (the halo dictionary `pack_one` reads each key's NEIGHBOUR cores from) MUST cover the full OCCUPIED
    // set, not just the resident keys: the GPU halo reads the actual neighbour core from the core store (which holds
    // every occupied brick here), so an occupied-but-not-resident neighbour contributes its REAL boundary voxels.
    // Building `by_key` over resident-only made `pack_one` pack AIR for those neighbours (the old residency-only
    // halo) and falsely diverge from the occupancy-correct GPU. (SSOT = the actual geometry, not the resident slice.)
    let all_keys: Vec<(IVec3, u32)> = source.occupied_keys().collect();
    let all_bricks: Vec<(IVec3, u32, Brick)> =
        all_keys.iter().map(|&(c, l)| (c, l, source.brick(c, l, &reg))).collect();
    let all_entries: Vec<ResidentBrick> =
        all_bricks.iter().map(|(c, l, b)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    let by_key = build_by_key(&all_entries);

    // The GPU pool as a GpuBrickPatch for `cell_block` decode.
    let gpu_patch = GpuBrickPatch {
        aabbs: Vec::new(),
        metas: gpu.pools.metas.clone(),
        voxels: gpu.pools.voxel.clone(),
        brick_palettes: gpu.pools.brick_palettes.clone(),
        palette: Vec::new(),
        lights: Vec::new(),
        alias: Vec::new(),
    };

    let mut checked = 0usize;
    let mut dense_seen = 0usize;
    for e in &entries {
        let key = (e.coord, e.lod);
        let slot = *gpu.pools.slot_of.get(&key).expect("resident key has a GPU slot");
        let gm = &gpu_patch.metas[slot as usize];
        // The CPU SSOT haloed cells for this key.
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
                "key {key:?} (slot {slot}) cell {cell}: GPU pool decoded {gpu_block} but CPU SSOT is {want}",
            );
        }
        checked += 1;
    }
    assert!(dense_seen > 0, "the scene must contain dense bricks to be a real gate");
    eprintln!(
        "[gpu-pack-parity] per-key content: OK — {checked} resident keys ({dense_seen} dense) byte-identical to the CPU SSOT"
    );
}

/// **(b) RAY-HIT PARITY.** Build the GPU-driven pool patch AND the CPU oracle pool patch, run the FAITHFUL CPU
/// DDA trace (`trace_faithful`, a line-for-line port of `voxel_raytrace.wgsl`) over a ray GRID against BOTH,
/// and assert the first-hit (block id + world position + normal) is identical — render-identity at the pool
/// level. The GPU `ray_query` layer over an identical pool is independently gated by `voxel_raytrace_gpu.rs`.
#[test]
fn gpu_driven_pack_ray_hits_match_cpu() {
    let Some((device, queue)) = common::headless_compute_device_with_storage(512, 48) else {
        eprintln!("[skip] no GPU adapter (or limits too low) — GPU-driven ray-hit parity skipped");
        return;
    };

    let map = scene();
    let source = StaticVoxSource::new(&map);
    let occ = SectorOccupancy::from_occupied_full(source.occupied_keys_full());
    let reg = registry();
    let cores = build_core_store(&source, &reg);

    let half = 8i32;
    let span0 = brick_span(0);
    let cam = [0.5 * span0, 1.5 * span0, 0.5 * span0];

    let max_resident = 8192u32;
    let list_cap = 200_000usize;
    let gpu = GpuDrive::run(device, queue, &occ, &cores, cam, half, max_resident, list_cap);

    // The GPU resident keys → CPU oracle pool (a fresh ResidentPacker cold fill over the SAME set). Both pools
    // hold the SAME resident bricks, so a faithful trace over either yields the same surface.
    let resident_keys: Vec<(IVec3, u32)> = gpu.pools.slot_of.keys().copied().collect();
    let bricks: Vec<(IVec3, u32, Brick)> =
        resident_keys.iter().map(|&(c, l)| (c, l, source.brick(c, l, &reg))).collect();
    let entries: Vec<ResidentBrick> =
        bricks.iter().map(|(c, l, b)| ResidentBrick { coord: *c, brick: b, lod: *l }).collect();
    let oracle = cpu_oracle(&entries, &reg);

    // The GPU-driven pool as a patch for the faithful trace (reads world_min from the meta + recomputes the
    // grown AABB). CANONICALIZE the meta order (by lod, world_min) on BOTH pools: the trace's `< best_t`
    // tiebreak between two EQUIDISTANT coplanar bricks is candidate-ORDER dependent (the documented `best_axis`
    // ambiguity — `voxel_normal_swap`), and the GPU slot order ≠ the CPU slot order. Sorting both metas into the
    // SAME canonical order makes the tiebreak order-INDEPENDENT, so a normal difference can ONLY mean a genuine
    // CONTENT divergence — which the per-key gate already proves there is none. (Reordering metas is safe: the
    // voxel/palette buffers are indexed by `meta.voxel_offset`/`palette_base`, unaffected by the meta order.)
    let gpu_patch = canonical_patch(&gpu.pools.metas, &gpu.pools.voxel, &gpu.pools.brick_palettes);
    let oracle_patch = canonical_patch(&oracle.patch.metas, &oracle.patch.voxels, &oracle.patch.brick_palettes);

    // A ray grid sweeping the slab from a few directions (mirrors voxel_raytrace_gpu's grid-of-rays oracle).
    let centre = Vec3::new(0.0, 1.0 * span0, 0.0);
    let mut hits_compared = 0usize;
    let mut gpu_hits = 0usize;
    let dirs = [
        Vec3::new(0.3, -1.0, 0.2),
        Vec3::new(-0.4, -0.8, 0.5),
        Vec3::new(0.6, -0.5, -0.3),
        Vec3::new(0.1, -1.0, 0.0),
        Vec3::new(-0.7, -0.6, -0.6),
    ];
    for du in -6..=6 {
        for dv in -6..=6 {
            let origin = centre
                + Vec3::new(du as f32 * span0 * 0.5, 6.0 * span0, dv as f32 * span0 * 0.5);
            for dir in dirs {
                let rd = dir.normalize();
                let gpu_hit = dda_oracle::trace_faithful(&gpu_patch, origin, rd, 1e-4);
                let cpu_hit = dda_oracle::trace_faithful(&oracle_patch, origin, rd, 1e-4);
                hits_compared += 1;
                match (gpu_hit, cpu_hit) {
                    (None, None) => {}
                    (Some(g), Some(c)) => {
                        gpu_hits += 1;
                        // Block id at the committed cell.
                        assert_eq!(
                            g.best.block_id, c.best.block_id,
                            "ray {origin:?}->{rd:?}: GPU hit block {} != CPU {}",
                            g.best.block_id, c.best.block_id
                        );
                        // Hit position (t along the same ray) — sub-millimetre.
                        assert!(
                            (g.best.hit_t - c.best.hit_t).abs() < 1e-3,
                            "ray {origin:?}->{rd:?}: GPU hit t {} != CPU {}",
                            g.best.hit_t, c.best.hit_t
                        );
                        // Normal.
                        assert_eq!(g.normal, c.normal, "ray {origin:?}->{rd:?}: GPU normal {:?} != CPU {:?}", g.normal, c.normal);
                    }
                    (g, c) => panic!("ray {origin:?}->{rd:?}: GPU hit {:?} but CPU {:?} (one missed)", g.is_some(), c.is_some()),
                }
            }
        }
    }
    assert!(gpu_hits > 0, "the ray grid must produce SOME hits to be a real gate");
    eprintln!(
        "[gpu-pack-parity] ray-hit: OK — {hits_compared} rays compared, {gpu_hits} hits identical (block/pos/normal) GPU-driven vs CPU pool"
    );
    let _ = &oracle.slot_of;
    let _ = &oracle.resident;
}

// CPU-side SSOT cross-check (no GPU): the core store round-trips a key's core.
#[test]
fn core_store_round_trips() {
    let map = scene();
    let source = StaticVoxSource::new(&map);
    let reg = registry();
    let cores = build_core_store(&source, &reg);
    // Look up a known occupied key and verify its core matches the source brick.
    let key = (IVec3::new(0, 0, 0), 0u32);
    let mask = cores.table_size - 1;
    let mut s = (brick_key_hash(key.0, key.1) & mask) as usize;
    let mut found = None;
    for _ in 0..cores.table_size {
        if cores.table[s * 5 + 3] == EMPTY_LOD {
            break;
        }
        if cores.table[s * 5 + 3] == key.1
            && cores.table[s * 5] as i32 == key.0.x
            && cores.table[s * 5 + 1] as i32 == key.0.y
            && cores.table[s * 5 + 2] as i32 == key.0.z
        {
            found = Some(cores.table[s * 5 + 4]);
            break;
        }
        s = (s + 1) & (mask as usize);
    }
    let ci = found.expect("origin brick must be in the core store") as usize;
    let brick = source.brick(key.0, key.1, &reg);
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let vi = (x + y * BRICK_EDGE + z * BRICK_EDGE * BRICK_EDGE) as usize;
                assert_eq!(cores.cores[ci * BRICK_VOXELS + vi], brick.get(x, y, z).0 as u32);
            }
        }
    }
    let _ = NEIGHBOUR_ABSENT;
    let _: HashSet<(IVec3, u32)> = HashSet::new();
}
