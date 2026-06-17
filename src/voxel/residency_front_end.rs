//! **Phase G "G-c.4" — the LIVE readback-free GPU residency FRONT END** (`docs/PHASE_G_GC_PLAN.md` §1, §3, §4).
//!
//! This is the production home of the GPU-driven residency pipeline that, until G-c.4, ran ONLY in
//! `tests/voxel_gpu_residency_converge.rs`'s `GpuFrontEnd`. It is a faithful port of that proven, parity- and
//! convergence-gated driver, with three production changes:
//!
//! 1. **External pool buffers.** The persistent POOL (`meta`/`voxel`/`brick_palettes`/`aabb`) is NOT owned here —
//!    it is the LIVE scene's `SceneKeepAlive` buffers (`raytrace.rs`), so the GPU front end writes the SAME pool
//!    the renderer / GI / ReSTIR / DLSS consume, with NO copy and NO second scene. [`rebind_pool`] (re)builds the
//!    pool-touching bind groups when the scene (re)allocates its buffers (a scene switch / epoch).
//! 2. **Caller's encoder + fill-then-build.** [`record_frame`] records the whole pipeline (Pass A→D + the landed
//!    classify/pack/write_aabb tail, all `record_indirect`) into a caller-supplied encoder, so the dirty-chunk
//!    BLAS build can ride the SAME encoder + submit (fill-then-build, exactly like `apply_gpu_pack`).
//! 3. **Non-blocking 1-frame-late `change_count` mirror** (§3.1). [`record_frame`] copies `change_count` into a
//!    mappable staging ring; [`poll_change_count`] reads the PREVIOUS frame's value out-of-band (the one permitted
//!    CPU↔GPU sync) so the caller knows — one frame late — whether to record the AS build. A converged static
//!    camera reads 0 and idles; no host stall.
//!
//! The slot table / free-list / quarantine / per-frame hashes / lists / slab allocators / indirect dispatch
//! buffers all PERSIST across frames in this struct (the GPU analogue of the CPU `ResidentPacker`). The clipmap
//! is never materialized — `ResidencyParams` (per-LOD camera brick + clip_half) is the only per-frame CPU→GPU
//! write, computed from the live camera (the §5 CPU↔GPU boundary).

use bevy::math::IVec3;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::brickmap::{MAX_LOD, brick_span};
use super::residency_gpu::{GpuBrickCoreBuffers, GpuResidencyBuffers};
use super::streaming::{camera_brick_coord_lod, level_box_pub};

/// LOD count (`MAX_LOD + 1`).
const LODS: usize = (MAX_LOD + 1) as usize;
/// The WG-cell edge in bricks (one shell cell is `WG_CELL³` bricks) — mirrors `enumerate_shells`'s `@workgroup_size(512)`.
const WG_CELL: i32 = 8;
/// Keep-old-until-revealed refine descent cap — MUST equal `streaming.rs`'s private `REFINE_DESCENT_CAP`.
const REFINE_DESCENT_CAP: u32 = 5;
/// Hash empty-slot sentinel (`lod == EMPTY_LOD` ⇒ free), shared with the WGSL.
const EMPTY_LOD: u32 = 0xFFFF_FFFF;
/// 48-byte `GpuBrickMeta` in u32 words.
const META_WORDS: usize = 12;

// =========================================================================================================
//  Uniform structs — the SSOT shared with `voxel_residency.wgsl` (same layout as the converge gate).
// =========================================================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LevelParams {
    cam_brick_coord: [i32; 3],
    _pad_a: i32,
    cell_lo: [i32; 3],
    cell_offset: u32,
    cell_dims: [u32; 3],
    cell_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResidencyParams {
    levels: [LevelParams; LODS],
    clip_half_bricks: i32,
    total_cells: u32,
    /// ENTER-CAP: candidate distance → histogram bucket = `floor(dist * hist_scale)`.
    hist_scale: f32,
    _pad1: u32,
    /// ENTER-CAP: the camera world position (the nearest-priority distance rank centre).
    cam_world: [f32; 3],
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DiffConfig {
    slot_table_size: u32,
    present_size: u32,
    max_resident: u32,
    refine_descent_cap: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackConfig {
    core_table_size: u32,
    max_resident: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Build the per-frame [`ResidencyParams`] from the live camera world position + clip half-extent. The per-LOD
/// `level_box` (the clipmap shell on each grid) + the WG-cell tiling of it are computed here (the only per-frame
/// CPU work) — bit-identical to the converge gate's `build_params` SSOT.
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
    // ENTER-CAP histogram scale: span the worst-case candidate distance (the coarsest LOD's level box corner,
    // `clip_half` bricks out on each axis ⇒ the box DIAGONAL) across `HIST_BUCKETS` buckets. `brick_span(MAX_LOD)`
    // is the coarsest brick edge in metres; a small +1 guard avoids a zero scale at half==0.
    let max_dist = (half.max(1) as f32) * brick_span(MAX_LOD) * 3.0_f32.sqrt();
    let hist_scale = HIST_BUCKETS as f32 / max_dist;
    ResidencyParams {
        levels,
        clip_half_bricks: half,
        total_cells: offset,
        hist_scale,
        _pad1: 0,
        cam_world: cam,
        _pad2: 0,
    }
}

/// Enter-cap distance histogram buckets — MUST equal `HIST_BUCKETS` in `voxel_residency.wgsl`.
const HIST_BUCKETS: u32 = 4096;

/// The maximum `total_cells` (shell WG-cells across all LODs) the front end's `shell_idx`/list buffers are sized
/// for, AND the candidate/enter/drop/pack/aabb list capacity. It bounds the transient per-frame work, NOT the
/// resident pool (that is `max_resident`). At clip_half=160 a measured cold-fill settles to ~143k resident
/// surface bricks; the per-frame candidate set (surface + halo) and the shell WG-cell union across 8 LODs stay
/// well under this. **Capped so the derived `present_size = next_pow2(2·LIST_CAP)` clear dispatch stays within the
/// 65535-workgroup 1D limit:** `present_size ≤ 2^21` ⇒ `present_size/64 = 32768 ≤ 65535`. The residency passes
/// index `gid.x` LINEARLY (no 2D fold), so every per-frame dispatch MUST be `≤ 65535` workgroups — the binding
/// constraint here. 1_000_000 ⇒ `present_size = next_pow2(2M) = 2^21`, list clear `= 1M/64 = 15625` WGs.
///
/// **Sized to fit the FULL per-frame DESIRED + CANDIDATE sets at the widest clip_half (160, Bistro).** At
/// clip_half=160 the Bistro `desired` (occupied-in-shell superset) measures ~870k and `cand` (surface) ~615k.
/// `desired_list`/`cand_list` MUST hold the WHOLE set — if `desired` overflows this cap, `build_present_flag`
/// (which runs over `LIST_CAP` invocations) never registers the tail of the desired set, so the resident bricks
/// beyond the cap fail `present_contains`, get DROPPED, and re-enter next frame ⇒ permanent THRASH (the G-c.4 BUG-2
/// non-convergence). 1M covers both with headroom while keeping `present_size = 2^21` (clear dispatch 32768 WGs).
const LIST_CAP: usize = 1_000_000;

// =========================================================================================================
//  The persistent live front end.
// =========================================================================================================

/// The live GPU-driven residency front end — a render-world resource holder. All residency-decision state
/// PERSISTS in these buffers across frames; the POOL (meta/voxel/palette/aabb) is the live scene's (rebound via
/// [`rebind_pool`] on a scene (re)allocation). One [`record_frame`] records the whole readback-free pipeline into
/// the caller's encoder.
pub struct GpuResidencyFrontEnd {
    half: i32,
    max_resident: u32,
    slot_table_size: u32,
    present_size: u32,

    // --- immutable scene inputs (the occupancy + core store + their config uniforms) ---
    diff_cfg_buf: wgpu::Buffer,
    pack_cfg_buf: wgpu::Buffer,
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

    // --- persistent counts + lists ---
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

    // --- the persistent slab allocators (the POOL itself is external — the scene's) ---
    index_slab_ctrl: wgpu::Buffer,
    index_slab_free: wgpu::Buffer,
    palette_slab_ctrl: wgpu::Buffer,
    palette_slab_free: wgpu::Buffer,
    /// The GPU `DenseSlot` table (binding 49): per resident slot, its current dense slab offsets + size selectors
    /// `[index_off, index_bits, palette_off, palette_k]` — read+updated by Pass D3/D0 to REUSE-in-place / FREE the
    /// OLD slab on re-pack/drop (the bound on the slab high-water). Mirror of the CPU `SlotState::dense`.
    slab_state: wgpu::Buffer,
    /// ENTER-CAP (binding 50): the per-frame candidate distance histogram (`HIST_BUCKETS` u32, cleared each frame).
    enter_hist: wgpu::Buffer,
    /// ENTER-CAP (binding 51): `[cut_bucket, room]` — the nearest-priority admission cut (computed each frame).
    enter_cap: wgpu::Buffer,

    // --- the change_count signal + its mappable staging ring (the non-blocking 1-frame-late mirror) ---
    change_count_buf: wgpu::Buffer,
    change_staging: Vec<wgpu::Buffer>,
    /// Round-robin staging index; `frame_parity` selects which ring slot this frame copies into and which the
    /// caller maps (the previous frame's). 2-deep so the map never aliases an in-flight copy.
    ring: usize,

    // --- the dummies for the comprehensive bind group ---
    dummy_in: wgpu::Buffer,
    dummy_out: wgpu::Buffer,
    dummy_dispatch: wgpu::Buffer,

    // --- pipelines (residency Pass A/A2/B/C/D + change-count) ---
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

    // --- pack-shader pipelines + their (pool-dependent) bind groups, rebuilt by `rebind_pool` ---
    cls_bgl: wgpu::BindGroupLayout,
    pack_bgl: wgpu::BindGroupLayout,
    aabb_bgl: wgpu::BindGroupLayout,
    p_classify: wgpu::ComputePipeline,
    p_pack: wgpu::ComputePipeline,
    p_aabb: wgpu::ComputePipeline,

    // --- the per-epoch occupancy/core bind group + the pool-dependent pack bind groups (rebuilt by `rebind_pool`) ---
    bound: Option<BoundScene>,
}

/// The bind groups that depend on either the per-epoch occupancy/core store OR the live scene's pool buffers.
/// Rebuilt by [`GpuResidencyFrontEnd::rebind_pool`] whenever the occupancy/core epoch changes or the scene
/// reallocates its pool (so the front end always writes the CURRENT scene's buffers).
struct BoundScene {
    /// The comprehensive residency bind group with the DUMMY at binding 7 (every pass but B0 binds it).
    res_bg: wgpu::BindGroup,
    /// The comprehensive residency bind group with the REAL `shell_dispatch` at binding 7 (Pass B0 only — wgpu
    /// forbids STORAGE+INDIRECT on one buffer in a single dispatch, so B0 alone binds it as STORAGE here).
    res_bg_b0: wgpu::BindGroup,
    /// `classify_brick` bind group (reads cores + neighbours; writes classify_out).
    cls_bg: wgpu::BindGroup,
    /// `pack_brick` bind group (writes the scene's voxel/palette/meta pool).
    pack_bg: wgpu::BindGroup,
    /// `write_aabb` bind group (writes the scene's aabb pool).
    aabb_bg: wgpu::BindGroup,
    /// The params uniform for THIS frame (re-uploaded each `record_frame`; held so the bind group's Arc is valid).
    params_buf: wgpu::Buffer,
}

impl GpuResidencyFrontEnd {
    /// Build the persistent front end (all pipelines + the persistent diff/slab/list/dispatch buffers). The POOL
    /// and the occupancy/core store are bound LATER by [`rebind_pool`] (they are the live scene's / per-epoch).
    /// `max_resident` is the scene capacity (== `SceneKeepAlive` slot capacity == the CPU packer cap).
    #[allow(clippy::too_many_lines)]
    pub fn new(device: &wgpu::Device, half: i32, max_resident: u32) -> Self {
        let slot_table_size = (max_resident as usize * 2).max(2).next_power_of_two() as u32;
        let present_size = (LIST_CAP * 2).max(2).next_power_of_two() as u32;
        let list_cap = LIST_CAP;

        let diff_cfg = DiffConfig { slot_table_size, present_size, max_resident, refine_descent_cap: REFINE_DESCENT_CAP };
        let diff_cfg_buf = buf_init(device, "res_diff_cfg", bytemuck::bytes_of(&diff_cfg), wgpu::BufferUsages::UNIFORM);
        // core_table_size is filled by rebind_pool (it depends on the per-epoch core store); start at 2 (a valid pow2).
        let pack_cfg = PackConfig { core_table_size: 2, max_resident, _pad0: 0, _pad1: 0 };
        let pack_cfg_buf = buf_init(device, "res_pack_cfg", bytemuck::bytes_of(&pack_cfg), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let index_pool_base = buf_init(device, "res_index_pool_base", bytemuck::cast_slice(&[0u32]), wgpu::BufferUsages::STORAGE);
        let palette_pool_base = buf_init(device, "res_palette_pool_base", bytemuck::cast_slice(&[0u32]), wgpu::BufferUsages::STORAGE);

        // persistent diff state — slot_table empty, free-list full (every slot free), quarantine empty.
        let mut slot_init = vec![0u32; slot_table_size as usize * 5];
        for s in slot_init.chunks_exact_mut(5) {
            s[3] = EMPTY_LOD;
        }
        let slot_table_buf = buf_init(device, "res_slot_table", bytemuck::cast_slice(&slot_init), storage_usage());
        let free_init: Vec<u32> = (0..max_resident).collect();
        let free_ring_buf = buf_init(device, "res_free_ring", bytemuck::cast_slice(&free_init), storage_usage());
        let free_ctrl_buf = buf_init(device, "res_free_ctrl", bytemuck::cast_slice(&[0u32, max_resident]), storage_usage());
        let quar_ring_buf = storage_buf(device, "res_quar_ring", (max_resident as u64) * 4);
        let quar_ctrl_buf = buf_init(device, "res_quar_ctrl", bytemuck::cast_slice(&[0u32, 0u32]), storage_usage());

        // persistent per-frame hashes (init EMPTY; Pass A2 re-clears each frame).
        let present_init = vec![EMPTY_LOD; present_size as usize * 4];
        let present_flag = buf_init(device, "res_present_flag", bytemuck::cast_slice(&present_init), storage_usage());
        let dirty_init = vec![EMPTY_LOD; slot_table_size as usize * 4];
        let dirty_flag = buf_init(device, "res_dirty_flag", bytemuck::cast_slice(&dirty_init), storage_usage());

        // persistent counts + lists.
        let mk0 = |label: &str| buf_init(device, label, bytemuck::bytes_of(&0u32), storage_usage());
        let shell_count = mk0("res_shell_count");
        let shell_idx = storage_buf(device, "res_shell_idx", (list_cap * 4) as u64);
        let cand_count = mk0("res_cand_count");
        let cand_list = storage_buf(device, "res_cand_list", (list_cap * 16) as u64);
        let desired_count = mk0("res_desired_count");
        let desired_list = storage_buf(device, "res_desired_list", (list_cap * 16) as u64);
        let enter_count = mk0("res_enter_count");
        let enter_list = storage_buf(device, "res_enter_list", (list_cap * 16) as u64);
        let drop_count = mk0("res_drop_count");
        let drop_list = storage_buf(device, "res_drop_list", (list_cap * 16) as u64);
        let drop_decision = storage_buf(device, "res_drop_decision", (slot_table_size as u64) * 4);
        let dirty_count = mk0("res_dirty_count");
        let dirty_list = storage_buf(device, "res_dirty_list", (list_cap * 16) as u64);
        let dirty_slot = storage_buf(device, "res_dirty_slot", (list_cap * 4) as u64);
        let pack_count = mk0("res_pack_count");
        let pack_commands = storage_buf(device, "res_pack_commands", (list_cap * 60) as u64);
        let aabb_count = mk0("res_aabb_count");
        let aabb_commands = storage_buf(device, "res_aabb_commands", (list_cap * 32) as u64);
        let classify_commands = storage_buf(device, "res_classify_commands", (list_cap * 16) as u64);
        let neighbour_indices = storage_buf(device, "res_neighbour_indices", (list_cap as u64) * 27 * 4);
        let classify_out = storage_buf(device, "res_classify_out", (list_cap * 4 * 4) as u64);

        // indirect dispatch buffers — seeded (0,1,1); Pass A re-seeds each frame.
        let shell_dispatch = buf_init(device, "res_shell_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let pack_dispatch = buf_init(device, "res_pack_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let aabb_dispatch = buf_init(device, "res_aabb_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());
        let classify_dispatch = buf_init(device, "res_classify_dispatch", bytemuck::cast_slice(&[0u32, 1, 1]), dispatch_usage());

        // slab allocators (the index/palette arenas — bump+free-list per size class, pre-sized to capacity).
        let index_ctrl_init = vec![0u32; 1 + 5 * 2];
        let index_slab_ctrl = buf_init(device, "res_index_slab_ctrl", bytemuck::cast_slice(&index_ctrl_init), storage_usage());
        let index_slab_free = storage_buf(device, "res_index_slab_free", (5 * max_resident as u64) * 4);
        let palette_ctrl_init = vec![0u32; 1 + 16 * 2];
        let palette_slab_ctrl = buf_init(device, "res_palette_slab_ctrl", bytemuck::cast_slice(&palette_ctrl_init), storage_usage());
        let palette_slab_free = storage_buf(device, "res_palette_slab_free", (16 * max_resident as u64) * 4);
        // The GPU DenseSlot table — 4 u32 / slot, all 0 (no slot has a dense slab yet).
        let slab_state = storage_buf(device, "res_slab_state", (max_resident as u64) * 4 * 4);
        // ENTER-CAP — the candidate distance histogram + the `[cut_bucket, room]` cut.
        let enter_hist = storage_buf(device, "res_enter_hist", (HIST_BUCKETS as u64) * 4);
        let enter_cap = buf_init(device, "res_enter_cap", bytemuck::cast_slice(&[HIST_BUCKETS, 0u32]), storage_usage());

        // the change_count signal + a 2-deep mappable staging ring (the non-blocking mirror).
        let change_count_buf = buf_init(device, "res_change_count", bytemuck::bytes_of(&0u32), storage_usage());
        let change_staging: Vec<wgpu::Buffer> = (0..2)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "res_change_staging0" } else { "res_change_staging1" }),
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let dummy_in = buf_init(device, "res_dummy_in", bytemuck::cast_slice(&[0u32; 4]), wgpu::BufferUsages::STORAGE);
        let dummy_out = storage_buf(device, "res_dummy_out", 16);
        let dummy_dispatch = storage_buf(device, "res_dummy_dispatch", 16);

        // shaders.
        let res_src = include_str!("../../assets/shaders/voxel_residency.wgsl");
        let res_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel_residency_live"),
            source: wgpu::ShaderSource::Wgsl(res_src.into()),
        });
        let pack_src = include_str!("../../assets/shaders/voxel_pack.wgsl");
        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel_pack_live"),
            source: wgpu::ShaderSource::Wgsl(pack_src.into()),
        });

        // the comprehensive residency bind-group layout (0..=48). Each pipeline references only its subset.
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
        entries.push(storage_entry(49, false)); // slab_state (GPU DenseSlot table — read+write in Pass D3/D0)
        entries.push(storage_entry(50, false)); // enter_hist (enter-cap distance histogram)
        entries.push(storage_entry(51, false)); // enter_cap ([cut_bucket, room])
        let res_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("res_live_bgl"),
            entries: &entries,
        });
        let res_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("res_live_pl"),
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

        // pack shader pipelines (their bind GROUPS are pool-dependent → built in rebind_pool).
        let cls_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("res_cls_bgl"),
            entries: &[storage_entry(1, true), storage_entry(2, true), storage_entry(8, false), storage_entry(9, true)],
        });
        let cls_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("res_cls_pl"),
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
        let pack_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("res_pack_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let pack_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("res_pack_pl"),
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
        let aabb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("res_aabb_bgl"),
            entries: &[storage_entry(6, false), storage_entry(7, true)],
        });
        let aabb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("res_aabb_pl"),
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

        Self {
            half,
            max_resident,
            slot_table_size,
            present_size,
            diff_cfg_buf,
            pack_cfg_buf,
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
            index_slab_ctrl,
            index_slab_free,
            palette_slab_ctrl,
            palette_slab_free,
            slab_state,
            enter_hist,
            enter_cap,
            change_count_buf,
            change_staging,
            ring: 0,
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
            cls_bgl,
            pack_bgl,
            aabb_bgl,
            p_classify,
            p_pack,
            p_aabb,
            bound: None,
        }
    }

    /// (Re)bind the per-epoch occupancy + core store AND the live scene's pool buffers. Called on a scene
    /// (re)allocation or an occupancy/core epoch change. Resets the persistent diff state to EMPTY (a fresh scene
    /// streams in cold) so the slot table never references a stale pool's slots, and patches `pack_cfg`'s
    /// `core_table_size` for the new store. `meta`/`voxel`/`brick_palettes`/`aabb` are the scene's persistent pool
    /// buffers (group-0 in the renderer). The bind groups reference these directly — the GPU front end writes the
    /// SAME pool the renderer consumes.
    #[allow(clippy::too_many_arguments)]
    pub fn rebind_pool(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        occ: &GpuResidencyBuffers,
        core: &GpuBrickCoreBuffers,
        meta: &wgpu::Buffer,
        voxel: &wgpu::Buffer,
        brick_palettes: &wgpu::Buffer,
        aabb: &wgpu::Buffer,
    ) {
        // Patch pack_cfg.core_table_size for the new core store.
        let pack_cfg = PackConfig { core_table_size: core.table_size, max_resident: self.max_resident, _pad0: 0, _pad1: 0 };
        queue.write_buffer(&self.pack_cfg_buf, 0, bytemuck::bytes_of(&pack_cfg));

        // Reset the persistent diff state to EMPTY (cold-stream the new scene). The slab allocators reset too so
        // the new scene allocates from a fresh arena (the scene's pool is freshly (re)allocated alongside).
        self.reset_state(queue);

        // The params uniform is uploaded per frame; create a placeholder so the bind group is valid pre-first-frame.
        let placeholder = build_params([0.0; 3], self.half);
        let params_buf = buf_init(device, "res_params", bytemuck::bytes_of(&placeholder), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        let res_bg = self.build_res_bg(device, occ, &params_buf, &self.dummy_dispatch, core, meta);
        let res_bg_b0 = self.build_res_bg(device, occ, &params_buf, &self.shell_dispatch, core, meta);

        let cls_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_cls_bg"),
            layout: &self.cls_bgl,
            entries: &[bind(1, &core.cores), bind(2, &self.neighbour_indices), bind(8, &self.classify_out), bind(9, &self.classify_commands)],
        });
        let pack_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_pack_bg"),
            layout: &self.pack_bgl,
            entries: &[
                bind(0, &self.pack_commands),
                bind(1, &core.cores),
                bind(2, &self.neighbour_indices),
                bind(3, voxel),
                bind(4, brick_palettes),
                bind(5, meta),
            ],
        });
        let aabb_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_aabb_bg"),
            layout: &self.aabb_bgl,
            entries: &[bind(6, aabb), bind(7, &self.aabb_commands)],
        });
        self.bound = Some(BoundScene { res_bg, res_bg_b0, cls_bg, pack_bg, aabb_bg, params_buf });
    }

    /// Drop the bound scene (a scene switch away to a non-GPU-residency scene). The next [`rebind_pool`] cold-starts.
    pub fn unbind(&mut self) {
        self.bound = None;
    }

    /// Would driving `cam` this frame OVERFLOW the transient list/dispatch capacity (the shell WG-cell union >
    /// `LIST_CAP`, or a per-frame dispatch > the 65535-workgroup 1D limit)? The caller skips the GPU drive (CPU
    /// fallback) when true, so an over-wide clip_half / over-large scene never submits an invalid dispatch. (For
    /// the in-RAM scenes the front end binds today this never trips; it is the guard for the future streamed path.)
    pub fn would_overflow(&self, cam: [f32; 3]) -> bool {
        let params = build_params(cam, self.half);
        let b0_wgs = params.total_cells.div_ceil(64);
        params.total_cells as usize > LIST_CAP || b0_wgs > 65535
    }

    /// Whether the front end currently has a scene bound (occupancy + core + pool).
    pub fn is_bound(&self) -> bool {
        self.bound.is_some()
    }

    /// Reset the persistent diff/slab state to EMPTY (GPU-side via `queue.write_buffer`). Called by `rebind_pool`
    /// so a freshly-(re)allocated scene pool streams in cold (no stale slot→pool references).
    fn reset_state(&self, queue: &wgpu::Queue) {
        let mut slot_init = vec![0u32; self.slot_table_size as usize * 5];
        for s in slot_init.chunks_exact_mut(5) {
            s[3] = EMPTY_LOD;
        }
        queue.write_buffer(&self.slot_table_buf, 0, bytemuck::cast_slice(&slot_init));
        let free_init: Vec<u32> = (0..self.max_resident).collect();
        queue.write_buffer(&self.free_ring_buf, 0, bytemuck::cast_slice(&free_init));
        queue.write_buffer(&self.free_ctrl_buf, 0, bytemuck::cast_slice(&[0u32, self.max_resident]));
        queue.write_buffer(&self.quar_ctrl_buf, 0, bytemuck::cast_slice(&[0u32, 0u32]));
        let present_init = vec![EMPTY_LOD; self.present_size as usize * 4];
        queue.write_buffer(&self.present_flag, 0, bytemuck::cast_slice(&present_init));
        let dirty_init = vec![EMPTY_LOD; self.slot_table_size as usize * 4];
        queue.write_buffer(&self.dirty_flag, 0, bytemuck::cast_slice(&dirty_init));
        let index_ctrl_init = vec![0u32; 1 + 5 * 2];
        queue.write_buffer(&self.index_slab_ctrl, 0, bytemuck::cast_slice(&index_ctrl_init));
        let palette_ctrl_init = vec![0u32; 1 + 16 * 2];
        queue.write_buffer(&self.palette_slab_ctrl, 0, bytemuck::cast_slice(&palette_ctrl_init));
        // Zero the GPU DenseSlot table (no slot has a dense slab in a freshly cold-started scene).
        let slab_state_init = vec![0u32; self.max_resident as usize * 4];
        queue.write_buffer(&self.slab_state, 0, bytemuck::cast_slice(&slab_state_init));
        queue.write_buffer(&self.change_count_buf, 0, bytemuck::bytes_of(&0u32));
    }

    /// Build the comprehensive residency bind group, binding `slot7` at binding 7 (the dummy for every pass but
    /// B0, the real `shell_dispatch` for B0). `core`'s table + cores feed the Pass-D halo lookup.
    fn build_res_bg(
        &self,
        device: &wgpu::Device,
        occ: &GpuResidencyBuffers,
        params_buf: &wgpu::Buffer,
        slot7: &wgpu::Buffer,
        core: &GpuBrickCoreBuffers,
        meta: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_bg"),
            layout: &self.res_bgl,
            entries: &[
                bind(0, &occ.header),
                bind(1, &occ.entries),
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
                bind(25, &core.table),
                bind(26, &core.cores),
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
                bind(37, meta), // binding 37 (meta) — the LIVE scene pool; Pass D's drop/uniform path GPU-writes it
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

    /// Record ONE whole readback-free residency frame for `cam` into the caller's `enc`: Pass A → A2 → B0 →
    /// B(indirect) → C → write_change_count → D → classify/pack/write_aabb (INDIRECT, self-gating). The pack tail
    /// writes the LIVE scene pool buffers. Also copies `change_count` into this frame's staging-ring slot for the
    /// non-blocking 1-frame-late mirror. Does NOT submit — the caller closes the encoder (so the AS build rides
    /// the same submit). Returns the `total_cells` for this frame (a diagnostic / overflow guard).
    ///
    /// Caller MUST have called [`rebind_pool`] (asserts otherwise — a programming error).
    pub fn record_frame(&mut self, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, cam: [f32; 3]) -> u32 {
        let bound = self.bound.as_ref().expect("record_frame without a bound scene (call rebind_pool)");
        let params = build_params(cam, self.half);
        // The shell WG-cell union must fit `shell_idx` (sized to LIST_CAP). If a (very wide clip_half / very large
        // scene) frame would overflow, the caller's `total_cells > LIST_CAP` check (see `record_frame`'s return)
        // skips the GPU drive for this frame — here we still record but the B0 dispatch is clamped by its own
        // bound check, so an overflow can't corrupt past the buffer (it just under-enumerates the farthest cells).
        queue.write_buffer(&bound.params_buf, 0, bytemuck::bytes_of(&params));

        let clear_wgs = self.slot_table_size.max(self.present_size).div_ceil(64).max(1);
        let list_wgs = (LIST_CAP as u32).div_ceil(64).max(1);
        let bg = &bound.res_bg;
        let bg_b0 = &bound.res_bg_b0;

        // Pass A0 — clear enumerate/pack counts + SEED the indirect dispatches (0,1,1) (self-gating).
        compute(enc, &self.p_seed, bg, 1);
        // Pass A — drain the previous frame's quarantine + clear diff enter/drop counts.
        compute(enc, &self.p_release, bg, 1);
        // Pass A2 — clear the per-frame hashes + change_count.
        compute(enc, &self.p_clear, bg, clear_wgs);
        // Pass B0 — shell dispatch prep (binds the real shell_dispatch at 7).
        compute(enc, &self.p_b0, bg_b0, params.total_cells.div_ceil(64).max(1));
        // Pass B — enumerate (INDIRECT over the GPU-written shell_dispatch).
        record_indirect(enc, &self.p_b, bg, &self.shell_dispatch);
        // Pass C — present-flag, drop-mark, drop-apply, enter (sized over the LIST CAP / slot-table, no readback).
        compute(enc, &self.p_present, bg, list_wgs);
        compute(enc, &self.p_mark, bg, self.slot_table_size.div_ceil(64).max(1));
        compute(enc, &self.p_apply, bg, self.slot_table_size.div_ceil(64).max(1));
        // Enter-cap (BUG-2 nearest-priority): build the candidate distance histogram, compute the cut bucket from
        // the live pool room, THEN enter only the nearest-`room` candidates (≤ pool cap, stable ⇒ converges).
        compute(enc, &self.p_cap_hist, bg, list_wgs);
        compute(enc, &self.p_cap_compute, bg, 1);
        compute(enc, &self.p_enter, bg, list_wgs);
        // publish change_count (= enter + drop).
        compute(enc, &self.p_chg, bg, 1);
        // Pass D — dirty build, drops, neighbours, then classify/pack/write_aabb INDIRECT (the self-gating tail).
        compute(enc, &self.p_d_dirty, bg, list_wgs);
        compute(enc, &self.p_d_drops, bg, list_wgs);
        compute(enc, &self.p_d_nbr, bg, list_wgs);
        record_indirect(enc, &self.p_classify, &bound.cls_bg, &self.classify_dispatch);
        compute(enc, &self.p_d_cmd, bg, list_wgs);
        record_indirect(enc, &self.p_pack, &bound.pack_bg, &self.pack_dispatch);
        record_indirect(enc, &self.p_aabb, &bound.aabb_bg, &self.aabb_dispatch);

        // Copy change_count into THIS frame's staging-ring slot (the next poll reads the OTHER slot — last frame).
        enc.copy_buffer_to_buffer(&self.change_count_buf, 0, &self.change_staging[self.ring], 0, 4);

        params.total_cells
    }

    /// Read the PREVIOUS frame's `change_count` out-of-band (the one permitted CPU↔GPU sync, §3.1) and advance the
    /// staging ring so the NEXT [`record_frame`] copies into the slot we just read. Non-blocking: maps the slot
    /// the LAST frame's copy targeted (a frame ago, so its submit has completed by now — no host stall on the
    /// hot path). Returns `None` on the very first frame (no prior copy yet) or if the map isn't ready.
    ///
    /// Call this ONCE per frame, BEFORE `record_frame`, to decide whether to record the AS build (a converged
    /// static camera returns 0 → skip the BLAS rebuild → fully idle). `>0` ⇒ record the AS build this frame.
    pub fn poll_change_count(&mut self, device: &wgpu::Device) -> Option<u32> {
        // The slot the PREVIOUS frame's record_frame copied into is `1 - self.ring` (we toggle after each frame).
        // On the first ever frame nothing has been copied — return None (the caller records the AS build anyway).
        let read_slot = 1 - self.ring;
        let staging = &self.change_staging[read_slot];
        // Try to map. We poll (non-blocking) — if the previous submit hasn't landed yet, treat as "changed" (None).
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        // Pump the device once WITHOUT blocking the GPU timeline indefinitely. `poll` here is cheap; the copy is a
        // 4-byte D2H that completed a frame ago.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        match rx.try_recv() {
            Ok(true) => {
                let v = match staging.slice(..).get_mapped_range() {
                    Ok(d) => Some(u32::from_le_bytes([d[0], d[1], d[2], d[3]])),
                    Err(_) => None,
                };
                staging.unmap();
                v
            }
            _ => None,
        }
    }

    /// Advance the staging ring after this frame's `record_frame` (toggle which slot the next frame writes / the
    /// next poll reads). Call AFTER `record_frame` for this frame.
    pub fn advance_ring(&mut self) {
        self.ring = 1 - self.ring;
    }

    /// **Diagnostic ONLY** — the resident-pool capacity (`max_resident`), for the paged-drive diagnostic log line.
    pub fn max_resident_diag(&self) -> u32 {
        self.max_resident
    }

    /// **Diagnostic ONLY** (blocking) — the index + palette slab high-water marks (WORDS used) vs the pool reserve.
    /// An overflow (high-water > the live scene pool's word capacity) ⇒ the GPU slab allocator wrote out of bounds.
    pub fn diag_slab_highwater(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (u32, u32) {
        let rb = |b: &wgpu::Buffer| -> u32 {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("res_diag_slab_rb"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("res_diag_slab") });
            enc.copy_buffer_to_buffer(b, 0, &staging, 0, 4);
            queue.submit(core::iter::once(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let v = match staging.slice(..).get_mapped_range() {
                Ok(d) => u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
                Err(_) => u32::MAX,
            };
            staging.unmap();
            v
        };
        (rb(&self.index_slab_ctrl), rb(&self.palette_slab_ctrl))
    }

    /// **Diagnostic ONLY** (blocking) — read back the per-frame counts (aabb_count, pack_count, enter, drop, change)
    /// after a recorded frame. Localizes whether Pass B/C/D produced work. Returns (aabb, pack, cand, desired, change).
    pub fn diag_counts(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> (u32, u32, u32, u32, u32) {
        let rb = |b: &wgpu::Buffer| -> u32 {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("res_diag_cnt_rb"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("res_diag_cnt") });
            enc.copy_buffer_to_buffer(b, 0, &staging, 0, 4);
            queue.submit(core::iter::once(enc.finish()));
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let v = match staging.slice(..).get_mapped_range() {
                Ok(d) => u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
                Err(_) => u32::MAX,
            };
            staging.unmap();
            v
        };
        (
            rb(&self.aabb_count),
            rb(&self.pack_count),
            rb(&self.cand_count),
            rb(&self.desired_count),
            rb(&self.change_count_buf),
        )
    }

    /// **Diagnostic ONLY** (blocking readback — never on the hot path). The number of LIVE slots in the persistent
    /// slot table (resident `(coord,lod)` keys), read back by mapping the slot-table buffer. Used by the paged-drive
    /// diagnostic gate to confirm the front end converges (and is not entering the full pool). Blocks the queue.
    pub fn diag_resident_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        let words = self.slot_table_size as usize * 5;
        let size = (words * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("res_diag_slot_rb"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("res_diag_rb") });
        enc.copy_buffer_to_buffer(&self.slot_table_buf, 0, &staging, 0, size);
        queue.submit(core::iter::once(enc.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = match staging.slice(..).get_mapped_range() {
            Ok(d) => d,
            Err(_) => return u32::MAX,
        };
        let words_u32: &[u32] = bytemuck::cast_slice(&data);
        let mut live = 0u32;
        for s in words_u32.chunks_exact(5) {
            if s[3] != EMPTY_LOD {
                live += 1;
            }
        }
        drop(data);
        staging.unmap();
        live
    }
}

// =========================================================================================================
//  small wgpu helpers (mirror of the converge gate's; one SSOT for the live + test buffer shapes).
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
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
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
fn record_indirect(enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, indirect: &wgpu::Buffer) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(p);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups_indirect(indirect, 0);
}
fn compute(enc: &mut wgpu::CommandEncoder, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, wgs: u32) {
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(p);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(wgs.max(1), 1, 1);
}

/// The span (world meters) of a LOD0 brick — re-exported for the caller's `clip_half` reach diagnostics.
pub fn lod0_span() -> f32 {
    brick_span(0)
}

/// Words per slot in the meta buffer (48-byte `GpuBrickMeta`). Re-exported so the caller can size the pool.
pub const fn meta_words() -> usize {
    META_WORDS
}
