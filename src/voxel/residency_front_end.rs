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
/// Slots per BLAS chunk (a slot-band) — MUST equal `raytrace.rs::CHUNK_SLOTS` and `voxel_pack.wgsl`'s `CHUNK_SLOTS`.
/// The per-frame dirty-chunk mask `write_aabb` fills (one bit per `chunk = slot / CHUNK_SLOTS`) drives the CPU's
/// targeted BLAS rebuild — only the chunks that actually changed, not a blind sweep.
const CHUNK_SLOTS: u32 = 512;
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
    /// 4-S1: LODs >= this are the always-resident coarse BACKDROP (exempt from the budget cut). `MAX_LOD + 1` = OFF.
    backdrop_lod: u32,
    /// ENTER-CAP: the camera world position (the nearest-priority distance rank centre).
    cam_world: [f32; 3],
    /// 4-S2/S3: the current frame counter (rides the old `_pad2`; no layout change). The residency reads
    /// `frame - last_used[slot]` to keep ray-recently-used bricks (ray-guided) + age the LRU. 0 when demand is off.
    frame: u32,
    /// 4-S2/S3: ray-guided keep + LRU master toggle (1 = on). When 0 the residency ignores `last_used` (distance cut).
    demand: u32,
    /// 4-S4: backdrop LODs reach `clip_half · backdrop_reach` (live; 1 = no extension).
    backdrop_reach: u32,
    /// 4-S2/S3: a brick ray-hit within this many frames is kept beyond the cut (live ray-keep window).
    ray_keep_frames: u32,
    _pad3: u32,
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
    index_stride: u32,   // WORDS per slot in the index (voxel) pool — fixed per-slot slab (no allocator)
    palette_stride: u32, // WORDS per slot in the palette pool
}

/// Build the per-frame [`ResidencyParams`] from the live camera world position + clip half-extent. The per-LOD
/// `level_box` (the clipmap shell on each grid) + the WG-cell tiling of it are computed here (the only per-frame
/// CPU work) — bit-identical to the converge gate's `build_params` SSOT.
#[allow(clippy::too_many_arguments)]
fn build_params(
    cam: [f32; 3],
    half: i32,
    backdrop_lod: u32,
    frame: u32,
    demand: bool,
    backdrop_reach: u32,
    ray_keep_frames: u32,
) -> ResidencyParams {
    let mut levels = [LevelParams::zeroed(); LODS];
    let mut offset = 0u32;
    for lod in 0..=MAX_LOD {
        // 4-S4: backdrop LODs use the extended reach so the CPU cell grid covers exactly what the WGSL
        // `level_resident` (which applies `backdrop_reach` internally) accepts — else the extra backdrop bricks
        // would never be enumerated/entered.
        let (lo, hi) = level_box_pub(cam, lod, lod_clip_half(lod, half, backdrop_lod, backdrop_reach));
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
        backdrop_lod,
        cam_world: cam,
        frame,
        demand: u32::from(demand),
        backdrop_reach: backdrop_reach.max(1),
        ray_keep_frames,
        _pad3: 0,
    }
}

/// Enter-cap distance histogram buckets — MUST equal `HIST_BUCKETS` in `voxel_residency.wgsl`.
const HIST_BUCKETS: u32 = 4096;

/// 4-S4 — the effective clip half-extent for `lod`: `clip_half` for fine LODs, `clip_half · backdrop_reach` for the
/// pinned coarse backdrop (LODs >= `backdrop_lod`). `backdrop_lod > MAX_LOD` (off) ⇒ always `half` (no extension).
/// `backdrop_reach` is the LIVE editor lever (`VoxelRtResidencySettings`); MUST match the WGSL `lod_half` (which uses
/// `params.backdrop_reach`) + the pager's `desired_regions`.
#[inline]
pub fn lod_clip_half(lod: u32, half: i32, backdrop_lod: u32, backdrop_reach: u32) -> i32 {
    if lod >= backdrop_lod { half * backdrop_reach.max(1) as i32 } else { half }
}

/// The maximum `total_cells` (shell WG-cells across all LODs) the front end's `shell_idx`/list buffers are sized
/// for, AND the candidate/enter/drop/pack/aabb list capacity. It bounds the transient per-frame work, NOT the
/// resident pool (that is `max_resident`).
///
/// **Size-agnostic dispatch (memory-neutral):** the per-frame residency passes run at @workgroup_size(256) and
/// the indirect shell enumerate (`enumerate_shells`) is 2D-folded (`finalize_shell_dispatch_2d`), so no per-frame
/// dispatch hits the 65535-workgroup-per-dimension cap regardless of how many cells/bricks a frame touches. This
/// FIXES a real correctness ceiling: a dense scene with > 65535 solid 8³ WG-cells previously under-ran the 1D
/// enumerate dispatch (silent holes). It costs ZERO extra memory — it removes a dispatch limit, not a size cap.
///
/// **Why LIST_CAP stays 1M (memory budget — must run on 8 GB VRAM):** the transient lists scale with LIST_CAP;
/// the largest, `neighbour_indices`, is 27 u32/entry = `LIST_CAP·108 B` = 108 MB at 1M (fits wgpu's DEFAULT
/// 128 MB `max_storage_buffer_binding_size` — no device-limit raise needed anywhere). Growing LIST_CAP to widen
/// the view would balloon transient VRAM (4M ⇒ 432 MB for this one buffer) — the WRONG lever on an 8 GB device.
/// A wider view at BOUNDED transient memory comes from TILING the shell enumeration (process the cell union in
/// LIST_CAP-sized windows), and a scene-size-independent resident set from demand/LRU streaming — see
/// docs/DYNAMIC_LARGE_SCENE_PLAN.md (Phases 2-tiling / 4-demand). NOT from a bigger transient cap.
///
/// `desired_list`/`cand_list` MUST hold the WHOLE per-frame desired+candidate set — if `desired` overflows this
/// cap, `build_present_flag` never registers the tail of the desired set, so resident bricks beyond the cap fail
/// `present_contains`, get DROPPED, and re-enter next frame ⇒ permanent THRASH (the G-c.4 BUG-2 non-convergence).
/// At clip_half=160 Bistro measures `desired` ~870k / `cand` ~615k — under 1M. `would_overflow` skips the drive
/// (never silently truncates) rather than exceed this; tiling will replace that skip with windowed enumeration.
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
    /// 4-S1: the coarse-backdrop LOD threshold (LODs >= this are pinned, exempt from the budget cut). Set once at
    /// construction from `ADVENTURE_BACKDROP_LOD` (default `MAX_LOD + 1` = OFF). Fed into `ResidencyParams` per frame.
    backdrop_lod: u32,
    /// 4-S2/S3: ray-guided keep + LRU master toggle (live; set per-frame from `VoxelRtResidencySettings`).
    demand: bool,
    /// 4-S4: live backdrop reach multiplier.
    backdrop_reach: u32,
    /// 4-S2/S3: live ray-keep window (frames).
    ray_keep_frames: u32,
    /// 4-S2/S3: frame counter (bumped per `record_frame`), fed into `ResidencyParams` to age `last_used`.
    frame: u32,
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

    /// ENTER-CAP (binding 50): the per-frame candidate distance histogram (`HIST_BUCKETS` u32, cleared each frame).
    enter_hist: wgpu::Buffer,
    /// ENTER-CAP (binding 51): `[cut_bucket, room]` — the nearest-priority admission cut (computed each frame).
    enter_cap: wgpu::Buffer,

    // --- the change_count signal + its mappable staging ring (the non-blocking 1-frame-late mirror) ---
    change_count_buf: wgpu::Buffer,
    change_staging: Vec<wgpu::Buffer>,

    // --- the per-frame DIRTY-CHUNK bitmask (the targeted AS-rebuild driver) + its mappable staging ring ---
    /// One bit per BLAS chunk (`chunk = slot / CHUNK_SLOTS`); `write_aabb` atomically sets the bit for every
    /// changed slot's chunk. Cleared each frame, copied to the staging ring, read back 1-frame-late so the CPU
    /// rebuilds ONLY the chunks that changed (not a blind sweep of the mostly-empty pool).
    dirty_chunk_buf: wgpu::Buffer,
    dirty_chunk_staging: Vec<wgpu::Buffer>,
    /// `ceil(n_chunks / 32)` — the u32 word count of the dirty-chunk mask.
    dirty_mask_words: u32,
    /// `ceil(max_resident / CHUNK_SLOTS)` — the number of BLAS chunks (the mask's bit count upper bound).
    n_chunks: u32,
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
    p_mark: wgpu::ComputePipeline,
    p_apply: wgpu::ComputePipeline,
    p_cap_compute: wgpu::ComputePipeline,
    /// FUSED enter side (live): enumerate surface bricks → global histogram (Pass A) / enter below cut (Pass B).
    /// No candidate_list — the view is bounded by the surface-cell count, not a per-frame candidate cap.
    p_enum_hist: wgpu::ComputePipeline,
    p_enum_enter: wgpu::ComputePipeline,
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
    p_fin_classify: wgpu::ComputePipeline,
    p_fin_pack: wgpu::ComputePipeline,
    p_fin_shell: wgpu::ComputePipeline,

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
        // Phase 4: the dynamic-residency levers are driven LIVE per-frame from `VoxelRtResidencySettings` (the editor
        // panel) via `set_residency_levers`. Initialise OFF (identical to pre-Phase-4) until the first frame sets them.
        let backdrop_lod = MAX_LOD + 1;

        let diff_cfg = DiffConfig { slot_table_size, present_size, max_resident, refine_descent_cap: REFINE_DESCENT_CAP };
        let diff_cfg_buf = buf_init(device, "res_diff_cfg", bytemuck::bytes_of(&diff_cfg), wgpu::BufferUsages::UNIFORM);
        // core_table_size is filled by rebind_pool (it depends on the per-epoch core store); start at 2 (a valid pow2).
        let pack_cfg = PackConfig { core_table_size: 2, max_resident, index_stride: 0, palette_stride: 0 };
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

        // ENTER-CAP — the candidate distance histogram + the `[cut_bucket, room]` cut.
        // HIST_BUCKETS distance bins + 1 trailing slot = the 4-S1 backdrop-reserve counter (BACKDROP_RESERVE_SLOT).
        let enter_hist = storage_buf(device, "res_enter_hist", (HIST_BUCKETS as u64 + 1) * 4);
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

        // the per-frame DIRTY-CHUNK bitmask + its 2-deep mappable staging ring (1-frame-late, like change_count).
        let n_chunks = max_resident.div_ceil(CHUNK_SLOTS).max(1);
        let dirty_mask_words = n_chunks.div_ceil(32).max(1);
        let dirty_chunk_buf =
            buf_init(device, "res_dirty_chunk", bytemuck::cast_slice(&vec![0u32; dirty_mask_words as usize]), storage_usage());
        let dirty_chunk_staging: Vec<wgpu::Buffer> = (0..2)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(if i == 0 { "res_dirty_chunk_staging0" } else { "res_dirty_chunk_staging1" }),
                    size: (dirty_mask_words as u64) * 4,
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
        for b in 27..=40 {
            entries.push(storage_entry(b, false));
        }
        entries.push(storage_entry(45, true)); // index_pool_base (word base of the fixed per-slot index pool)
        entries.push(storage_entry(46, true)); // palette_pool_base
        entries.push(storage_entry(47, true));
        entries.push(storage_entry(48, false));
        entries.push(storage_entry(50, false)); // enter_hist (enter-cap distance histogram)
        entries.push(storage_entry(51, false)); // enter_cap ([cut_bucket, room])
        entries.push(storage_entry(52, false)); // 4-S2/S3: last_used_frame per slot (ray-guided keep + LRU)
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
        let p_mark = mk_res("diff_drop_mark");
        let p_apply = mk_res("diff_drop_apply");
        let p_cap_compute = mk_res("enter_cap_compute");
        let p_enum_hist = mk_res("enumerate_histogram");
        let p_enum_enter = mk_res("enumerate_enter");
        let p_chg = mk_res("write_change_count");
        let p_d_dirty = mk_res("pack_build_dirty");
        let p_d_nbr = mk_res("pack_build_neighbours");
        let p_d_cmd = mk_res("pack_build_commands");
        let p_d_drops = mk_res("pack_build_drops");
        // Convert the per-brick classify/pack indirect dispatches from [count,1,1] to a 2D [x,y,1] grid so they can
        // exceed the 65535 workgroups-per-dimension limit (large scenes — Bistro ~610k bricks). 1 invocation each.
        let p_fin_classify = mk_res("finalize_classify_dispatch_2d");
        let p_fin_pack = mk_res("finalize_pack_dispatch_2d");
        let p_fin_shell = mk_res("finalize_shell_dispatch_2d");

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
                storage_entry(12, true), // pack_cmd_count (read) — the 2D-dispatch over-run guard
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
            // 11 = aabb_count (read): write_aabb_dirty gates on the real command count, not the buffer capacity.
            entries: &[storage_entry(6, false), storage_entry(7, true), storage_entry(10, false), storage_entry(11, true)],
        });
        let aabb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("res_aabb_pl"),
            bind_group_layouts: &[Some(&aabb_bgl)],
            immediate_size: 0,
        });
        let p_aabb = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("write_aabb_dirty"),
            layout: Some(&aabb_pl),
            module: &pack_module,
            entry_point: Some("write_aabb_dirty"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            half,
            backdrop_lod,
            demand: false,
            backdrop_reach: 4,
            ray_keep_frames: 30,
            frame: 0,
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
            enter_hist,
            enter_cap,
            change_count_buf,
            change_staging,
            dirty_chunk_buf,
            dirty_chunk_staging,
            dirty_mask_words,
            n_chunks,
            ring: 0,
            dummy_in,
            dummy_out,
            dummy_dispatch,
            res_bgl,
            p_seed,
            p_release,
            p_clear,
            p_b0,
            p_mark,
            p_apply,
            p_cap_compute,
            p_enum_hist,
            p_enum_enter,
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
            p_fin_classify,
            p_fin_pack,
            p_fin_shell,
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
        last_used: &wgpu::Buffer,
    ) {
        // Patch pack_cfg for the new core store + the per-slot slab strides. The index/palette pools are
        // FIXED-per-slot (each slot owns `stride` words at `slot·stride`) — derive the stride from the ACTUAL pool
        // buffer size ÷ max_resident so it always matches the allocation (mirrors RESERVE_*_WORDS_PER_BRICK). This
        // replaces the shared bump+free-list allocator (which raced ⇒ two live bricks aliasing one slab ⇒ garbage).
        let index_stride = (voxel.size() as u32 / 4) / self.max_resident;
        let palette_stride = (brick_palettes.size() as u32 / 4) / self.max_resident;
        let pack_cfg = PackConfig { core_table_size: core.table_size, max_resident: self.max_resident, index_stride, palette_stride };
        queue.write_buffer(&self.pack_cfg_buf, 0, bytemuck::bytes_of(&pack_cfg));

        // Reset the persistent diff state to EMPTY (cold-stream the new scene). The slab allocators reset too so
        // the new scene allocates from a fresh arena (the scene's pool is freshly (re)allocated alongside).
        self.reset_state(queue);

        // The params uniform is uploaded per frame; create a placeholder so the bind group is valid pre-first-frame.
        let placeholder = build_params([0.0; 3], self.half, self.backdrop_lod, 0, self.demand, self.backdrop_reach, self.ray_keep_frames);
        let params_buf = buf_init(device, "res_params", bytemuck::bytes_of(&placeholder), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);

        let res_bg = self.build_res_bg(device, occ, &params_buf, &self.dummy_dispatch, core, meta, last_used);
        let res_bg_b0 = self.build_res_bg(device, occ, &params_buf, &self.shell_dispatch, core, meta, last_used);

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
                bind(12, &self.pack_count), // pack_brick's 2D-dispatch over-run guard (gate on the real count)
            ],
        });
        let aabb_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("res_aabb_bg"),
            layout: &self.aabb_bgl,
            entries: &[bind(6, aabb), bind(7, &self.aabb_commands), bind(10, &self.dirty_chunk_buf), bind(11, &self.aabb_count)],
        });
        self.bound = Some(BoundScene { res_bg, res_bg_b0, cls_bg, pack_bg, aabb_bg, params_buf });
    }

    /// Drop the bound scene (a scene switch away to a non-GPU-residency scene). The next [`rebind_pool`] cold-starts.
    pub fn unbind(&mut self) {
        self.bound = None;
    }

    /// Would driving `cam` this frame exceed the ONE remaining hard dispatch limit — Pass B0
    /// (`prepare_shell_dispatch`) at @workgroup_size(256) over `total_cells` exceeding the 65535-workgroup 1D cap
    /// (i.e. `total_cells > ~16.7M` RAW shell cells)? The caller skips the GPU drive when true.
    ///
    /// NOTE: the old `total_cells > LIST_CAP` bail is GONE — the enter side is now candidate-list-free (the fused
    /// `enumerate_histogram`/`enumerate_enter` consume surface bricks on the fly), so a far view / large scene is
    /// bounded by the SURFACE-CELL count (`shell_wg_indices`, OOB-guarded + clamped in `prepare_shell_dispatch`),
    /// not a per-frame candidate cap. A view with ≤ `shell_wg_indices` solid cells renders the nearest pool-worth;
    /// only an enormous (> the solid-cell buffer, or > 16.7M raw cells) view is skipped. (`docs/DYNAMIC_LARGE_SCENE_PLAN.md` Phase 2b.)
    /// Phase 4 — push the editor's dynamic-residency levers (live, per-frame, from `VoxelRtResidencySettings`).
    /// Must be called BEFORE `would_overflow`/`record_frame` each frame (both build the params from these).
    pub fn set_residency_levers(&mut self, demand: bool, backdrop_lod: u32, backdrop_reach: u32, ray_keep_frames: u32) {
        self.demand = demand;
        self.backdrop_lod = backdrop_lod;
        self.backdrop_reach = backdrop_reach.max(1);
        self.ray_keep_frames = ray_keep_frames.max(1);
    }

    pub fn would_overflow(&self, cam: [f32; 3]) -> bool {
        let params = build_params(cam, self.half, self.backdrop_lod, self.frame, self.demand, self.backdrop_reach, self.ray_keep_frames);
        params.total_cells.div_ceil(256) > 65535
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
        last_used: &wgpu::Buffer,
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
                bind(45, &self.index_pool_base),
                bind(46, &self.palette_pool_base),
                bind(47, &self.classify_out),
                bind(48, &self.change_count_buf),
                bind(50, &self.enter_hist),
                bind(51, &self.enter_cap),
                bind(52, last_used), // 4-S2/S3: the scene's per-slot last_used_frame (READ for ray-guided keep + LRU)
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
        self.frame = self.frame.wrapping_add(1); // 4-S2/S3: age clock for ray-guided keep + LRU (bump BEFORE the borrow)
        let bound = self.bound.as_ref().expect("record_frame without a bound scene (call rebind_pool)");
        let params = build_params(cam, self.half, self.backdrop_lod, self.frame, self.demand, self.backdrop_reach, self.ray_keep_frames);
        // The shell WG-cell union must fit `shell_idx` (sized to LIST_CAP). If a (very wide clip_half / very large
        // scene) frame would overflow, the caller's `total_cells > LIST_CAP` check (see `record_frame`'s return)
        // skips the GPU drive for this frame — here we still record but the B0 dispatch is clamped by its own
        // bound check, so an overflow can't corrupt past the buffer (it just under-enumerates the farthest cells).
        queue.write_buffer(&bound.params_buf, 0, bytemuck::bytes_of(&params));
        // Clear the per-frame dirty-chunk mask BEFORE the passes run (queue writes are ordered before this encoder's
        // submit), so `write_aabb`'s atomicOrs accumulate only THIS frame's changed chunks.
        queue.write_buffer(&self.dirty_chunk_buf, 0, bytemuck::cast_slice(&vec![0u32; self.dirty_mask_words as usize]));

        // The per-frame residency passes run at @workgroup_size(256) (size-agnostic headroom: a 4× higher
        // workgroup-count ceiling than the old 64, so LIST_CAP can grow to 4M while every 1D dispatch stays
        // <= the 65535-workgroup limit). MUST match the WGSL @workgroup_size on these entry points.
        const RES_WG: u32 = 256;
        let clear_wgs = self.slot_table_size.max(self.present_size).div_ceil(RES_WG).max(1);
        let list_wgs = (LIST_CAP as u32).div_ceil(RES_WG).max(1);
        let bg = &bound.res_bg;
        let bg_b0 = &bound.res_bg_b0;

        // Pass A0 — clear enumerate/pack counts + SEED the indirect dispatches (0,1,1) (self-gating).
        compute(enc, &self.p_seed, bg, 1);
        // Pass A — drain the previous frame's quarantine + clear diff enter/drop counts.
        compute(enc, &self.p_release, bg, 1);
        // Pass A2 — clear the per-frame hashes + change_count.
        compute(enc, &self.p_clear, bg, clear_wgs);
        // Pass B0 — shell dispatch prep (binds the real shell_dispatch at 7).
        compute(enc, &self.p_b0, bg_b0, params.total_cells.div_ceil(RES_WG).max(1));
        // shell_dispatch [n,1,1] → 2D [x,y,1] (size-agnostic past the 65535 workgroup-per-dim cap).
        compute(enc, &self.p_fin_shell, bg, 1);
        // BUDGET cut FIRST (Phase 4): Pass A bins EVERY desired surface brick (resident + non-resident) into the
        // GLOBAL enter_hist (cleared by p_clear); enter_cap_compute derives the nearest-`max_resident` cut radius.
        // Must precede drop/enter — both read the cut: drop EVICTS resident bricks beyond it, enter ADMITS
        // non-resident below it. (FUSED — no candidate_list, so the view is bounded by the surface-CELL count.)
        record_indirect(enc, &self.p_enum_hist, bg, &self.shell_dispatch);
        compute(enc, &self.p_cap_compute, bg, 1);
        // Pass C — drop-mark, drop-apply (no readback): drops left-clipmap bricks (keep-old-until-revealed) AND
        // evicts desired bricks beyond the budget cut. Scans only the slot_table (enumeration-independent).
        // Freed slots go to quarantine (released next frame) so the evict→enter handoff is 1-frame-lagged.
        compute(enc, &self.p_mark, bg, self.slot_table_size.div_ceil(RES_WG).max(1));
        compute(enc, &self.p_apply, bg, self.slot_table_size.div_ceil(RES_WG).max(1));
        // ENTER (FUSED Pass B): re-enumerate; enter each non-resident surface brick the cut admits — correct by
        // construction (no nearest brick dropped for list capacity).
        record_indirect(enc, &self.p_enum_enter, bg, &self.shell_dispatch);
        // publish change_count (= enter + drop).
        compute(enc, &self.p_chg, bg, 1);
        // Pass D — dirty build, drops, neighbours, then classify/pack/write_aabb INDIRECT (the self-gating tail).
        compute(enc, &self.p_d_dirty, bg, list_wgs);
        compute(enc, &self.p_d_drops, bg, list_wgs);
        compute(enc, &self.p_d_nbr, bg, list_wgs);
        compute(enc, &self.p_fin_classify, bg, 1); // classify_dispatch [n,1,1] → 2D [x,y,1] (size-agnostic)
        record_indirect(enc, &self.p_classify, &bound.cls_bg, &self.classify_dispatch);
        compute(enc, &self.p_d_cmd, bg, list_wgs);
        compute(enc, &self.p_fin_pack, bg, 1); // pack_dispatch [n,1,1] → 2D [x,y,1] (size-agnostic)
        record_indirect(enc, &self.p_pack, &bound.pack_bg, &self.pack_dispatch);
        record_indirect(enc, &self.p_aabb, &bound.aabb_bg, &self.aabb_dispatch);

        // Copy change_count into THIS frame's staging-ring slot (the next poll reads the OTHER slot — last frame).
        enc.copy_buffer_to_buffer(&self.change_count_buf, 0, &self.change_staging[self.ring], 0, 4);
        // Likewise mirror the per-frame dirty-chunk mask (read back 1-frame-late by `poll_dirty_chunks`).
        enc.copy_buffer_to_buffer(&self.dirty_chunk_buf, 0, &self.dirty_chunk_staging[self.ring], 0, (self.dirty_mask_words as u64) * 4);

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

    /// Read the PREVIOUS frame's DIRTY-CHUNK mask out-of-band (non-blocking, 1-frame-late — same ring discipline as
    /// [`poll_change_count`](Self::poll_change_count)) and return the chunk indices that changed. The caller rebuilds
    /// EXACTLY those chunks' BLASes (not a blind sweep). `None` on the first frame (no prior copy) or if the map
    /// isn't ready yet — the caller then leaves its pending set unchanged (no chunks newly dirtied this poll).
    ///
    /// Call ONCE per frame, BEFORE `record_frame`, paired with [`poll_change_count`](Self::poll_change_count) (they
    /// share the staging ring; [`advance_ring`](Self::advance_ring) advances both).
    pub fn poll_dirty_chunks(&self, device: &wgpu::Device) -> Option<Vec<u32>> {
        let read_slot = 1 - self.ring;
        let staging = &self.dirty_chunk_staging[read_slot];
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        match rx.try_recv() {
            Ok(true) => {
                let out = match staging.slice(..).get_mapped_range() {
                    Ok(d) => {
                        let words: &[u32] = bytemuck::cast_slice(&d);
                        let mut chunks = Vec::new();
                        for (wi, &w) in words.iter().enumerate() {
                            if w == 0 {
                                continue;
                            }
                            for b in 0..32u32 {
                                if w & (1 << b) != 0 {
                                    let chunk = wi as u32 * 32 + b;
                                    if chunk < self.n_chunks {
                                        chunks.push(chunk);
                                    }
                                }
                            }
                        }
                        Some(chunks)
                    }
                    Err(_) => None,
                };
                staging.unmap();
                out
            }
            _ => None,
        }
    }

    /// Advance the staging ring after this frame's `record_frame` (toggle which slot the next frame writes / the
    /// next poll reads). Call AFTER `record_frame` for this frame. Shared by the change_count + dirty-chunk mirrors.
    pub fn advance_ring(&mut self) {
        self.ring = 1 - self.ring;
    }

    /// **Diagnostic ONLY** — the resident-pool capacity (`max_resident`), for the paged-drive diagnostic log line.
    pub fn max_resident_diag(&self) -> u32 {
        self.max_resident
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
        // A live key has lod ∈ 0..=MAX_LOD; EMPTY_LOD (free) and EMPTY_LOD-1 (TOMBSTONE, a deleted-key hole) are not.
        const TOMBSTONE_LOD: u32 = EMPTY_LOD - 1;
        for s in words_u32.chunks_exact(5) {
            if s[3] != EMPTY_LOD && s[3] != TOMBSTONE_LOD {
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
