//! Incremental clipmap bake scheduling in **chunk units**.
//!
//! The main thread does cheap integer chunk-ring window diffs (enqueue entered chunks,
//! evict exited chunks) + per-brick topology (BVH cull, palette, tile alloc) and emits GPU
//! compute bake jobs; the actual per-voxel eval runs on the GPU (`sdf_brick_bake.wgsl`), so
//! camera motion never blocks.
//!
//! Eager eviction is safe because addressing is **absolute** (chunk keys, not a
//! camera-relative ring origin — see [`super::chunk`]): a not-yet-baked chunk is simply
//! absent from the GPU chunk table, and the nested coarser LOD shell already covers that
//! region, so the leading edge shows coarser-correct terrain that refines in — never a
//! hole, never a shift.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::camera::primitives::Frustum;
use bevy::tasks::{ComputeTaskPool, ParallelSlice};

use super::atlas::{self, SdfAtlas};
use super::chunk;
use super::{
    SdfCamera, SdfGridConfig, SdfMaterial, SdfOp, SdfPrimitive, SdfVolume, VolumeQueryData, bvh,
    edits, gather_sorted_edits,
};
// Stress-scene generator — only the bake-cache regression test consumes it.
#[cfg(test)]
use super::tower_field;

// Pure chunk-ring window geometry. The names are re-imported here so production code (and the
// in-file tests, via their `use super::*`) call them unqualified; `ring_chunk_origin` is also `pub`
// (the GPU rig + the editor LOD-ring overlay assert/draw against this source-of-truth window).
mod window;
pub use window::ring_chunk_origin;
use window::{
    FULL_CHUNK_MASK, bricks_in_aabb_windowed, chunk_conservative_mask, chunk_has_geometry_with,
    chunk_in_inner_hole, chunk_in_shell, for_each_brick_key, for_each_brick_key_masked,
    for_each_entered_chunk, for_each_entered_shell, for_each_exited_chunk, for_each_exited_shell,
    ivec_floor_div,
};
#[cfg(test)]
use window::chunk_in_window;
#[cfg(test)]
use window::{chunk_brick_keys, chunk_window_keys, chunks_in_aabb_windowed};

// The read-only classify core (Send; runs in parallel across the compute pool). Bare names
// re-imported so the bake path + the in-file tests use them unqualified.
mod classify;
use classify::{Verdict, classify_candidates, narrow_band_keep, snapshot_hash_peek};

/// One brick the GPU compute bake must fill this frame. The CPU has already done the
/// topology work (BVH cull → `edit_indices` into the frame's flat edit list, palette,
/// tile allocation); the compute shader runs the 512-voxel `fold_csg` eval and writes the
/// brick's texels straight into the atlas tile at `tile`. `lod`/`coord` give the shader
/// each voxel's world position; `dist_band` is the per-LOD snorm clamp band so the shader
/// needs no `SdfGridConfig`.
#[derive(Clone)]
pub struct GpuBakeJob {
    pub tile: u32,
    /// Material atlas tile for a MULTI-material brick; `None` for a single-material brick (it stores
    /// no material texels — the bake node skips its material copy and the reader uses `palette[0]`).
    pub mat_tile: Option<u32>,
    pub lod: u32,
    pub coord: IVec3,
    pub voxel_size: f32,
    pub dist_band: f32,
    pub palette: edits::Palette,
    /// Range into [`PendingGpuBakes::edits`] of this brick's culled candidate edits.
    pub edit_start: u32,
    pub edit_count: u32,
}

/// Main-world hand-off for the GPU brick bake: this frame's jobs plus the flat edit list
/// they index. Filled by `schedule_bakes` in [`BakeBackend::Gpu`] mode and drained by the
/// render-world extract.
#[derive(Resource, Default)]
pub struct PendingGpuBakes {
    pub jobs: Vec<GpuBakeJob>,
    pub edits: Vec<edits::GpuEdit>,
}

impl PendingGpuBakes {
    fn clear(&mut self) {
        self.jobs.clear();
        self.edits.clear();
    }
}

/// Last frame's per-edit world AABB, keyed by entity. Lets the scheduler dirty an edit's
/// *former* footprint (not just where it moved to) so vacated chunks get rebuilt/removed.
/// Also serves as the previous entity set for add/remove detection.
#[derive(Resource, Default)]
pub struct PrevEditAabbs {
    map: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d>,
}

impl PrevEditAabbs {
    /// Number of tracked edits (for add/remove detection in the diagnostic sync bake).
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Whether `entity` was present last frame.
    pub fn contains(&self, entity: &Entity) -> bool {
        self.map.contains_key(entity)
    }
    /// Replace the tracked set (diagnostic sync bake bookkeeping).
    pub fn set_map(&mut self, map: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d>) {
        self.map = map;
    }
}

/// Drives incremental clipmap baking in chunk units (see module docs).
#[derive(Resource)]
pub struct BakeScheduler {
    /// Snapshot of the current edits + BVH used to emit GPU bake jobs (cheap Arc clone).
    edits: Arc<Vec<edits::ResolvedEdit>>,
    bvh: Arc<bvh::Bvh>,
    /// Decoded height maps for bake-time displacement, snapshotted alongside edits/BVH.
    /// Rebuilt when the material registry's displacement columns change (see
    /// `update_height_field`).
    height: super::height::SharedHeightField,
    /// Per-LOD chunk-ring origin currently resident (index = lod), in chunk coords. Used
    /// to diff which chunks entered/exited as the camera moves. Empty until first run.
    ring_chunk_origin: Vec<IVec3>,
    /// Chunks awaiting a bake, each mapped to a `u64` DIRTY-BRICK MASK (bit `i` set ⇒ local brick `i`
    /// needs re-classify). A moved edit dirties only the bricks its footprint actually reaches (see
    /// `dirty_edit_footprints`/`bricks_in_aabb_windowed`); a recenter-entered or structurally-rebaked
    /// chunk dirties all 64 (`FULL_CHUNK_MASK`). `gather_candidates` expands only the set bits, so a
    /// small drag classifies its handful of touched bricks instead of every brick of every straddled
    /// chunk. FxHash: integer `ChunkKey` keys, mutated per dirty per frame.
    pending: rustc_hash::FxHashMap<chunk::ChunkKey, u64>,
    /// Entity → its index in the sorted `edits`/BVH. Lets a LOCALIZED edit (a drag) re-resolve only
    /// the changed entities and refit their BVH leaves in O(changed · depth), instead of re-gathering
    /// and rebuilding the whole BVH over all ~14.6k edits every frame. Rebuilt only on a full gather
    /// (add/remove/`rebake_all`); a pure move keeps every index stable so the map stays valid.
    entity_index: rustc_hash::FxHashMap<Entity, u32>,
    /// Reusable emit scratch (cleared + refilled each frame) so a continuous drag does zero
    /// growth reallocation. `mem::take`n into `emit_gpu_bakes` and restored at the end.
    emit_scratch: EmitScratch,
    /// Monotonic counter bumped whenever the edit set changes (an edit moves/adds/removes →
    /// `edits`/`bvh` are replaced). Stamped onto an async bake task's input snapshot; when the
    /// task lands, a mismatch means the edits changed mid-flight, so its (now stale) classify is
    /// discarded and its chunks re-queued. See the async dispatch in [`schedule_bakes`].
    edit_gen: u64,
    /// Classified-but-over-budget Keep groups carried across frames so the expensive classify
    /// (palette build) runs ONCE per brick instead of re-running on the whole backlog every frame.
    /// Each entry is a whole chunk's Keep set (chunk-atomic). Drained coarse-LOD/nearest-first
    /// before any new `pending` classify (see `apply_ready`/`refresh_ready`). Bounded by the
    /// low-water refill gate in the dispatch paths so it stays ~1–2 budgets deep.
    ready: Vec<ReadyChunk>,
    /// The `edit_gen` the indices held in `ready` reference. On a mismatch the edit set changed and
    /// the carried indices may be stale (an add/remove shifts positions), so `ready` is invalidated
    /// — fully on an add/remove, selectively (only the dirtied footprint) on a pure move. See
    /// `refresh_ready` and the step-1 invalidation in `schedule_bakes`.
    ready_edit_gen: u64,
    /// Camera position `ready` was last window-filtered + priority-re-sorted at. Window membership
    /// and priority only change when the camera moves, so `refresh_ready` skips that O(ready) pass
    /// while the camera is stationary (a cold bake) — the only time `ready` is large.
    ready_maint_cam: Vec3,
    /// Camera FORWARD direction `ready` was last priority-sorted at. A rotation-only move (same
    /// position, new look direction) changes the in-frustum priority bucket without moving the camera,
    /// so `refresh_ready` re-sorts when EITHER position OR forward changed. Sentinel until first sort.
    ready_maint_fwd: Vec3,
}

/// Per-frame scratch buffers for `emit_gpu_bakes`, held on the scheduler so their capacity
/// persists across frames instead of allocating from empty each emit.
#[derive(Default)]
struct EmitScratch {
    /// This frame's drained batch: each chunk paired with its dirty-brick mask (see [`BakeScheduler::pending`]).
    drained: Vec<(chunk::ChunkKey, u64)>,
    candidates: Vec<(chunk::ChunkKey, atlas::BrickKey)>,
}

/// OR `mask` into chunk `ck`'s dirty-brick entry in `pending`, creating it if absent. The single
/// mutation point for `pending` so chunk-level membership and per-brick masks stay consistent.
#[inline]
fn dirty_mask(pending: &mut rustc_hash::FxHashMap<chunk::ChunkKey, u64>, ck: chunk::ChunkKey, mask: u64) {
    *pending.entry(ck).or_insert(0) |= mask;
}

/// One deferred-but-classified chunk carried in [`BakeScheduler::ready`]: every brick of the chunk
/// that classified `Keep`, with its already-built palette, culled edit indices, and content hash —
/// enough to emit the GPU bake job later with NO re-classify (see [`push_bake_job`]). Chunk-atomic:
/// the whole `keeps` set bakes on one frame or waits together (mirrors `apply_verdicts`' run logic),
/// so the finer LOD never shows a half-baked chunk. The bricks were evicted from the atlas on the
/// first defer (the drag-trail eviction in `apply_verdicts`), so they are absent until re-applied.
struct ReadyChunk {
    ck: chunk::ChunkKey,
    keeps: Vec<(atlas::BrickKey, edits::Palette, Vec<u32>, u64)>,
}

impl ReadyChunk {
    /// The dirty-brick mask of this carried group: a bit per carried Keep's local slot. Used to
    /// re-queue exactly these bricks into `pending` when an edit change invalidates the group — the
    /// carried bricks were evicted on their first defer, so a re-queue is the only way they bake.
    fn carried_mask(&self, config: &SdfGridConfig) -> u64 {
        let mut mask = 0u64;
        for (key, ..) in &self.keeps {
            let (_, local) = chunk::chunk_of(*key, config);
            mask |= 1u64 << local;
        }
        mask
    }
}

impl Default for BakeScheduler {
    fn default() -> Self {
        Self {
            edits: Arc::new(Vec::new()),
            bvh: Arc::new(bvh::Bvh::default()),
            height: Arc::new(super::height::HeightField::default()),
            ring_chunk_origin: Vec::new(),
            pending: rustc_hash::FxHashMap::default(),
            entity_index: rustc_hash::FxHashMap::default(),
            emit_scratch: EmitScratch::default(),
            edit_gen: 0,
            ready: Vec::new(),
            ready_edit_gen: 0,
            // Sentinel ≠ any real camera so the first `refresh_ready` with a non-empty ready runs
            // the window-filter/sort once.
            ready_maint_cam: Vec3::splat(f32::INFINITY),
            ready_maint_fwd: Vec3::ZERO,
        }
    }
}

impl BakeScheduler {
    /// Replace the height-field snapshot used by subsequent bakes (rebuilt when the material
    /// registry's displacement columns change).
    pub fn set_height(&mut self, height: super::height::SharedHeightField) {
        self.height = height;
    }

    /// Drop all queued + carried bake state so the next frame re-bakes the window from scratch. Used on
    /// a scene switch (pair with [`super::atlas::SdfAtlas::reset`]). Clearing `ring_chunk_origin` forces
    /// a full window re-enter (otherwise the diff sees no movement against the now-empty atlas and never
    /// re-enqueues); bumping `edit_gen` discards any async bake task still in flight for the old scene
    /// when it lands. Edits/BVH are left as-is — they re-snapshot on the next gather from the new scene.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.ready.clear();
        self.ring_chunk_origin.clear();
        self.edit_gen = self.edit_gen.wrapping_add(1);
        self.ready_edit_gen = self.edit_gen;
    }
}

/// Max brick bake JOBS emitted to the GPU per frame. Each job writes its
/// brick's texels into two storage buffers sized `jobs × tile`; the material buffer
/// dominates at `1024 u32 × 4 B = 4096 B/job`, so the GPU's `maxStorageBufferBindingSize`
/// (128 MB default) caps us at ~32768 jobs. 8192 sits well under that (32 MB mat + 16 MB dist)
/// AND keeps the per-frame bake-node dispatch small enough to avoid a visible stutter when the
/// cap is hit every frame (a continuously-dragged edit). A single huge edit can dirty 70k+
/// bricks; the overflow spills back to `pending` and bakes over the next frames (coarse LOD
/// covers the gap meanwhile — see `emit_gpu_bakes`). Without this cap a giant edit overflows the
/// buffer binding and wgpu aborts the frame.
const GPU_BAKE_JOB_CAP: usize = 8192;

/// SOFT per-frame bake budget — the smoothing knob (distinct from the hard `GPU_BAKE_JOB_CAP`
/// GPU-buffer ceiling). When the camera crosses a coarse LOD-ring snap boundary a whole shell of
/// geometry chunks enters at once (~8k bricks in the stress scene), and baking them all on one
/// frame is a visible hitch (~8 ms CPU emit + ~5 ms GPU dispatch). Capping emitted jobs at this
/// lower budget spreads the shell over a few frames; the coarse LOD already covers the not-yet-
/// baked fine shell (same fallback the hard cap relies on), so the only cost is a brief slightly-
/// coarser band right after a snap, which the LOD blend hides. The coarse-first + nearest-first
/// drain order means the most-visible bricks bake first. Sized so a snap spreads over ~4 frames
/// rather than spiking one; a continuously-dragged edit also stays under it frame-to-frame.
const SOFT_BAKE_BUDGET: usize = 2048;
const _: () = assert!(SOFT_BAKE_BUDGET <= GPU_BAKE_JOB_CAP, "soft budget must stay under the GPU buffer ceiling");

/// Max chunks CLASSIFIED in one frame's batch. The carry queue (`ready`) classifies each brick once,
/// but classifying the WHOLE backlog in a single batch makes the apply+carry that lands it an
/// O(backlog) main-thread hitch that GROWS with scene size — the wrong scaling. Bounding the batch
/// caps that landing cost to O(batch) regardless of how big the scene is; the un-drained chunks wait
/// in `pending` as cheap keys (still classified once, on a later frame). Sized ABOVE
/// `ASYNC_BAKE_THRESHOLD`/`CHUNK_VOLUME` so a full batch still crosses into the off-thread classify
/// (its bricks never hit the main thread), while `apply_ready` drains the previous batch's carry
/// during the classify — so no idle frames. Apply (`SOFT_BAKE_BUDGET`/frame) stays the settle-rate
/// governor; total classify work is O(bricks), once each.
///
/// Sized large (not just over the budget) on purpose: each bake frame's `drain_priority_batch`
/// drains + sorts ALL of `pending` to take the top batch, so FEWER, larger batches mean fewer of
/// those whole-`pending` sorts — and with FxHash the spill `remove_brick`s a big batch leaves are
/// cheap, so shrinking the batch only adds frame overhead (measured: 72 chunks regressed both total
/// and flythrough MAX vs 128). Stays above `ASYNC_BAKE_THRESHOLD` so a full batch still goes off-thread.
const CLASSIFY_REFILL_CHUNKS: usize = 128;

/// √3, the chunk bounding-sphere radius factor: a cube of edge `s` has half-diagonal `s·√3/2`.
const SQRT_3: f32 = 1.732_050_8;

/// A camera frustum reduced to its 6 inward half-spaces (`normal·p + d ≥ 0` inside), copied from Bevy's
/// `Frustum` so the scheduler holds no ECS borrow. Used ONLY for bake PRIORITY (in-view bakes first),
/// never for residency (off-screen geometry stays resident for shadows/GI).
#[derive(Clone, Copy)]
struct FrustumPlanes([Vec4; 6]);

impl FrustumPlanes {
    /// Conservative in-view test for a bounding sphere: 1 (out of view ⇒ sorts later) when the sphere
    /// lies fully OUTSIDE any plane within `margin` slack; 0 (in view / straddling) otherwise.
    fn out_rank(&self, center: Vec3, radius: f32, margin: f32) -> u32 {
        let p = center.extend(1.0);
        for plane in self.0 {
            if plane.dot(p) < -(radius + margin) {
                return 1;
            }
        }
        0
    }
}

/// Camera state threaded into the bake-PRIORITY path: position (proximity), look direction (so a
/// rotation-only move still re-sorts the carry queue), and the optional frustum (in-view first). A
/// `None` frustum degrades to distance-only ordering (camera without a `Frustum` component, or a
/// headless settle/perf driver).
#[derive(Clone, Copy)]
struct BakeView {
    pos: Vec3,
    fwd: Vec3,
    frustum: Option<FrustumPlanes>,
    margin: f32,
}

impl BakeView {
    /// Distance-only view (no frustum) — the graceful fallback and the headless drive helpers.
    fn pos_only(pos: Vec3) -> Self {
        Self { pos, fwd: Vec3::ZERO, frustum: None, margin: 0.0 }
    }
}

/// So the headless settle/perf drivers (and any caller without a frustum) can pass a bare camera
/// position where a [`BakeView`] is expected — distance-only ordering.
impl From<Vec3> for BakeView {
    fn from(pos: Vec3) -> Self {
        Self::pos_only(pos)
    }
}

/// Take the `max_chunks` highest-priority (coarse-LOD/nearest-camera first) chunks from `pending` into
/// `out`, leaving the rest IN `pending` for a later frame. Bounds the per-frame classify batch.
///
/// O(pending) per call: one packed-integer priority key per chunk (computed ONCE), a `select_nth`
/// partition (NO full sort), then remove only the taken chunks. The previous version drained the
/// WHOLE set, sorted it by an expensive per-comparison key, and re-inserted all-but-`max_chunks`
/// EVERY frame — an O(n log n) hitch (~36 ms on a big cold-bake `pending`).
fn drain_priority_batch(
    pending: &mut rustc_hash::FxHashMap<chunk::ChunkKey, u64>,
    out: &mut Vec<(chunk::ChunkKey, u64)>,
    config: &SdfGridConfig,
    view: &BakeView,
    max_chunks: usize,
) {
    out.clear();
    if pending.is_empty() {
        return;
    }
    // Each taken chunk carries its dirty-brick mask forward so gather/classify touch only those bricks.
    let mut keyed: Vec<(u128, chunk::ChunkKey, u64)> = pending
        .iter()
        .map(|(&ck, &mask)| (chunk_priority_key(ck, config, view), ck, mask))
        .collect();
    let k = max_chunks.min(keyed.len());
    if keyed.len() > k {
        keyed.select_nth_unstable_by_key(k - 1, |&(key, ..)| key);
        keyed.truncate(k);
    }
    // Order the small taken batch so the bake applies coarse/nearest first.
    keyed.sort_unstable_by_key(|&(key, ..)| key);
    for (_, ck, mask) in &keyed {
        pending.remove(ck);
        out.push((*ck, *mask));
    }
}

/// Packed bake-drain priority for a chunk, ordered coarsest → in-view → nearest (smaller = bakes
/// first):
/// 1. **Coarse LOD first** (`u32::MAX - lod`) — UNCHANGED and kept the most-significant field: the
///    coarse ring fills before fine, so a budget cap only ever spills fine detail whose coarser
///    fallback is already resident (the hole-free coverage invariant the chunk-atomic make-before-break
///    relies on). Frustum must NOT outrank this.
/// 2. **In view** (`frustum_rank`: 0 in/straddling, 1 out) — within an LOD, what the camera is looking
///    at bakes before off-screen regions. Distance-only (rank 0 everywhere) when no frustum.
/// 3. **Nearest camera** (`d2`) — finest tiebreak.
///
/// Packed into a `u128` so it sorts/partitions with cheap integer comparisons (key math runs once each).
fn chunk_priority_key(ck: chunk::ChunkKey, config: &SdfGridConfig, view: &BakeView) -> u128 {
    let size = chunk::chunk_world_size(ck.lod, config);
    let center = chunk::chunk_min_world(ck, config) + Vec3::splat(size * 0.5);
    let d2 = view.pos.distance_squared(center);
    let frustum_rank = match &view.frustum {
        Some(planes) => planes.out_rank(center, size * 0.5 * SQRT_3, view.margin),
        None => 0,
    };
    (u128::from(u32::MAX - ck.lod) << 33) | (u128::from(frustum_rank) << 32) | u128::from(d2.to_bits())
}

/// GEOMETRY-DRIVEN bake dirtying: enqueue every chunk each AABB's footprint reaches, at every LOD,
/// clamped to that LOD's ring window. The bake's dirty set is derived from the SPARSE geometry (edit
/// footprints), NEVER from the dense window (`R³ · lod_count` chunks, almost all empty) — so an empty
/// region is never enqueued no matter how large the window or how sparse the scene. `chunks_in_aabb_
/// windowed` enumerates only the AABB's chunks (and never outside the window), so a terrain-scale
/// heightmap costs O(its window footprint), not O(world). Shared by the full-rebake (ALL edits) and
/// the incremental move (the changed edits' old∪new) paths, so add/remove and move dirty identically.
fn dirty_edit_footprints(
    pending: &mut rustc_hash::FxHashMap<chunk::ChunkKey, u64>,
    aabbs: &[bevy::math::bounding::Aabb3d],
    config: &SdfGridConfig,
    camera_pos: Vec3,
) {
    let r = config.ring_chunks_per_axis();
    for lod in 0..config.lod_count {
        let origin = ring_chunk_origin(config, camera_pos, lod);
        for aabb in aabbs {
            for (ck, mask) in bricks_in_aabb_windowed(config, aabb, lod, origin, r) {
                // Skip chunks inside this LOD's inner hole — a finer LOD already covers them, so baking
                // them here is the redundant-stack work we're eliminating ({native..native+overlap}).
                if chunk_in_inner_hole(config, camera_pos, ck) {
                    continue;
                }
                dirty_mask(pending, ck, mask);
            }
        }
    }
}

/// Recompute the CONSERVATIVE occupancy (the empty-space DDA's traversal grid) for every FULL-RING
/// chunk the edits' footprints touch — the edit-side counterpart of the recenter's cons pass. The
/// recenter only fires on camera motion, so an add/remove/move with a stationary camera must refresh
/// cons here or a moved object's new (coarse-LOD) position would keep stale `cons_occ`. No shell hole
/// (full ring); reads the post-refit BVH. Pass the changed edits' old∪new AABBs (incremental) or all
/// gathered AABBs (full re-gather).
#[expect(clippy::too_many_arguments)]
fn refresh_conservative_footprints(
    atlas: &mut SdfAtlas,
    aabbs: &[bevy::math::bounding::Aabb3d],
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    camera_pos: Vec3,
    scratch: &mut Vec<u32>,
    stack: &mut Vec<u32>,
) {
    let r = config.ring_chunks_per_axis();
    for lod in 0..config.lod_count {
        let origin = ring_chunk_origin(config, camera_pos, lod);
        for aabb in aabbs {
            for (ck, _mask) in bricks_in_aabb_windowed(config, aabb, lod, origin, r) {
                let mask = chunk_conservative_mask(ck, edits, bvh, config, scratch, stack);
                atlas.set_conservative_chunk(ck, mask, config);
            }
        }
    }
}

/// SURFACE-PRUNED dirtying for the MOVE path: dirty only the bricks the moving edit's surface SHELL
/// crosses across its old∪new position — never its solid interior. A coarse-to-fine descent over each
/// LOD's window∩footprint box prunes any block where the edit is SATURATED (deep interior or far
/// exterior) and SAME-SIGN at BOTH the old and new positions: there the edit's band-clamped
/// contribution is an unchanged constant, so no brick in the block can change (true for any CSG op —
/// a constant input to min/max/smin leaves the fold output unchanged). So a massive sphere's interior
/// LODs prune at the ROOT in O(1) and only the surface-crossing band descends to bricks → per-frame
/// cost scales with SURFACE AREA, not volume (the fix for the 25 ms `sched_refit_incremental` hitch
/// dragging a window-spanning sphere).
///
/// Correct vs eviction: a trailing brick that just LEFT the shell is within band of the OLD surface
/// (tested via `old`), so it is NOT pruned → reclassified → evicted. A block the surface swept fully
/// across in one frame flips sign (`old`/`new` differ) → also not pruned. The pruned blocks are
/// interior/exterior, whose bricks are never resident (deep interior bakes to `Drop`), so pruning them
/// is a no-op for residency. The `band` margin matches `narrow_band_keep`'s reach (LOD band + ¼·k
/// smoothing) so we never prune a brick the classifier would have Kept.
fn dirty_moving_edit(
    pending: &mut rustc_hash::FxHashMap<chunk::ChunkKey, u64>,
    old: &edits::ResolvedEdit,
    new: &edits::ResolvedEdit,
    config: &SdfGridConfig,
    camera_pos: Vec3,
) {
    let s = config.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let r = config.ring_chunks_per_axis();
    let old_aabb = edits::edit_world_aabb(&old.prim, &old.transform, old.op.smoothing);
    let new_aabb = edits::edit_world_aabb(&new.prim, &new.transform, new.op.smoothing);
    // Smoothing widens the edit's influence by ≤ k/4 (smin's max deviation from min) — match
    // `narrow_band_keep`'s `0.25·smooth` reach so the prune never drops a brick classify would keep.
    let smooth_pad = 0.25 * old.op.smoothing.max(new.op.smoothing).max(0.0);
    for lod in 0..config.lod_count {
        let brick_world = config.brick_world_size(lod);
        let band = atlas::dist_band_world(config, lod) + smooth_pad;
        let pad = Vec3::splat(atlas::SNORM_CLAMP_DIST + brick_world);
        // Union footprint in brick-index space, clamped to this LOD's ring window.
        let win_origin = ring_chunk_origin(config, camera_pos, lod);
        let win_lo = win_origin * c;
        let win_hi = (win_origin + IVec3::splat(r)) * c - IVec3::ONE;
        let lo_w = Vec3::from(old_aabb.min).min(Vec3::from(new_aabb.min)) - pad;
        let hi_w = Vec3::from(old_aabb.max).max(Vec3::from(new_aabb.max)) + pad;
        let bi_lo = ivec_floor_div(lo_w, brick_world).max(win_lo);
        let bi_hi = ivec_floor_div(hi_w, brick_world).min(win_hi);
        if bi_lo.x > bi_hi.x || bi_lo.y > bi_hi.y || bi_lo.z > bi_hi.z {
            continue;
        }
        // Coarse-to-fine descent over brick-index boxes; prune saturated-same-sign blocks whole.
        let mut stack: Vec<(IVec3, IVec3)> = vec![(bi_lo, bi_hi)];
        while let Some((lo, hi)) = stack.pop() {
            // Brick index `i` spans world `[i·bw, (i+1)·bw]`, so the box spans `[lo·bw, (hi+1)·bw]`.
            let center = (lo + hi + IVec3::ONE).as_vec3() * (0.5 * brick_world);
            let half = (hi - lo + IVec3::ONE).as_vec3() * (0.5 * brick_world);
            let margin = half.length() + band;
            let d_old = edits::eval_world_inv(&old.prim, &old.inv_model, center, 0.0);
            let d_new = edits::eval_world_inv(&new.prim, &new.inv_model, center, 0.0);
            if d_old.abs() > margin && d_new.abs() > margin && (d_old > 0.0) == (d_new > 0.0) {
                continue; // saturated, same sign at both positions → nothing here can change
            }
            if lo == hi {
                let (ck, local) = chunk::chunk_of(atlas::BrickKey::new(lod, lo * s), config);
                // Skip the inner hole (finer LOD covers it) — keeps a near edit's coarse LODs out of
                // the redundant stack.
                if !chunk_in_inner_hole(config, camera_pos, ck) {
                    dirty_mask(pending, ck, 1u64 << local);
                }
                continue;
            }
            // Split the LONGEST axis at its midpoint (binary BSP — interior prunes in O(surface)).
            // `lo + (hi-lo)/2` keeps the cut in `[lo, hi-1]` so both halves are non-empty (incl. neg).
            let ext = hi - lo;
            let (mut a_hi, mut b_lo) = (hi, lo);
            if ext.x >= ext.y && ext.x >= ext.z {
                let mid = lo.x + (hi.x - lo.x) / 2;
                a_hi.x = mid;
                b_lo.x = mid + 1;
            } else if ext.y >= ext.z {
                let mid = lo.y + (hi.y - lo.y) / 2;
                a_hi.y = mid;
                b_lo.y = mid + 1;
            } else {
                let mid = lo.z + (hi.z - lo.z) / 2;
                a_hi.z = mid;
                b_lo.z = mid + 1;
            }
            stack.push((lo, a_hi));
            stack.push((b_lo, hi));
        }
    }
}

/// Any component that affects an edit's baked result. A change to one of these
/// triggers a targeted rebake of the bricks the edit touches. Exposed as
/// [`ChangedEditFilter`] so the diagnostic sync bake can reuse the same change filter.
pub type ChangedEditFilter = Or<(
    // `GlobalTransform` (not local `Transform`) so a volume rebakes when an ancestor
    // moves it via hierarchy propagation, not only when its own transform is edited.
    Changed<GlobalTransform>,
    Changed<SdfOp>,
    Changed<SdfPrimitive>,
    Changed<SdfMaterial>,
)>;
type ChangedEdit = ChangedEditFilter;

/// Main-thread scheduling + GPU job emission — no per-voxel baking on the CPU. Diffs the
/// per-LOD chunk-ring window as the camera moves (enqueue entered chunks, evict exited
/// chunks), dirties edited regions, does per-brick topology (BVH cull, palette, tile alloc),
/// and emits GPU compute bake jobs. The per-voxel eval runs on the GPU. All integer window
/// math + Arc clones + a 9-point palette per dirty brick — microseconds.
#[expect(clippy::too_many_arguments, clippy::type_complexity)]
pub fn schedule_bakes(
    mut atlas: ResMut<SdfAtlas>,
    mut bvh: ResMut<bvh::Bvh>,
    mut sched: ResMut<BakeScheduler>,
    mut prev_aabbs: ResMut<PrevEditAabbs>,
    mut gpu_bakes: ResMut<PendingGpuBakes>,
    config: Res<SdfGridConfig>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    changed: Query<Entity, (With<SdfVolume>, ChangedEdit)>,
    mut removed: RemovedComponents<SdfVolume>,
    camera: Query<(&Transform, Option<&Frustum>), (With<SdfCamera>, Without<SdfVolume>)>,
    mut baked_dbg: ResMut<super::BakedBrickDebug>,
    time: Res<Time>,
) {
    let _span = crate::instrument::span("bake schedule");
    // Diagnostic: prune baked-brick markers older than the fade window (entries ACCUMULATE across
    // frames so they can fade out — see BakedBrickDebug); `emit_gpu_bakes` appends new ones.
    let now = time.elapsed_secs();
    if baked_dbg.enabled {
        baked_dbg.bricks.retain(|&(_, _, t)| now - t < super::BAKED_BRICK_FADE_SECS);
    } else {
        baked_dbg.bricks.clear();
    }
    // GPU bake jobs are rebuilt from scratch each frame; the render world consumed last
    // frame's. `gpu_baked_tiles` likewise holds only THIS frame's GPU-written tiles.
    gpu_bakes.clear();
    atlas.gpu_baked_tiles.clear();
    let cam = camera.iter().next();
    let camera_pos = cam.map(|(t, _)| t.translation).unwrap_or(Vec3::ZERO);
    // Bake-priority view: position (proximity) + optional frustum (in-view first). The frustum is a
    // PRIORITY hint only — residency stays view-independent. `update_frusta` runs each frame, so a
    // one-frame-stale frustum is fine for ordering.
    let bake_view: BakeView = match cam {
        Some((t, Some(frustum))) => BakeView {
            pos: t.translation,
            fwd: *t.forward(),
            frustum: Some(FrustumPlanes(frustum.half_spaces.map(|hs| hs.normal_d()))),
            margin: config.frustum_priority_margin,
        },
        Some((t, None)) => BakeView::pos_only(t.translation),
        None => BakeView::pos_only(Vec3::ZERO),
    };
    let lod_count = config.lod_count;
    let first_run = sched.ring_chunk_origin.is_empty();
    if first_run {
        sched.ring_chunk_origin = vec![IVec3::splat(i32::MIN); lod_count as usize];
    }

    // --- 1. Edit changes → dirty affected chunks (within current windows) ------------
    // CHANGE-DETECTION GATE: `gather_sorted_edits` (collect + sort all volumes + a matrix-inverse +
    // 8-corner AABB per edit) and the per-edit BVH rebuild are the ~3 ms every-frame floor. SKIP them
    // on a frame where no edit changed — detected cheaply, WITHOUT touching all ~14.6k volumes, from
    // Bevy change detection: `changed` (`Changed` covers both moves AND adds — a fresh component reads
    // as changed) plus `RemovedComponents` for despawns. The cached `sched.edits`/`bvh` stay valid, so
    // the recenter still resolves geometry. `read().count()` drains the events so they don't re-fire.
    let any_removed = removed.read().count() > 0;
    if atlas.rebake_all || !changed.is_empty() || any_removed {
        // FULL re-gather + BVH REBUILD only when the edit SET changed (an add/remove shifts every
        // edit's index) or a global rebake is forced. A pure MOVE / property change of EXISTING
        // edits keeps each edit's index stable (sorted by SdfOrder, which a move doesn't touch), so
        // it takes the INCREMENTAL path below: re-resolve just the changed entities + refit their
        // BVH leaves in O(changed · depth) — no O(n log n) rebuild and no 14.6k-edit re-gather (the
        // ~6 ms/frame drag hitch). An ADD shows up as a `changed` entity not yet in `entity_index`.
        let added = changed.iter().any(|e| !sched.entity_index.contains_key(&e));
        if atlas.rebake_all || any_removed || added {
            let _g_gather = info_span!("sched_gather").entered();
            let gathered = gather_sorted_edits(&volumes);
            let current: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d> =
                gathered.iter().map(|g| (g.entity, g.aabb)).collect();
            let resolved: Vec<edits::ResolvedEdit> = gathered.iter().map(|g| g.edit.clone()).collect();
            let aabbs: Vec<bevy::math::bounding::Aabb3d> = gathered.iter().map(|g| g.aabb).collect();
            let new_bvh = bvh::Bvh::build(&aabbs);
            *bvh = new_bvh.clone();
            sched.bvh = Arc::new(new_bvh);
            sched.edits = Arc::new(resolved);
            sched.entity_index = gathered.iter().enumerate().map(|(i, g)| (g.entity, i as u32)).collect();
            sched.edit_gen = sched.edit_gen.wrapping_add(1);

            // GEOMETRY-DRIVEN dirty: the edit SET changed → re-dirty every edit's FOOTPRINT (so the
            // current geometry re-bakes; the per-brick content-hash skip keeps the real work to bricks
            // that actually changed) PLUS every currently-RESIDENT chunk (so a removal that emptied a
            // chunk gets its orphaned bricks evicted by `gather_candidates`). The dirty set is derived
            // from the SPARSE geometry, NEVER the dense window — flooding `pending` with the whole
            // (mostly-empty) window made the bounded drain grind through empties for thousands of
            // frames (the gaps between LOD bakes). Same `dirty_edit_footprints` the move path uses.
            dirty_edit_footprints(&mut sched.pending, &aabbs, &config, camera_pos);
            // A structural change can empty/relocate any resident chunk; without the old geometry we
            // can't tell WHICH bricks, so re-examine all 64 (the eviction sweep). Not the hot path.
            for ck in atlas.live_chunks.resident_chunk_keys().collect::<Vec<_>>() {
                dirty_mask(&mut sched.pending, ck, FULL_CHUNK_MASK);
            }
            atlas.rebake_all = false;
            // Edit-side CONSERVATIVE refresh for the changed geometry (add/remove/scene change with a
            // stationary camera — the recenter won't fire). On the FIRST run the recenter's
            // sentinel→entered pass populates the whole ring, so skip the redundant footprint sweep.
            if !first_run {
                let redits = Arc::clone(&sched.edits);
                let mut cscratch: Vec<u32> = Vec::new();
                let mut cstack: Vec<u32> = Vec::new();
                refresh_conservative_footprints(
                    &mut atlas, &aabbs, &redits, &bvh, &config, camera_pos, &mut cscratch, &mut cstack,
                );
            }
            invalidate_ready_on_edit_change(&mut sched, &config, true);
            prev_aabbs.map = current;
        } else {
            // INCREMENTAL: existing edits moved/changed, indices stable. Re-resolve ONLY those, refit
            // their BVH leaves (both the `Res` copy and the scheduler's Arc snapshot), and dirty each
            // edit's old∪new footprint. `Arc::make_mut` is in-place here — between frames the snapshot
            // Arcs have refcount 1 (the per-frame `Arc::clone` in `emit_gpu_bakes` is already dropped).
            let _g = info_span!("sched_refit_incremental").entered();
            sched.edit_gen = sched.edit_gen.wrapping_add(1);
            let mut move_cstack: Vec<u32> = Vec::new();
            let mut move_cscratch: Vec<u32> = Vec::new();
            for entity in &changed {
                let Some(&idx) = sched.entity_index.get(&entity) else { continue };
                let Ok((_, t, p, op, _order, m)) = volumes.get(entity) else { continue };
                let transform = t.compute_transform();
                let new_aabb = edits::edit_world_aabb(p, &transform, op.smoothing);
                let resolved = edits::ResolvedEdit::new(p.clone(), transform, *op, m.registry_id as u16);
                // SURFACE-PRUNED dirty over the moved edit's old∪new position — only the moving
                // surface shell, never the solid interior. Reads the PREVIOUS resolved edit (still in
                // `sched.edits[idx]`) for the old surface before overwriting it.
                let old_resolved = sched.edits[idx as usize].clone();
                dirty_moving_edit(&mut sched.pending, &old_resolved, &resolved, &config, camera_pos);
                Arc::make_mut(&mut sched.edits)[idx as usize] = resolved;
                Arc::make_mut(&mut sched.bvh).refit_edit(idx, new_aabb);
                bvh.refit_edit(idx, new_aabb);
                // Refresh cons over the moved edit's old∪new footprint (post-refit BVH) so the DDA sees
                // it at its new coarse-LOD position and drops it at the old one.
                let old_aabb = edits::edit_world_aabb(
                    &old_resolved.prim,
                    &old_resolved.transform,
                    old_resolved.op.smoothing,
                );
                let medits = Arc::clone(&sched.edits);
                refresh_conservative_footprints(
                    &mut atlas,
                    &[old_aabb, new_aabb],
                    &medits,
                    &bvh,
                    &config,
                    camera_pos,
                    &mut move_cscratch,
                    &mut move_cstack,
                );
                prev_aabbs.map.insert(entity, new_aabb);
            }
            // Indices stable → keep the far carry backlog, drop only the re-dirtied footprint.
            invalidate_ready_on_edit_change(&mut sched, &config, false);
        }
    }

    // --- 2. Camera chunk-ring recenter (eager enter + immediate evict, absolute addressing) ----
    // Entered chunks with geometry are enqueued for a bake; EXITED chunks are evicted immediately
    // (no make-before-break deferral — see the exited-chunk handler). `atlas.remove_brick` bumps the
    // upload + topology generations itself, so an evict-only frame (flying away from the scene) still
    // makes the render world re-extract and drop the stale bricks — it doesn't depend on a bake
    // being applied that frame.
    let g_recenter = info_span!("sched_recenter").entered();
    let r = config.ring_chunks_per_axis();
    let mut bvh_stack: Vec<u32> = Vec::new();
    let mut cons_scratch: Vec<u32> = Vec::new();
    let cons_edits = Arc::clone(&sched.edits);
    for lod in 0..lod_count {
        let li = lod as usize;
        let new_origin = ring_chunk_origin(&config, camera_pos, lod);
        let old_origin = sched.ring_chunk_origin[li];
        if new_origin == old_origin {
            continue;
        }
        // Entered SHELL chunks → enqueue a bake, but skip empty ones: an entered chunk has no
        // resident bricks yet, so a chunk no edit reaches has nothing to bake. Enqueuing it
        // anyway would burn the per-frame budget on all-`None` bakes and starve the real
        // geometry entering far rings (the fly-away-from-scene LOD-stall bug).
        // The resident region is a hollow SHELL (outer ring minus the finer-covered inner hole), so
        // "entered" = the gained outer rim ∪ the hole boundary the receding hole uncovered — two thin
        // slabs, never the R³ interior. This is what keeps each region to `{native..native+overlap}`
        // instead of the whole LOD stack: a near surface's coarse LODs sit inside their holes and are
        // never enqueued.
        let first = old_origin == IVec3::splat(i32::MIN);
        for_each_entered_shell(&config, lod, new_origin, old_origin, first, |coord| {
            let ck = chunk::ChunkKey::new(lod, coord);
            if chunk_has_geometry_with(ck, &bvh, &config, &mut bvh_stack) {
                // A freshly-entered chunk has no resident bricks yet → bake all 64.
                dirty_mask(&mut sched.pending, ck, FULL_CHUNK_MASK);
            }
        });
        // Entered FULL-RING chunks → set their CONSERVATIVE occupancy (the empty-space DDA's traversal
        // grid). NO shell holes: a near surface's coarse LODs keep a traversal entry even though their
        // tiles are shelled out, so the DDA's coarse jumps survive (this is what fixes the 2.5× march
        // regression). One early-exit BVH probe pre-filters empties before the 64-brick mask.
        for_each_entered_chunk(new_origin, old_origin, r, |coord| {
            let ck = chunk::ChunkKey::new(lod, coord);
            let mask = if chunk_has_geometry_with(ck, &bvh, &config, &mut bvh_stack) {
                chunk_conservative_mask(ck, &cons_edits, &bvh, &config, &mut cons_scratch, &mut bvh_stack)
            } else {
                0
            };
            atlas.set_conservative_chunk(ck, mask, &config);
        });
        // Exited SHELL chunks → cancel any pending bake and EVICT IMMEDIATELY (no deferral). Two cases,
        // both safe: a chunk that left the OUTER edge renders at the next resident coarser LOD
        // (`resolve_march` serves the finest RESIDENT level — a brief LOD pop during handoff, never a
        // hole, since the coarser ring is larger and bakes first); a chunk the INNER hole newly covers
        // became redundant — two finer levels are already resident there, so dropping it is hole-free by
        // construction (no make-before-break needed). Evicting inline spreads the cost over the small
        // per-frame exited slab. Skipped on the first run (the sentinel origin isn't a real region).
        if !first {
            for_each_exited_shell(&config, lod, new_origin, old_origin, |coord| {
                let ck = chunk::ChunkKey::new(lod, coord);
                sched.pending.remove(&ck);
                for_each_brick_key(ck, &config, |bk| {
                    atlas.remove_brick(&bk, &config);
                });
            });
            // Exited FULL-RING chunks → drop their conservative occupancy (removes the directory entry
            // if no tiles remain). Mirrors the shell eviction but over the whole ring.
            for_each_exited_chunk(new_origin, old_origin, r, |coord| {
                atlas.set_conservative_chunk(chunk::ChunkKey::new(lod, coord), 0, &config);
            });
        }
        sched.ring_chunk_origin[li] = new_origin;
    }
    drop(g_recenter);

    // --- 3. Emit GPU bake jobs for the dirty chunks ----------------------------------
    // The CPU does only topology (BVH cull + palette + tile alloc) and emits a GpuBakeJob per
    // brick; the compute shader fills the texels. Bounded + parallel per frame so even a big cold
    // shell bakes a slice every frame (no off-thread single-flight wait), settling in ~bricks/budget.
    {
        let _g_dispatch = info_span!("sched_dispatch").entered();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu_bakes, &config, bake_view, &mut baked_dbg, now);
    }

    // DIAGNOSTIC: per-LOD baked-job histogram this frame + remaining `pending` backlog. Shows
    // whether a small edit's cost is spread across coarse LODs (each coarse chunk = 64 huge
    // bricks) and how much spilled. Gated on the overlay toggle so it's silent in normal use.
    if baked_dbg.enabled && !gpu_bakes.jobs.is_empty() {
        let mut by_lod = [0u32; 16];
        for j in &gpu_bakes.jobs {
            if (j.lod as usize) < by_lod.len() {
                by_lod[j.lod as usize] += 1;
            }
        }
        let nonzero: Vec<(usize, u32)> = by_lod.iter().enumerate()
            .filter(|(_, n)| **n > 0).map(|(l, n)| (l, *n)).collect();
        info!("  baked jobs={} pending_backlog={} by_lod={:?}", gpu_bakes.jobs.len(), sched.pending.len(), nonzero);
    }
}

/// Emit one bake job for `key` from already-culled edit indices `indices` and a known `palette`.
/// No re-cull, no palette rebuild — the caller supplies both. `tile` must be allocated.
#[allow(clippy::too_many_arguments)]
fn push_bake_job(
    gpu_bakes: &mut PendingGpuBakes,
    edits_snapshot: &[edits::ResolvedEdit],
    config: &SdfGridConfig,
    key: atlas::BrickKey,
    tile: u32,
    mat_tile: Option<u32>,
    indices: &[u32],
    palette: edits::Palette,
) {
    let edit_start = gpu_bakes.edits.len() as u32;
    for &i in indices {
        gpu_bakes.edits.push(edits::to_gpu_edit(&edits_snapshot[i as usize]));
    }
    gpu_bakes.jobs.push(GpuBakeJob {
        tile,
        mat_tile,
        lod: key.lod,
        coord: key.coord,
        voxel_size: config.voxel_size_at(key.lod),
        dist_band: atlas::dist_band_world(config, key.lod),
        palette,
        edit_start,
        edit_count: indices.len() as u32,
    });
}


/// Phase 1 (serial, cheap, MAIN THREAD): gather candidate bricks from the drained chunks, restricted
/// to each chunk's DIRTY-BRICK MASK. A chunk no edit reaches has nothing to bake, but it may hold
/// RESIDENT bricks that must be evicted the same frame (a moved edit's vacated chunks — the drag
/// trail), so `evict` is invoked for the chunk's DIRTY bricks if it is empty. Non-empty chunks
/// contribute only their dirty brick keys to `candidates`. The empty pre-cull is one BVH query per
/// chunk instead of one per brick. Eviction stays synchronous here even when the classify is
/// offloaded, so the drag trail never lags behind the async bake.
///
/// Masking the eviction is correct: a brick resident in a now-empty chunk must lie in the vacating
/// edit's old footprint (or it'd belong to other geometry, contradicting "empty"), so it is dirty.
fn gather_candidates(
    drained: &[(chunk::ChunkKey, u64)],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    candidates: &mut Vec<(chunk::ChunkKey, atlas::BrickKey)>,
    mut evict: impl FnMut(atlas::BrickKey),
) {
    let _g = info_span!("emit_phase1_gather").entered();
    let mut cull_stack: Vec<u32> = Vec::new();
    for &(ck, mask) in drained.iter() {
        if !chunk_has_geometry_with(ck, bvh, config, &mut cull_stack) {
            for_each_brick_key_masked(ck, mask, config, &mut evict);
            continue;
        }
        for_each_brick_key_masked(ck, mask, config, |key| candidates.push((ck, key)));
    }
}

/// Phase 3 (serial, MAIN THREAD): apply the verdicts — evict, skip, or insert + push a GPU bake job
/// under the soft budget. Mutates the atlas + job list, so it must run on the main thread before the
/// render-world Extract.
///
/// CHUNK-ATOMIC budgeting (make-before-break): a chunk's WHOLE Keep-set bakes this frame, or the
/// whole chunk spills to a later one — never partially. Every brick of a chunk that bakes calls
/// `set_brick` this same frame, so the chunk appears in the GPU lookup table ATOMICALLY at the next
/// extract; the finer LOD never shows half-baked (some bricks resident, neighbours not), which is the
/// sub-chunk mixed-LOD "garbled ripple" seen during a LOD 4→3 handoff. While a chunk waits, the
/// coarser LOD that covers it (resident in its larger window) keeps serving the region — hole-free.
///
/// `candidates` are grouped by chunk (consecutive — `gather_candidates` emits each chunk's bricks in
/// a run, and the parallel classify preserves order), so one pass detects chunk runs by key change.
///
/// A spilled chunk's Keep verdicts are RETAINED into `deferred` (one [`ReadyChunk`] per spilled
/// chunk, appended — the caller owns the running queue) so the expensive classify runs once per
/// brick; a later frame bakes them via [`apply_ready`] with no re-classify. The drag-trail eviction
/// of a stale-resident spilled brick is unchanged.
#[expect(clippy::too_many_arguments)]
fn apply_verdicts(
    atlas: &mut SdfAtlas,
    gpu_bakes: &mut PendingGpuBakes,
    edits_snapshot: &[edits::ResolvedEdit],
    config: &SdfGridConfig,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
    mut verdicts: Vec<Verdict>,
    deferred: &mut Vec<ReadyChunk>,
    job_budget: usize,
) {
    let _g3 = info_span!("emit_phase3_apply").entered();
    let n = candidates.len();
    // Latches once a chunk doesn't fit: every later (lower-priority — `sort_drained` ordered them)
    // chunk spills too, so we never bake out of priority order to squeeze a small chunk in.
    let mut budget_full = false;
    let mut i = 0;
    while i < n {
        let ck = candidates[i].0;
        let mut j = i;
        let mut keep_count = 0usize;
        while j < n && candidates[j].0 == ck {
            if matches!(verdicts[j], Verdict::Keep(..)) {
                keep_count += 1;
            }
            j += 1;
        }
        // Spill the whole chunk if its Keep-set wouldn't FULLY fit this frame's remaining budget.
        // (A chunk holds ≤ CHUNK_VOLUME bricks ≪ the budget, so a chunk is never permanently
        // unbakeable — it just waits for a frame with room.)
        let spill = keep_count > 0 && (budget_full || gpu_bakes.jobs.len() + keep_count > job_budget);
        budget_full |= spill;
        // A spilled chunk accumulates its Keep set here to carry forward (classified once).
        let mut carry = spill.then(|| ReadyChunk { ck, keeps: Vec::with_capacity(keep_count) });
        for k in i..j {
            let key = candidates[k].1;
            match std::mem::replace(&mut verdicts[k], Verdict::Skip) {
                Verdict::Empty | Verdict::Drop => {
                    atlas.remove_brick(&key, config);
                }
                // Resident brick, content unchanged (hash matched in classify) → texels still valid,
                // leave it as-is. This is what keeps a sphere dragged over the heightmap cheap.
                Verdict::Skip => {}
                Verdict::Keep(palette, indices, hash) => {
                    if let Some(carry) = carry.as_mut() {
                        // Deferred to a later frame. If the brick is currently RESIDENT with a stale
                        // hash (a re-bake — its content changed) it holds STALE texels; evict it so the
                        // lookup misses and falls back to the correct coarser LOD until its real bake
                        // lands ("old surface band left behind while dragging"). A genuinely new brick
                        // isn't resident, so this is a no-op. The Keep itself is RETAINED (carried,
                        // not re-classified) — `apply_ready` re-inserts + bakes it later.
                        atlas.remove_brick(&key, config);
                        carry.keeps.push((key, palette, indices, hash));
                        continue;
                    }
                    let tile = atlas.insert_gpu_brick(key, palette, hash, config);
                    let mat_tile = atlas.mat_tiles.tile(&key);
                    push_bake_job(
                        gpu_bakes, edits_snapshot, config, key, tile, mat_tile, &indices, palette,
                    );
                    if baked_dbg.enabled {
                        let bw = config.brick_world_size(key.lod);
                        let center = config.brick_min_world(key.coord, key.lod) + Vec3::splat(0.5 * bw);
                        baked_dbg.bricks.push((center, bw, now_secs));
                    }
                }
            }
        }
        // `spill` ⇒ keep_count > 0 ⇒ carry.keeps is non-empty, so this never carries an empty group.
        if let Some(carry) = carry.take() {
            deferred.push(carry);
        }
        i = j;
    }
}

/// Bake carried Keep groups from `ready` up to `job_budget`, oldest-priority first (`ready` is kept
/// sorted coarse-LOD/nearest-first by `refresh_ready`). Chunk-atomic: a chunk whose Keep set won't
/// FULLY fit this frame's remaining budget stays in `ready`, and — latching like `apply_verdicts` —
/// every lower-priority chunk after it waits too, holding the bake in priority order. The applied
/// chunks are a prefix of `ready` (priority-sorted), so they drain off the front. Each carried brick
/// was evicted on its first defer, so `insert_gpu_brick` allocates a fresh (fungible) tile.
#[expect(clippy::too_many_arguments)]
fn apply_ready(
    atlas: &mut SdfAtlas,
    gpu_bakes: &mut PendingGpuBakes,
    edits_snapshot: &[edits::ResolvedEdit],
    config: &SdfGridConfig,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
    ready: &mut Vec<ReadyChunk>,
    job_budget: usize,
) {
    if ready.is_empty() {
        return;
    }
    let _g = info_span!("apply_ready", ready = ready.len()).entered();
    // How many priority-ordered chunks fit this frame (stop at the first that doesn't — latch).
    let mut jobs = gpu_bakes.jobs.len();
    let mut fit = 0;
    for rc in ready.iter() {
        if jobs + rc.keeps.len() > job_budget {
            break;
        }
        jobs += rc.keeps.len();
        fit += 1;
    }
    for rc in ready.drain(0..fit) {
        for (key, palette, indices, hash) in &rc.keeps {
            let tile = atlas.insert_gpu_brick(*key, *palette, *hash, config);
            let mat_tile = atlas.mat_tiles.tile(key);
            push_bake_job(
                gpu_bakes, edits_snapshot, config, *key, tile, mat_tile, indices, *palette,
            );
            if baked_dbg.enabled {
                let bw = config.brick_world_size(key.lod);
                let center = config.brick_min_world(key.coord, key.lod) + Vec3::splat(0.5 * bw);
                baked_dbg.bricks.push((center, bw, now_secs));
            }
        }
    }
}

/// Per-frame `ready`-queue maintenance, run BEFORE any apply in every bake frame:
///   1. Stale-snapshot invalidation: if `ready_edit_gen != edit_gen` the edit set changed since the
///      carried indices were built. Conservatively re-queue every carried chunk to `pending` (it
///      re-classifies against the new snapshot) and clear `ready`. (Refined to a selective,
///      footprint-scoped invalidation in `schedule_bakes` step 1 so a small move keeps the far
///      backlog — but this full flush is the always-correct fallback and the only path the
///      headless test/settle harness exercises.)
///   2. Window-filter: drop carried chunks that left their LOD's current ring window — the recenter
///      already evicted their bricks, and re-inserting would collide with the in-window chunk
///      sharing their toroidal `c mod R` directory slot (the hazard `apply_async_result` guards).
///      Dropped entries are NOT re-queued (their region is gone).
///   3. Re-sort by the CURRENT camera (coarse-LOD-first, nearest-first — same key as `sort_drained`)
///      so the most-visible deferred work bakes first even as the camera moves. Per-chunk, cheap.
///
/// Steps 2–3 (the O(ready) pass) run ONLY when the camera moved since the last maintenance — window
/// membership + priority can't change while it's stationary, which is exactly when `ready` is large
/// (a cold bake), so a fixed-camera settle pays O(1) here per frame.
/// Reconcile the carry queue (`ready`) against a new edit snapshot, called from `schedule_bakes`
/// step 1 AFTER the changed edit's footprint is dirtied into `pending`:
///
/// `index_shift` (whole-set rebuild, or an add/remove) shifts every edit's position in the snapshot,
/// so all carried indices are stale → flush all `ready` back to `pending` to re-classify against the
/// new snapshot. Otherwise (a pure move/property change) positions are stable, so only the chunks the
/// edit's old∪new footprint re-dirtied (now in `pending`) are stale → drop just those from `ready`. A
/// kept entry's chunk wasn't in the footprint, so it can't fold the moved edit → its indices stay
/// valid. This is what keeps a small edit cheap even mid-bake: the far carried backlog survives
/// instead of re-classifying. Either way `ready_edit_gen` is synced so `refresh_ready`'s conservative
/// full flush is a no-op.
fn invalidate_ready_on_edit_change(sched: &mut BakeScheduler, config: &SdfGridConfig, index_shift: bool) {
    if index_shift {
        // Every carried index is stale → re-queue every group's bricks to re-classify (they were
        // evicted on defer, so a re-queue is the only path that bakes them).
        for rc in std::mem::take(&mut sched.ready) {
            dirty_mask(&mut sched.pending, rc.ck, rc.carried_mask(config));
        }
    } else {
        // Indices stable. A carried group is stale only where the moved edit's re-dirtied footprint
        // INTERSECTS its carried bricks (those bricks fold the moved edit ⇒ their palette is now
        // wrong). Re-queue exactly those groups' bricks; groups whose bricks the footprint didn't
        // touch keep their valid carried palette and stay in `ready`.
        let ready = std::mem::take(&mut sched.ready);
        let pending = &mut sched.pending;
        sched.ready = ready
            .into_iter()
            .filter_map(|rc| {
                let carried = rc.carried_mask(config);
                let dirtied = pending.get(&rc.ck).copied().unwrap_or(0);
                if dirtied & carried != 0 {
                    dirty_mask(pending, rc.ck, carried);
                    None // dropped from ready; its bricks now re-classify via pending
                } else {
                    Some(rc)
                }
            })
            .collect();
    }
    sched.ready_edit_gen = sched.edit_gen;
}

fn refresh_ready(sched: &mut BakeScheduler, config: &SdfGridConfig, view: &BakeView) {
    if sched.ready_edit_gen != sched.edit_gen {
        for rc in std::mem::take(&mut sched.ready) {
            dirty_mask(&mut sched.pending, rc.ck, rc.carried_mask(config));
        }
        sched.ready_edit_gen = sched.edit_gen;
    }
    // Window/shell membership AND priority only change when the camera MOVES or TURNS, so skip the
    // O(ready) pass while both are unchanged (a stationary cold bake — the only time `ready` is large).
    if sched.ready.is_empty() || (view.pos == sched.ready_maint_cam && view.fwd == sched.ready_maint_fwd) {
        return;
    }
    sched.ready_maint_cam = view.pos;
    sched.ready_maint_fwd = view.fwd;
    // Window-filter only once a window exists (the recenter has populated `ring_chunk_origin`). With
    // no window, no chunk can have exited, so there is nothing to drop — and a missing origin would
    // wrongly reject everything. (The direct-dirty unit tests drive emit without a recenter.)
    if !sched.ring_chunk_origin.is_empty() {
        let origins = &sched.ring_chunk_origin;
        // Drop carried chunks that left the resident SHELL — either past the outer edge OR newly
        // covered by a finer LOD (now in the inner hole, so redundant).
        sched.ready.retain(|rc| {
            let origin = origins.get(rc.ck.lod as usize).copied().unwrap_or(IVec3::splat(i32::MIN));
            chunk_in_shell(config, rc.ck.lod, rc.ck.coord, origin)
        });
    }
    // `by_cached_key` computes the (non-trivial) priority key ONCE per entry, not once per
    // comparison — important when a moving camera re-sorts a large `ready` mid-cold-bake.
    sched.ready.sort_by_cached_key(|rc| chunk_priority_key(rc.ck, config, view));
}

/// THE per-frame bake: bake carried (already-classified) work first, then — if the budget has room —
/// drain a bounded batch of dirty `pending`, classify it in PARALLEL on the compute pool, and apply.
/// Each brick is classified exactly once (over-budget Keeps carry forward in `ready`); the bounded
/// batch keeps the per-frame classify + apply small, and the parallel classify means even a big cold
/// shell bakes every frame (no off-thread single-flight wait), so a cold scene settles in ~bricks/
/// budget frames. Used by both `schedule_bakes` (production) and the unit/perf settle drivers.
fn emit_gpu_bakes(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    view: impl Into<BakeView>,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
) {
    let _span = info_span!("sdf_emit_gpu_bakes").entered();
    let view = view.into();
    let edits_snapshot = Arc::clone(&sched.edits);
    let bvh_snapshot = Arc::clone(&sched.bvh);

    // 1. Maintain the carry queue (stale-flush, shell-filter, re-sort), then bake carried work
    //    FIRST — it's already classified, and draining it before new work honors priority.
    refresh_ready(sched, config, &view);
    apply_ready(atlas, gpu_bakes, &edits_snapshot, config, baked_dbg, now_secs, &mut sched.ready, SOFT_BAKE_BUDGET);

    // 2. Classify fresh `pending` ONLY if this frame's budget isn't already spent by carried work.
    //    When it is, leave `pending` intact (don't drain what we won't classify) — a later frame
    //    picks it up. The whole drained set is classified ONCE; its over-budget Keeps are carried in
    //    `ready` (not re-queued), so a settle never re-classifies the backlog.
    if gpu_bakes.jobs.len() < SOFT_BAKE_BUDGET && !sched.pending.is_empty() {
        let mut scratch = std::mem::take(&mut sched.emit_scratch);
        scratch.candidates.clear();
        drain_priority_batch(&mut sched.pending, &mut scratch.drained, config, &view, CLASSIFY_REFILL_CHUNKS);
        gather_candidates(&scratch.drained, &bvh_snapshot, config, &mut scratch.candidates, |key| {
            atlas.remove_brick(&key, config);
        });
        let hash_peek = snapshot_hash_peek(atlas, &scratch.candidates);
        let verdicts = classify_candidates(&scratch.candidates, &edits_snapshot, &bvh_snapshot, config, &hash_peek);
        apply_verdicts(
            atlas, gpu_bakes, &edits_snapshot, config, baked_dbg, now_secs, &scratch.candidates, verdicts,
            &mut sched.ready, SOFT_BAKE_BUDGET,
        );
        sched.emit_scratch = scratch;
    }
}

// --- Shared test/perf drive helpers ---------------------------------------------
//
// The recenter + scheduler-priming helpers used to drive the bake lifecycle directly (no ECS
// App) from BOTH the `tests` unit module and the `perf` harness. Kept here, on the shared
// parent, so there is a SINGLE source for the recenter mirror (it tracks `schedule_bakes`
// step 2 — see below) rather than a copy per test module.

/// Mirror `schedule_bakes` step 2 (camera recenter): enqueue entered geometry chunks into
/// `pending`, evict exited chunks eagerly. Returns the number of geometry chunks enqueued
/// (for the fly-away starvation bound). `ring_chunk_origin` lives on the scheduler.
#[cfg(test)]
fn recenter_step(sched: &mut BakeScheduler, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, cam: Vec3) -> usize {
    if sched.ring_chunk_origin.is_empty() {
        sched.ring_chunk_origin = vec![IVec3::splat(i32::MIN); cfg.lod_count as usize];
    }
    let r = cfg.ring_chunks_per_axis();
    let mut stack: Vec<u32> = Vec::new();
    let mut cscratch: Vec<u32> = Vec::new();
    let cons_edits = Arc::clone(&sched.edits);
    let mut enqueued = 0usize;
    for lod in 0..cfg.lod_count {
        let li = lod as usize;
        let new_origin = ring_chunk_origin(cfg, cam, lod);
        let old_origin = sched.ring_chunk_origin[li];
        if new_origin == old_origin {
            continue;
        }
        let first = old_origin == IVec3::splat(i32::MIN);
        for_each_entered_shell(cfg, lod, new_origin, old_origin, first, |coord| {
            let ck = chunk::ChunkKey::new(lod, coord);
            if chunk_has_geometry_with(ck, &sched.bvh, cfg, &mut stack) {
                if !sched.pending.contains_key(&ck) {
                    enqueued += 1;
                }
                dirty_mask(&mut sched.pending, ck, FULL_CHUNK_MASK);
            }
        });
        // Mirror production: full-ring CONSERVATIVE occupancy (the empty-space DDA's traversal grid).
        for_each_entered_chunk(new_origin, old_origin, r, |coord| {
            let ck = chunk::ChunkKey::new(lod, coord);
            let mask = if chunk_has_geometry_with(ck, &sched.bvh, cfg, &mut stack) {
                chunk_conservative_mask(ck, &cons_edits, &sched.bvh, cfg, &mut cscratch, &mut stack)
            } else {
                0
            };
            atlas.set_conservative_chunk(ck, mask, cfg);
        });
        if !first {
            for_each_exited_shell(cfg, lod, new_origin, old_origin, |coord| {
                let ck = chunk::ChunkKey::new(lod, coord);
                sched.pending.remove(&ck);
                for bk in chunk_brick_keys(ck, cfg) {
                    atlas.remove_brick(&bk, cfg);
                }
            });
            for_each_exited_chunk(new_origin, old_origin, r, |coord| {
                atlas.set_conservative_chunk(chunk::ChunkKey::new(lod, coord), 0, cfg);
            });
        }
        sched.ring_chunk_origin[li] = new_origin;
    }
    enqueued
}

/// Build a BVH over the world AABBs of `resolved` (one leaf per edit), exactly as `schedule_bakes`
/// does, so the topology cull a settle exercises is the production cull.
#[cfg(test)]
fn build_bvh(resolved: &[edits::ResolvedEdit]) -> bvh::Bvh {
    let aabbs: Vec<bevy::math::bounding::Aabb3d> = resolved
        .iter()
        .map(|e| edits::edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    bvh::Bvh::build(&aabbs)
}

/// A scheduler primed with `resolved` + its BVH (the rest defaulted) — the starting state a
/// settle/perf drive needs before the first recenter.
#[cfg(test)]
fn primed_sched(resolved: &[edits::ResolvedEdit]) -> BakeScheduler {
    BakeScheduler {
        edits: Arc::new(resolved.to_vec()),
        bvh: Arc::new(build_bvh(resolved)),
        ..BakeScheduler::default()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod perf;
