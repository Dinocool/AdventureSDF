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
use bevy::tasks::{AsyncComputeTaskPool, ComputeTaskPool, ParallelSlice, Task, block_on, poll_once};

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
    chunk_has_geometry_with, chunk_in_window, chunk_window_keys, chunks_in_aabb_windowed,
    for_each_brick_key, for_each_entered_chunk, for_each_exited_chunk,
};
#[cfg(test)]
use window::chunk_brick_keys;

// The read-only classify core (Send; runs across the compute pool or on a background task). Bare
// names re-imported so the dispatch path + the in-file tests use them unqualified.
mod classify;
use classify::{Verdict, classify_candidates, classify_candidates_serial, snapshot_hash_peek};

/// One brick the GPU compute bake must fill this frame. The CPU has already done the
/// topology work (BVH cull → `edit_indices` into the frame's flat edit list, palette,
/// tile allocation); the compute shader runs the 512-voxel `fold_csg` eval and writes the
/// brick's texels straight into the atlas tile at `tile`. `lod`/`coord` give the shader
/// each voxel's world position; `dist_band` is the per-LOD snorm clamp band so the shader
/// needs no `SdfGridConfig`.
#[derive(Clone)]
pub struct GpuBakeJob {
    pub tile: u32,
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
    /// Chunk keys awaiting a bake (deduped).
    pending: std::collections::HashSet<chunk::ChunkKey>,
    /// Reusable emit scratch (cleared + refilled each frame) so a continuous drag does zero
    /// growth reallocation. `mem::take`n into `emit_gpu_bakes` and restored at the end.
    emit_scratch: EmitScratch,
    /// Monotonic counter bumped whenever the edit set changes (an edit moves/adds/removes →
    /// `edits`/`bvh` are replaced). Stamped onto an async bake task's input snapshot; when the
    /// task lands, a mismatch means the edits changed mid-flight, so its (now stale) classify is
    /// discarded and its chunks re-queued. See the async dispatch in [`schedule_bakes`].
    edit_gen: u64,
}

/// Per-frame scratch buffers for `emit_gpu_bakes`, held on the scheduler so their capacity
/// persists across frames instead of allocating from empty each emit.
#[derive(Default)]
struct EmitScratch {
    drained: Vec<chunk::ChunkKey>,
    candidates: Vec<(chunk::ChunkKey, atlas::BrickKey)>,
    spilled: std::collections::HashSet<chunk::ChunkKey>,
}

impl Default for BakeScheduler {
    fn default() -> Self {
        Self {
            edits: Arc::new(Vec::new()),
            bvh: Arc::new(bvh::Bvh::default()),
            height: Arc::new(super::height::HeightField::default()),
            ring_chunk_origin: Vec::new(),
            pending: std::collections::HashSet::new(),
            emit_scratch: EmitScratch::default(),
            edit_gen: 0,
        }
    }
}

/// Holds the single in-flight async bake-classify task (the offload path for large bakes). The
/// task runs the read-only [`classify_candidates`] on snapshots; its result is applied
/// synchronously on the main thread in [`schedule_bakes`] (atlas mutation can't be off-thread —
/// the render-world Extract reads the atlas the same frame). Single-flight: while a task is in
/// flight, newly dirtied chunks accumulate in `pending` and the next spawn (after this one lands)
/// drains them. Modelled on `TextureStreamState` in render.rs.
#[derive(Resource, Default)]
pub struct BakeTaskState {
    task: Option<Task<BakeTaskOutput>>,
}

/// The result a background classify task hands back to the main thread, plus the snapshot identity
/// needed to reconcile staleness on apply.
struct BakeTaskOutput {
    candidates: Vec<(chunk::ChunkKey, atlas::BrickKey)>,
    verdicts: Vec<Verdict>,
    /// `edit_gen` the snapshot was built from — if it no longer matches the scheduler's, the edit
    /// set changed mid-flight and every verdict is stale.
    edit_gen: u64,
}

/// Candidate-count threshold above which a bake is offloaded to a background task instead of run
/// synchronously. Small bakes (dragging an object, a single-LOD nudge — a few hundred bricks) stay
/// fully synchronous so editing feels instant; only a big shell (a coarse-LOD snap, ~10k–30k
/// candidates) crosses this and goes async, where it costs a few frames of latency (coarse LOD
/// covers the gap) instead of a main-thread hitch.
const ASYNC_BAKE_THRESHOLD: usize = 4096;

impl BakeScheduler {
    /// Replace the height-field snapshot used by subsequent bakes (rebuilt when the material
    /// registry's displacement columns change).
    pub fn set_height(&mut self, height: super::height::SharedHeightField) {
        self.height = height;
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
#[expect(clippy::too_many_arguments)]
pub fn schedule_bakes(
    mut atlas: ResMut<SdfAtlas>,
    mut bvh: ResMut<bvh::Bvh>,
    mut sched: ResMut<BakeScheduler>,
    mut prev_aabbs: ResMut<PrevEditAabbs>,
    mut gpu_bakes: ResMut<PendingGpuBakes>,
    mut bake_task: ResMut<BakeTaskState>,
    config: Res<SdfGridConfig>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    changed: Query<Entity, (With<SdfVolume>, ChangedEdit)>,
    mut removed: RemovedComponents<SdfVolume>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
    mut baked_dbg: ResMut<super::BakedBrickDebug>,
    time: Res<Time>,
) {
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
    let camera_pos = camera.iter().next().map(|t| t.translation).unwrap_or(Vec3::ZERO);
    let lod_count = config.lod_count;
    let r = config.ring_chunks_per_axis();
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
        let _g_gather = info_span!("sched_gather").entered();
        let gathered = gather_sorted_edits(&volumes);
        let current: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d> =
            gathered.iter().map(|g| (g.entity, g.aabb)).collect();
        let set_changed = current.len() != prev_aabbs.map.len()
            || current.keys().any(|e| !prev_aabbs.map.contains_key(e));
        let resolved: Vec<edits::ResolvedEdit> = gathered.iter().map(|g| g.edit.clone()).collect();
        let aabbs: Vec<bevy::math::bounding::Aabb3d> = gathered.iter().map(|g| g.aabb).collect();
        let new_bvh = bvh::Bvh::build(&aabbs);
        *bvh = new_bvh.clone();
        sched.bvh = Arc::new(new_bvh);
        sched.edits = Arc::new(resolved);
        // Bump the edit generation so any in-flight async classify task built from the OLD edits is
        // recognised as stale when it lands (its verdicts are dropped + chunks re-queued).
        sched.edit_gen = sched.edit_gen.wrapping_add(1);
        // NOTE: no global cache-invalidation here. Each brick is memoised by a CONTENT HASH of
        // the edits it folds (`PackedBrick::baked_hash`), so a re-dirtied chunk re-bakes only the
        // bricks whose folded content actually changed — moving one edit no longer invalidates
        // every brick its coarse footprint overlaps (see the Phase-3 hash skip in emit_gpu_bakes).

        // DIAGNOSTIC: log WHY a rebake fired and WHICH entities changed, so we can see if moving
        // one small edit wrongly drags a terrain-scale edit into `changed` (→ full-ring re-dirty).
        if baked_dbg.enabled {
            let changed_list: Vec<(String, f32)> = changed
                .iter()
                .filter_map(|e| {
                    current.get(&e).map(|aabb| {
                        let he = Vec3::from(aabb.max) - Vec3::from(aabb.min);
                        // a size bucket label + the AABB's largest extent
                        let span = he.max_element();
                        let label = if span > 100.0 { "HUGE" } else if span > 5.0 { "big" } else { "small" };
                        (label.to_string(), span)
                    })
                })
                .collect();
            info!(
                "rebake: rebake_all={} set_changed={} changed_n={} whole_ring={} | {:?}",
                atlas.rebake_all, set_changed, changed.iter().count(),
                atlas.rebake_all || set_changed, changed_list,
            );
        }

        if atlas.rebake_all || set_changed {
            // Whole set changed → re-dirty every resident-window chunk at each LOD.
            for lod in 0..lod_count {
                let origin = ring_chunk_origin(&config, camera_pos, lod);
                for ck in chunk_window_keys(origin, r, lod) {
                    sched.pending.insert(ck);
                }
            }
            atlas.rebake_all = false;
        } else {
            // Existing edits moved → dirty the chunks over each changed edit's old∪new footprint,
            // clamped to the resident window for that LOD.
            for entity in &changed {
                let old = prev_aabbs.map.get(&entity);
                let new = current.get(&entity);

                for lod in 0..lod_count {
                    let origin = ring_chunk_origin(&config, camera_pos, lod);
                    // `chunks_in_aabb_windowed` already clamps to the window, so no per-key
                    // `chunk_in_window` filter is needed — and crucially it never ENUMERATES
                    // outside the window, so a terrain-scale heightmap AABB costs O(window),
                    // not O(world).
                    let mut dirty_one = |aabb: &bevy::math::bounding::Aabb3d| {
                        for ck in chunks_in_aabb_windowed(&config, aabb, lod, origin, r) {
                            sched.pending.insert(ck);
                        }
                    };
                    if let Some(old) = old {
                        dirty_one(old);
                    }
                    if let Some(new) = new {
                        dirty_one(new);
                    }
                }
            }
        }
        prev_aabbs.map = current;
    }

    // --- 2. Camera chunk-ring recenter (eager enter + immediate evict, absolute addressing) ----
    // Entered chunks with geometry are enqueued for a bake; EXITED chunks are evicted immediately
    // (no make-before-break deferral — see the exited-chunk handler). `atlas.remove_brick` bumps the
    // upload + topology generations itself, so an evict-only frame (flying away from the scene) still
    // makes the render world re-extract and drop the stale bricks — it doesn't depend on a bake
    // being applied that frame.
    let g_recenter = info_span!("sched_recenter").entered();
    let mut bvh_stack: Vec<u32> = Vec::new();
    for lod in 0..lod_count {
        let li = lod as usize;
        let new_origin = ring_chunk_origin(&config, camera_pos, lod);
        let old_origin = sched.ring_chunk_origin[li];
        if new_origin == old_origin {
            continue;
        }
        // Entered chunks → enqueue a bake, but skip empty ones: an entered chunk has no
        // resident bricks yet, so a chunk no edit reaches has nothing to bake. Enqueuing it
        // anyway would burn the per-frame budget on all-`None` bakes and starve the real
        // geometry entering far rings (the fly-away-from-scene LOD-stall bug).
        // Only the entered SHELL is visited (slab difference), not the whole R³ interior — the
        // overlap stays resident and unchanged, so re-scanning it every snap was pure waste.
        for_each_entered_chunk(new_origin, old_origin, r, |coord| {
            let ck = chunk::ChunkKey::new(lod, coord);
            if chunk_has_geometry_with(ck, &bvh, &config, &mut bvh_stack) {
                sched.pending.insert(ck);
            }
        });
        // Exited chunks → cancel any pending bake and EVICT IMMEDIATELY (no deferral). `resolve_march`
        // serves the finest RESIDENT LOD, so once these fine bricks are gone the region just renders at
        // the next resident coarser LOD — a brief LOD pop during the handoff, never a hole, because a
        // coarser ring is larger and bakes first so a coarser level is already resident. Evicting inline
        // here also spreads the cost over the small per-frame exited shell instead of letting a held set
        // accumulate and drain in O(N)-sized bursts.
        // Skipped on the first run (no prior window — the sentinel origin isn't a real region).
        if !first_run {
            for_each_exited_chunk(new_origin, old_origin, r, |coord| {
                let ck = chunk::ChunkKey::new(lod, coord);
                sched.pending.remove(&ck);
                for_each_brick_key(ck, &config, |bk| {
                    atlas.remove_brick(&bk, &config);
                });
            });
        }
        sched.ring_chunk_origin[li] = new_origin;
    }
    drop(g_recenter);

    // --- 3. Emit GPU bake jobs for the dirty chunks ----------------------------------
    // The CPU does only topology (BVH cull + palette + tile alloc) and emits a GpuBakeJob per
    // brick; the compute shader fills the texels. Small bakes run synchronously; a large shell
    // (coarse-LOD snap) offloads its classify to a background task (see `dispatch_bake`).
    {
        let _g_dispatch = info_span!("sched_dispatch").entered();
        dispatch_bake(
            &mut atlas, &mut sched, &mut bake_task, &mut gpu_bakes, &config, camera_pos, &mut baked_dbg, now,
        );
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
fn push_bake_job(
    gpu_bakes: &mut PendingGpuBakes,
    edits_snapshot: &[edits::ResolvedEdit],
    config: &SdfGridConfig,
    key: atlas::BrickKey,
    tile: u32,
    indices: &[u32],
    palette: edits::Palette,
) {
    let edit_start = gpu_bakes.edits.len() as u32;
    for &i in indices {
        gpu_bakes.edits.push(edits::to_gpu_edit(&edits_snapshot[i as usize]));
    }
    gpu_bakes.jobs.push(GpuBakeJob {
        tile,
        lod: key.lod,
        coord: key.coord,
        voxel_size: config.voxel_size_at(key.lod),
        dist_band: atlas::dist_band_world(config, key.lod),
        palette,
        edit_start,
        edit_count: indices.len() as u32,
    });
}

/// Sort drained dirty chunks coarsest-LOD first, then nearest-camera first within an LOD — so the
/// ring fills 8→0 (the cap only ever spills fine detail whose coarse fallback is already resident)
/// and the chunks the viewer is looking at bake before distant ones.
fn sort_drained(drained: &mut [chunk::ChunkKey], config: &SdfGridConfig, camera_pos: Vec3) {
    drained.sort_unstable_by_key(|ck| {
        let size = chunk::chunk_world_size(ck.lod, config);
        let center = chunk::chunk_min_world(*ck, config) + Vec3::splat(size * 0.5);
        let d2 = center.distance_squared(camera_pos);
        (std::cmp::Reverse(ck.lod), d2.to_bits())
    });
}

/// Phase 1 (serial, cheap, MAIN THREAD): gather candidate bricks from the drained chunks. A chunk
/// no edit reaches has nothing to bake, but it may hold RESIDENT bricks that must be evicted the
/// same frame (a moved edit's vacated chunks — the drag trail), so `evict` is invoked for every
/// brick of an empty chunk. Non-empty chunks contribute their `CHUNK_BRICKS³` brick keys to
/// `candidates`. The empty pre-cull is one BVH query per chunk instead of 64 per-brick queries.
/// Eviction stays synchronous here even when the classify is offloaded, so the drag trail never
/// lags behind the async bake.
fn gather_candidates(
    drained: &[chunk::ChunkKey],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    candidates: &mut Vec<(chunk::ChunkKey, atlas::BrickKey)>,
    mut evict: impl FnMut(atlas::BrickKey),
) {
    let _g = info_span!("emit_phase1_gather").entered();
    let mut cull_stack: Vec<u32> = Vec::new();
    for ck in drained.iter() {
        if !chunk_has_geometry_with(*ck, bvh, config, &mut cull_stack) {
            for_each_brick_key(*ck, config, &mut evict);
            continue;
        }
        for_each_brick_key(*ck, config, |key| candidates.push((*ck, key)));
    }
}

/// Phase 3 (serial, MAIN THREAD): apply the verdicts — evict, skip, or insert + push a GPU bake job
/// under the soft budget. Mutates the atlas + job list, so it must run on the main thread before the
/// render-world Extract. `spilled` is a reusable set (cleared on entry).
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
    spilled: &mut std::collections::HashSet<chunk::ChunkKey>,
    job_budget: usize,
) {
    let _g3 = info_span!("emit_phase3_apply").entered();
    spilled.clear();
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
                    if spill {
                        // Deferred to a later frame. If the brick is currently RESIDENT with a stale
                        // hash (a re-bake — its content changed) it holds STALE texels; evict it so the
                        // lookup misses and falls back to the correct coarser LOD until its real bake
                        // lands ("old surface band left behind while dragging"). A genuinely new brick
                        // isn't resident, so this is a no-op — it simply never appears until its chunk
                        // bakes whole.
                        atlas.remove_brick(&key, config);
                        continue;
                    }
                    let tile = atlas.insert_gpu_brick(key, palette, hash, config);
                    push_bake_job(gpu_bakes, edits_snapshot, config, key, tile, &indices, palette);
                    if baked_dbg.enabled {
                        let bw = config.brick_world_size(key.lod);
                        let center = config.brick_min_world(key.coord, key.lod) + Vec3::splat(0.5 * bw);
                        baked_dbg.bricks.push((center, bw, now_secs));
                    }
                }
            }
        }
        if spill {
            spilled.insert(ck);
        }
        i = j;
    }
}

/// Synchronous bake emit: drain `pending`, gather candidates (evicting empties), classify INLINE,
/// and apply. The production path is [`dispatch_bake`] (sync for small bakes, async for large);
/// this fully-synchronous variant is the deterministic test/settle harness used by the unit tests,
/// sharing the same `gather_candidates` / `classify_candidates` / `apply_verdicts` building blocks.
#[cfg(test)]
fn emit_gpu_bakes(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    camera_pos: Vec3,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
) {
    let _span = info_span!("sdf_emit_gpu_bakes").entered();
    let edits_snapshot = Arc::clone(&sched.edits);
    let bvh_snapshot = Arc::clone(&sched.bvh);

    // Take the reusable scratch out of the scheduler so `sched` is free for `pending` below;
    // restored at the end (keeps its capacity across frames → zero growth realloc while dragging).
    let mut scratch = std::mem::take(&mut sched.emit_scratch);
    scratch.drained.clear();
    scratch.candidates.clear();
    scratch.spilled.clear();
    scratch.drained.extend(sched.pending.drain());
    sort_drained(&mut scratch.drained, config, camera_pos);

    gather_candidates(
        &scratch.drained,
        &bvh_snapshot,
        config,
        &mut scratch.candidates,
        |key| {
            atlas.remove_brick(&key, config);
        },
    );

    let hash_peek = snapshot_hash_peek(atlas, &scratch.candidates);
    let verdicts = classify_candidates(&scratch.candidates, &edits_snapshot, &bvh_snapshot, config, &hash_peek);
    apply_verdicts(
        atlas,
        gpu_bakes,
        &edits_snapshot,
        config,
        baked_dbg,
        now_secs,
        &scratch.candidates,
        verdicts,
        &mut scratch.spilled,
        SOFT_BAKE_BUDGET,
    );

    // Re-queue spilled chunks for the next frame(s). Their evictions already happened above;
    // only their deferred bakes retry. The atlas grows naturally as the spill drains, and the
    // render world preserves existing texels across the grow (see `prepare_sdf_atlas_gpu`), so
    // no re-emit of the already-baked set is needed.
    for &ck in scratch.spilled.iter() {
        sched.pending.insert(ck);
    }

    // Return the scratch (with its grown capacity) to the scheduler for next frame's reuse.
    sched.emit_scratch = scratch;
}

/// Apply a classify result (from either the sync path or a landed async task) to the atlas + job
/// list, then re-queue spilled chunks. Shared tail of both bake paths. `spilled` reuses the
/// scheduler's scratch set.
#[expect(clippy::too_many_arguments)]
fn finish_bake_apply(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
    verdicts: Vec<Verdict>,
) {
    let edits_snapshot = Arc::clone(&sched.edits);
    let mut spilled = std::mem::take(&mut sched.emit_scratch.spilled);
    // Apply at the SOFT budget even though the classify ran off-thread. The apply itself
    // (atlas inserts + tile alloc) AND the downstream render `extract_sdf` (rebuilds the O(bricks)
    // chunk lookup tables on a topology change) are BOTH main-thread and scale with jobs/frame —
    // dumping a full 8192-brick shell in one apply just moves the hitch from classify to
    // apply+extract (measured ~14 ms apply + ~20 ms extract). The budget bounds ALL per-frame
    // main-thread bake work; the shell still drains over a few frames (coarse LOD covers the gap).
    apply_verdicts(
        atlas, gpu_bakes, &edits_snapshot, config, baked_dbg, now_secs, candidates, verdicts, &mut spilled, SOFT_BAKE_BUDGET,
    );
    for &ck in spilled.iter() {
        sched.pending.insert(ck);
    }
    sched.emit_scratch.spilled = spilled;
}

/// Reconcile a landed async classify result against the CURRENT scheduler state, dropping verdicts
/// that went stale while the task ran, then apply the survivors. Two staleness cases:
///   1. `out.edit_gen != sched.edit_gen` — the edit set changed mid-flight, so EVERY verdict is
///      computed from stale geometry. Drop them all and re-queue every candidate chunk to `pending`
///      (the next bake re-classifies against the new edits). Safe + simple; snaps rarely coincide
///      with an edit change.
///   2. Otherwise, per candidate: if its chunk is no longer in its LOD's CURRENT window (the camera
///      moved between spawn and apply; the recenter already advanced the origin + evicted it), drop
///      its verdict — inserting it would collide with the in-window chunk sharing its `c mod R`
///      toroidal-directory slot.
fn apply_async_result(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
    out: BakeTaskOutput,
) {
    let r = config.ring_chunks_per_axis();
    if out.edit_gen != sched.edit_gen {
        // Whole result stale — re-queue every candidate's chunk (deduped) and bail.
        for (ck, _key) in &out.candidates {
            sched.pending.insert(*ck);
        }
        return;
    }
    // Filter candidates whose chunk is no longer in its LOD's CURRENT window — the camera may have
    // moved between when this task was spawned and now, and the recenter (step 2, this frame) has
    // already advanced `ring_chunk_origin` and evicted the exited chunks. Inserting an out-of-window
    // chunk would collide with the in-window chunk that shares its toroidal-directory slot
    // (`dir_index` is `c mod R`), so we MUST check the current origin, not the snapshot's. The
    // verdicts vector is parallel to candidates, so filter both together.
    let mut kept_candidates: Vec<(chunk::ChunkKey, atlas::BrickKey)> = Vec::with_capacity(out.candidates.len());
    let mut kept_verdicts: Vec<Verdict> = Vec::with_capacity(out.verdicts.len());
    for ((ck, key), verdict) in out.candidates.into_iter().zip(out.verdicts) {
        let li = ck.lod as usize;
        let origin = sched.ring_chunk_origin.get(li).copied().unwrap_or(IVec3::splat(i32::MIN));
        if chunk_in_window(ck.coord, origin, r) {
            kept_candidates.push((ck, key));
            kept_verdicts.push(verdict);
        }
        // else: chunk exited; recenter already evicted its bricks — drop, don't re-queue.
    }
    finish_bake_apply(atlas, sched, gpu_bakes, config, baked_dbg, now_secs, &kept_candidates, kept_verdicts);
}

/// Bake dispatch: poll any in-flight async classify (apply it if ready), then drain this frame's
/// `pending` into candidates and EITHER classify+apply synchronously (small bake — instant) OR
/// spawn a background classify task (large bake — offloaded so the main thread doesn't hitch).
/// Single-flight: only one async task at a time; while it runs, new dirt accumulates in `pending`.
#[expect(clippy::too_many_arguments)]
fn dispatch_bake(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    bake_task: &mut BakeTaskState,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    camera_pos: Vec3,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
) {
    // 1. Poll the in-flight task. If it finished, reconcile + apply its result this frame (before
    //    the render-world Extract). If still running, leave it and skip spawning a new one.
    let mut task_in_flight = false;
    if let Some(task) = bake_task.task.as_mut() {
        match block_on(poll_once(task)) {
            Some(out) => {
                bake_task.task = None;
                apply_async_result(atlas, sched, gpu_bakes, config, baked_dbg, now_secs, out);
            }
            None => task_in_flight = true,
        }
    }

    // 2. Gather this frame's candidates from pending (sync — includes the same-frame empty-chunk
    //    eviction, which must never lag the drag). Done even while a task is in flight, so empties
    //    still evict; but we only CLASSIFY when no task is running (single-flight).
    let bvh_snapshot = Arc::clone(&sched.bvh);
    let mut scratch = std::mem::take(&mut sched.emit_scratch);
    scratch.drained.clear();
    scratch.candidates.clear();
    if task_in_flight {
        // A task owns the bake this cycle. Don't drain pending (let it accumulate for the next
        // spawn), but DO evict empties for any chunks already dirtied — no: empties only matter for
        // chunks we'd otherwise bake. Leave pending intact; restore scratch and return.
        sched.emit_scratch = scratch;
        return;
    }
    scratch.drained.extend(sched.pending.drain());
    sort_drained(&mut scratch.drained, config, camera_pos);
    gather_candidates(&scratch.drained, &bvh_snapshot, config, &mut scratch.candidates, |key| {
        atlas.remove_brick(&key, config);
    });

    let candidate_count = scratch.candidates.len();
    if candidate_count <= ASYNC_BAKE_THRESHOLD {
        // SMALL bake → classify + apply synchronously this frame (instant, snappy).
        let edits_snapshot = Arc::clone(&sched.edits);
        let hash_peek = snapshot_hash_peek(atlas, &scratch.candidates);
        let verdicts = classify_candidates(&scratch.candidates, &edits_snapshot, &bvh_snapshot, config, &hash_peek);
        let mut spilled = std::mem::take(&mut scratch.spilled);
        spilled.clear();
        apply_verdicts(
            atlas, gpu_bakes, &edits_snapshot, config, baked_dbg, now_secs, &scratch.candidates, verdicts, &mut spilled, SOFT_BAKE_BUDGET,
        );
        for &ck in spilled.iter() {
            sched.pending.insert(ck);
        }
        scratch.spilled = spilled;
        sched.emit_scratch = scratch;
        return;
    }

    // LARGE bake → offload the classify to a background task. Snapshot everything the read-only
    // classify needs (Arc edits/bvh, owned candidates + hash_peek + config), stamp the snapshot's
    // identity (edit_gen + ring origins) for staleness reconciliation, and spawn single-flight.
    let edits_snapshot = Arc::clone(&sched.edits);
    let hash_peek = snapshot_hash_peek(atlas, &scratch.candidates);
    let candidates = std::mem::take(&mut scratch.candidates);
    let config_snapshot = config.clone();
    let edit_gen = sched.edit_gen;
    sched.emit_scratch = scratch;

    let pool = AsyncComputeTaskPool::get();
    bake_task.task = Some(pool.spawn(async move {
        // SERIAL classify — nesting the ComputeTaskPool scope inside this async task would deadlock.
        let verdicts = classify_candidates_serial(&candidates, &edits_snapshot, &bvh_snapshot, &config_snapshot, &hash_peek);
        BakeTaskOutput { candidates, verdicts, edit_gen }
    }));
}

#[cfg(test)]
mod tests;
