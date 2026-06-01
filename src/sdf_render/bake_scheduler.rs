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
    /// Per-LOD ring origins the snapshot was built from — a candidate whose chunk is no longer in
    /// its LOD's current window has exited and must be dropped (already evicted by the recenter).
    ring_origins: Vec<IVec3>,
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

/// The chunk coord (per axis) of the chunk-ring window corner for `camera_pos` at `lod`:
/// the camera's chunk minus half the ring (in chunks) on each axis, so the ring is
/// centred on the camera. `ring_bricks / CHUNK_BRICKS` chunks per axis.
///
/// `pub` so the GPU rig (`tests/sdf_gpu_rig.rs`) can assert the shader's `in_ring_chunk`
/// agrees with THIS source-of-truth window — they're hand-duplicated across Rust/WGSL, and
/// a silent divergence would make the chunk-DDA skip step past real geometry.
pub fn ring_chunk_origin(config: &SdfGridConfig, camera_pos: Vec3, lod: u32) -> IVec3 {
    let s = config.cell_stride();
    let cam_brick = config.world_to_brick_lod(camera_pos, lod);
    let cam_brick_idx = IVec3::new(
        cam_brick.x.div_euclid(s),
        cam_brick.y.div_euclid(s),
        cam_brick.z.div_euclid(s),
    );
    let cam_chunk = IVec3::new(
        cam_brick_idx.x.div_euclid(chunk::CHUNK_BRICKS),
        cam_brick_idx.y.div_euclid(chunk::CHUNK_BRICKS),
        cam_brick_idx.z.div_euclid(chunk::CHUNK_BRICKS),
    );
    // Hysteresis: snap the camera chunk to the coarse `recenter_snap_chunks` lattice so
    // the window only recenters on discrete jumps, not every chunk crossing. `div_euclid`
    // floors toward -inf so the snapped lattice is continuous across the world origin.
    let snap = config.recenter_snap_chunks.max(1);
    let cam_chunk_snapped = IVec3::new(
        cam_chunk.x.div_euclid(snap) * snap,
        cam_chunk.y.div_euclid(snap) * snap,
        cam_chunk.z.div_euclid(snap) * snap,
    );
    let half = (config.ring_bricks / chunk::CHUNK_BRICKS as u32 / 2) as i32;
    cam_chunk_snapped - IVec3::splat(half)
}

/// Chunks per axis in a ring window.
fn ring_chunks_per_axis(config: &SdfGridConfig) -> i32 {
    (config.ring_bricks / chunk::CHUNK_BRICKS as u32) as i32
}

/// Whether any edit reaches chunk `ck` (its world AABB overlaps an edit in the BVH). A
/// chunk's world AABB is exactly the union of its `CHUNK_BRICKS³` brick AABBs, so a BVH miss
/// here guarantees *every* brick in the chunk would bake to empty space. Used to skip
/// enqueuing empty entered chunks: they would consume budget producing nothing, starving the
/// real geometry entering far (coarse-LOD) rings — the cause of LOD never refreshing when
/// flying away from the scene. Safe only for camera-*entered* chunks (no resident bricks to
/// evict yet); the edit-dirty path must still enqueue emptied chunks so vacated bricks get
/// removed.
/// Whether any edit reaches chunk `ck`, reusing a caller-owned BVH traversal `stack` (cleared on
/// entry) so a recenter that runs thousands of these per snap frame does zero heap allocation per
/// query. Uses the BVH's EARLY-EXIT `any_overlap_with`: this only needs an occupancy boolean, so
/// stopping at the first overlapping leaf (instead of collecting every edit a dense chunk overlaps —
/// often hundreds, all discarded) roughly halves the per-query cost on tower-dense chunks.
fn chunk_has_geometry_with(
    ck: chunk::ChunkKey,
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    stack: &mut Vec<u32>,
) -> bool {
    let size = chunk::chunk_world_size(ck.lod, config);
    let min = chunk::chunk_min_world(ck, config);
    let aabb = bevy::math::bounding::Aabb3d::from_min_max(min, min + Vec3::splat(size));
    bvh.any_overlap_with(&aabb, stack)
}

/// Whether chunk coord `c` is inside the `R³` chunk window with corner `origin`.
fn chunk_in_window(c: IVec3, origin: IVec3, r: i32) -> bool {
    let rel = c - origin;
    rel.x >= 0 && rel.y >= 0 && rel.z >= 0 && rel.x < r && rel.y < r && rel.z < r
}


/// Every chunk key in the `R³` window with corner `origin` at `lod`.
fn chunk_window_keys(origin: IVec3, r: i32, lod: u32) -> impl Iterator<Item = chunk::ChunkKey> {
    (0..r).flat_map(move |iz| {
        (0..r).flat_map(move |iy| {
            (0..r).map(move |ix| chunk::ChunkKey::new(lod, origin + IVec3::new(ix, iy, iz)))
        })
    })
}

/// Invoke `f` only for the chunk coords in the `R³` window at `new_origin` that are NOT in the
/// `R³` window at `old_origin` (the chunks that *entered* on a recenter). For two equal-size
/// axis-aligned windows the difference is a thin boundary slab, so this visits only the new shell
/// (a few hundred chunks) instead of the full `R³` interior (4096 at R=16) — the recenter only
/// cares about entered chunks, and the overlap is unchanged. Yields nothing when the windows are
/// identical; yields the whole new window when they don't overlap (a teleport). Each coord is
/// visited at most once (axis-partitioned: x-slabs, then the x-overlap's y-slabs, then the xy-
/// overlap's z-slabs), so no dedup is needed.
fn for_each_entered_chunk(new_origin: IVec3, old_origin: IVec3, r: i32, mut f: impl FnMut(IVec3)) {
    // Overlap box of the two windows on each axis (empty if they don't overlap). Use saturating
    // arithmetic so a sentinel old_origin (i32::MIN on the first run) can't overflow `+ r`; it just
    // yields an empty overlap → the whole new window is "entered", which is the correct first-run
    // behaviour (and `for_each_exited_chunk` then evicts nothing, since the old window is degenerate).
    let new_end = IVec3::new(
        new_origin.x.saturating_add(r),
        new_origin.y.saturating_add(r),
        new_origin.z.saturating_add(r),
    );
    let old_end = IVec3::new(
        old_origin.x.saturating_add(r),
        old_origin.y.saturating_add(r),
        old_origin.z.saturating_add(r),
    );
    let ov_min = new_origin.max(old_origin);
    let ov_max = new_end.min(old_end);
    // x in new-window but outside the x-overlap → whole yz cross-section entered.
    let x_overlap_empty = ov_min.x >= ov_max.x;
    for x in new_origin.x..new_end.x {
        let x_entered = x_overlap_empty || x < ov_min.x || x >= ov_max.x;
        if x_entered {
            // Entire yz face at this x is new.
            for y in new_origin.y..new_origin.y + r {
                for z in new_origin.z..new_origin.z + r {
                    f(IVec3::new(x, y, z));
                }
            }
        } else {
            // x is shared; partition y the same way, then z within the xy-overlap.
            let y_overlap_empty = ov_min.y >= ov_max.y;
            for y in new_origin.y..new_origin.y + r {
                let y_entered = y_overlap_empty || y < ov_min.y || y >= ov_max.y;
                if y_entered {
                    for z in new_origin.z..new_origin.z + r {
                        f(IVec3::new(x, y, z));
                    }
                } else {
                    // x,y shared; only the z-slab outside the z-overlap entered.
                    let z_overlap_empty = ov_min.z >= ov_max.z;
                    for z in new_origin.z..new_origin.z + r {
                        if z_overlap_empty || z < ov_min.z || z >= ov_max.z {
                            f(IVec3::new(x, y, z));
                        }
                    }
                }
            }
        }
    }
}

/// Invoke `f` for the chunk coords that *exited* on a recenter — in the OLD window but not the
/// new. Symmetric to [`for_each_entered_chunk`] (args swapped), used to evict the trailing shell.
fn for_each_exited_chunk(new_origin: IVec3, old_origin: IVec3, r: i32, f: impl FnMut(IVec3)) {
    for_each_entered_chunk(old_origin, new_origin, r, f);
}

/// All brick keys belonging to chunk `ck` (its `CHUNK_BRICKS³` local slots).
#[cfg(test)]
fn chunk_brick_keys(ck: chunk::ChunkKey, config: &SdfGridConfig) -> Vec<atlas::BrickKey> {
    let mut keys = Vec::with_capacity(chunk::CHUNK_VOLUME as usize);
    for_each_brick_key(ck, config, |k| keys.push(k));
    keys
}

/// Allocation-free counterpart of [`chunk_brick_keys`]: invoke `f` for each of a chunk's 64
/// brick keys without building a Vec. The bake emit's serial gather/apply loops run this over
/// the entire dirty set (thousands of chunks on a terrain-scale heightmap move), so avoiding a
/// per-chunk 64-element heap alloc there is a measurable win (emit phases 1+3 were ~20ms spikes).
#[inline]
fn for_each_brick_key(ck: chunk::ChunkKey, config: &SdfGridConfig, mut f: impl FnMut(atlas::BrickKey)) {
    let s = config.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let base = ck.coord * c; // brick-index space
    for lz in 0..c {
        for ly in 0..c {
            for lx in 0..c {
                let bi = base + IVec3::new(lx, ly, lz);
                f(atlas::BrickKey::new(ck.lod, bi * s)); // back to coord space
            }
        }
    }
}

/// The chunks at `lod` whose world extent overlaps `aabb` (grown by the bake footprint
/// pad so a moved edit re-dirties every chunk that could fold it), **clamped to the resident
/// ring window** `[win_origin, win_origin + r)`. Computed directly in chunk-coord space — no
/// per-brick enumeration.
///
/// The clamp is essential for terrain-scale edits: a heightmap's AABB spans the whole world
/// in XZ, so the unclamped chunk range is millions of chunks. The caller only ever keeps the
/// in-window ones anyway, so intersecting the loop bounds with the window up front makes the
/// work O(AABB ∩ window) ≤ r³ instead of O(AABB volume) — the fix for the multi-hundred-ms
/// `schedule_bakes` freeze when a heightmap edit changes (it used to allocate + enumerate the
/// entire terrain-sized chunk volume per LOD, then discard 99.99% via `chunk_in_window`).
fn chunks_in_aabb_windowed(
    config: &SdfGridConfig,
    aabb: &bevy::math::bounding::Aabb3d,
    lod: u32,
    win_origin: IVec3,
    r: i32,
) -> Vec<chunk::ChunkKey> {
    let chunk_world = chunk::chunk_world_size(lod, config);
    let pad = Vec3::splat(atlas::SNORM_CLAMP_DIST + config.brick_world_size(lod));
    let lo = (Vec3::from(aabb.min) - pad) / chunk_world;
    let hi = (Vec3::from(aabb.max) + pad) / chunk_world;
    // Intersect the AABB chunk-range with the window box BEFORE enumerating.
    let lo = IVec3::new(lo.x.floor() as i32, lo.y.floor() as i32, lo.z.floor() as i32)
        .max(win_origin);
    let hi = IVec3::new(hi.x.ceil() as i32, hi.y.ceil() as i32, hi.z.ceil() as i32)
        .min(win_origin + IVec3::splat(r - 1));

    let mut out = Vec::new();
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                out.push(chunk::ChunkKey::new(lod, IVec3::new(x, y, z)));
            }
        }
    }
    out
}

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
    let r = ring_chunks_per_axis(&config);
    let first_run = sched.ring_chunk_origin.is_empty();
    if first_run {
        sched.ring_chunk_origin = vec![IVec3::splat(i32::MIN); lod_count as usize];
    }

    // --- 1. Edit changes → dirty affected chunks (within current windows) ------------
    let gathered = gather_sorted_edits(&volumes);
    let current: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d> =
        gathered.iter().map(|g| (g.entity, g.aabb)).collect();
    let set_changed = current.len() != prev_aabbs.map.len()
        || current.keys().any(|e| !prev_aabbs.map.contains_key(e));
    let edits_changed = atlas.rebake_all || set_changed || !changed.is_empty();

    if edits_changed {
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

    // --- 2. Camera chunk-ring recenter (eager enter/evict, absolute addressing) ------
    // `atlas.remove_brick` bumps the upload + topology generations itself, so an evict-only
    // frame (e.g. flying away from the scene) still makes the render world re-extract and
    // drop the stale bricks — it doesn't depend on a bake being applied that frame.
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
        // Exited chunks → drop all their bricks (and cancel any pending bake). Skipped on the first
        // run: there is no prior window to evict (the sentinel origin isn't a real chunk region).
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

    // --- 3. Emit GPU bake jobs for the dirty chunks ----------------------------------
    // The CPU does only topology (BVH cull + palette + tile alloc) and emits a GpuBakeJob per
    // brick; the compute shader fills the texels. Small bakes run synchronously; a large shell
    // (coarse-LOD snap) offloads its classify to a background task (see `dispatch_bake`).
    dispatch_bake(
        &mut atlas, &mut sched, &mut bake_task, &mut gpu_bakes, &config, camera_pos, &mut baked_dbg, now,
    );

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

/// Narrow-band cull decision for one candidate brick: KEEP iff the folded isosurface can pass
/// through it. Drops the deep INTERIOR of a solid (`|d| ≫ 0`) and the FAR-EXTERIOR corners the
/// coarse AABB query leaves in — the r³ bulk the march never samples (rays approach from
/// outside, read only the thin zero-band, and stop). `indices` are the brick's BVH-culled edit
/// indices into `edits`. Pure + allocation-free; unit-tested directly.
///
/// SMOOTHING SAFETY: a plain center test `|fold(center)| > circumradius + band` assumes the
/// folded field is 1-Lipschitz — true ONLY at smoothing 0. iq polynomial `smin`/`smax` have a
/// correction term `k·h(1−h)` that peaks at `k/4`, so the smoothed field can sit up to `Σ kᵢ/4`
/// (additive down the fold chain) NEARER the true surface than `fold_hard` — the bound that
/// kept the cull safe. Without padding for it, a brick at a SMOOTHED subtract-carve crease whose
/// center reads "far" but whose corner actually crosses zero gets wrongly dropped → a hole at
/// the rim (the reported bug). We pad `reach` by `Σ kᵢ/4`, restoring the conservative bound.
///
/// FORCE-KEEP on sign change: as a belt-and-suspenders that can only ever KEEP (never drop, so
/// it cannot add a hole), if the folded field changes sign across the brick's 9 palette sample
/// points (8 corners + center) the surface provably crosses the brick → keep regardless of the
/// center distance. This also catches an enclosed cavity whose surface the center eval misses.
fn narrow_band_keep(
    edits: &[edits::ResolvedEdit],
    indices: &[u32],
    config: &SdfGridConfig,
    key: atlas::BrickKey,
) -> bool {
    let brick_world = config.brick_world_size(key.lod);
    let brick_min = config.brick_min_world(key.coord, key.lod);
    let center = brick_min + Vec3::splat(0.5 * brick_world);

    // Smoothing pad: additive Σ(kᵢ)/4 over the brick's smoothed candidate edits.
    let smooth_sum: f32 = indices
        .iter()
        .map(|&i| edits[i as usize].op.smoothing.max(0.0))
        .sum();
    // CONSERVATIVE reach: circumradius + the LOD's distance band + smoothing pad. The
    // `dist_band` term is a deliberate safety margin — it only ever makes us KEEP a brick the
    // center test would otherwise drop, never the reverse, so it cannot introduce a hole. (It is
    // largely subsumed by `cull_edit_indices` already culling on the raw brick AABB, but is kept
    // because proving it strictly redundant at coarse-LOD corners is fragile and the cost is one
    // add. The over-keep halo it allows is a one-time first-bake cost, not a per-frame drag cost.)
    let reach = brick_world * (0.5 * 3.0_f32.sqrt())
        + atlas::dist_band_world(config, key.lod)
        + 0.25 * smooth_sum;

    // Force-keep if the surface provably crosses the brick (sign change over the 9 samples).
    // Only meaningful when smoothing inflates the gradient; the common k=0 path stays 1 eval.
    if smooth_sum > 0.0 {
        let voxel_size = config.voxel_size_at(key.lod);
        let samples = atlas::SdfAtlas::brick_palette_samples(key, voxel_size);
        let mut neg = false;
        let mut pos = false;
        for p in samples {
            if edits::fold_csg_dist_indexed(edits, indices, p) <= 0.0 {
                neg = true;
            } else {
                pos = true;
            }
            if neg && pos {
                return true;
            }
        }
    }

    edits::fold_csg_dist_indexed(edits, indices, center).abs() <= reach
}

/// Turn this frame's dirty chunks into [`GpuBakeJob`]s: the CPU does only the topology work
/// (BVH cull → which bricks exist + their palette + a stable tile), and the compute shader
/// fills each brick's 512 texels straight into the atlas. No main-thread voxel loop.
/// One candidate brick's classification, produced by [`classify_candidates`] (read-only, can run
/// on a background task) and consumed by [`apply_verdicts`] (mutates the atlas, main thread only).
pub(crate) enum Verdict {
    /// No edit reaches it → evict.
    Empty,
    /// Narrow-band cull → evict.
    Drop,
    /// Resident with matching content hash → leave its texels as-is.
    Skip,
    /// Bake: palette + culled edit indices + content hash.
    Keep(edits::Palette, Vec<u32>, u64),
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

/// Classify ONE chunk's slice of candidate bricks into `Verdict`s (read-only). The shared core of
/// both the parallel (sync) and serial (async task) classify paths. `scratch`/`stack`/`hash_memo`
/// are caller-owned reusable buffers (the memo lets an edit-set folded by many bricks hash once per
/// unique set). Reads only the edit/BVH snapshots, config, and the `hash_peek` resident-hash
/// snapshot — no atlas borrow — so it is `Send` and safe on a background task.
#[expect(clippy::too_many_arguments)]
fn classify_chunk(
    chunk: &[(chunk::ChunkKey, atlas::BrickKey)],
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    hash_peek: &std::collections::HashMap<atlas::BrickKey, u64>,
    scratch: &mut Vec<u32>,
    stack: &mut Vec<u32>,
    hash_memo: &mut std::collections::HashMap<Box<[u32]>, u64>,
    out: &mut Vec<Verdict>,
) {
    for &(_ck, key) in chunk {
        if atlas::SdfAtlas::cull_edit_indices_with(key, bvh, config, scratch, stack).is_none() {
            out.push(Verdict::Empty);
            continue;
        }
        if !narrow_band_keep(edits, scratch, config, key) {
            out.push(Verdict::Drop);
            continue;
        }
        // Content hash of exactly the edits this brick folds — its bake-cache key. Memoised per
        // unique culled index-set (the costly part is the same for identical sets).
        let hash = *hash_memo
            .entry(scratch.clone().into_boxed_slice())
            .or_insert_with(|| edits::bake_content_hash(edits, scratch));
        // HASH-PEEK EARLY-OUT: a resident brick with the same content keeps valid texels — skip
        // BEFORE the (dominant-cost) palette build. This is what makes a sphere dragged over the
        // heightmap cheap. The peek reads a SNAPSHOT of resident hashes (no atlas borrow); a stale
        // snapshot is harmless — a wrong Skip re-bakes identically next frame, a wrong Keep is
        // overwritten by `insert_gpu_brick`'s authoritative hash.
        if hash_peek.get(&key).is_some_and(|&h| h == hash) {
            out.push(Verdict::Skip);
            continue;
        }
        let voxel_size = config.voxel_size_at(key.lod);
        let samples = atlas::SdfAtlas::brick_palette_samples(key, voxel_size);
        let palette = edits::build_palette_indexed(edits, scratch, &samples);
        out.push(Verdict::Keep(palette, scratch.clone(), hash));
    }
}

/// Phase 2 (PARALLEL, READ-ONLY): classify every candidate via `par_chunk_map` across the compute
/// pool. The SYNCHRONOUS bake path. Reads only snapshots — see [`classify_chunk`].
pub(crate) fn classify_candidates(
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    hash_peek: &std::collections::HashMap<atlas::BrickKey, u64>,
) -> Vec<Verdict> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let _g = info_span!("emit_phase2_classify", candidates = candidates.len()).entered();
    let classify = |_idx: usize, chunk: &[(chunk::ChunkKey, atlas::BrickKey)]| -> Vec<Verdict> {
        let mut scratch: Vec<u32> = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        let mut hash_memo: std::collections::HashMap<Box<[u32]>, u64> = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(chunk.len());
        classify_chunk(chunk, edits, bvh, config, hash_peek, &mut scratch, &mut stack, &mut hash_memo, &mut out);
        out
    };
    // `get_or_init` so headless tests (which don't boot the full app that sets up the pool) still
    // run; in production the pool already exists.
    let pool = ComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
    let chunk_size = candidates.len().div_ceil(pool.thread_num().max(1)).max(1);
    candidates
        .par_chunk_map(pool, chunk_size, classify)
        .into_iter()
        .flatten()
        .collect()
}

/// SERIAL classify of all candidates — for the background async task, where nesting the
/// `ComputeTaskPool` scope (as `classify_candidates` does) inside an `AsyncComputeTaskPool` task
/// would deadlock. Single-threaded is fine off the main thread: the whole point is to not block the
/// frame, and one background thread chewing through the shell over a few frames is exactly the goal.
pub(crate) fn classify_candidates_serial(
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
    edits: &[edits::ResolvedEdit],
    bvh: &bvh::Bvh,
    config: &SdfGridConfig,
    hash_peek: &std::collections::HashMap<atlas::BrickKey, u64>,
) -> Vec<Verdict> {
    let mut scratch: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    let mut hash_memo: std::collections::HashMap<Box<[u32]>, u64> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(candidates.len());
    classify_chunk(candidates, edits, bvh, config, hash_peek, &mut scratch, &mut stack, &mut hash_memo, &mut out);
    out
}

/// Phase 3 (serial, MAIN THREAD): apply each candidate's verdict — evict, skip, or insert + push a
/// GPU bake job under the soft budget. Mutates the atlas + job list, so it must run on the main
/// thread before the render-world Extract. A Keep over [`SOFT_BAKE_BUDGET`] spills its chunk back to
/// `sched.pending` (NOT inserted → stays non-resident → coarse-LOD fallback, hole-free) and evicts
/// any stale resident texels. `spilled` is a reusable set (cleared on entry).
#[expect(clippy::too_many_arguments)]
fn apply_verdicts(
    atlas: &mut SdfAtlas,
    gpu_bakes: &mut PendingGpuBakes,
    edits_snapshot: &[edits::ResolvedEdit],
    config: &SdfGridConfig,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
    verdicts: Vec<Verdict>,
    spilled: &mut std::collections::HashSet<chunk::ChunkKey>,
    job_budget: usize,
) {
    let _g3 = info_span!("emit_phase3_apply").entered();
    spilled.clear();
    for ((ck, key), verdict) in candidates.iter().zip(verdicts) {
        match verdict {
            Verdict::Empty | Verdict::Drop => {
                atlas.remove_brick(key, config);
            }
            // Resident brick, content unchanged (hash matched in classify) → texels still valid,
            // leave it as-is. This is what keeps a sphere dragged over the heightmap cheap.
            Verdict::Skip => {}
            Verdict::Keep(palette, indices, hash) => {
                if gpu_bakes.jobs.len() >= job_budget {
                    // Over this frame's job budget → defer this brick's bake to a later frame, so a
                    // big shell spreads instead of spiking. The SYNC path uses the low SOFT budget
                    // (the classify is on the main thread, so keep per-frame work small); the ASYNC
                    // apply uses the full `GPU_BAKE_JOB_CAP` (its classify already ran off-thread, so
                    // the only per-frame cost left is the cheap HashMap/Vec apply — drain faster).
                    // Same spill mechanics either way. CRUCIAL: if the brick is currently RESIDENT
                    // with a DIFFERENT hash it holds STALE texels (its content changed) — leaving it
                    // resident would serve its stale shape over the freshly-baked coarse level (the
                    // "old surface band left behind while dragging"). Evict it so the lookup misses
                    // and falls back to the correct coarse LOD until its real bake lands.
                    atlas.remove_brick(key, config);
                    spilled.insert(*ck);
                    continue;
                }
                let tile = atlas.insert_gpu_brick(*key, palette, hash, config);
                push_bake_job(gpu_bakes, edits_snapshot, config, *key, tile, &indices, palette);
                if baked_dbg.enabled {
                    let bw = config.brick_world_size(key.lod);
                    let center = config.brick_min_world(key.coord, key.lod) + Vec3::splat(0.5 * bw);
                    baked_dbg.bricks.push((center, bw, now_secs));
                }
            }
        }
    }
    atlas.last_bake_was_full = false;
}

/// Build the `hash_peek` snapshot: each candidate's resident `baked_hash` (absent → not resident).
/// Lets [`classify_candidates`] do the content-hash skip without borrowing the atlas, so it can run
/// on a background task.
fn snapshot_hash_peek(
    atlas: &SdfAtlas,
    candidates: &[(chunk::ChunkKey, atlas::BrickKey)],
) -> std::collections::HashMap<atlas::BrickKey, u64> {
    let mut map = std::collections::HashMap::with_capacity(candidates.len());
    for &(_ck, key) in candidates {
        if let Some(b) = atlas.bricks.get(&key) {
            map.insert(key, b.baked_hash);
        }
    }
    map
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
///   2. Otherwise, per candidate: if its chunk has exited its LOD's window since the snapshot
///      (compared against the snapshot's `ring_origins`, which still match because edits didn't
///      change the window), the recenter already evicted it — drop its verdict, don't re-queue.
fn apply_async_result(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    baked_dbg: &mut super::BakedBrickDebug,
    now_secs: f32,
    out: BakeTaskOutput,
) {
    let r = ring_chunks_per_axis(config);
    if out.edit_gen != sched.edit_gen {
        // Whole result stale — re-queue every candidate's chunk (deduped) and bail.
        for (ck, _key) in &out.candidates {
            sched.pending.insert(*ck);
        }
        return;
    }
    // Filter candidates whose chunk has exited its LOD window since the snapshot. The verdicts
    // vector is parallel to candidates, so filter both together.
    let mut kept_candidates: Vec<(chunk::ChunkKey, atlas::BrickKey)> = Vec::with_capacity(out.candidates.len());
    let mut kept_verdicts: Vec<Verdict> = Vec::with_capacity(out.verdicts.len());
    for ((ck, key), verdict) in out.candidates.into_iter().zip(out.verdicts) {
        let li = ck.lod as usize;
        let origin = out.ring_origins.get(li).copied().unwrap_or(IVec3::splat(i32::MIN));
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
    let ring_origins = sched.ring_chunk_origin.clone();
    sched.emit_scratch = scratch;

    let pool = AsyncComputeTaskPool::get();
    bake_task.task = Some(pool.spawn(async move {
        // SERIAL classify — nesting the ComputeTaskPool scope inside this async task would deadlock.
        let verdicts = classify_candidates_serial(&candidates, &edits_snapshot, &bvh_snapshot, &config_snapshot, &hash_peek);
        BakeTaskOutput { candidates, verdicts, edit_gen, ring_origins }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::edits::{edit_world_aabb, CsgKind, ResolvedEdit};
    use bevy::math::bounding::Aabb3d;
    use std::collections::HashSet;

    fn config() -> SdfGridConfig {
        SdfGridConfig::default()
    }

    // --- GPU recenter convergence harness -------------------------------------------
    //
    // Drives the real recenter (step 2 of `schedule_bakes`) + `emit_gpu_bakes` directly on a
    // scheduler/atlas pair, so the resident-set convergence invariants are tested against the
    // exact production topology code (no ECS App needed). The GPU bake emits synchronously, so
    // there's no async lag to model — what's dirtied this frame is baked this frame.

    /// Mirror `schedule_bakes` step 2 (camera recenter): enqueue entered geometry chunks into
    /// `pending`, evict exited chunks eagerly. Returns the number of geometry chunks enqueued
    /// (for the fly-away starvation bound). `ring_chunk_origin` lives on the scheduler.
    fn recenter_step(sched: &mut BakeScheduler, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, cam: Vec3) -> usize {
        let r = ring_chunks_per_axis(cfg);
        if sched.ring_chunk_origin.is_empty() {
            sched.ring_chunk_origin = vec![IVec3::splat(i32::MIN); cfg.lod_count as usize];
        }
        let mut stack: Vec<u32> = Vec::new();
        let mut enqueued = 0usize;
        for lod in 0..cfg.lod_count {
            let li = lod as usize;
            let new_origin = ring_chunk_origin(cfg, cam, lod);
            let old_origin = sched.ring_chunk_origin[li];
            if new_origin == old_origin {
                continue;
            }
            let first = old_origin == IVec3::splat(i32::MIN);
            for_each_entered_chunk(new_origin, old_origin, r, |coord| {
                let ck = chunk::ChunkKey::new(lod, coord);
                if chunk_has_geometry_with(ck, &sched.bvh, cfg, &mut stack)
                    && sched.pending.insert(ck)
                {
                    enqueued += 1;
                }
            });
            if !first {
                for_each_exited_chunk(new_origin, old_origin, r, |coord| {
                    let ck = chunk::ChunkKey::new(lod, coord);
                    sched.pending.remove(&ck);
                    for bk in chunk_brick_keys(ck, cfg) {
                        atlas.remove_brick(&bk, cfg);
                    }
                });
            }
            sched.ring_chunk_origin[li] = new_origin;
        }
        enqueued
    }

    /// Recenter to `cam` and drain the GPU bake emission until idle (the cap may spill over
    /// several frames). Returns the resident chunk set — the GPU equivalent of a fresh settle.
    fn settle_gpu(sched: &mut BakeScheduler, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, cam: Vec3) -> HashSet<chunk::ChunkKey> {
        recenter_step(sched, atlas, cfg, cam);
        let mut gpu = PendingGpuBakes::default();
        let mut guard = 0;
        loop {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
            guard += 1;
            assert!(guard < 1000, "settle did not converge");
            if sched.pending.is_empty() {
                break;
            }
        }
        atlas.bricks.keys().map(|k| chunk::chunk_of(*k, cfg).0).collect()
    }

    fn build_bvh(edits: &[ResolvedEdit]) -> bvh::Bvh {
        let aabbs: Vec<Aabb3d> = edits
            .iter()
            .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
            .collect();
        bvh::Bvh::build(&aabbs)
    }

    fn box_edit(pos: Vec3, half: f32, mat: u16) -> ResolvedEdit {
        ResolvedEdit::new(
            SdfPrimitive::Box { half_extents: Vec3::splat(half) },
            Transform::from_translation(pos),
            SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
            mat,
        )
    }

    fn sphere_edit(pos: Vec3, radius: f32, mat: u16) -> ResolvedEdit {
        ResolvedEdit::new(
            SdfPrimitive::Sphere { radius },
            Transform::from_translation(pos),
            SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
            mat,
        )
    }

    fn subtract_sphere(pos: Vec3, radius: f32) -> ResolvedEdit {
        ResolvedEdit::new(
            SdfPrimitive::Sphere { radius },
            Transform::from_translation(pos),
            SdfOp { kind: CsgKind::Subtract, smoothing: 0.0 },
            0,
        )
    }

    /// Regression guard for the narrow-band cull on a SUBTRACTED (hollow / bitten) solid: the
    /// cull must not drop any brick the TRUE folded surface passes through, and every resident
    /// brick's per-brick CULLED candidate set must agree in SIGN with the full edit list at the
    /// brick corners (so the GPU bakes the carve, not solid). Covers both an enclosed cavity and
    /// an open bite. (Proven the cull is innocent of the interior-hole artefact — that lives in
    /// the GPU bake/march, not here.)
    #[test]
    fn cull_preserves_subtracted_surface_bricks() {
        for (r_in, off) in [(4.0_f32, 0.0_f32), (5.0, 10.0)] {
            let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
            let r_out = 10.0;
            let edits = vec![
                sphere_edit(Vec3::ZERO, r_out, 0),
                subtract_sphere(Vec3::new(off, 0.0, 0.0), r_in),
            ];
            let mut sched = primed_sched(&edits);
            let mut atlas = SdfAtlas::default();
            for x in -5..=5 {
                for y in -5..=5 {
                    for z in -5..=5 {
                        sched.pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                    }
                }
            }
            let mut gpu = PendingGpuBakes::default();
            let mut guard = 0;
            loop {
                gpu.clear();
                atlas.gpu_baked_tiles.clear();
                emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
                guard += 1;
                assert!(guard < 1000);
                if sched.pending.is_empty() { break; }
            }

            let all_edits = sched.edits.clone();
            let bw = cfg.brick_world_size(0);
            let mut scratch: Vec<u32> = Vec::new();
            let corner = |bmin: Vec3| {
                let mut cs = [Vec3::ZERO; 8];
                let mut i = 0;
                for cx in [0.0, bw] {
                    for cy in [0.0, bw] {
                        for cz in [0.0, bw] {
                            cs[i] = bmin + Vec3::new(cx, cy, cz);
                            i += 1;
                        }
                    }
                }
                cs
            };

            // (1) No surface-bearing brick dropped.
            for ck in chunk_window_keys(IVec3::splat(-5), 11, 0) {
                for key in chunk_brick_keys(ck, &cfg) {
                    if atlas::SdfAtlas::cull_edit_indices(key, &sched.bvh, &cfg, &mut scratch).is_none() {
                        continue;
                    }
                    let cs = corner(cfg.brick_min_world(key.coord, 0));
                    let (mut neg, mut pos) = (false, false);
                    for p in cs {
                        if edits::fold_csg(&all_edits, p).dist <= 0.0 { neg = true; } else { pos = true; }
                    }
                    if neg && pos {
                        assert!(
                            atlas.bricks.contains_key(&key),
                            "r_in={r_in} off={off}: dropped a brick the surface passes through at {:?}",
                            key.coord
                        );
                    }
                }
            }

            // (2) Per-brick culled candidate set agrees in sign with the full edit list.
            for key in atlas.bricks.keys() {
                if atlas::SdfAtlas::cull_edit_indices(*key, &sched.bvh, &cfg, &mut scratch).is_none() {
                    continue;
                }
                let culled: Vec<edits::ResolvedEdit> =
                    scratch.iter().map(|&i| all_edits[i as usize].clone()).collect();
                for p in corner(cfg.brick_min_world(key.coord, 0)) {
                    let d_full = edits::fold_csg(&all_edits, p).dist;
                    let d_cull = edits::fold_csg(&culled, p).dist;
                    assert_eq!(
                        d_full <= 0.0, d_cull <= 0.0,
                        "r_in={r_in} off={off}: culled-set sign mismatch at brick {:?} (full={d_full:.3} cull={d_cull:.3})",
                        key.coord
                    );
                }
            }
        }
    }

    /// A scheduler primed with `edits` + their BVH, as `schedule_bakes` would leave it after
    /// the edit-change step — so `emit_gpu_bakes` can be driven directly in a test.
    fn primed_sched(edits: &[ResolvedEdit]) -> BakeScheduler {
        BakeScheduler {
            edits: Arc::new(edits.to_vec()),
            bvh: Arc::new(build_bvh(edits)),
            ..BakeScheduler::default()
        }
    }

    /// Drive `dispatch_bake` (the production sync/async hybrid) frame-by-frame until the bake
    /// settles: pending empty AND no task in flight. Mirrors the per-frame `schedule_bakes` call so
    /// the async path (task spawn → poll → reconcile → apply, spread over frames) is exercised
    /// end-to-end with a real `AsyncComputeTaskPool`. Returns the resident chunk set.
    fn settle_dispatch(
        sched: &mut BakeScheduler,
        atlas: &mut SdfAtlas,
        bake_task: &mut BakeTaskState,
        cfg: &SdfGridConfig,
        cam: Vec3,
    ) -> HashSet<chunk::ChunkKey> {
        recenter_step(sched, atlas, cfg, cam);
        let mut gpu = PendingGpuBakes::default();
        let mut guard = 0;
        loop {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            dispatch_bake(atlas, sched, bake_task, &mut gpu, cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
            // Tiny yield so the background classify thread gets CPU between polls (in the real app
            // frames are milliseconds apart; this test loop would otherwise busy-spin the main thread
            // and starve the pool). Not needed in production.
            if bake_task.task.is_some() {
                std::thread::yield_now();
            }
            guard += 1;
            assert!(guard < 50000, "async dispatch settle did not converge");
            if sched.pending.is_empty() && bake_task.task.is_none() {
                break;
            }
        }
        atlas.bricks.keys().map(|k| chunk::chunk_of(*k, cfg).0).collect()
    }

    /// The async offload must converge to the SAME resident set as the synchronous settle. Uses a
    /// scene big enough to cross `ASYNC_BAKE_THRESHOLD` so the large-bake (task) path actually runs.
    /// Proves the snapshot/spawn/poll/reconcile/apply round-trip drops nothing and bakes everything.
    #[test]
    fn async_dispatch_converges_to_sync_settle() {
        // Headless tests don't boot TaskPoolPlugin, so init the async pool the dispatch needs.
        AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
        // Ring sized so the sphere's in-window shell modestly exceeds the async threshold (4096) —
        // enough to take the task path, small enough that the SERIAL background classify finishes
        // quickly under the test's busy-poll loop (a huge shell would serial-classify for seconds).
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 24, recenter_snap_chunks: 1, ..config() };
        let edits = vec![sphere_edit(Vec3::ZERO, 9.0, 0)];
        let cam = Vec3::ZERO;

        // Async path.
        let mut a_atlas = SdfAtlas::default();
        let mut a_sched = primed_sched(&edits);
        let mut a_task = BakeTaskState::default();
        let a_chunks = settle_dispatch(&mut a_sched, &mut a_atlas, &mut a_task, &cfg, cam);

        // Synchronous reference.
        let mut s_atlas = SdfAtlas::default();
        let mut s_sched = primed_sched(&edits);
        let s_chunks = settle_gpu(&mut s_sched, &mut s_atlas, &cfg, cam);

        assert!(a_atlas.bricks.len() > ASYNC_BAKE_THRESHOLD, "scene must exceed the async threshold to test the task path (got {})", a_atlas.bricks.len());
        let a_bricks: HashSet<_> = a_atlas.bricks.keys().copied().collect();
        let s_bricks: HashSet<_> = s_atlas.bricks.keys().copied().collect();
        assert_eq!(a_bricks, s_bricks, "async bake resident set diverged from the synchronous settle");
        assert_eq!(a_chunks, s_chunks);
    }

    /// Staleness reconciliation: if the edit set changes (`edit_gen` bumped) while an async classify
    /// is in flight, the landed result must be discarded wholesale and its chunks re-queued — never
    /// applied against the new geometry.
    #[test]
    fn async_stale_edit_gen_requeues_all() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        let edits = vec![sphere_edit(Vec3::ZERO, 2.0, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();

        // Fabricate a landed task output tagged with an OLD edit_gen (current is 0, stamp 99 stale).
        let cand = (chunk::ChunkKey::new(0, IVec3::ZERO), atlas::BrickKey::new(0, IVec3::ZERO));
        let out = BakeTaskOutput {
            candidates: vec![cand],
            verdicts: vec![Verdict::Keep([edits::PALETTE_EMPTY; edits::PALETTE_K], vec![0], 1234)],
            edit_gen: 99, // != sched.edit_gen (0)
            ring_origins: sched.ring_chunk_origin.clone(),
        };
        sched.pending.clear();
        apply_async_result(&mut atlas, &mut sched, &mut gpu, &cfg, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0, out);

        // Nothing baked (verdict dropped); the candidate's chunk is back in pending for re-classify.
        assert!(gpu.jobs.is_empty(), "stale result must not bake");
        assert!(atlas.bricks.is_empty(), "stale result must not insert bricks");
        assert!(sched.pending.contains(&cand.0), "stale result's chunk must be re-queued");
    }

    /// `emit_gpu_bakes` never emits more than `GPU_BAKE_JOB_CAP` jobs in one frame, and the
    /// dirty chunks whose bricks didn't fit are spilled back to `pending` for the next frame.
    #[test]
    fn gpu_emit_caps_jobs_and_spills_overflow() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        // A big SOLID sphere. The narrow-band cull drops its deep interior, so the overflow
        // must come from the SHELL alone — radius 22 gives a surface band of ~30k+ bricks,
        // comfortably over the 16384 cap. (A solid box would now cull to almost nothing.)
        let edits = vec![sphere_edit(Vec3::ZERO, 22.0, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();

        // Dirty a chunk cube bounding the whole sphere (chunk_world ≈ 2.8, so ±22 ⇒ chunk ±8).
        for x in -9..=9 {
            for y in -9..=9 {
                for z in -9..=9 {
                    sched.pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                }
            }
        }

        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

        assert!(
            gpu.jobs.len() <= GPU_BAKE_JOB_CAP,
            "emitted {} jobs, over the cap {}",
            gpu.jobs.len(),
            GPU_BAKE_JOB_CAP
        );
        assert!(
            !sched.pending.is_empty(),
            "overflow chunks must spill back to pending for the next frame"
        );
        // Every emitted job corresponds to a resident brick (so the shader can read it).
        assert_eq!(
            gpu.jobs.len(),
            atlas.bricks.len(),
            "resident brick count must equal the emitted job count (no half-inserted bricks)"
        );
    }

    /// A spilled brick must NOT be inserted into the atlas — it stays non-resident so the
    /// shader falls back to the coarser LOD. Conversely, empty bricks are evicted even on a
    /// capped frame. Here: resident bricks == emitted jobs, and the spill is purely deferred.
    #[test]
    fn gpu_emit_spilled_bricks_stay_non_resident() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        // Big SOLID sphere: the cull drops its deep interior, so the >cap overflow rides on the
        // shell + bounding-box exterior alone (~100k+ bricks). (A solid box would cull to ~0.)
        let edits = vec![sphere_edit(Vec3::ZERO, 22.0, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();
        for x in -9..=9 {
            for y in -9..=9 {
                for z in -9..=9 {
                    sched.pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                }
            }
        }

        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

        // No brick is resident without a corresponding job this frame: a capped (spilled)
        // brick is never inserted, so the atlas never exposes an un-baked (zero) tile.
        assert_eq!(atlas.bricks.len(), gpu.jobs.len());
        assert!(atlas.bricks.len() <= GPU_BAKE_JOB_CAP);

        // Draining the spill over subsequent frames eventually bakes everything (no chunk is
        // dropped). Run until pending empties; the resident set grows monotonically.
        let mut guard = 0;
        let mut last = atlas.bricks.len();
        while !sched.pending.is_empty() {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
            assert!(atlas.bricks.len() >= last, "resident set must not shrink while draining spill");
            last = atlas.bricks.len();
            guard += 1;
            assert!(guard < 100, "spill drain did not converge");
        }
        assert!(atlas.bricks.len() > GPU_BAKE_JOB_CAP, "all dirty bricks eventually resident");
    }

    /// Narrow-band interior cull: a solid object bakes only its surface SHELL, not its deep
    /// interior. The march reads the field from OUTSIDE (rays shrink to the surface and stop),
    /// so interior bricks are write-only waste — and for a big solid they're the r³ bulk that
    /// drives the approach-bake hitch. Assert: (a) the brick at the centre of a large solid is
    /// NOT resident, (b) a brick straddling the surface IS, (c) the resident count is far below
    /// the solid's full bounding-box brick count (what the old AABB-only cull kept).
    #[test]
    fn gpu_emit_culls_deep_interior_of_solid() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        let radius = 10.0;
        let edits = vec![sphere_edit(Vec3::ZERO, radius, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();

        // Dirty a chunk cube bounding the sphere (chunk_world = 2.8 ⇒ ±10 ⇒ chunk ±4).
        for x in -5..=5 {
            for y in -5..=5 {
                for z in -5..=5 {
                    sched.pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                }
            }
        }
        // Drain fully (cap may spill across frames) so the resident set is the final one.
        let mut gpu = PendingGpuBakes::default();
        let mut guard = 0;
        loop {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
            guard += 1;
            assert!(guard < 1000, "cull-test settle did not converge");
            if sched.pending.is_empty() {
                break;
            }
        }

        let brick_at = |p: Vec3| atlas::BrickKey::new(0, cfg.world_to_brick_lod(p, 0));
        // (a) Dead centre of the solid: surface is `radius` away ≫ brick reach → culled.
        assert!(
            !atlas.bricks.contains_key(&brick_at(Vec3::ZERO)),
            "deep-interior brick at the sphere centre must be culled (write-only waste)"
        );
        // (b) A brick straddling the surface (just inside it) must stay resident.
        assert!(
            atlas.bricks.contains_key(&brick_at(Vec3::new(radius - 0.05, 0.0, 0.0))),
            "surface-shell brick must remain resident (the march reads it)"
        );
        // (c) Resident ≪ what the OLD (AABB-only) cull kept. That cull kept every brick whose
        // box overlapped the sphere's bounding BOX — i.e. the full (2r)³ cube. The narrow-band
        // cull keeps only the shell, so it must be well under half that cube.
        let bw = cfg.brick_world_size(0);
        let bbox_bricks = ((2.0 * radius / bw).ceil() as usize).pow(3);
        assert!(
            atlas.bricks.len() < bbox_bricks / 2,
            "resident {} should be far below the AABB-cull bounding-box count {}",
            atlas.bricks.len(),
            bbox_bricks
        );
    }

    /// Bake-cache skip: re-emitting an already-baked chunk within the SAME edit epoch produces
    /// ZERO jobs (the bricks' `baked_epoch` matches → skipped, no re-cull/re-bake), but bumping
    /// `edit_epoch` (as an edit change does) lapses every stamp so the next emit re-bakes them.
    /// This is the core of the multi-frame-bake hitch fix: a spilled chunk re-queued each frame
    /// no longer re-processes the bricks it already baked.
    #[test]
    fn gpu_emit_skips_already_baked_within_epoch() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        let edits = vec![sphere_edit(Vec3::ZERO, 3.0, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();

        let chunks: Vec<_> = (-2..=2)
            .flat_map(|x| (-2..=2).flat_map(move |y| (-2..=2).map(move |z| chunk::ChunkKey::new(0, IVec3::new(x, y, z)))))
            .collect();

        // Frame 1: bake the sphere shell. Some bricks become resident.
        for ck in &chunks { sched.pending.insert(*ck); }
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        let baked = atlas.bricks.len();
        assert!(baked > 0, "first emit must bake the shell");

        // Frame 2: same chunks dirtied again, edits UNCHANGED → every brick's content hash
        // matches its resident hash → all skipped, no jobs.
        gpu.clear();
        atlas.gpu_baked_tiles.clear();
        for ck in &chunks { sched.pending.insert(*ck); }
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        assert_eq!(gpu.jobs.len(), 0, "re-emit with unchanged edits must skip all baked bricks (content hash)");
        assert_eq!(atlas.bricks.len(), baked, "resident set unchanged on a pure re-emit");

        // Frame 3: the edit MOVED → the bricks it folds now hash differently → they re-bake.
        let moved = vec![sphere_edit(Vec3::new(0.5, 0.0, 0.0), 3.0, 0)];
        sched = primed_sched(&moved);
        gpu.clear();
        atlas.gpu_baked_tiles.clear();
        for ck in &chunks { sched.pending.insert(*ck); }
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
        assert!(!gpu.jobs.is_empty(), "after the edit moves, its bricks must re-bake (content hash changed)");
    }

    /// Fold the shared `tower_field_edits` list into a `(ResolvedEdit, world AABB)` pair list — the
    /// exact geometry the runtime `TowerSpawner` produces, so the bake-cache test exercises the real
    /// stress scene. Roles map to arbitrary distinct material ids.
    fn gallery_resolved() -> Vec<(ResolvedEdit, Aabb3d)> {
        use edits::TowerRole;
        edits::tower_field_edits(&edits::TowerFieldParams::default())
            .into_iter()
            .map(|(_order, transform, prim, role)| {
                let mat = match role {
                    TowerRole::Ground => 0u16,
                    TowerRole::Cube => 1,
                    TowerRole::Cap => 2,
                };
                let op = SdfOp { kind: CsgKind::Union, smoothing: 0.0 };
                let aabb = edit_world_aabb(&prim, &transform, op.smoothing);
                (ResolvedEdit::new(prim, transform, op, mat), aabb)
            })
            .collect()
    }

    /// Drive the production moved-edit dirty path (step 1 of `schedule_bakes`): for the changed
    /// edit, dirty every window chunk over its old∪new world AABB at each LOD. Mirrors the real
    /// code exactly so the test's dirty set is the one the app would produce.
    fn dirty_moved_edit(
        sched: &mut BakeScheduler,
        cfg: &SdfGridConfig,
        cam: Vec3,
        old_aabb: &Aabb3d,
        new_aabb: &Aabb3d,
    ) {
        let r = ring_chunks_per_axis(cfg);
        for lod in 0..cfg.lod_count {
            let origin = ring_chunk_origin(cfg, cam, lod);
            for ck in chunks_in_aabb_windowed(cfg, old_aabb, lod, origin, r) {
                sched.pending.insert(ck);
            }
            for ck in chunks_in_aabb_windowed(cfg, new_aabb, lod, origin, r) {
                sched.pending.insert(ck);
            }
        }
    }

    /// REGRESSION GUARD for the content-hash bake cache (the "moving one object re-bakes the
    /// terrain" bug). Uses the REAL gallery — `gallery_demo_edits`: a procedural heightmap ground +
    /// six cube-towers (rotated cubes) capped by red spheres — at the production 8-LOD config.
    /// Procedure:
    ///   1. Settle the full multi-LOD resident set at the gallery camera (dominated by heightmap).
    ///   2. NUDGE one tower's red sphere a few cm and dirty exactly the chunks the production
    ///      moved-edit path would (old∪new footprint, all LODs).
    ///   3. Assert the re-bake job count is a small fraction of the resident set — only the moved
    ///      sphere's own bricks re-bake; every heightmap / neighbour-tower brick its coarse
    ///      footprint overlaps is content-hash-skipped. Before the per-edit AABB refine in
    ///      `cull_edit_indices`, the moved sphere leaked into every brick sharing its BVH leaf,
    ///      flipping their content hash and re-baking the whole overlapping set.
    #[test]
    fn moving_sphere_near_heightmap_does_not_rebake_heightmap() {
        let cfg = SdfGridConfig { recenter_snap_chunks: 1, ..config() };
        // Gallery camera (orbit default sits ~10 units out, looking at origin).
        let cam = Vec3::new(0.0, 5.0, 10.0);

        let pairs = gallery_resolved();
        let edits0: Vec<ResolvedEdit> = pairs.iter().map(|(e, _)| e.clone()).collect();
        // Move a capping red sphere from a tower NEAR the camera (so it sits in the fine LOD ring
        // and actually re-bakes). Pick the sphere whose XZ is closest to the origin.
        let moved_idx = edits0
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e.prim, SdfPrimitive::Sphere { .. }))
            .min_by(|(_, a), (_, b)| {
                let da = a.transform.translation.xz().length_squared();
                let db = b.transform.translation.xz().length_squared();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i)
            .expect("gallery must contain capping spheres");

        let mut sched = primed_sched(&edits0);
        let mut atlas = SdfAtlas::default();

        // 1) Settle the full resident set through the production recenter + emit path.
        settle_gpu(&mut sched, &mut atlas, &cfg, cam);
        let resident = atlas.bricks.len();
        assert!(resident > 500, "gallery heightmap should make the resident set large (got {resident})");

        // 2) Nudge the capping sphere a few cm; everything else UNCHANGED. Rebuild edits/BVH and
        //    dirty exactly the production old∪new footprint chunks.
        let old_aabb = pairs[moved_idx].1;
        let mut new_edits = edits0.clone();
        let moved_tf = {
            let t = new_edits[moved_idx].transform;
            Transform { translation: t.translation + Vec3::new(0.04, 0.0, 0.0), ..t }
        };
        new_edits[moved_idx] = ResolvedEdit::new(new_edits[moved_idx].prim.clone(), moved_tf, new_edits[moved_idx].op, new_edits[moved_idx].material_id);
        let new_aabb = edit_world_aabb(&new_edits[moved_idx].prim, &moved_tf, 0.0);
        sched.edits = Arc::new(new_edits);
        sched.bvh = Arc::new(build_bvh(&sched.edits));

        dirty_moved_edit(&mut sched, &cfg, cam, &old_aabb, &new_aabb);
        let mut gpu = PendingGpuBakes::default();
        atlas.gpu_baked_tiles.clear();
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

        // 3) Only the moved sphere's own bricks re-bake — a tiny fraction of the resident set. The
        //    scatter gallery is ~14k edits settling ~78k resident bricks; the content-hash cache
        //    keeps the rebake to a few dozen (the moved sphere's own shell). A leak would re-bake
        //    hundreds-to-thousands as the moved edit's coarse footprint dragged in unchanged terrain.
        let rebaked = gpu.jobs.len();
        assert!(
            rebaked > 0,
            "the moved sphere's bricks MUST re-bake (content changed)"
        );
        assert!(
            rebaked < 200,
            "moving one sphere re-baked {rebaked} of {resident} resident bricks — unchanged terrain \
             / neighbour towers are being re-baked too (content-hash cache leak)"
        );
    }

    /// Empty-space bricks are evicted the same frame even under the job cap (eviction is
    /// CPU-only, never spilled) — the fix for the drag trail must survive the cap.
    #[test]
    fn gpu_emit_evicts_empties_under_cap() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        // Small edit at the origin; most chunks are empty space.
        let edits = vec![box_edit(Vec3::ZERO, 0.5, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();

        // Pre-populate a far brick as if it were resident from a previous position, then dirty
        // its chunk: the edit doesn't reach it, so it must be evicted this frame. Use a real
        // brick key from the chunk's enumeration so it's stride-aligned (chunk_brick_keys must
        // actually visit it).
        let far_chunk = chunk::ChunkKey::new(0, IVec3::new(100, 0, 0));
        let far = chunk_brick_keys(far_chunk, &cfg)[0];
        atlas.insert_gpu_brick(far, [edits::PALETTE_EMPTY; edits::PALETTE_K], 0, &cfg);
        assert!(atlas.bricks.contains_key(&far));
        // The edit doesn't reach this far brick, so it classifies as Empty and is evicted this
        // frame regardless of any content hash — the content-hash skip only applies to a Keep.
        sched.pending.insert(far_chunk);
        sched.pending.insert(chunk::ChunkKey::new(0, IVec3::ZERO));

        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);

        assert!(
            !atlas.bricks.contains_key(&far),
            "a now-empty brick must be evicted, not left as a trail"
        );
    }

    /// The headline correctness guarantee: drive the camera back and forth across geometry
    /// (so chunks repeatedly exit and re-enter windows) via the real recenter + GPU emit, then
    /// settle at the final camera. The resident set must equal a fresh settle there — no stale
    /// leading edge, no missing bricks. Absolute addressing makes a brick that exits and
    /// re-enters identical, so the walk must converge to the same set as arriving directly.
    #[test]
    fn recenter_walk_converges_to_fresh_settle() {
        let cfg = SdfGridConfig { lod_count: 3, ring_bricks: 8, recenter_snap_chunks: 1, ..Default::default() };
        let edits: Vec<ResolvedEdit> = (-6i32..=6).map(|i| box_edit(Vec3::new(i as f32 * 1.2, 0.0, 0.0), 0.4, (i.rem_euclid(3)) as u16)).collect();

        let mut atlas = SdfAtlas::default();
        let mut sched = primed_sched(&edits);

        // Walk a winding path forward and back across several brick/chunk boundaries.
        let path = [0.0f32, 2.0, 4.0, 1.0, -3.0, -1.0, 5.0, 0.0, 3.0, -4.0, 0.0];
        for &x in &path {
            settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(x, 0.0, 0.0));
        }
        let final_cam = Vec3::new(*path.last().unwrap(), 0.0, 0.0);
        let walked: HashSet<_> = atlas.bricks.keys().copied().collect();

        // A fresh arrival at the same camera (independent scheduler/atlas).
        let mut fresh_atlas = SdfAtlas::default();
        let mut fresh_sched = primed_sched(&edits);
        settle_gpu(&mut fresh_sched, &mut fresh_atlas, &cfg, final_cam);
        let fresh: HashSet<_> = fresh_atlas.bricks.keys().copied().collect();

        assert_eq!(walked, fresh, "recenter walk diverged from a fresh settle (stale/missing bricks)");
    }

    /// PERF SIMULATION of the camera-movement hitch in the stress scene. Settles the full tower
    /// field at the production 8-LOD config, then walks the camera in small steps (crossing several
    /// recenter snap boundaries) and, for each step, counts the recenter's per-frame work:
    /// `window_scans` (entered-shell chunk slots visited across all recentering LODs),
    /// `geom_queries` (how many ran a BVH AABB query), and `enqueued` (geometry chunks newly
    /// dirtied this frame). Printed so we can SEE the spike on the frames where coarse LODs snap.
    /// Not an assertion-heavy test, a measurement rig (run with --ignored --nocapture).
    #[test]
    #[ignore = "perf measurement rig; run explicitly with --ignored --nocapture"]
    fn lod_recenter_cost_walk() {
        use edits::TowerRole;
        let cfg = SdfGridConfig::default(); // production: 8 LODs, ring 64, snap 2
        let edits: Vec<ResolvedEdit> = edits::tower_field_edits(&edits::TowerFieldParams::default())
            .into_iter()
            .map(|(_o, t, p, role)| {
                let mat = match role { TowerRole::Ground => 0u16, TowerRole::Cube => 1, TowerRole::Cap => 2 };
                ResolvedEdit::new(p, t, SdfOp { kind: CsgKind::Union, smoothing: 0.0 }, mat)
            })
            .collect();
        eprintln!("LOD-WALK: {} edits, building BVH...", edits.len());

        let mut atlas = SdfAtlas::default();
        let mut sched = primed_sched(&edits);
        // Settle at the start camera (orbit default ~10 units out).
        let r = ring_chunks_per_axis(&cfg);
        settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(0.0, 5.0, 10.0));
        eprintln!("LOD-WALK: settled, resident bricks = {}", atlas.bricks.len());

        // Walk along +X in 1.5 m steps — at base voxel 0.1 / chunk_world ≈ 3.2 m, snap 2 ⇒ a LOD0
        // recenter every ~6.4 m, coarser LODs every 2^L × that, so steps cross staggered boundaries.
        let mut max_scans = 0usize;
        let mut max_geom = 0usize;
        let mut max_enq = 0usize;
        for step in 1..=40i32 {
            let cam = Vec3::new(step as f32 * 1.5, 5.0, 10.0);
            // Instrumented mirror of step 2 recenter: count window scans + geometry queries + time.
            let mut scans = 0usize;
            let mut geom = 0usize;
            let mut enqueued = 0usize;
            let mut stack: Vec<u32> = Vec::new();
            let t_recenter = std::time::Instant::now();
            for lod in 0..cfg.lod_count {
                let li = lod as usize;
                let new_origin = ring_chunk_origin(&cfg, cam, lod);
                let old_origin = sched.ring_chunk_origin[li];
                if new_origin == old_origin { continue; }
                for_each_entered_chunk(new_origin, old_origin, r, |coord| {
                    scans += 1;
                    geom += 1;
                    let ck = chunk::ChunkKey::new(lod, coord);
                    if chunk_has_geometry_with(ck, &sched.bvh, &cfg, &mut stack) && sched.pending.insert(ck) {
                        enqueued += 1;
                    }
                });
                for_each_exited_chunk(new_origin, old_origin, r, |coord| {
                    let ck = chunk::ChunkKey::new(lod, coord);
                    sched.pending.remove(&ck);
                    for_each_brick_key(ck, &cfg, |bk| { atlas.remove_brick(&bk, &cfg); });
                });
                sched.ring_chunk_origin[li] = new_origin;
            }
            let recenter_us = t_recenter.elapsed().as_micros();
            let mut gpu = PendingGpuBakes::default();
            let t_emit = std::time::Instant::now();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
            let emit_us = t_emit.elapsed().as_micros();
            if scans > 0 {
                eprintln!("step {step:2}: scans={scans:6} geom_q={geom:5} enq={enqueued:4} candidates~{:6} baked={:5} | recenter={recenter_us:5}us emit={emit_us:6}us emit_per_job={:.2}us",
                    enqueued * 64, gpu.jobs.len(), emit_us as f64 / (gpu.jobs.len().max(1)) as f64);
            }
            max_scans = max_scans.max(scans);
            max_geom = max_geom.max(geom);
            max_enq = max_enq.max(enqueued);
        }
        eprintln!("LOD-WALK MAX: window_scans={max_scans} geom_queries={max_geom} enqueued={max_enq}");
    }

    /// Flying *away* from a localized scene must still refresh the scene's bricks into their
    /// new (coarser) LOD rings — the same resident set as a fresh settle at the destination,
    /// and the bake enqueues stay bounded by the scene's shell footprint (NOT the empty volume
    /// swept), so flying 4× farther does not enqueue ~4× the chunks (the empty-chunk cull).
    #[test]
    fn flying_away_still_refreshes_scene_lod() {
        let cfg = SdfGridConfig { lod_count: 4, ring_bricks: 8, recenter_snap_chunks: 1, ..Default::default() };
        let edits: Vec<ResolvedEdit> =
            (-1i32..=1).map(|i| box_edit(Vec3::new(i as f32 * 0.5, 0.0, 0.0), 0.4, 0)).collect();

        // Fly `steps` small steps away from the scene; return (resident chunk set, total
        // geometry-chunk enqueues over the flight, excluding the initial fill).
        let run = |sign: f32, steps: i32| -> (HashSet<chunk::ChunkKey>, usize) {
            let mut atlas = SdfAtlas::default();
            let mut sched = primed_sched(&edits);
            settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::ZERO); // initial fill
            let mut enqueued = 0usize;
            let mut gpu = PendingGpuBakes::default();
            for i in 1..=steps {
                let cam = Vec3::new(sign * i as f32 * 0.4, 0.0, 0.0);
                enqueued += recenter_step(&mut sched, &mut atlas, &cfg, cam);
                // Drain this frame's emission (the GPU bake; spill drains over frames).
                let mut guard = 0;
                loop {
                    gpu.jobs.clear();
                    gpu.edits.clear();
                    atlas.gpu_baked_tiles.clear();
                    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, Vec3::ZERO, &mut crate::sdf_render::BakedBrickDebug::default(), 0.0);
                    guard += 1;
                    assert!(guard < 1000, "frame drain did not converge");
                    if sched.pending.is_empty() { break; }
                }
            }
            let set = atlas.bricks.keys().map(|k| chunk::chunk_of(*k, &cfg).0).collect();
            (set, enqueued)
        };

        // 1) Symmetry + correctness: flying away either direction leaves the same resident set
        //    as a fresh settle at the destination (no stale fine bricks, nothing missing).
        for (label, sign, steps) in [("forward", 1.0, 16), ("backward", -1.0, 16)] {
            let (flown, _) = run(sign, steps);
            let mut fresh_atlas = SdfAtlas::default();
            let mut fresh_sched = primed_sched(&edits);
            let fresh = settle_gpu(&mut fresh_sched, &mut fresh_atlas, &cfg, Vec3::new(sign * steps as f32 * 0.4, 0.0, 0.0));
            assert_eq!(flown, fresh, "{label}: flew-in resident chunks diverged from a fresh settle");
            assert!(!flown.is_empty(), "{label}: scene vanished after flying away");
        }

        // 2) The cull's core guarantee: enqueues while flying away are bounded by the scene's
        //    shell footprint, NOT the empty volume swept — flying 4× farther must NOT enqueue
        //    ~4× the chunks.
        let (_, near) = run(-1.0, 8);
        let (_, far) = run(-1.0, 32); // 4× the distance
        assert!(
            far <= near * 2,
            "enqueues scaled with flight distance (near={near}, far={far}) — empty chunks not culled, scene will starve"
        );
    }

    /// `ring_bricks / CHUNK_BRICKS` chunks per axis. With the defaults (12 / 4) that is 3.
    #[test]
    fn ring_window_is_chunks_per_axis() {
        let cfg = config();
        let r = ring_chunks_per_axis(&cfg);
        assert_eq!(r, (cfg.ring_bricks / chunk::CHUNK_BRICKS as u32) as i32);
        assert!(r >= 1, "ring must be at least one chunk wide");
    }

    /// The ring is centred on the camera: the camera's own chunk sits at the window's
    /// middle (`origin + half`), so re-deriving the camera chunk from the world position
    /// lands inside the window.
    #[test]
    fn ring_origin_centres_camera_chunk() {
        let cfg = config();
        let r = ring_chunks_per_axis(&cfg);
        let half = r / 2;
        for lod in 0..cfg.lod_count {
            // A few world positions, including off-origin and negative.
            for cam in [
                Vec3::ZERO,
                Vec3::new(37.0, -12.0, 250.0),
                Vec3::new(-400.0, 8.0, -130.0),
            ] {
                let origin = ring_chunk_origin(&cfg, cam, lod);
                // Camera's chunk = origin + half on each axis.
                let cam_chunk = origin + IVec3::splat(half);
                assert!(
                    chunk_in_window(cam_chunk, origin, r),
                    "camera chunk must be inside its own ring (lod={lod}, cam={cam:?})"
                );
            }
        }
    }

    /// `chunk_in_window` is a half-open `[origin, origin+r)` box on every axis.
    #[test]
    fn chunk_in_window_boundaries() {
        let origin = IVec3::new(5, -2, 0);
        let r = 3;
        assert!(chunk_in_window(origin, origin, r), "corner is inside");
        assert!(chunk_in_window(origin + IVec3::splat(r - 1), origin, r), "far corner inside");
        assert!(!chunk_in_window(origin + IVec3::splat(r), origin, r), "one past is outside");
        assert!(!chunk_in_window(origin - IVec3::X, origin, r), "one before is outside");
    }

    /// `chunk_window_keys` yields exactly `r³` distinct keys, all inside the window and
    /// all at the requested LOD.
    #[test]
    fn chunk_window_keys_cover_the_box() {
        use std::collections::HashSet;
        let origin = IVec3::new(-1, 4, 2);
        let r = 3;
        let lod = 2u32;
        let keys: Vec<_> = chunk_window_keys(origin, r, lod).collect();
        assert_eq!(keys.len(), (r * r * r) as usize, "must enumerate r^3 chunks");
        let set: HashSet<_> = keys.iter().map(|k| k.coord).collect();
        assert_eq!(set.len(), keys.len(), "no duplicate chunk coords");
        for k in &keys {
            assert_eq!(k.lod, lod);
            assert!(chunk_in_window(k.coord, origin, r));
        }
    }

    /// Each brick key a chunk emits maps back to that exact chunk + a unique local slot
    /// 0..CHUNK_VOLUME — the round-trip the GPU resolve relies on.
    #[test]
    fn chunk_brick_keys_roundtrip_through_chunk_of() {
        use std::collections::HashSet;
        let cfg = config();
        let ck = chunk::ChunkKey::new(1, IVec3::new(-2, 0, 3));
        let bricks = chunk_brick_keys(ck, &cfg);
        assert_eq!(bricks.len(), chunk::CHUNK_VOLUME as usize);
        let mut locals = HashSet::new();
        for bk in &bricks {
            let (back, local) = chunk::chunk_of(*bk, &cfg);
            assert_eq!(back, ck, "brick must belong to the chunk that emitted it");
            assert!(local < chunk::CHUNK_VOLUME, "local slot in range");
            assert!(locals.insert(local), "each brick occupies a distinct local slot");
        }
        assert_eq!(locals.len(), chunk::CHUNK_VOLUME as usize, "all 64 slots covered");
    }

    /// A small edit AABB at the origin dirties the chunk(s) it overlaps at LOD 0, and the
    /// chunk containing the origin is always among them (footprint pad ⇒ never misses).
    #[test]
    fn chunks_in_aabb_covers_origin_chunk() {
        let cfg = config();
        let aabb = Aabb3d::new(Vec3::ZERO, Vec3::splat(0.3));
        // A window comfortably containing the origin.
        let win = IVec3::splat(-8);
        let r = 16;
        let chunks = chunks_in_aabb_windowed(&cfg, &aabb, 0, win, r);
        assert!(!chunks.is_empty(), "an edit must dirty at least one chunk");
        let origin_chunk = chunk::chunk_of(atlas::BrickKey::new(0, IVec3::ZERO), &cfg).0;
        assert!(
            chunks.contains(&origin_chunk),
            "the origin's chunk must be in the dirtied set"
        );
        // All returned chunks are at the requested LOD.
        assert!(chunks.iter().all(|c| c.lod == 0));
    }

    /// The windowed clamp is the heightmap-freeze fix: a terrain-scale AABB (huge in XZ) must
    /// only ever enumerate chunks INSIDE the window — never the millions of chunks its full
    /// extent spans. Asserts the result is bounded by r³ and every chunk is in-window, even
    /// though the AABB is vastly larger than the window.
    #[test]
    fn chunks_in_aabb_windowed_is_bounded_by_window() {
        let cfg = config();
        // A heightmap-like AABB: enormous in XZ, thin in Y.
        let aabb = Aabb3d::new(Vec3::ZERO, Vec3::new(100_000.0, 2.0, 100_000.0));
        let win = IVec3::splat(-8);
        let r = 16;
        let chunks = chunks_in_aabb_windowed(&cfg, &aabb, 0, win, r);
        assert!(
            chunks.len() <= (r * r * r) as usize,
            "windowed dirty set must be bounded by r³ = {}, got {}",
            r * r * r,
            chunks.len()
        );
        assert!(
            chunks.iter().all(|c| chunk_in_window(c.coord, win, r)),
            "every dirtied chunk must lie inside the window"
        );
        assert!(!chunks.is_empty(), "the AABB overlaps the window, so some chunks dirty");
    }

    /// A one-chunk camera shift exposes only a thin shell: the entered chunks are a face
    /// of the window (`r²`), never the whole `r³` volume. This is what keeps incremental
    /// recenter cheap (vs re-baking the full ring).
    #[test]
    fn one_chunk_shift_exposes_only_a_shell() {
        let r = 3;
        let old_origin = IVec3::ZERO;
        let new_origin = IVec3::new(1, 0, 0); // shift +1 chunk on X
        let entered = chunk_window_keys(new_origin, r, 0)
            .filter(|k| !chunk_in_window(k.coord, old_origin, r))
            .count();
        assert_eq!(entered, (r * r) as usize, "a 1-chunk shift enters exactly one r^2 face");
        assert!(entered < (r * r * r) as usize, "shell is far smaller than the volume");
    }

    /// Hysteresis: with `recenter_snap_chunks > 1`, camera motion that stays within one
    /// snap cell must not move the ring origin at all — so no shell is entered/exited and
    /// no rebake is triggered. The window only jumps when the snapped camera chunk changes.
    #[test]
    fn snap_holds_origin_within_a_snap_cell() {
        let snap = 4;
        let cfg = SdfGridConfig { recenter_snap_chunks: snap, ..config() };
        let chunk_world = chunk::chunk_world_size(0, &cfg);
        let lod = 0u32;

        // Origin at the world origin, then nudge the camera across most of a snap cell
        // (snap chunks wide) without crossing the next snap boundary.
        let base = ring_chunk_origin(&cfg, Vec3::ZERO, lod);
        let within = (snap as f32 - 0.5) * chunk_world; // just under one snap cell
        for d in [0.0, 0.25, 0.5, 0.9] {
            let cam = Vec3::new(within * d, 0.0, 0.0);
            assert_eq!(
                ring_chunk_origin(&cfg, cam, lod),
                base,
                "origin moved within a snap cell (cam={cam:?}); hysteresis not holding"
            );
        }

        // Crossing a full snap cell must move the origin by exactly `snap` chunks.
        let past = Vec3::new(snap as f32 * chunk_world + 0.1, 0.0, 0.0);
        let moved = ring_chunk_origin(&cfg, past, lod);
        assert_eq!(moved.x - base.x, snap, "a full snap-cell crossing shifts the origin by snap chunks");
    }

}
