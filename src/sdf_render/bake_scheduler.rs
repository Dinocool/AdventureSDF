//! Incremental, async clipmap bake scheduling in **chunk units**.
//!
//! The main thread only does cheap integer chunk-ring window diffs (enqueue entered
//! chunks, evict exited chunks) and applies finished task results; the actual
//! `bake_brick` work runs on `AsyncComputeTaskPool`, so camera motion never blocks.
//!
//! Eager eviction is safe because addressing is **absolute** (chunk keys, not a
//! camera-relative ring origin — see [`super::chunk`]): a not-yet-baked chunk is simply
//! absent from the GPU chunk table, and the nested coarser LOD shell already covers that
//! region, so the leading edge shows coarser-correct terrain that refines in — never a
//! hole, never a shift.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use super::atlas::{self, ATLAS_TILES_PER_ROW, SdfAtlas};
use super::chunk;
use super::{
    BakeBackend, SdfCamera, SdfGridConfig, SdfMaterial, SdfOp, SdfPrimitive, SdfVolume,
    VolumeQueryData, bvh, edits, gather_sorted_edits,
};

/// One-frame request to bake this frame's dirty chunks **synchronously** (on the main
/// thread, same frame) instead of deferring them to the async task pool.
///
/// Reusable seam for "I need this edit visible *now*": set `.0 = true` from any system
/// that mutates an [`SdfVolume`] and wants the result on screen immediately — a live
/// gizmo drag, an inspector slider, a programmatic edit. Without it the async path
/// applies a frame or two later, and during a *continuous* edit (e.g. a drag) every
/// frame bumps `edit_epoch`, so the in-flight async results are discarded as stale and
/// nothing lands until the edit stops. `schedule_bakes` consumes (clears) the flag each
/// frame after honoring it.
#[derive(Resource, Default)]
pub struct SyncBakeRequest(pub bool);

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
/// they index. Filled by `schedule_bakes`/`apply_bakes` in [`BakeBackend::Gpu`] mode and
/// drained by the render-world extract. `atlas_rows` is how many tile rows the atlas spans
/// this frame so the render world can size the destination/scratch consistently.
#[derive(Resource, Default)]
pub struct PendingGpuBakes {
    pub jobs: Vec<GpuBakeJob>,
    pub edits: Vec<edits::GpuEdit>,
    pub atlas_rows: u32,
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

/// One in-flight async bake: a pool task baking the bricks of one or more chunks, tagged
/// with the `edit_epoch` it was scheduled under (results baked against superseded edits
/// are detected and requeued).
struct BakeJob {
    epoch: u64,
    task: Task<Vec<(atlas::BrickKey, Option<atlas::PackedBrick>)>>,
}

/// Drives incremental, async clipmap baking in chunk units (see module docs).
#[derive(Resource)]
pub struct BakeScheduler {
    /// Bumped whenever the edit set changes (add/remove/move). Results carrying an
    /// older epoch are stale and get requeued.
    edit_epoch: u64,
    /// Snapshot of the current edits + BVH handed to bake tasks (cheap Arc clone).
    edits: Arc<Vec<edits::ResolvedEdit>>,
    bvh: Arc<bvh::Bvh>,
    /// Decoded height maps for bake-time displacement, snapshotted alongside edits/BVH so
    /// async bake tasks can sample it. Rebuilt when the material registry's displacement
    /// columns change (see `update_height_field`).
    height: super::height::SharedHeightField,
    /// Per-LOD chunk-ring origin currently resident (index = lod), in chunk coords. Used
    /// to diff which chunks entered/exited as the camera moves. Empty until first run.
    ring_chunk_origin: Vec<IVec3>,
    /// Chunk keys awaiting a bake (deduped).
    pending: std::collections::HashSet<chunk::ChunkKey>,
    /// Tasks currently baking.
    inflight: Vec<BakeJob>,
    /// (GPU bake mode) How many atlas tile rows the render world has committed to the GPU
    /// atlas texture. When `high_water` pushes `required_rows` past this, the render world
    /// reallocs (recreates + zero-fills) the texture, dropping every GPU-written tile — so
    /// on that frame we must re-emit the WHOLE resident set as bake jobs, not just the
    /// dirty shell. Mirrors `AtlasCapacity::rows` in render.rs from the main world.
    gpu_atlas_rows: u32,
}

impl Default for BakeScheduler {
    fn default() -> Self {
        Self {
            edit_epoch: 0,
            edits: Arc::new(Vec::new()),
            bvh: Arc::new(bvh::Bvh::default()),
            height: Arc::new(super::height::HeightField::default()),
            ring_chunk_origin: Vec::new(),
            pending: std::collections::HashSet::new(),
            inflight: Vec::new(),
            gpu_atlas_rows: 0,
        }
    }
}

impl BakeScheduler {
    /// Replace the height-field snapshot used by subsequent bakes (rebuilt when the material
    /// registry's displacement columns change). Async tasks clone the `Arc`.
    pub fn set_height(&mut self, height: super::height::SharedHeightField) {
        self.height = height;
    }

    /// The current height-field snapshot (for the synchronous diagnostic bake, which baked
    /// off the scheduler's edits/bvh too).
    pub fn height_field(&self) -> &super::height::HeightField {
        &self.height
    }
}

/// Max chunks baked per pool task. Each chunk is up to `CHUNK_VOLUME` (64) `bake_brick`
/// calls, so keep the per-task chunk count small to stream results back promptly.
const BAKE_CHUNKS_PER_TASK: usize = 2;

/// Per-frame schedule budget: at most this many chunks are turned into bake tasks each
/// frame. A large camera jump (or first bake) can dirty thousands of chunks; spawning
/// them all at once bursts the task pool and stalls the frame. Capping the drain bounds
/// worst-case per-frame scheduling cost — leftover chunks stay in `pending` (a dedup set)
/// and drain over the next frames, refining in. Coarser LODs already cover the not-yet-
/// baked region, so the delay shows as coarse-correct terrain, never a hole.
const SCHEDULE_BUDGET_CHUNKS: usize = 64;

/// Any component that affects an edit's baked result. A change to one of these
/// triggers a targeted rebake of the bricks the edit touches. Exposed as
/// [`ChangedEditFilter`] so the diagnostic sync bake can reuse the same change filter.
pub type ChangedEditFilter = Or<(
    Changed<Transform>,
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

/// Squared world distance from the camera to a chunk's centre. Used to drain the bake
/// queue nearest-first so the visible shell fills before far chunks.
fn chunk_dist_sq(ck: chunk::ChunkKey, camera_pos: Vec3, config: &SdfGridConfig) -> f32 {
    let half = chunk::chunk_world_size(ck.lod, config) * 0.5;
    let centre = chunk::chunk_min_world(ck, config) + Vec3::splat(half);
    (centre - camera_pos).length_squared()
}

/// Whether any edit reaches chunk `ck` (its world AABB overlaps an edit in the BVH). A
/// chunk's world AABB is exactly the union of its `CHUNK_BRICKS³` brick AABBs (the bare
/// brick footprint `bake_coord` queries), so a BVH miss here guarantees *every* brick in
/// the chunk would bake to empty space. Used to skip enqueuing empty entered chunks: they
/// would consume the per-frame budget producing nothing, starving the real geometry that
/// enters far (coarse-LOD) rings — the cause of LOD never refreshing when flying away from
/// the scene. Safe only for camera-*entered* chunks (no resident bricks to evict yet); the
/// edit-dirty path must still enqueue emptied chunks so vacated bricks get removed.
fn chunk_has_geometry(ck: chunk::ChunkKey, bvh: &bvh::Bvh, config: &SdfGridConfig, scratch: &mut Vec<u32>) -> bool {
    let size = chunk::chunk_world_size(ck.lod, config);
    let min = chunk::chunk_min_world(ck, config);
    let aabb = bevy::math::bounding::Aabb3d::from_min_max(min, min + Vec3::splat(size));
    bvh.query_aabb(&aabb, scratch);
    !scratch.is_empty()
}

/// Remove and return up to `budget` chunks from `pending`, nearest-camera-first. When
/// `pending` fits in the budget it is drained whole (cheap). Otherwise the nearest
/// `budget` chunks are taken and the rest left for subsequent frames. Pure (no ECS), so
/// the budgeting + ordering is unit-testable without an App or task pool.
fn drain_budget(
    pending: &mut std::collections::HashSet<chunk::ChunkKey>,
    camera_pos: Vec3,
    config: &SdfGridConfig,
    budget: usize,
) -> Vec<chunk::ChunkKey> {
    if pending.len() <= budget {
        return pending.drain().collect();
    }
    let mut all: Vec<chunk::ChunkKey> = pending.iter().copied().collect();
    // Sort by squared distance of the chunk centre to the camera (finer LODs sit nearer
    // the camera, so this also naturally prioritises high-detail chunks).
    all.sort_by(|a, b| {
        chunk_dist_sq(*a, camera_pos, config)
            .partial_cmp(&chunk_dist_sq(*b, camera_pos, config))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all.truncate(budget);
    for ck in &all {
        pending.remove(ck);
    }
    all
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

/// All brick keys belonging to chunk `ck` (its `CHUNK_BRICKS³` local slots).
fn chunk_brick_keys(ck: chunk::ChunkKey, config: &SdfGridConfig) -> Vec<atlas::BrickKey> {
    let s = config.cell_stride();
    let c = chunk::CHUNK_BRICKS;
    let base = ck.coord * c; // brick-index space
    let mut keys = Vec::with_capacity(chunk::CHUNK_VOLUME as usize);
    for lz in 0..c {
        for ly in 0..c {
            for lx in 0..c {
                let bi = base + IVec3::new(lx, ly, lz);
                keys.push(atlas::BrickKey::new(ck.lod, bi * s)); // back to coord space
            }
        }
    }
    keys
}

/// The chunks at `lod` whose world extent overlaps `aabb` (grown by the bake footprint
/// pad so a moved edit re-dirties every chunk that could fold it). Computed directly in
/// chunk-coord space — no per-brick enumeration.
fn chunks_in_aabb(
    config: &SdfGridConfig,
    aabb: &bevy::math::bounding::Aabb3d,
    lod: u32,
) -> Vec<chunk::ChunkKey> {
    let chunk_world = chunk::chunk_world_size(lod, config);
    let pad = Vec3::splat(atlas::SNORM_CLAMP_DIST + config.brick_world_size(lod));
    let lo = (Vec3::from(aabb.min) - pad) / chunk_world;
    let hi = (Vec3::from(aabb.max) + pad) / chunk_world;
    let lo = IVec3::new(lo.x.floor() as i32, lo.y.floor() as i32, lo.z.floor() as i32);
    let hi = IVec3::new(hi.x.ceil() as i32, hi.y.ceil() as i32, hi.z.ceil() as i32);

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

/// Main-thread scheduling only — no baking. Diffs the per-LOD chunk-ring window as the
/// camera moves (enqueue entered chunks, evict exited chunks), dirties edited regions,
/// and spawns async bake tasks. All integer window math + Arc clones — microseconds.
#[expect(clippy::too_many_arguments)]
pub fn schedule_bakes(
    mut atlas: ResMut<SdfAtlas>,
    mut bvh: ResMut<bvh::Bvh>,
    mut sched: ResMut<BakeScheduler>,
    mut prev_aabbs: ResMut<PrevEditAabbs>,
    mut sync_request: ResMut<SyncBakeRequest>,
    mut gpu_bakes: ResMut<PendingGpuBakes>,
    backend: Res<BakeBackend>,
    config: Res<SdfGridConfig>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    changed: Query<Entity, (With<SdfVolume>, ChangedEdit)>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    // GPU bake jobs are rebuilt from scratch each frame; the render world consumed last
    // frame's. `gpu_baked_tiles` likewise holds only THIS frame's GPU-written tiles.
    gpu_bakes.clear();
    atlas.gpu_baked_tiles.clear();
    let gpu_mode = *backend == BakeBackend::Gpu;
    // Consume the one-frame "bake now, synchronously" request (see SyncBakeRequest).
    let bake_sync = std::mem::take(&mut sync_request.0);
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
        sched.edit_epoch = sched.edit_epoch.wrapping_add(1);

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
            // Existing edits moved → dirty the chunks over each changed edit's old∪new
            // footprint, clamped to the resident window for that LOD.
            for entity in &changed {
                for lod in 0..lod_count {
                    let origin = ring_chunk_origin(&config, camera_pos, lod);
                    let mut dirty_one = |aabb: &bevy::math::bounding::Aabb3d| {
                        for ck in chunks_in_aabb(&config, aabb, lod) {
                            if chunk_in_window(ck.coord, origin, r) {
                                sched.pending.insert(ck);
                            }
                        }
                    };
                    if let Some(old) = prev_aabbs.map.get(&entity) {
                        dirty_one(old);
                    }
                    if let Some(new) = current.get(&entity) {
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
    let mut bvh_scratch: Vec<u32> = Vec::new();
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
        for ck in chunk_window_keys(new_origin, r, lod) {
            let entered = first_run || !chunk_in_window(ck.coord, old_origin, r);
            if entered && chunk_has_geometry(ck, &bvh, &config, &mut bvh_scratch) {
                sched.pending.insert(ck);
            }
        }
        // Exited chunks → drop all their bricks (and cancel any pending bake).
        if !first_run {
            for ck in chunk_window_keys(old_origin, r, lod) {
                if !chunk_in_window(ck.coord, new_origin, r) {
                    sched.pending.remove(&ck);
                    for bk in chunk_brick_keys(ck, &config) {
                        atlas.remove_brick(&bk);
                    }
                }
            }
        }
        sched.ring_chunk_origin[li] = new_origin;
    }

    // --- 3. Bake the dirty chunks (GPU compute, sync inline, or async tasks) ----------
    //
    // GPU mode: the CPU does only topology (BVH cull + palette + tile alloc) and emits a
    // GpuBakeJob per brick; the compute shader fills the texels. A realloc on the render
    // side (the atlas grew taller) zero-fills the texture and drops every GPU-written tile,
    // so on a grow frame we re-emit the WHOLE resident set, not just this frame's shell.
    if gpu_mode {
        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu_bakes, &config, camera_pos);
        return;
    }

    if sched.pending.is_empty() {
        return;
    }

    // Take only this frame's budget, nearest-camera-first, so the visible shell fills
    // before far chunks. Leftover pending chunks persist (dedup set) for next frame.
    let drained = drain_budget(&mut sched.pending, camera_pos, &config, SCHEDULE_BUDGET_CHUNKS);

    // Sync path: bake on the main thread and apply this frame, so a live edit (gizmo
    // drag, slider) is visible immediately. Skips the task pool entirely, so there's no
    // epoch race to lose — what we drain, we apply now. Same per-frame budget bounds cost.
    if bake_sync {
        let mut applied = false;
        for ck in &drained {
            for key in chunk_brick_keys(*ck, &config) {
                match SdfAtlas::bake_brick(key, &sched.edits, &sched.bvh, &config, &sched.height) {
                    Some(brick) => {
                        atlas.insert_brick(key, brick);
                        applied = true;
                    }
                    None => applied |= atlas.remove_brick(&key),
                }
            }
        }
        if applied {
            atlas.last_bake_was_full = false;
            atlas.bump_generation();
        }
        return;
    }

    let pool = AsyncComputeTaskPool::get();
    let epoch = sched.edit_epoch;

    for group in drained.chunks(BAKE_CHUNKS_PER_TASK) {
        // Expand the group's chunks into their brick keys for the task.
        let mut keys: Vec<atlas::BrickKey> = Vec::new();
        for ck in group {
            keys.extend(chunk_brick_keys(*ck, &config));
        }
        let edits = Arc::clone(&sched.edits);
        let bvh_snapshot = Arc::clone(&sched.bvh);
        let height = Arc::clone(&sched.height);
        let cfg = config.clone();
        let task = pool.spawn(async move {
            keys.into_iter()
                .map(|key| {
                    let baked = SdfAtlas::bake_brick(key, &edits, &bvh_snapshot, &cfg, &height);
                    (key, baked)
                })
                .collect::<Vec<_>>()
        });
        sched.inflight.push(BakeJob { epoch, task });
    }
}

/// (GPU bake mode) Turn this frame's dirty chunks into [`GpuBakeJob`]s: the CPU does only
/// the topology work (BVH cull → which bricks exist + their palette + a stable tile), and
/// the compute shader fills each brick's 512 texels straight into the atlas. No main-thread
/// voxel loop — that's the whole point.
///
/// Two-pass so a leading-edge brick that grows the atlas is handled correctly: pass 1 culls
/// and allocates tiles for the dirty shell (growing the tile high-water), then we learn
/// whether the render world will realloc (recreate + zero-fill) the atlas this frame. On a
/// realloc every previously-GPU-written tile is about to be wiped, so pass 2 emits jobs for
/// the WHOLE resident set; otherwise just the dirty shell.
fn emit_gpu_bakes(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
    camera_pos: Vec3,
) {
    let edits_snapshot = Arc::clone(&sched.edits);
    let bvh_snapshot = Arc::clone(&sched.bvh);
    let mut scratch: Vec<u32> = Vec::new();

    // Pass 1: cull + allocate the dirty shell (or drop empty bricks). Collect the live keys
    // so pass 2 can build their jobs once the realloc decision is known.
    let drained = drain_budget(&mut sched.pending, camera_pos, config, SCHEDULE_BUDGET_CHUNKS);
    let mut dirty_live: Vec<atlas::BrickKey> = Vec::new();
    for ck in &drained {
        for key in chunk_brick_keys(*ck, config) {
            if atlas::SdfAtlas::cull_edit_indices(key, &bvh_snapshot, config, &mut scratch).is_some()
            {
                let voxel_size = config.voxel_size_at(key.lod);
                let positions = atlas::SdfAtlas::brick_voxel_positions(key, voxel_size);
                let culled: Vec<edits::ResolvedEdit> =
                    scratch.iter().map(|&i| edits_snapshot[i as usize].clone()).collect();
                let palette = edits::build_palette(&culled, &positions);
                atlas.insert_gpu_brick(key, palette);
                dirty_live.push(key);
            } else {
                atlas.remove_brick(&key);
            }
        }
    }

    // Will the render world recreate the atlas texture this frame? Only when the tile
    // high-water needs more rows than it has committed (a grow → recreate + zero-fill,
    // dropping every GPU-written tile, so we must re-emit the whole resident set).
    //
    // Do NOT also OR in `atlas.last_bake_was_full` here: this function WRITES that flag (as
    // the realloc signal to the render world), so reading it back would be a feedback loop —
    // a single realloc would latch it `true` and re-emit every brick every frame forever.
    // The grow comparison is self-clearing: once `gpu_atlas_rows == required_rows`, the next
    // idle frame sees no grow and emits nothing.
    let required_rows = atlas.tiles.high_water().div_ceil(ATLAS_TILES_PER_ROW).max(1);
    let realloc = required_rows > sched.gpu_atlas_rows;

    // Pass 2: build the jobs. On a realloc, every resident brick; otherwise the dirty shell.
    let bake_keys: Vec<atlas::BrickKey> = if realloc {
        atlas.bricks.keys().copied().collect()
    } else {
        dirty_live
    };

    for key in bake_keys {
        // The tile was allocated in pass 1 (dirty shell) or on a prior frame (already
        // resident); either way it exists. Re-cull for the shader's edit list + palette.
        let Some(tile) = atlas.tiles.tile(&key) else { continue };
        if atlas::SdfAtlas::cull_edit_indices(key, &bvh_snapshot, config, &mut scratch).is_none() {
            continue;
        }
        let voxel_size = config.voxel_size_at(key.lod);
        let dist_band = atlas::dist_band_world(config, key.lod);
        let positions = atlas::SdfAtlas::brick_voxel_positions(key, voxel_size);
        let culled: Vec<edits::ResolvedEdit> =
            scratch.iter().map(|&i| edits_snapshot[i as usize].clone()).collect();
        let palette = edits::build_palette(&culled, &positions);

        let edit_start = gpu_bakes.edits.len() as u32;
        for e in &culled {
            gpu_bakes.edits.push(edits::to_gpu_edit(e));
        }
        gpu_bakes.jobs.push(GpuBakeJob {
            tile,
            lod: key.lod,
            coord: key.coord,
            voxel_size,
            dist_band,
            palette,
            edit_start,
            edit_count: culled.len() as u32,
        });
        atlas.gpu_baked_tiles.insert(tile);
    }

    if realloc {
        // Tell the render world to recreate the texture (then our bake node fills it), and
        // remember the committed height so we don't re-emit the full set every frame.
        atlas.last_bake_was_full = true;
        sched.gpu_atlas_rows = required_rows;
    } else {
        atlas.last_bake_was_full = false;
    }
    gpu_bakes.atlas_rows = required_rows;
}

/// Main-thread, non-blocking drain of finished bake tasks. Inserts baked bricks (per-brick
/// tiles as before) and bumps the generation so the incremental GPU upload picks up the
/// changed tiles. Stale results (superseded edit epoch) are requeued by chunk.
pub fn apply_bakes(
    mut atlas: ResMut<SdfAtlas>,
    mut sched: ResMut<BakeScheduler>,
    config: Res<SdfGridConfig>,
) {
    if sched.inflight.is_empty() {
        return;
    }
    let current_epoch = sched.edit_epoch;
    let lod_count = config.lod_count;
    let r = ring_chunks_per_axis(&config);
    let mut applied = false;
    let mut requeue: Vec<chunk::ChunkKey> = Vec::new();

    let mut i = 0;
    while i < sched.inflight.len() {
        let Some(results) = block_on(poll_once(&mut sched.inflight[i].task)) else {
            i += 1;
            continue;
        };
        let job_epoch = sched.inflight[i].epoch;
        sched.inflight.swap_remove(i);

        if job_epoch != current_epoch {
            // Stale → requeue the affected chunks (deduped) under the current edits.
            for (key, _) in &results {
                requeue.push(chunk::chunk_of(*key, &config).0);
            }
            continue;
        }

        for (key, baked) in results {
            // Skip a brick whose chunk left its LOD window while baking.
            let ck = chunk::chunk_of(key, &config).0;
            let li = ck.lod as usize;
            if li < lod_count as usize {
                let origin = sched.ring_chunk_origin[li];
                if !chunk_in_window(ck.coord, origin, r) {
                    atlas.remove_brick(&key);
                    continue;
                }
            }
            match baked {
                Some(brick) => {
                    atlas.insert_brick(key, brick);
                    applied = true;
                }
                None => {
                    if atlas.remove_brick(&key) {
                        applied = true;
                    }
                }
            }
        }
    }

    for ck in requeue {
        sched.pending.insert(ck);
    }
    if applied {
        atlas.last_bake_was_full = false;
        atlas.bump_generation();
    }
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

    // --- Async-scheduler emulation (no task pool) -----------------------------------
    //
    // Reproduces exactly what `schedule_bakes` + `apply_bakes` do to the atlas, but with
    // tasks completing **out of order and late** — the worst case for the leading-edge
    // staleness bug (a chunk that exits and re-enters a window before its in-flight bake
    // lands). The emulation shares the real window-diff + apply-guard helpers, so a hole
    // here would be a hole in production.

    /// One emulated in-flight bake: the chunk and its per-brick bake results.
    type EmuJob = (chunk::ChunkKey, Vec<(atlas::BrickKey, Option<atlas::PackedBrick>)>);

    struct EmuSched {
        ring_chunk_origin: Vec<IVec3>,
        pending: HashSet<chunk::ChunkKey>,
        /// Jobs queued but not yet applied — drained partially and out of order to model
        /// async lag.
        inflight: std::collections::VecDeque<EmuJob>,
        /// Total chunk enqueues over this scheduler's life (every `pending.insert` from a
        /// recenter). The cull bounds this by the scene's shell footprint; without it, it
        /// grows with the empty volume the camera sweeps.
        enqueued_total: usize,
    }

    impl EmuSched {
        fn new(lod_count: u32) -> Self {
            Self {
                ring_chunk_origin: vec![IVec3::splat(i32::MIN); lod_count as usize],
                pending: HashSet::new(),
                inflight: std::collections::VecDeque::new(),
                enqueued_total: 0,
            }
        }

        /// Mirror `schedule_bakes` step 2 (camera recenter): enqueue entered chunks (culling
        /// empty ones the same way the real path does), evict exited chunks eagerly.
        fn recenter(&mut self, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, bvh: &bvh::Bvh, cam: Vec3) {
            let r = ring_chunks_per_axis(cfg);
            let first = self.ring_chunk_origin.iter().all(|o| *o == IVec3::splat(i32::MIN));
            let mut scratch: Vec<u32> = Vec::new();
            for lod in 0..cfg.lod_count {
                let li = lod as usize;
                let new_origin = ring_chunk_origin(cfg, cam, lod);
                let old_origin = self.ring_chunk_origin[li];
                if new_origin == old_origin {
                    continue;
                }
                for ck in chunk_window_keys(new_origin, r, lod) {
                    let entered = first || !chunk_in_window(ck.coord, old_origin, r);
                    if entered && chunk_has_geometry(ck, bvh, cfg, &mut scratch) && self.pending.insert(ck) {
                        self.enqueued_total += 1;
                    }
                }
                if !first {
                    for ck in chunk_window_keys(old_origin, r, lod) {
                        if !chunk_in_window(ck.coord, new_origin, r) {
                            self.pending.remove(&ck);
                            for bk in chunk_brick_keys(ck, cfg) {
                                atlas.remove_brick(&bk);
                            }
                        }
                    }
                }
                self.ring_chunk_origin[li] = new_origin;
            }
        }

        /// Move up to `budget` pending chunks into the in-flight queue (baking now, but
        /// "delivered" later via `apply`). Order is arbitrary (HashSet) — for tests that
        /// don't exercise the nearest-first priority.
        fn schedule(&mut self, cfg: &SdfGridConfig, edits: &[ResolvedEdit], bvh: &bvh::Bvh, budget: usize) {
            let take: Vec<chunk::ChunkKey> = self.pending.iter().copied().take(budget).collect();
            for ck in take {
                self.pending.remove(&ck);
                let results: Vec<_> = chunk_brick_keys(ck, cfg)
                    .into_iter()
                    .map(|bk| (bk, SdfAtlas::bake_brick(bk, edits, bvh, cfg, &super::super::height::HeightField::default())))
                    .collect();
                self.inflight.push_back((ck, results));
            }
        }

        /// Like `schedule` but drains via the REAL `drain_budget` (nearest-camera-first),
        /// so the starvation regime — empty near chunks crowding out far real geometry — is
        /// reproduced faithfully. This is the priority the live `schedule_bakes` uses.
        fn schedule_nearest(&mut self, cfg: &SdfGridConfig, edits: &[ResolvedEdit], bvh: &bvh::Bvh, cam: Vec3, budget: usize) {
            let take = drain_budget(&mut self.pending, cam, cfg, budget);
            for ck in take {
                let results: Vec<_> = chunk_brick_keys(ck, cfg)
                    .into_iter()
                    .map(|bk| (bk, SdfAtlas::bake_brick(bk, edits, bvh, cfg, &super::super::height::HeightField::default())))
                    .collect();
                self.inflight.push_back((ck, results));
            }
        }

        /// Mirror `apply_bakes`: apply up to `count` in-flight results, popping from the
        /// FRONT or BACK to force out-of-order delivery, with the same window-recheck guard.
        fn apply(&mut self, atlas: &mut SdfAtlas, cfg: &SdfGridConfig, count: usize, from_back: bool) {
            let r = ring_chunks_per_axis(cfg);
            for _ in 0..count {
                let Some((ck, results)) = (if from_back { self.inflight.pop_back() } else { self.inflight.pop_front() }) else {
                    break;
                };
                let li = ck.lod as usize;
                let in_window = li < self.ring_chunk_origin.len()
                    && chunk_in_window(ck.coord, self.ring_chunk_origin[li], r);
                for (key, baked) in results {
                    if !in_window {
                        atlas.remove_brick(&key);
                        continue;
                    }
                    match baked {
                        Some(brick) => atlas.insert_brick(key, brick),
                        None => {
                            atlas.remove_brick(&key);
                        }
                    }
                }
            }
        }

        fn idle(&self) -> bool {
            self.pending.is_empty() && self.inflight.is_empty()
        }
    }

    fn build_bvh(edits: &[ResolvedEdit]) -> bvh::Bvh {
        let aabbs: Vec<Aabb3d> = edits
            .iter()
            .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
            .collect();
        bvh::Bvh::build(&aabbs)
    }

    fn box_edit(pos: Vec3, half: f32, mat: u16) -> ResolvedEdit {
        ResolvedEdit {
            prim: SdfPrimitive::Box { half_extents: Vec3::splat(half) },
            transform: Transform::from_translation(pos),
            op: SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
            material_id: mat,
        }
    }

    /// The headline correctness guarantee for reviving the async path: drive the camera
    /// back and forth across geometry (so chunks repeatedly exit and re-enter windows)
    /// while applying bakes **late and out of order**, then flush. The resident set must
    /// equal a from-scratch `full_bake` at the final camera — no stale leading edge, no
    /// hole. This is the invariant whose absence kept the async path disabled.
    #[test]
    fn async_emulation_converges_under_out_of_order_lag() {
        let cfg = SdfGridConfig { lod_count: 3, ring_bricks: 8, recenter_snap_chunks: 1, ..Default::default() };
        let edits: Vec<ResolvedEdit> = (-6i32..=6).map(|i| box_edit(Vec3::new(i as f32 * 1.2, 0.0, 0.0), 0.4, (i.rem_euclid(3)) as u16)).collect();
        let bvh = build_bvh(&edits);

        let mut atlas = SdfAtlas::default();
        let mut sched = EmuSched::new(cfg.lod_count);

        // First fill at the origin (mirrors first_run: everything pending, then baked).
        let cam0 = Vec3::ZERO;
        sched.recenter(&mut atlas, &cfg, &bvh, cam0);
        // Drain the initial fill fully so we start from a clean resident set.
        while !sched.idle() {
            sched.schedule(&cfg, &edits, &bvh, 3);
            sched.apply(&mut atlas, &cfg, 2, false);
        }

        // A winding path that crosses several brick/chunk boundaries forward and back.
        let path = [2.0f32, 4.0, 1.0, -3.0, -1.0, 5.0, 0.0, 3.0, -4.0, 0.0];
        let mut from_back = false;
        for (i, &x) in path.iter().enumerate() {
            let cam = Vec3::new(x, 0.0, 0.0);
            sched.recenter(&mut atlas, &cfg, &bvh, cam);
            // Deliberately under-drain so tasks lag across recenters, and alternate the
            // apply order to force out-of-order delivery.
            sched.schedule(&cfg, &edits, &bvh, 2);
            sched.apply(&mut atlas, &cfg, 1, from_back);
            from_back = !from_back;
            // No per-step assert: mid-flight the atlas is legitimately incomplete (coarse
            // LOD covers the gap). We assert only after a full flush below.
            let _ = i;
        }

        // Flush everything at the final camera position.
        let final_cam = Vec3::new(*path.last().unwrap(), 0.0, 0.0);
        sched.recenter(&mut atlas, &cfg, &bvh, final_cam);
        let mut guard = 0;
        while !sched.idle() {
            sched.schedule(&cfg, &edits, &bvh, 8);
            sched.apply(&mut atlas, &cfg, 8, from_back);
            from_back = !from_back;
            guard += 1;
            assert!(guard < 1000, "flush did not converge");
        }

        // The payoff: byte-identical to a fresh full_bake at the final camera.
        let mut reference = SdfAtlas::default();
        reference.full_bake(&edits, &bvh, &cfg, &super::super::height::HeightField::default(), final_cam);
        let inc: HashSet<_> = atlas.bricks.keys().copied().collect();
        let refk: HashSet<_> = reference.bricks.keys().copied().collect();
        assert_eq!(inc, refk, "async emulation diverged from full_bake (stale/missing bricks)");
        for (key, rb) in &reference.bricks {
            assert_eq!(atlas.bricks[key].dist, rb.dist, "dist mismatch at {key:?}");
        }
    }

    /// Settle the chunk-windowed scheduler at `cam` from an empty atlas with an unlimited
    /// budget: the canonical "correct resident set for this camera position" (no stale, no
    /// starved). Reference oracle for the fly-in tests — apples-to-apples with the live path
    /// (both chunk-windowed), unlike `full_bake` which uses the brick-level ring and so
    /// differs at window boundaries.
    fn settle_fresh(cfg: &SdfGridConfig, edits: &[ResolvedEdit], bvh: &bvh::Bvh, cam: Vec3) -> HashSet<chunk::ChunkKey> {
        let mut atlas = SdfAtlas::default();
        let mut sched = EmuSched::new(cfg.lod_count);
        sched.recenter(&mut atlas, cfg, bvh, cam);
        let mut guard = 0;
        while !sched.idle() {
            sched.schedule(cfg, edits, bvh, 64);
            sched.apply(&mut atlas, cfg, 64, false);
            guard += 1;
            assert!(guard < 1000, "settle did not converge");
        }
        atlas.bricks.keys().map(|k| chunk::chunk_of(*k, cfg).0).collect()
    }

    /// Flying *away* from a localized scene under a tight per-frame budget must still
    /// refresh the scene's bricks into their new (coarser) LOD rings — the same as flying
    /// *toward* it. Before the empty-chunk cull, backing into empty space enqueued
    /// near-but-empty chunks that won the nearest-first budget every frame and starved the
    /// real geometry entering far rings, so its LOD never updated. With the cull, only
    /// chunks that actually contain geometry consume budget, so flying in (either direction)
    /// converges to the same resident set as a fresh settle at that position — no stale fine
    /// bricks left behind, no coarse bricks starved.
    #[test]
    fn flying_away_still_refreshes_scene_lod() {
        let cfg = SdfGridConfig { lod_count: 4, ring_bricks: 8, recenter_snap_chunks: 1, ..Default::default() };
        // A small scene clustered near the origin (so most entered chunks are empty space).
        let edits: Vec<ResolvedEdit> =
            (-1i32..=1).map(|i| box_edit(Vec3::new(i as f32 * 0.5, 0.0, 0.0), 0.4, 0)).collect();
        let bvh = build_bvh(&edits);

        // Fly `steps` small steps away from the scene, one bounded frame each, draining
        // NEAREST-camera-first (the live priority). Returns (resident chunk set, total
        // chunk enqueues over the whole flight).
        let run = |sign: f32, steps: i32| -> (HashSet<chunk::ChunkKey>, usize) {
            let mut atlas = SdfAtlas::default();
            let mut sched = EmuSched::new(cfg.lod_count);
            sched.recenter(&mut atlas, &cfg, &bvh, Vec3::ZERO);
            while !sched.idle() {
                sched.schedule(&cfg, &edits, &bvh, 64);
                sched.apply(&mut atlas, &cfg, 64, false);
            }
            let baseline_enqueued = sched.enqueued_total; // exclude the initial fill
            let mut from_back = false;
            for i in 1..=steps {
                let cam = Vec3::new(sign * i as f32 * 0.4, 0.0, 0.0);
                sched.recenter(&mut atlas, &cfg, &bvh, cam);
                sched.schedule_nearest(&cfg, &edits, &bvh, cam, 2);
                sched.apply(&mut atlas, &cfg, 2, from_back);
                from_back = !from_back;
            }
            let set = atlas.bricks.keys().map(|k| chunk::chunk_of(*k, &cfg).0).collect();
            (set, sched.enqueued_total - baseline_enqueued)
        };

        // 1) Symmetry + correctness: flying away in either direction leaves the same resident
        //    set as a fresh settle at the destination (no stale fine bricks, nothing missing).
        for (label, sign, steps) in [("forward", 1.0, 16), ("backward", -1.0, 16)] {
            let (flown, _) = run(sign, steps);
            let fresh = settle_fresh(&cfg, &edits, &bvh, Vec3::new(sign * steps as f32 * 0.4, 0.0, 0.0));
            assert_eq!(flown, fresh, "{label}: flew-in resident chunks diverged from a fresh settle");
            assert!(!flown.is_empty(), "{label}: scene vanished after flying away");
        }

        // 2) The cull's core guarantee: total bake enqueues while flying away are bounded by
        //    the scene's shell footprint, NOT the empty volume swept. So flying 4× farther
        //    must NOT enqueue ~4× the chunks. Without the cull every empty leading-edge chunk
        //    is enqueued, the count scales with distance, and the budget starves the scene.
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
        let chunks = chunks_in_aabb(&cfg, &aabb, 0);
        assert!(!chunks.is_empty(), "an edit must dirty at least one chunk");
        let origin_chunk = chunk::chunk_of(atlas::BrickKey::new(0, IVec3::ZERO), &cfg).0;
        assert!(
            chunks.contains(&origin_chunk),
            "the origin's chunk must be in the dirtied set"
        );
        // All returned chunks are at the requested LOD.
        assert!(chunks.iter().all(|c| c.lod == 0));
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

    /// The per-frame budget takes at most `budget` chunks, leaves the rest pending, and
    /// prefers the ones nearest the camera so the visible shell fills first.
    #[test]
    fn drain_budget_is_bounded_and_nearest_first() {
        let cfg = config();
        // A line of LOD-0 chunks marching away from the camera on +X.
        let mut pending: HashSet<chunk::ChunkKey> = HashSet::new();
        for x in 0..100 {
            pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, 0, 0)));
        }
        let camera = chunk::chunk_min_world(chunk::ChunkKey::new(0, IVec3::ZERO), &cfg);

        let budget = 10;
        let taken = drain_budget(&mut pending, camera, &cfg, budget);
        assert_eq!(taken.len(), budget, "must take exactly the budget");
        assert_eq!(pending.len(), 90, "the rest stay pending for next frame");

        // The taken chunks are the 10 nearest (x = 0..=9); none of the far ones.
        let max_x = taken.iter().map(|c| c.coord.x).max().unwrap();
        assert!(max_x < budget as i32, "budget drained far chunks before near ones (max_x={max_x})");

        // Draining repeatedly eventually empties pending (no chunk is dropped).
        let mut total = taken.len();
        while !pending.is_empty() {
            total += drain_budget(&mut pending, camera, &cfg, budget).len();
        }
        assert_eq!(total, 100, "every pending chunk is eventually drained");
    }
}
