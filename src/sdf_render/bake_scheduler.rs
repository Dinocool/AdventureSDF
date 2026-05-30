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

use super::atlas::{self, SdfAtlas};
use super::chunk;
use super::{
    SdfCamera, SdfGridConfig, SdfMaterial, SdfOp, SdfPrimitive, SdfVolume, VolumeQueryData, bvh,
    edits, gather_sorted_edits,
};

/// Last frame's per-edit world AABB, keyed by entity. Lets the scheduler dirty an edit's
/// *former* footprint (not just where it moved to) so vacated chunks get rebuilt/removed.
/// Also serves as the previous entity set for add/remove detection.
#[derive(Resource, Default)]
pub struct PrevEditAabbs {
    map: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d>,
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
    /// Per-LOD chunk-ring origin currently resident (index = lod), in chunk coords. Used
    /// to diff which chunks entered/exited as the camera moves. Empty until first run.
    ring_chunk_origin: Vec<IVec3>,
    /// Chunk keys awaiting a bake (deduped).
    pending: std::collections::HashSet<chunk::ChunkKey>,
    /// Tasks currently baking.
    inflight: Vec<BakeJob>,
}

impl Default for BakeScheduler {
    fn default() -> Self {
        Self {
            edit_epoch: 0,
            edits: Arc::new(Vec::new()),
            bvh: Arc::new(bvh::Bvh::default()),
            ring_chunk_origin: Vec::new(),
            pending: std::collections::HashSet::new(),
            inflight: Vec::new(),
        }
    }
}

/// Max chunks baked per pool task. Each chunk is up to `CHUNK_VOLUME` (64) `bake_brick`
/// calls, so keep the per-task chunk count small to stream results back promptly.
const BAKE_CHUNKS_PER_TASK: usize = 2;

/// Any component that affects an edit's baked result. A change to one of these
/// triggers a targeted rebake of the bricks the edit touches.
type ChangedEdit = Or<(
    Changed<Transform>,
    Changed<SdfOp>,
    Changed<SdfPrimitive>,
    Changed<SdfMaterial>,
)>;

/// The chunk coord (per axis) of the chunk-ring window corner for `camera_pos` at `lod`:
/// the camera's chunk minus half the ring (in chunks) on each axis, so the ring is
/// centred on the camera. `ring_bricks / CHUNK_BRICKS` chunks per axis.
fn ring_chunk_origin(config: &SdfGridConfig, camera_pos: Vec3, lod: u32) -> IVec3 {
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
    let half = (config.ring_bricks / chunk::CHUNK_BRICKS as u32 / 2) as i32;
    cam_chunk - IVec3::splat(half)
}

/// Chunks per axis in a ring window.
fn ring_chunks_per_axis(config: &SdfGridConfig) -> i32 {
    (config.ring_bricks / chunk::CHUNK_BRICKS as u32) as i32
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
    config: Res<SdfGridConfig>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    changed: Query<Entity, (With<SdfVolume>, ChangedEdit)>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
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
    for lod in 0..lod_count {
        let li = lod as usize;
        let new_origin = ring_chunk_origin(&config, camera_pos, lod);
        let old_origin = sched.ring_chunk_origin[li];
        if new_origin == old_origin {
            continue;
        }
        // Entered chunks → enqueue a bake.
        for ck in chunk_window_keys(new_origin, r, lod) {
            if first_run || !chunk_in_window(ck.coord, old_origin, r) {
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

    // --- 3. Spawn async bake tasks (chunk-batched) -----------------------------------
    if sched.pending.is_empty() {
        return;
    }
    let pool = AsyncComputeTaskPool::get();
    let epoch = sched.edit_epoch;
    let drained: Vec<chunk::ChunkKey> = sched.pending.drain().collect();
    for group in drained.chunks(BAKE_CHUNKS_PER_TASK) {
        // Expand the group's chunks into their brick keys for the task.
        let mut keys: Vec<atlas::BrickKey> = Vec::new();
        for ck in group {
            keys.extend(chunk_brick_keys(*ck, &config));
        }
        let edits = Arc::clone(&sched.edits);
        let bvh_snapshot = Arc::clone(&sched.bvh);
        let cfg = config.clone();
        let task = pool.spawn(async move {
            keys.into_iter()
                .map(|key| {
                    let baked = SdfAtlas::bake_brick(key, &edits, &bvh_snapshot, &cfg);
                    (key, baked)
                })
                .collect::<Vec<_>>()
        });
        sched.inflight.push(BakeJob { epoch, task });
    }
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
