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
}

impl Default for BakeScheduler {
    fn default() -> Self {
        Self {
            edits: Arc::new(Vec::new()),
            bvh: Arc::new(bvh::Bvh::default()),
            height: Arc::new(super::height::HeightField::default()),
            ring_chunk_origin: Vec::new(),
            pending: std::collections::HashSet::new(),
        }
    }
}

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
/// (128 MB default) caps us at ~32768 jobs. 16384 leaves headroom (64 MB mat + 32 MB dist)
/// and is a clean 256×64 dispatch grid. A single huge edit can dirty 70k+ bricks; the
/// overflow spills back to `pending` and bakes over the next frames (coarse LOD covers the
/// gap meanwhile — see `emit_gpu_bakes`). Without this cap a giant edit overflows the buffer
/// binding and wgpu aborts the frame.
const GPU_BAKE_JOB_CAP: usize = 16384;

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

/// Whether any edit reaches chunk `ck` (its world AABB overlaps an edit in the BVH). A
/// chunk's world AABB is exactly the union of its `CHUNK_BRICKS³` brick AABBs, so a BVH miss
/// here guarantees *every* brick in the chunk would bake to empty space. Used to skip
/// enqueuing empty entered chunks: they would consume budget producing nothing, starving the
/// real geometry entering far (coarse-LOD) rings — the cause of LOD never refreshing when
/// flying away from the scene. Safe only for camera-*entered* chunks (no resident bricks to
/// evict yet); the edit-dirty path must still enqueue emptied chunks so vacated bricks get
/// removed.
fn chunk_has_geometry(ck: chunk::ChunkKey, bvh: &bvh::Bvh, config: &SdfGridConfig, scratch: &mut Vec<u32>) -> bool {
    let size = chunk::chunk_world_size(ck.lod, config);
    let min = chunk::chunk_min_world(ck, config);
    let aabb = bevy::math::bounding::Aabb3d::from_min_max(min, min + Vec3::splat(size));
    bvh.query_aabb(&aabb, scratch);
    !scratch.is_empty()
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
    config: Res<SdfGridConfig>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    changed: Query<Entity, (With<SdfVolume>, ChangedEdit)>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
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
    // Retention margin (in chunks): we BAKE the inner ring but RETAIN bricks out to
    // `ring + 2·margin`, evicting only when a brick leaves that larger window. Keeping the
    // old bricks resident through a ring shift holds the LOD nesting invariant true *during*
    // the transition — the coarser-LOD fallback (and any rebaked replacement) is always
    // resident, so there's never a 1-frame hole at the LOD boundary. The shader resolves
    // bricks purely by chunk-table presence (not ring membership), so a retained brick stays
    // fully sampleable. `m ≥ recenter_snap_chunks` guarantees survival of one snap crossing.
    let m = config.retention_margin_chunks.max(0);
    let mr = r + 2 * m; // margin window edge length (chunks/axis)
    let mut bvh_scratch: Vec<u32> = Vec::new();
    for lod in 0..lod_count {
        let li = lod as usize;
        let new_origin = ring_chunk_origin(&config, camera_pos, lod);
        let old_origin = sched.ring_chunk_origin[li];
        if new_origin == old_origin {
            continue;
        }
        // Entered chunks → enqueue a bake (INNER ring only; we bake the ring, retain a margin
        // around it). Skip empty ones: a chunk no edit reaches has nothing to bake, and
        // enqueuing it would starve real geometry entering far rings (fly-away LOD-stall bug).
        for ck in chunk_window_keys(new_origin, r, lod) {
            let entered = first_run || !chunk_in_window(ck.coord, old_origin, r);
            if entered && chunk_has_geometry(ck, &bvh, &config, &mut bvh_scratch) {
                sched.pending.insert(ck);
            }
        }
        // Exited chunks → evict ONLY those that left the MARGIN window (not just the inner
        // ring). A brick still inside `ring + margin` is retained — sampleable, not rebaked.
        if !first_run {
            let old_margin = old_origin - IVec3::splat(m);
            let new_margin = new_origin - IVec3::splat(m);
            for ck in chunk_window_keys(old_margin, mr, lod) {
                if !chunk_in_window(ck.coord, new_margin, mr) {
                    sched.pending.remove(&ck);
                    for bk in chunk_brick_keys(ck, &config) {
                        atlas.remove_brick(&bk);
                    }
                }
            }
        }
        sched.ring_chunk_origin[li] = new_origin;
    }

    // --- 3. Emit GPU bake jobs for the dirty chunks ----------------------------------
    // The CPU does only topology (BVH cull + palette + tile alloc) and emits a GpuBakeJob per
    // brick; the compute shader fills the texels.
    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu_bakes, &config);
}

/// Turn this frame's dirty chunks into [`GpuBakeJob`]s: the CPU does only the topology work
/// (BVH cull → which bricks exist + their palette + a stable tile), and the compute shader
/// fills each brick's 512 texels straight into the atlas. No main-thread voxel loop.
fn emit_gpu_bakes(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    gpu_bakes: &mut PendingGpuBakes,
    config: &SdfGridConfig,
) {
    let _span = info_span!("sdf_emit_gpu_bakes").entered();
    let edits_snapshot = Arc::clone(&sched.edits);
    let bvh_snapshot = Arc::clone(&sched.bvh);
    let mut scratch: Vec<u32> = Vec::new();

    // Emit one bake job for `key` from already-culled edit indices `indices` and a known
    // `palette`. No re-cull, no palette rebuild — the caller supplies both. `tile` must be
    // allocated.
    #[expect(clippy::too_many_arguments)]
    fn push_job(
        atlas: &mut SdfAtlas,
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
        atlas.gpu_baked_tiles.insert(tile);
    }

    // Drain ALL pending each frame and EVICT empties immediately (no per-frame eviction
    // budget). When an edit moves it dirties its old∪new footprint; the vacated chunks MUST
    // be evicted the same frame or their stale bricks linger as a trail behind the dragged
    // object (the "parts left behind" artifact). Eviction is CPU-only (`remove_brick`, no GPU
    // job), so it's always cheap and always immediate.
    //
    // BAKE emission, by contrast, is capped: each bake job writes into GPU storage buffers
    // sized `jobs × tile`, and a huge edit (70k+ bricks) would overflow the 128 MB buffer
    // binding and abort the frame. So we emit at most `GPU_BAKE_JOB_CAP` jobs; the rest spill
    // back to `pending` and bake over the next frames.
    //
    // Two invariants keep the cap hole-free:
    //  1. A spilled brick is NOT inserted into the atlas (stays non-resident) — the shader's
    //     chunk lookup misses it and falls back to the coarser LOD, which IS baked. (Inserting
    //     it would expose a resident tile with un-baked zero texels: worse than a hole.)
    //  2. Coarse LODs emit before fine (sort dirty chunks by DESCENDING lod). Coarse rings
    //     have ~8× fewer bricks per level, so they always fit; if the cap is hit it only ever
    //     spills the finest detail, whose coarse fallback is already resident this frame.
    let mut drained: Vec<chunk::ChunkKey> = sched.pending.drain().collect();
    drained.sort_unstable_by_key(|ck| std::cmp::Reverse(ck.lod)); // coarsest (highest lod) first
    let mut spilled: Vec<chunk::ChunkKey> = Vec::new();
    for ck in &drained {
        let mut chunk_spilled = false;
        for key in chunk_brick_keys(*ck, config) {
            if atlas::SdfAtlas::cull_edit_indices(key, &bvh_snapshot, config, &mut scratch).is_some()
            {
                // Over the per-frame job cap → defer this brick's BAKE. Do NOT insert it
                // (must stay non-resident → coarse-LOD fallback). The chunk is re-queued.
                if gpu_bakes.jobs.len() >= GPU_BAKE_JOB_CAP {
                    chunk_spilled = true;
                    continue;
                }
                let voxel_size = config.voxel_size_at(key.lod);
                let samples = atlas::SdfAtlas::brick_palette_samples(key, voxel_size);
                let culled: Vec<edits::ResolvedEdit> =
                    scratch.iter().map(|&i| edits_snapshot[i as usize].clone()).collect();
                let palette = edits::build_palette(&culled, &samples);
                let tile = atlas.insert_gpu_brick(key, palette);
                push_job(
                    atlas, gpu_bakes, &edits_snapshot, config, key, tile, &scratch, palette,
                );
            } else {
                // Empty space → evict immediately (no job, no trail).
                atlas.remove_brick(&key);
            }
        }
        if chunk_spilled {
            spilled.push(*ck);
        }
    }
    // Re-queue spilled chunks for the next frame(s). Their evictions already happened above;
    // only their deferred bakes retry. The atlas grows naturally as the spill drains, and the
    // render world preserves existing texels across the grow (see `prepare_sdf_atlas_gpu`), so
    // no re-emit of the already-baked set is needed.
    for ck in spilled {
        sched.pending.insert(ck);
    }
    atlas.last_bake_was_full = false;
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
        let m = cfg.retention_margin_chunks.max(0);
        let mr = r + 2 * m;
        if sched.ring_chunk_origin.is_empty() {
            sched.ring_chunk_origin = vec![IVec3::splat(i32::MIN); cfg.lod_count as usize];
        }
        let first = sched.ring_chunk_origin.iter().all(|o| *o == IVec3::splat(i32::MIN));
        let mut scratch: Vec<u32> = Vec::new();
        let mut enqueued = 0usize;
        for lod in 0..cfg.lod_count {
            let li = lod as usize;
            let new_origin = ring_chunk_origin(cfg, cam, lod);
            let old_origin = sched.ring_chunk_origin[li];
            if new_origin == old_origin {
                continue;
            }
            for ck in chunk_window_keys(new_origin, r, lod) {
                let entered = first || !chunk_in_window(ck.coord, old_origin, r);
                if entered && chunk_has_geometry(ck, &sched.bvh, cfg, &mut scratch) && sched.pending.insert(ck) {
                    enqueued += 1;
                }
            }
            if !first {
                let old_margin = old_origin - IVec3::splat(m);
                let new_margin = new_origin - IVec3::splat(m);
                for ck in chunk_window_keys(old_margin, mr, lod) {
                    if !chunk_in_window(ck.coord, new_margin, mr) {
                        sched.pending.remove(&ck);
                        for bk in chunk_brick_keys(ck, cfg) {
                            atlas.remove_brick(&bk);
                        }
                    }
                }
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
            emit_gpu_bakes(atlas, sched, &mut gpu, cfg);
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
        ResolvedEdit {
            prim: SdfPrimitive::Box { half_extents: Vec3::splat(half) },
            transform: Transform::from_translation(pos),
            op: SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
            material_id: mat,
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

    /// `emit_gpu_bakes` never emits more than `GPU_BAKE_JOB_CAP` jobs in one frame, and the
    /// dirty chunks whose bricks didn't fit are spilled back to `pending` for the next frame.
    #[test]
    fn gpu_emit_caps_jobs_and_spills_overflow() {
        let cfg = SdfGridConfig { lod_count: 1, ring_bricks: 8, recenter_snap_chunks: 1, ..config() };
        // One big edit covering a wide region so many chunks are dirty and non-empty.
        let edits = vec![box_edit(Vec3::ZERO, 40.0, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();

        // Dirty far more chunks than the cap can bake in one frame: each chunk = 64 bricks,
        // so 512 chunks ≈ 32768 candidate bricks > GPU_BAKE_JOB_CAP (16384).
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    sched.pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                }
            }
        }

        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg);

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
        let edits = vec![box_edit(Vec3::ZERO, 40.0, 0)];
        let mut sched = primed_sched(&edits);
        let mut atlas = SdfAtlas::default();
        let mut gpu = PendingGpuBakes::default();
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    sched.pending.insert(chunk::ChunkKey::new(0, IVec3::new(x, y, z)));
                }
            }
        }

        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg);

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
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg);
            assert!(atlas.bricks.len() >= last, "resident set must not shrink while draining spill");
            last = atlas.bricks.len();
            guard += 1;
            assert!(guard < 100, "spill drain did not converge");
        }
        assert!(atlas.bricks.len() > GPU_BAKE_JOB_CAP, "all dirty bricks eventually resident");
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
        atlas.insert_gpu_brick(far, [edits::PALETTE_EMPTY; edits::PALETTE_K]);
        assert!(atlas.bricks.contains_key(&far));
        sched.pending.insert(far_chunk);
        sched.pending.insert(chunk::ChunkKey::new(0, IVec3::ZERO));

        emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg);

        assert!(
            !atlas.bricks.contains_key(&far),
            "a now-empty brick must be evicted, not left as a trail"
        );
    }

    /// The headline correctness guarantee: drive the camera back and forth across geometry
    /// (so chunks repeatedly exit and re-enter windows) via the real recenter + GPU emit, then
    /// settle at the final camera. The resident set must equal a fresh settle there — no stale
    /// leading edge, no missing bricks. Absolute addressing makes a brick that exits and
    /// re-enters identical, so the walk's resident set must be a SUPERSET of a fresh arrival —
    /// it can never be MISSING a brick a fresh settle has (that would be a hole), though with
    /// the retention margin it may keep extra bricks the camera passed through (sampleable,
    /// not stale — they're valid geometry, just outside the inner ring).
    #[test]
    fn recenter_walk_never_misses_fresh_bricks() {
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

        let missing: Vec<_> = fresh.difference(&walked).collect();
        assert!(missing.is_empty(), "recenter walk is missing {} fresh bricks (holes): {missing:?}", missing.len());
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
                    emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg);
                    guard += 1;
                    assert!(guard < 1000, "frame drain did not converge");
                    if sched.pending.is_empty() { break; }
                }
            }
            let set = atlas.bricks.keys().map(|k| chunk::chunk_of(*k, &cfg).0).collect();
            (set, enqueued)
        };

        // 1) Correctness: flying away either direction leaves a resident set that COVERS a
        //    fresh settle at the destination (nothing missing → no hole). The retention margin
        //    may keep extra bricks the camera passed through, so flown ⊇ fresh (not ==).
        for (label, sign, steps) in [("forward", 1.0, 16), ("backward", -1.0, 16)] {
            let (flown, _) = run(sign, steps);
            let mut fresh_atlas = SdfAtlas::default();
            let mut fresh_sched = primed_sched(&edits);
            let fresh = settle_gpu(&mut fresh_sched, &mut fresh_atlas, &cfg, Vec3::new(sign * steps as f32 * 0.4, 0.0, 0.0));
            let missing: Vec<_> = fresh.difference(&flown).collect();
            assert!(missing.is_empty(), "{label}: flew-in resident chunks missing {} fresh chunks (holes)", missing.len());
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

    /// Retention margin: a small camera move that crosses the inner-ring edge but stays within
    /// `ring + margin` must NOT evict geometry bricks — they're retained (sampleable, not
    /// rebaked) so the LOD nesting holds through the shift. With a generous margin, a one-snap
    /// move drops zero geometry bricks; the same move with margin 0 (legacy) would evict the
    /// trailing shell. This is the fix for the LOD-boundary hole flicker.
    #[test]
    fn retention_margin_keeps_bricks_through_a_ring_shift() {
        // Wide flat slab so every ring shift crosses real geometry at the trailing edge.
        let edits: Vec<ResolvedEdit> = (-8i32..=8)
            .map(|i| box_edit(Vec3::new(i as f32 * 0.6, 0.0, 0.0), 0.4, 0))
            .collect();

        // Count geometry bricks evicted over a small forward move, for a given margin.
        let evicted_over_move = |margin: i32| -> usize {
            let cfg = SdfGridConfig {
                lod_count: 2,
                ring_bricks: 8,
                recenter_snap_chunks: 1,
                retention_margin_chunks: margin,
                ..config()
            };
            let mut atlas = SdfAtlas::default();
            let mut sched = primed_sched(&edits);
            settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::ZERO);
            let before: HashSet<_> = atlas.bricks.keys().copied().collect();
            // Nudge one snap-cell forward (crosses the inner-ring edge once).
            let chunk_world = chunk::chunk_world_size(0, &cfg);
            settle_gpu(&mut sched, &mut atlas, &cfg, Vec3::new(chunk_world, 0.0, 0.0));
            let after: HashSet<_> = atlas.bricks.keys().copied().collect();
            before.difference(&after).count()
        };

        let with_margin = evicted_over_move(cfg_default_margin());
        let no_margin = evicted_over_move(0);
        assert!(
            with_margin < no_margin,
            "retention margin must evict fewer bricks than margin=0 (margin={with_margin}, none={no_margin})"
        );
    }

    fn cfg_default_margin() -> i32 {
        super::super::DEFAULT_RETENTION_MARGIN_CHUNKS
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

}
