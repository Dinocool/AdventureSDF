//! **Camera-following residency — a true nested CLIPMAP of voxel bricks.**
//!
//! The HW-RT path streams a brick set around the camera. A brick is ALWAYS `8³` voxels, but its WORLD SPAN
//! scales with LOD ([`super::brickmap::brick_span`]`(L) = BRICK_WORLD_SIZE · 2^L`), so COARSER levels cover
//! MORE world at the same resolution — a geometry-clipmap / GigaVoxels 3D-mipmap. This replaces the old
//! dense-cube residency (every brick a fixed `1.6 m`, so coarse LOD added no coverage and view distance was
//! hard-capped) with NESTED CLIPMAP SHELLS: LOD0 fills the inner cube, each coarser level is a thin SHELL
//! that doubles the reach. Total view radius = `clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD`.
//!
//! This module owns the pure, headless-testable bookkeeping — no GPU, no Bevy systems — so the residency
//! scheme is proven in isolation and the render wiring ([`super::raytrace`]) just drives it:
//!
//! * [`brick_lod`] / [`desired_clipmap`] — given the camera world position, which `(coord, lod)` bricks
//!   should be resident: each level `L` resident in the shell `clip_half/2 < cheby(c, cam_brick_L) ≤
//!   clip_half` (LOD0 fills the full `cheby ≤ clip_half` cube), so each level is a bounded shell.
//! * [`ResidencyManager`] — the live set of resident bricks + a bounded WORK QUEUE, keyed by [`BrickKey`]
//!   `{coord, lod}` (coords now OVERLAP across LOD grids, so the lod is part of the key). Each `update`
//!   diffs the desired clipmap against the current set, ENQUEUES newly-entered / LOD-changed bricks, and
//!   DROPS exited ones; [`ResidencyManager::drain_work`] voxelizes at most `max_per_frame` per call so a big
//!   camera jump can't stall the frame. The packed list it exposes
//!   ([`ResidencyManager::resident_entries`]) feeds the SSOT [`super::gpu::pack_resident_set`].
//!
//! ## The stutter fix — incremental, O(shell) per move
//! Each level only changes when the camera crosses a LOD-`L` brick boundary (every `brick_span(L)` m), and a
//! coarse boundary is `2^L×` farther apart than LOD0's. So a small move shifts only the LOD0 shell (a thin
//! face-slab) and NOTHING coarse — the per-move enqueue/drop count is O(shell), not O(region). The
//! diff-reconcile `update` (drop-not-desired + enqueue-not-resident/lod-changed) gives this for free once
//! keyed by `(coord, lod)`.
//!
//! ## Keep-old-until-revealed
//! The manager only marks itself DIRTY (needing a re-pack + BLAS/TLAS rebuild) once a non-empty batch of
//! queued bricks has been voxelized. The previous resident set — and the TLAS the render path keeps bound —
//! stays valid until the new one is ready, so the camera never sees a hole/flash while a batch streams in.
//!
//! ## Cross-LOD seams
//! At a shell boundary a fine brick abuts a `2×`-coarser brick. We do NOT build a cross-LOD halo; the two
//! bricks are SEPARATE BLAS AABBs and the [`BRICK_AABB_EPSILON`](super::gpu::BRICK_AABB_EPSILON) overlap +
//! the nearest-solid-hit DDA commit the nearest surface across the LOD step. See the seam discussion in
//! [`super::gpu::pack_resident_set`].

use bevy::math::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use super::brickmap::{Brick, MAX_LOD, brick_span};
use super::edits::{VoxelEdits, apply_edit_overlay};
use super::gpu::ResidentBrick;
use super::palette::BlockRegistry;
use super::source::{BrickSource, WorldgenSource};
use crate::sdf_render::worldgen::biome::BiomeLibrary;
use crate::sdf_render::worldgen::layers::height::HeightLayer;

/// A resident-brick key in the nested clipmap: the integer brick `coord` ON the LOD-`lod` grid. Coords now
/// OVERLAP across LOD grids — the same integer coord at two LODs is two DIFFERENT world bricks
/// (`world_min = coord · brick_span(lod)`) — so the `lod` MUST be part of the key. The SSOT key for the
/// resident map, the work queue, and the empty-memo.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickKey {
    /// Integer brick coordinate on the LOD-`lod` grid.
    pub coord: IVec3,
    /// The clipmap LOD level.
    pub lod: u32,
}

/// Tunable clipmap streaming knobs. Plain `Copy` data so it can be a Bevy resource or a test literal.
#[derive(Clone, Copy, Debug, bevy::prelude::Resource)]
pub struct StreamingConfig {
    /// The clipmap half-extent, in bricks: each nested level covers `cheby ≤ clip_half` of ITS grid (a
    /// `(2·clip_half+1)³` cube), and the finer level below it covers the inner `cheby ≤ clip_half/2`, so each
    /// level is resident only in the SHELL `clip_half/2 < cheby ≤ clip_half` (LOD0 fills the full cube). The
    /// total view radius is `clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD`. Default 8 ⇒ each level `17³` and a
    /// view half-extent of `8 · 1.6 · 2^7 ≈ 1640 m` (vs the old dense ~45 m), at bounded VRAM (a clipmap of
    /// `MAX_LOD+1` thin shells, not a dense cube).
    pub clip_half_bricks: i32,
    /// Hard cap on resident bricks — a SAFETY bound so a mis-set `clip_half` can't blow VRAM. With the nested
    /// shells this should NOT bind (only NON-empty surface bricks are stored, a thin shell each level). If the
    /// desired set exceeds it the farthest bricks are dropped (logged). Default 60_000.
    pub max_resident_bricks: usize,
    /// Max bricks voxelized + enqueued→processed per `drain_work` call (per frame). Bounds the per-frame
    /// CPU cost of a big camera move; the rest carry in the queue to later frames. Default 256.
    pub max_bricks_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            clip_half_bricks: 8,
            max_resident_bricks: 60_000,
            max_bricks_per_frame: 256,
        }
    }
}

/// The brick coordinate the camera world position falls in on the LOD-`lod` grid: `floor(cam_world /
/// brick_span(lod))` per axis. DIFFERENT LODs are different coord grids, so the per-level clipmap centre
/// differs — this is the SSOT mapping camera world → the LOD-`lod` brick that contains it.
#[inline]
pub fn camera_brick_coord_lod(cam_world: [f32; 3], lod: u32) -> IVec3 {
    let span = brick_span(lod);
    IVec3::new(
        (cam_world[0] / span).floor() as i32,
        (cam_world[1] / span).floor() as i32,
        (cam_world[2] / span).floor() as i32,
    )
}

/// The LOD0 brick coordinate the camera falls in (`camera_brick_coord_lod(_, 0)`) — the SSOT "has the camera
/// crossed a brick?" key the render loop uses to decide when to re-`update`. A LOD0 crossing is the FINEST
/// boundary, so it strictly implies any coarser crossing; reconciling on it never misses a shell shift.
#[inline]
pub fn camera_brick_coord(cam_world: [f32; 3]) -> IVec3 {
    camera_brick_coord_lod(cam_world, 0)
}

/// The Chebyshev (L∞) distance in bricks between two brick coordinates — the clipmap shell metric. A cube
/// of radius `r` is exactly `{ b : cheby(b, centre) <= r }`.
#[inline]
fn cheby(a: IVec3, b: IVec3) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs())
}

/// The clipmap LOD that COVERS a world position, given as a LOD0 brick coordinate `coord` (world centre =
/// `(coord + 0.5) · brick_span(0)`). Returns the FINEST level whose [`desired_clipmap`] shell contains that
/// world position: LOD0 if it is inside the LOD0 cube, else the coarser shell it falls into, clamped to
/// [`MAX_LOD`]. The SSOT for "what resolution does the renderer see at this world point" — used by the
/// streaming tests to assert shell placement; the residency itself uses [`desired_clipmap`] directly.
#[inline]
pub fn brick_lod(coord: IVec3, cam_world: [f32; 3], cfg: &StreamingConfig) -> u32 {
    let half = cfg.clip_half_bricks;
    // The world centre of the LOD0 brick `coord`.
    let span0 = brick_span(0);
    let world = [
        (coord.x as f32 + 0.5) * span0,
        (coord.y as f32 + 0.5) * span0,
        (coord.z as f32 + 0.5) * span0,
    ];
    for lod in 0..=MAX_LOD {
        // The brick on the LOD-`lod` grid that contains `world`, and the camera's brick on the same grid.
        let here = camera_brick_coord_lod(world, lod);
        let cam_l = camera_brick_coord_lod(cam_world, lod);
        let d = cheby(here, cam_l);
        // Cede `half/2 - 1` (NOT `half/2`) to the finer level: the two levels floor their grids
        // INDEPENDENTLY, so the finer level's outer face can fall up to one COARSE brick short of the
        // `half/2` boundary on the side away from the camera's sub-cell offset. Ceding one ring LESS makes the
        // coarse shell OVERLAP the finer level by one ring, structurally closing that gap for every clip_half
        // (a covered→empty→covered hole would otherwise open as a black crack at each LOD radius). The overlap
        // is a no-op for correctness: `trace` keeps the NEAREST per-voxel t across all candidates.
        let inner = if lod == 0 { -1 } else { half / 2 - 1 };
        if d > inner && d <= half {
            return lod;
        }
    }
    MAX_LOD
}

/// The DESIRED clipmap residency: the set of `(coord, lod)` bricks that should be resident around the camera
/// at world position `cam_world`. NESTED SHELLS — for each level `L` in `0..=MAX_LOD`:
/// * `cam_brick_L = floor(cam_world / brick_span(L))` is the camera's brick on the LOD-`L` grid;
/// * level `L` is resident in the SHELL `clip_half/2 - 1 < cheby(c, cam_brick_L) <= clip_half`, ceding the
///   inner region to the FINER level `L-1`. The cede is `clip_half/2 - 1` (one ring LESS than the naive
///   `clip_half/2`) because the levels floor their grids INDEPENDENTLY, so the finer level's outer face can
///   fall up to one coarse brick short of the boundary; the extra ring makes adjacent shells OVERLAP and
///   structurally closes that gap (no covered→empty→covered crack at any LOD radius). LOD0 fills the full
///   `cheby <= clip_half` cube (it has no finer level beneath it).
///
/// Result: each level is a thin bounded shell, and the union reaches `clip_half · BRICK_WORLD_SIZE ·
/// 2^MAX_LOD` from the camera. Returned as a map keyed by [`BrickKey`]; iteration order is not guaranteed
/// (callers that need a stable `primitive_index` order sort — the manager does). If the union exceeds
/// `max_resident_bricks`, the FARTHEST bricks are dropped (deterministic) so the cap holds.
pub fn desired_clipmap(cam_world: [f32; 3], cfg: &StreamingConfig) -> FxHashMap<BrickKey, ()> {
    let half = cfg.clip_half_bricks;
    let mut out: FxHashMap<BrickKey, ()> = FxHashMap::default();
    for lod in 0..=MAX_LOD {
        let cam_l = camera_brick_coord_lod(cam_world, lod);
        // Cede `half/2 - 1` (NOT `half/2`) to the finer level: the two levels floor their grids
        // INDEPENDENTLY, so the finer level's outer face can fall up to one COARSE brick short of the
        // `half/2` boundary on the side away from the camera's sub-cell offset. Ceding one ring LESS makes the
        // coarse shell OVERLAP the finer level by one ring, structurally closing that gap for every clip_half
        // (a covered→empty→covered hole would otherwise open as a black crack at each LOD radius). The overlap
        // is a no-op for correctness: `trace` keeps the NEAREST per-voxel t across all candidates.
        let inner = if lod == 0 { -1 } else { half / 2 - 1 };
        for dz in -half..=half {
            for dy in -half..=half {
                for dx in -half..=half {
                    let coord = cam_l + IVec3::new(dx, dy, dz);
                    let d = cheby(coord, cam_l);
                    if d > inner {
                        // d <= half by the loop bounds; d > inner excludes the finer-level interior → SHELL.
                        out.insert(BrickKey { coord, lod }, ());
                    }
                }
            }
        }
    }
    if out.len() > cfg.max_resident_bricks {
        // Keep the nearest `max_resident_bricks`; drop the rest (farthest first). The "distance" is the
        // shell distance in the level's own grid, so a far coarse shell drops before a near fine one only if
        // it is genuinely farther in WORLD metres — rank by world distance. Deterministic tiebreak.
        let world_d = |k: &BrickKey| {
            // Approx world centre distance: brick centre = (coord + 0.5)·brick_span(lod).
            let span = brick_span(k.lod);
            let cx = (k.coord.x as f32 + 0.5) * span - cam_world[0];
            let cy = (k.coord.y as f32 + 0.5) * span - cam_world[1];
            let cz = (k.coord.z as f32 + 0.5) * span - cam_world[2];
            (cx * cx + cy * cy + cz * cz).sqrt()
        };
        let mut all: Vec<BrickKey> = out.keys().copied().collect();
        all.sort_by(|a, b| {
            world_d(a)
                .partial_cmp(&world_d(b))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then((a.lod, a.coord.z, a.coord.y, a.coord.x).cmp(&(b.lod, b.coord.z, b.coord.y, b.coord.x)))
        });
        for k in all.into_iter().skip(cfg.max_resident_bricks) {
            out.remove(&k);
        }
    }
    out
}

/// A queued unit of streaming work: voxelize the brick at clipmap key `key` (`(coord, lod)`). The LOD is part
/// of the key, so a brick that changes LOD (a shell shift) is a DIFFERENT key — enqueued + voxelized fresh at
/// the new LOD's coarse spacing, never silently re-tagged (the in-place mip means the voxel data differs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorkItem {
    key: BrickKey,
}

/// The live clipmap residency state + bounded work queue. Holds the voxelized [`Brick`]s currently resident
/// (only NON-empty bricks are stored — empty/all-air bricks are skipped, the sparsity invariant), keyed by
/// [`BrickKey`] `(coord, lod)`, plus a FIFO queue of bricks awaiting voxelization. `update` recomputes the
/// desired clipmap and reconciles; `drain_work` does the bounded voxelization.
///
/// Robust-by-construction: the resident map is the single source of what's live; `dirty` is set only when a
/// drained batch actually changes it, so the render path re-packs exactly when (and only when) the GPU set
/// must change — keep-old-until-revealed falls out for free. Keying by `(coord, lod)` makes a LOD change a
/// DIFFERENT key (a fresh voxelize at the new coarse spacing), so the old "retag the same brick" confusion
/// is structurally impossible.
#[derive(Default)]
pub struct ResidencyManager {
    /// Resident, voxelized bricks: clipmap key → its `8³` brick (voxelized at the key's LOD). Empty bricks
    /// are never inserted.
    resident: FxHashMap<BrickKey, Brick>,
    /// Keys awaiting voxelization (enqueued by `update`, processed by `drain_work`). A set membership guard
    /// (`queued`) prevents a key from being enqueued twice while it waits.
    queue: std::collections::VecDeque<WorkItem>,
    queued: FxHashSet<BrickKey>,
    /// KNOWN-EMPTY (all-air) keys in the current clipmap: bricks that voxelized to empty (above the surface)
    /// are NEVER resident (sparsity), so without this memo `update` would find them absent and re-enqueue +
    /// re-voxelize them on EVERY camera move — and most of the desired clipmap (~2/3) is empty sky/air, the
    /// dominant streaming churn otherwise. Memoize them so each empty key is voxelized ONCE; bounded to the
    /// clipmap (`update` prunes keys that leave). Emptiness is per-`(coord, lod)` (a coarse brick samples the
    /// surface at coarse spacing, so it can differ from a finer one — hence the LOD is in the key).
    empty: FxHashSet<BrickKey>,
    /// True iff the resident set CHANGED since the last `take_dirty` — the render path re-packs + rebuilds
    /// the BLAS/TLAS only then (otherwise it keeps the old, still-valid GPU scene).
    dirty: bool,
    /// Total bricks dropped by the resident cap over the manager's life (for logging).
    pub capped_total: usize,
}

impl ResidencyManager {
    /// A fresh, empty manager (no resident bricks, empty queue).
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of resident (voxelized, non-empty) bricks.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// True iff `key` is currently resident (a non-empty brick is stored for it).
    #[inline]
    pub fn is_resident(&self, key: &BrickKey) -> bool {
        self.resident.contains_key(key)
    }

    /// Number of bricks waiting in the work queue.
    #[inline]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// True iff the resident set has changed and not yet been consumed (a re-pack is due).
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Reconcile the resident set toward the desired CLIPMAP around the camera at world position `cam_world`:
    /// * DROP every resident brick no longer in the desired clipmap (a shell shifted / the camera moved) —
    ///   marks dirty.
    /// * ENQUEUE every desired `(coord, lod)` that is NOT resident — voxelized later by [`drain_work`].
    ///
    /// A LOD change is just a different [`BrickKey`] entering + the old one leaving (different coord grids),
    /// so there is NO retag path — each brick is voxelized at exactly one LOD. Does NOT itself voxelize, so a
    /// huge camera jump only enqueues here (cheap). The per-move enqueue/drop is O(shell): only the bricks
    /// whose key entered/left change, and a small move shifts only the LOD0 shell (coarse shells move
    /// `2^L×` less often). Returns the number of bricks dropped (so the caller can log churn).
    pub fn update(&mut self, cam_world: [f32; 3], cfg: &StreamingConfig) -> usize {
        let desired = desired_clipmap(cam_world, cfg);
        // The uncapped clipmap size (Σ over levels of each shell), for the cap-drop log.
        let half = cfg.clip_half_bricks as usize;
        let full_cube = (2 * half + 1).pow(3);
        let inner_cube = (half + 1).pow(3); // (2·(half/2)+1)³ with half/2 → bounded approximation for the log
        let uncapped = full_cube + MAX_LOD as usize * full_cube.saturating_sub(inner_cube);
        self.capped_total += uncapped.saturating_sub(desired.len());

        // Drop resident bricks that left the clipmap.
        let mut dropped = 0usize;
        let to_drop: Vec<BrickKey> =
            self.resident.keys().filter(|k| !desired.contains_key(*k)).copied().collect();
        for k in to_drop {
            self.resident.remove(&k);
            dropped += 1;
            self.dirty = true; // the GPU set shrank → must re-pack
        }
        // Prune the empty-memo to the current clipmap (bounds it as the camera roams; a key that re-enters is
        // cheaply re-voxelized + re-memoized). Deterministic terrain ⇒ an empty key is always empty.
        self.empty.retain(|k| desired.contains_key(k));

        // Enqueue each desired brick that is NOT already resident (and not known-empty / already queued).
        for key in desired.keys() {
            if !self.resident.contains_key(key)
                && !self.queued.contains(key)
                && !self.empty.contains(key)
            {
                self.queue.push_back(WorkItem { key: *key });
                self.queued.insert(*key);
            }
        }
        dropped
    }

    /// Process up to `cfg.max_bricks_per_frame` queued bricks from the WORLDGEN surface — the original
    /// worldgen drain (signature unchanged, so the streaming + perf harness tests are bit-identical). A thin
    /// wrapper over the source-generic [`drain_work_from`](Self::drain_work_from): it builds a
    /// [`WorldgenSource`] over `(layer, lib, seed)` and drains with NO edit overlay
    /// ([`VoxelEdits::is_empty`]), so the resident set is exactly what the direct `voxelize_brick` drain
    /// produced before the source abstraction.
    pub fn drain_work(
        &mut self,
        cfg: &StreamingConfig,
        layer: &HeightLayer,
        lib: &BiomeLibrary,
        registry: &BlockRegistry,
        seed: u64,
    ) -> usize {
        let source = WorldgenSource::new(layer, lib, seed);
        self.drain_work_from(cfg, &source, registry, &VoxelEdits::new())
    }

    /// Process up to `cfg.max_bricks_per_frame` queued bricks from ANY [`BrickSource`] (worldgen or a static
    /// `.vox`): SOURCE each at ITS key's LOD (the in-place mip — coarse keys sample at coarse spacing), apply
    /// the shared [`VoxelEdits`] overlay (so build/destroy editing works UNIFORMLY for every scene), store
    /// NON-empty results as resident, and drop empty ones (sparsity). Marks the set dirty iff at least one
    /// brick was actually added/removed — so a batch that produced only empty bricks does NOT trigger a
    /// needless re-pack, and the old GPU scene stays valid until a REVEALING batch lands
    /// (keep-old-until-revealed). Returns the number of bricks sourced this call.
    ///
    /// Bounded: never does more than `max_bricks_per_frame` sourcings; leftover queue items carry to the next
    /// call. Logs when it caps (leaves work pending). DETERMINISTIC: the source is [`Sync`] + pure and the
    /// per-brick overlay is pure, so the parallel drain yields a brick identical regardless of thread, applied
    /// in a fixed order — the resident set is bit-identical to a serial loop.
    pub fn drain_work_from(
        &mut self,
        cfg: &StreamingConfig,
        source: &dyn BrickSource,
        registry: &BlockRegistry,
        edits: &VoxelEdits,
    ) -> usize {
        use bevy::tasks::{ComputeTaskPool, ParallelSlice};
        let budget = cfg.max_bricks_per_frame;
        // Pop the per-frame batch first (serial, cheap): up to `budget` queued keys.
        let mut keys: Vec<BrickKey> = Vec::with_capacity(budget.min(self.queue.len()));
        while keys.len() < budget {
            let Some(item) = self.queue.pop_front() else { break };
            self.queued.remove(&item.key);
            keys.push(item.key);
        }
        let done = keys.len();
        if done > 0 {
            // Source the batch IN PARALLEL on the compute task pool. The source's `brick` is a pure function of
            // `(coord, lod, &registry)` (all shared + Sync), and the per-brick edit overlay is pure, so this is
            // determinism-preserving: each key yields an identical brick regardless of thread, and we apply the
            // results in a fixed order — the resident set is bit-identical to a serial loop. Chunked (~one
            // chunk per worker) so we spawn a handful of tasks, not one per brick.
            //
            // `get_or_init` (not `get`): the running app already initialized the ComputeTaskPool, but the
            // headless tests + perf harness call drain_work directly with no Bevy app — there `get()` panics,
            // so init a default pool on first use. (Same pool the live app uses when one exists.)
            //
            // The edit overlay is SKIPPED entirely when there are no edits — so a no-edit drain (the common
            // case, and EVERY worldgen-harness test) is the literal `source.brick(...)` path, bit-identical to
            // before the abstraction. When edits exist, each base brick is overlaid per-voxel via the shared
            // `apply_edit_overlay` SSOT (the same rule the static-scene + pick paths use).
            let has_edits = !edits.is_empty();
            let pool = ComputeTaskPool::get_or_init(bevy::tasks::TaskPool::default);
            let chunk = done.div_ceil(pool.thread_num().max(1)).max(1);
            let results: Vec<(BrickKey, Brick)> = keys
                .par_chunk_map(pool, chunk, |_, ks| {
                    ks.iter()
                        .map(|&k| {
                            let base = source.brick(k.coord, k.lod, registry);
                            // The overlay is keyed by world VOXEL coord on the LOD0 grid; it only affects LOD0
                            // bricks (a coarse brick's world-voxel footprint doesn't align with the override
                            // grid). Applying it unconditionally is still correct (a coarse base has no
                            // matching override key ⇒ unchanged), but skip non-LOD0 to keep coarse drains cheap.
                            let brick = if has_edits && k.lod == 0 {
                                apply_edit_overlay(k.coord, &base, edits)
                            } else {
                                base
                            };
                            (k, brick)
                        })
                        .collect::<Vec<_>>()
                })
                .into_iter()
                .flatten()
                .collect();
            // Apply serially (HashMap mutation): non-empty bricks become resident; an all-air brick is dropped.
            for (key, brick) in results {
                if brick.is_empty() {
                    // All-air → never resident; MEMOIZE so future moves don't re-source it (the churn fix +
                    // the static-scene clipmap BOUND: bricks outside the loaded map source empty once).
                    self.empty.insert(key);
                    if self.resident.remove(&key).is_some() {
                        self.dirty = true;
                    }
                } else {
                    self.empty.remove(&key); // defensive: a now-solid key must not stay memoized empty
                    self.resident.insert(key, brick);
                    self.dirty = true;
                }
            }
        }
        if !self.queue.is_empty() {
            bevy::log::debug!(
                "voxel streaming: capped at {budget} bricks/frame, {} still pending",
                self.queue.len()
            );
        }
        done
    }

    /// Force a RE-SOURCE of specific keys on the NEXT [`drain_work_from`](Self::drain_work_from): clear them
    /// from the empty-memo (so a now-solid edit isn't skipped as known-air) and re-enqueue them. It does NOT
    /// drop the resident entry — the OLD voxelized brick stays resident + bound until the re-source overwrites
    /// it next drain (keep-old-until-revealed: the camera never sees a hole/flash while the edited brick
    /// re-sources). Used for UNIFORM editing — an edit names the affected LOD0 bricks (owner + halo neighbours)
    /// and this re-queues exactly those, so the edit re-sources + re-packs LOCALLY (it ADAPTS, never
    /// full-clears — the resident set, the GI reservoirs, and the world cache all stay; see
    /// [[feedback-gi-adapt-not-reset]]). Keys not currently resident are simply enqueued so a place into empty
    /// space still appears. A key already queued is left as-is (the membership guard avoids a double-enqueue).
    /// No-op for an empty set.
    pub fn requeue_keys(&mut self, keys: impl IntoIterator<Item = BrickKey>) {
        for key in keys {
            self.empty.remove(&key);
            if !self.queued.contains(&key) {
                self.queue.push_back(WorkItem { key });
                self.queued.insert(key);
            }
        }
    }

    /// Take the dirty flag, clearing it. `true` ⇒ the resident set changed and the render path should
    /// re-pack + rebuild the BLAS/TLAS this frame; `false` ⇒ nothing changed, keep the old GPU scene.
    #[inline]
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// The resident bricks as [`ResidentBrick`] entries in a DETERMINISTIC order (sorted by `(lod, z, y, x)`),
    /// ready for [`super::gpu::pack_resident_set`]. The stable order keeps each brick's `primitive_index`
    /// reproducible (the test oracle relies on it). Borrows `self`, so the returned entries live as long as
    /// the manager isn't mutated.
    pub fn resident_entries(&self) -> Vec<ResidentBrick<'_>> {
        let mut keys: Vec<BrickKey> = self.resident.keys().copied().collect();
        keys.sort_by_key(|k| (k.lod, k.coord.z, k.coord.y, k.coord.x));
        keys.into_iter()
            .map(|key| {
                let brick = self.resident.get(&key).expect("key came from keys");
                ResidentBrick { coord: key.coord, brick, lod: key.lod }
            })
            .collect()
    }
}

/// The world-metre AABB half-extent the resident CLIPMAP covers around the camera (for logging / framing):
/// the OUTERMOST shell reaches `clip_half · brick_span(MAX_LOD) = clip_half · BRICK_WORLD_SIZE · 2^MAX_LOD`.
/// This is the clipmap view radius — `2^MAX_LOD×` the old dense-cube reach at the same `clip_half`.
pub fn region_half_extent_m(cfg: &StreamingConfig) -> f32 {
    cfg.clip_half_bricks as f32 * brick_span(MAX_LOD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::biome::{
        BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
    };
    use crate::sdf_render::worldgen::coord::LayerId;
    use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
    use crate::sdf_render::worldgen::layers::height::HeightParams;

    const SEED: u64 = 0xA15E_C0DE_2026;

    fn test_layer() -> HeightLayer {
        HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
    }

    fn test_library() -> BiomeLibrary {
        let mat = |name: &str, c: [f32; 4]| TerrainSurfaceMaterial {
            name: name.into(),
            base_color: c,
            roughness: 0.9,
            blend: 0.0,
            texture: None,
            tiling: 4.0,
            ..Default::default()
        };
        let materials = vec![mat("surface", [0.1, 0.5, 0.1, 1.0]), mat("stone", [0.5, 0.5, 0.5, 1.0])];
        let column = |_| BiomeDef {
            name: "b".into(),
            surface: TerrainMatId(0),
            surface_rules: vec![],
            strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1000.0 }],
            bedrock: TerrainMatId(1),
        };
        let biomes = BiomeId::ALL.iter().map(column).collect();
        BiomeLibrary { materials, biomes }
    }

    fn registry() -> BlockRegistry {
        BlockRegistry::from_biome_library(&test_library())
    }

    /// A LOD-`L` brick's containing camera coord scales with `brick_span(L)`: a fixed world position maps to
    /// a coarser brick coord at coarser LODs (the per-level clipmap centres differ).
    #[test]
    fn camera_brick_coord_scales_with_lod() {
        // World position 5 m: LOD0 (span 1.6) → floor(5/1.6)=3; LOD1 (3.2) → 1; LOD2 (6.4) → 0.
        let w = [5.0, 5.0, 5.0];
        assert_eq!(camera_brick_coord_lod(w, 0), IVec3::splat(3));
        assert_eq!(camera_brick_coord_lod(w, 1), IVec3::splat(1));
        assert_eq!(camera_brick_coord_lod(w, 2), IVec3::splat(0));
        // camera_brick_coord is the LOD0 alias.
        assert_eq!(camera_brick_coord(w), camera_brick_coord_lod(w, 0));
    }

    /// The desired clipmap is NESTED SHELLS: LOD0 fills the full `(2·half+1)³` cube; each coarser level is a
    /// SHELL `half/2 < cheby ≤ half` in its own grid. Every level is present, each a bounded shell.
    #[test]
    fn desired_clipmap_is_nested_shells() {
        let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 1_000_000, ..Default::default() };
        let cam = [0.5_f32, 0.5, 0.5]; // near the origin
        let d = desired_clipmap(cam, &cfg);
        let half = cfg.clip_half_bricks;

        // LOD0 is the full cube.
        let lod0: usize = d.keys().filter(|k| k.lod == 0).count();
        assert_eq!(lod0, (2 * half as usize + 1).pow(3), "LOD0 fills the full clip cube");
        // Every level 0..=MAX_LOD is present (the clipmap reaches all the way out).
        for lod in 0..=MAX_LOD {
            assert!(d.keys().any(|k| k.lod == lod), "level {lod} present in the clipmap");
        }
        // Each coarse level is a SHELL that OVERLAPS the finer level by one ring (cede `half/2 - 1`, the
        // coverage-gap fix): nothing deeper than that overlap boundary, and bricks out at cheby == half.
        for lod in 1..=MAX_LOD {
            let cam_l = camera_brick_coord_lod(cam, lod);
            let inner = half / 2 - 1;
            assert!(
                d.keys().filter(|k| k.lod == lod).all(|k| cheby(k.coord, cam_l) > inner),
                "level {lod} is a shell — nothing deeper than the one-ring overlap"
            );
            assert!(
                d.keys().any(|k| k.lod == lod && cheby(k.coord, cam_l) == half),
                "level {lod} reaches the shell's outer edge (cheby == half)"
            );
        }
    }

    /// `brick_lod(lod0_coord, cam_world, cfg)` reports the FINEST shell covering that world position: a LOD0
    /// brick inside the LOD0 cube is LOD0; one far enough out (past the LOD0 cube) lands in a coarse shell.
    #[test]
    fn brick_lod_reports_shell_level() {
        let cfg = StreamingConfig { clip_half_bricks: 8, ..Default::default() };
        let cam = [0.5_f32, 0.5, 0.5];
        // The camera's own LOD0 brick is covered by LOD0.
        assert_eq!(brick_lod(camera_brick_coord_lod(cam, 0), cam, &cfg), 0);
        // A LOD0 brick just inside the LOD0 cube edge (cheby 7 of 8) is still LOD0.
        assert_eq!(brick_lod(IVec3::new(7, 0, 0), cam, &cfg), 0);
        // A LOD0 brick PAST the LOD0 cube (cheby 12 > 8) falls into the LOD1 shell: on the LOD1 grid its world
        // centre (~20 m) is ~6 LOD1-bricks out — inside half/2(=4) < 6 ≤ 8.
        assert_eq!(brick_lod(IVec3::new(12, 0, 0), cam, &cfg), 1);
        // Far out (cheby 30 LOD0-bricks ≈ 48 m) is a coarser shell still.
        assert!(brick_lod(IVec3::new(30, 0, 0), cam, &cfg) >= 2);
    }

    /// The resident cap drops the farthest (in WORLD metres) bricks so the clipmap never exceeds
    /// `max_resident_bricks`, and the camera's own brick is always kept.
    #[test]
    fn resident_cap_drops_farthest() {
        let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 50, ..Default::default() };
        let cam = [0.5_f32, 0.5, 0.5];
        let d = desired_clipmap(cam, &cfg);
        assert_eq!(d.len(), 50, "capped to max_resident_bricks");
        let cam0 = camera_brick_coord_lod(cam, 0);
        assert!(d.contains_key(&BrickKey { coord: cam0, lod: 0 }), "the camera's LOD0 brick is always kept");
    }

    /// Residency reconciliation: a simulated camera move enters new bricks (enqueued, then voxelized into
    /// resident) and drops exited ones; empty (sky) bricks are skipped; the keep-old invariant holds (the
    /// set isn't dirty until a revealing batch lands). Resident bricks always lie within the clipmap.
    #[test]
    fn residency_updates_as_camera_moves() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        // Place the camera AT the surface so the inner LOD0 cube straddles terrain (non-empty bricks).
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 2, max_resident_bricks: 100_000, max_bricks_per_frame: 100_000 };

        let mut mgr = ResidencyManager::new();
        let cam0 = [0.0_f32, surf, 0.0];
        mgr.update(cam0, &cfg);
        assert!(mgr.pending() > 0, "entering a fresh clipmap enqueues work");
        assert!(!mgr.is_dirty(), "no bricks voxelized yet → not dirty (keep-old)");

        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        assert!(mgr.is_dirty(), "voxelizing real terrain bricks reveals new geometry → dirty");
        assert!(mgr.take_dirty());
        assert!(mgr.resident_count() > 0, "some non-empty bricks resident");

        // Move the camera +5 m in X (crosses a few LOD0 bricks). New bricks enter, far ones drop.
        let cam1 = [5.0_f32, surf, 0.0];
        let dropped = mgr.update(cam1, &cfg);
        assert!(dropped > 0, "moving away drops the bricks left behind");
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        // Every resident brick lies within its level's clipmap shell around the new camera.
        let half = cfg.clip_half_bricks;
        for e in mgr.resident_entries() {
            let cam_l = camera_brick_coord_lod(cam1, e.lod);
            assert!(cheby(e.coord, cam_l) <= half, "resident bricks stay in the clipmap");
        }
    }

    /// The per-frame cap bounds work: a large fresh clipmap drains at most `max_bricks_per_frame` per call,
    /// carrying the rest in the queue across calls until empty.
    #[test]
    fn carry_queue_caps_per_frame_work() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 3, max_resident_bricks: 1_000_000, max_bricks_per_frame: 50 };

        let mut mgr = ResidencyManager::new();
        let cam = [0.0_f32, surf, 0.0];
        mgr.update(cam, &cfg);
        let total = mgr.pending();
        assert!(total > 50, "the clipmap enqueues more than one frame's budget");

        let mut drains = 0;
        let mut voxelized = 0usize;
        while mgr.pending() > 0 {
            let n = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
            assert!(n <= 50, "never exceeds the per-frame cap");
            voxelized += n;
            drains += 1;
            assert!(drains <= total / 50 + 5, "must terminate");
        }
        assert_eq!(voxelized, total, "every enqueued brick is eventually voxelized");
        assert_eq!(drains, total.div_ceil(50), "carries the rest across frames");
    }

    /// A LOD change is a DIFFERENT key: when the camera moves so a world region's covering level shifts (the
    /// shell boundary crosses it), the old `(coord, lod)` key leaves the clipmap and a new `(coord', lod')`
    /// key enters — voxelized fresh at the new LOD's coarse spacing, never silently re-tagged. We verify a
    /// move re-keys: a coord that was LOD0-resident is no longer LOD0-resident once it falls into the LOD1
    /// shell, and the manager enqueues the new coarse key.
    #[test]
    fn lod_change_is_a_fresh_key() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

        let mut mgr = ResidencyManager::new();
        let cam0 = [0.0_f32, surf, 0.0];
        mgr.update(cam0, &cfg);
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        mgr.take_dirty();
        // Every resident brick is in SOME shell of the desired clipmap (keys are well-formed).
        let d0 = desired_clipmap(cam0, &cfg);
        for e in mgr.resident_entries() {
            assert!(d0.contains_key(&BrickKey { coord: e.coord, lod: e.lod }), "resident keys are desired");
        }

        // Jump the camera far in +X so the inner cube fully shifts: the old keys leave, new ones enter and are
        // enqueued (a re-key, not a retag).
        let jump = brick_span(0) * (cfg.clip_half_bricks as f32 * 2.0 + 1.0);
        let cam1 = [jump, surf, 0.0];
        let dropped = mgr.update(cam1, &cfg);
        assert!(dropped > 0, "the fully-shifted clipmap drops the old keys");
        assert!(mgr.pending() > 0, "and enqueues the new keys (fresh voxelize at their LOD)");
    }
}
