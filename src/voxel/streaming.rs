//! **Stage 3 — camera-following residency + per-LOD voxel mips.**
//!
//! The Stage-2 HW-RT path traced a STATIC ~32 m patch at the origin. Stage 3 replaces that with a brick
//! set that STREAMS around the camera and stores far bricks at a coarser voxel mip, so the view distance is
//! large while per-frame work and VRAM stay bounded.
//!
//! This module owns the pure, headless-testable bookkeeping — no GPU, no Bevy systems — so the residency
//! scheme is proven in isolation and the render wiring ([`super::raytrace`]) just drives it:
//!
//! * [`brick_lod`] / [`desired_residency`] — given the camera brick coordinate, which bricks should be
//!   resident and at what LOD (concentric rings: near = LOD0 full res, farther = progressively coarser).
//! * [`ResidencyManager`] — the live set of resident bricks + a bounded WORK QUEUE. Each `update` diffs the
//!   desired set against the current one, ENQUEUES newly-entered / LOD-changed bricks, and DROPS exited
//!   ones; [`ResidencyManager::drain_work`] processes at most `max_per_frame` queue items per call
//!   (voxelizing them) so a big camera jump can't stall the frame. The packed brick list it exposes
//!   ([`ResidencyManager::resident_entries`]) is the input to the SSOT [`super::gpu::pack_resident_set`].
//!
//! ## Keep-old-until-revealed
//! The manager only marks itself DIRTY (needing a re-pack + BLAS/TLAS rebuild) once a non-empty batch of
//! queued bricks has been voxelized. The previous resident set — and therefore the previous TLAS the render
//! path keeps bound — stays valid until the new one is ready, so the camera never sees a hole/flash while a
//! batch streams in.
//!
//! ## Bounds (all `log`ged when hit)
//! * Region: a CUBE of bricks of radius [`StreamingConfig::residency_radius_bricks`] around the camera
//!   brick (Chebyshev distance), capped at [`StreamingConfig::max_resident_bricks`] resident bricks.
//! * Work: at most [`StreamingConfig::max_bricks_per_frame`] bricks voxelized+queued per `drain_work`.

use bevy::math::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};

use super::brickmap::{BRICK_WORLD_SIZE, Brick, MAX_LOD, brick_coord_of_voxel};
use super::gpu::ResidentBrick;
use super::palette::BlockRegistry;
use super::voxelize::voxelize_brick;
use crate::sdf_render::worldgen::biome::BiomeLibrary;
use crate::sdf_render::worldgen::layers::height::HeightLayer;

/// Tunable streaming + LOD knobs. All distances are in BRICKS (1 brick = [`BRICK_WORLD_SIZE`] = 1.6 m), so
/// the region scales cleanly with brick size. Defaults give a ~32 m-radius region (matching the Stage-2
/// patch's reach) with LOD rings, bounded per-frame work, and a hard resident cap. Plain `Copy` data so it
/// can be a Bevy resource or a test literal.
#[derive(Clone, Copy, Debug, bevy::prelude::Resource)]
pub struct StreamingConfig {
    /// Residency radius in bricks (Chebyshev): bricks within this many bricks of the camera brick on every
    /// axis are resident. A radius of `r` gives a `(2r+1)³` cube. Default 20 bricks ≈ 32 m.
    pub residency_radius_bricks: i32,
    /// LOD ring radii in bricks: a brick at Chebyshev distance `d` from the camera takes LOD = the number
    /// of thresholds it exceeds. `lod_ring_bricks[i]` is the distance at/after which LOD becomes `i+1`.
    /// Must be ascending. Distances beyond the last threshold clamp to [`MAX_LOD`]. Default `[6, 12, 18]`.
    pub lod_ring_bricks: [i32; MAX_LOD as usize],
    /// Hard cap on resident bricks. If the desired set exceeds it the farthest bricks are dropped (logged).
    /// A safety bound so a mis-set radius can't blow VRAM. Default 60_000.
    pub max_resident_bricks: usize,
    /// Max bricks voxelized + enqueued→processed per `drain_work` call (per frame). Bounds the per-frame
    /// CPU cost of a big camera move; the rest carry in the queue to later frames. Default 256.
    pub max_bricks_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            residency_radius_bricks: 20,
            lod_ring_bricks: [6, 12, 18],
            max_resident_bricks: 60_000,
            max_bricks_per_frame: 256,
        }
    }
}

/// The brick coordinate the camera world position falls in (its containing brick), via the SSOT
/// voxel→brick addressing. The streaming region is centred here.
#[inline]
pub fn camera_brick_coord(cam_world: [f32; 3]) -> IVec3 {
    // World metres → world voxel (floor by VOXEL_SIZE) → brick coord. Reuse the brick addressing SSOT by
    // going through a voxel coordinate at the camera position.
    let to_voxel = |m: f32| (m / super::brickmap::VOXEL_SIZE).floor() as i32;
    brick_coord_of_voxel(IVec3::new(to_voxel(cam_world[0]), to_voxel(cam_world[1]), to_voxel(cam_world[2])))
}

/// The Chebyshev (L∞) distance in bricks between two brick coordinates — the ring metric. A cube region of
/// radius `r` is exactly `{ b : cheby(b, centre) <= r }`.
#[inline]
fn cheby(a: IVec3, b: IVec3) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs())
}

/// The LOD level for a brick at `coord` given the camera brick `cam` and the ring config: the number of
/// `lod_ring_bricks` thresholds the Chebyshev distance meets/exceeds, clamped to [`MAX_LOD`]. Near bricks
/// (distance < first threshold) are LOD0 (full res); each farther ring is one coarser mip. Pure function —
/// the SSOT for distance→LOD, shared by [`desired_residency`] and the tests.
#[inline]
pub fn brick_lod(coord: IVec3, cam: IVec3, cfg: &StreamingConfig) -> u32 {
    let d = cheby(coord, cam);
    let mut lod = 0u32;
    for &threshold in &cfg.lod_ring_bricks {
        if d >= threshold {
            lod += 1;
        }
    }
    lod.min(MAX_LOD)
}

/// The DESIRED resident set: every brick coordinate within the cube region of radius
/// `residency_radius_bricks` around the camera brick, mapped to its [`brick_lod`]. Deterministic ITERATION
/// is not guaranteed (it's a map); callers that need a stable `primitive_index` order sort the keys (the
/// manager does). If the region exceeds `max_resident_bricks`, the FARTHEST bricks (largest Chebyshev
/// distance, then by coordinate) are dropped so the cap holds — the dropped count is the caller's to log.
pub fn desired_residency(cam: IVec3, cfg: &StreamingConfig) -> FxHashMap<IVec3, u32> {
    let r = cfg.residency_radius_bricks;
    let mut out: FxHashMap<IVec3, u32> = FxHashMap::default();
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let coord = cam + IVec3::new(dx, dy, dz);
                out.insert(coord, brick_lod(coord, cam, cfg));
            }
        }
    }
    if out.len() > cfg.max_resident_bricks {
        // Keep the nearest `max_resident_bricks`; drop the rest (farthest first). Deterministic tiebreak.
        let mut all: Vec<IVec3> = out.keys().copied().collect();
        all.sort_by_key(|c| (cheby(*c, cam), c.z, c.y, c.x));
        for c in all.into_iter().skip(cfg.max_resident_bricks) {
            out.remove(&c);
        }
    }
    out
}

/// A queued unit of streaming work: voxelize the brick at `coord` (its desired LOD is looked up at process
/// time so a brick re-queued after a LOD ring change picks up the latest LOD).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorkItem {
    coord: IVec3,
}

/// The live residency state + bounded work queue. Holds the voxelized [`Brick`]s currently resident (only
/// NON-empty bricks are stored — empty/all-air bricks are skipped, the sparsity invariant), each tagged
/// with the LOD it should pack at, plus a FIFO queue of bricks awaiting voxelization. `update` recomputes
/// the desired set and reconciles; `drain_work` does the bounded voxelization.
///
/// Robust-by-construction: the resident map is the single source of what's live; `dirty` is set only when a
/// drained batch actually changes it, so the render path re-packs exactly when (and only when) the GPU set
/// must change — keep-old-until-revealed falls out for free.
#[derive(Default)]
pub struct ResidencyManager {
    /// Resident, voxelized bricks: coord → (brick, current LOD). Empty bricks are never inserted.
    resident: FxHashMap<IVec3, (Brick, u32)>,
    /// Coords awaiting voxelization (enqueued by `update`, processed by `drain_work`). A set membership
    /// guard (`queued`) prevents a brick from being enqueued twice while it waits.
    queue: std::collections::VecDeque<WorkItem>,
    queued: FxHashSet<IVec3>,
    /// True iff the resident set CHANGED since the last `take_dirty` — the render path re-packs + rebuilds
    /// the BLAS/TLAS only then (otherwise it keeps the old, still-valid GPU scene).
    dirty: bool,
    /// The camera brick from the last `update`, so `drain_work` knows each queued brick's current LOD.
    last_cam: IVec3,
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

    /// Reconcile the resident set toward the desired region around camera brick `cam`:
    /// * DROP every resident brick no longer in the desired set (camera moved away) — marks dirty.
    /// * ENQUEUE every desired brick that is NOT resident, OR whose desired LOD differs from its resident
    ///   LOD (it crossed a ring) — these are voxelized later by [`drain_work`].
    ///
    /// Does NOT itself voxelize (that's bounded work in `drain_work`), so a huge camera jump only enqueues
    /// here — cheap. Returns the number of bricks dropped (so the caller can log churn).
    pub fn update(&mut self, cam: IVec3, cfg: &StreamingConfig) -> usize {
        self.last_cam = cam;
        let desired = desired_residency(cam, cfg);
        let capped = (2 * cfg.residency_radius_bricks as usize + 1).pow(3).saturating_sub(desired.len());
        self.capped_total += capped;

        // Drop resident bricks that left the region.
        let mut dropped = 0usize;
        let to_drop: Vec<IVec3> = self.resident.keys().filter(|c| !desired.contains_key(*c)).copied().collect();
        for c in to_drop {
            self.resident.remove(&c);
            dropped += 1;
            self.dirty = true; // the GPU set shrank → must re-pack
        }

        // Enqueue desired bricks that are missing or at the wrong LOD (and not already queued).
        for (&coord, &want_lod) in &desired {
            let needs = match self.resident.get(&coord) {
                Some((_, have_lod)) => *have_lod != want_lod, // crossed a LOD ring → re-voxelize at new LOD
                None => true,                                 // newly entered
            };
            if needs && !self.queued.contains(&coord) {
                self.queue.push_back(WorkItem { coord });
                self.queued.insert(coord);
            }
        }
        dropped
    }

    /// Process up to `cfg.max_bricks_per_frame` queued bricks: voxelize each at its CURRENT desired LOD
    /// (recomputed from `last_cam`), store NON-empty results as resident, and drop empty ones (sparsity).
    /// Marks the set dirty iff at least one brick was actually added/updated — so a batch that produced only
    /// empty bricks does NOT trigger a needless re-pack, and the old GPU scene stays valid until a
    /// REVEALING batch lands (keep-old-until-revealed). Returns the number of bricks voxelized this call.
    ///
    /// Bounded: never does more than `max_bricks_per_frame` voxelizations; leftover queue items carry to
    /// the next call. Logs when it caps (leaves work pending).
    pub fn drain_work(
        &mut self,
        cfg: &StreamingConfig,
        layer: &HeightLayer,
        lib: &BiomeLibrary,
        registry: &BlockRegistry,
        seed: u64,
    ) -> usize {
        let budget = cfg.max_bricks_per_frame;
        let mut done = 0usize;
        while done < budget {
            let Some(item) = self.queue.pop_front() else { break };
            self.queued.remove(&item.coord);
            done += 1;

            let want_lod = brick_lod(item.coord, self.last_cam, cfg);
            let brick = voxelize_brick(item.coord, layer, lib, registry, seed);
            if brick.is_empty() {
                // All-air at full res → never resident (and if it WAS resident at a coarser LOD, drop it).
                if self.resident.remove(&item.coord).is_some() {
                    self.dirty = true;
                }
                continue;
            }
            // Store the full-res brick + its target LOD; the packer downsamples at pack time.
            self.resident.insert(item.coord, (brick, want_lod));
            self.dirty = true;
        }
        if !self.queue.is_empty() {
            bevy::log::debug!(
                "voxel streaming: capped at {budget} bricks/frame, {} still pending",
                self.queue.len()
            );
        }
        done
    }

    /// Take the dirty flag, clearing it. `true` ⇒ the resident set changed and the render path should
    /// re-pack + rebuild the BLAS/TLAS this frame; `false` ⇒ nothing changed, keep the old GPU scene.
    #[inline]
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// The resident bricks as [`ResidentBrick`] entries in a DETERMINISTIC order (sorted by brick
    /// coordinate `(z,y,x)`), ready for [`super::gpu::pack_resident_set`]. The stable order keeps each
    /// brick's `primitive_index` reproducible (the test oracle relies on it). Borrows `self`, so the
    /// returned entries live as long as the manager isn't mutated.
    pub fn resident_entries(&self) -> Vec<ResidentBrick<'_>> {
        let mut coords: Vec<IVec3> = self.resident.keys().copied().collect();
        coords.sort_by_key(|c| (c.z, c.y, c.x));
        coords
            .into_iter()
            .map(|coord| {
                let (brick, lod) = self.resident.get(&coord).expect("coord came from keys");
                ResidentBrick { coord, brick, lod: *lod }
            })
            .collect()
    }
}

/// The world-metre AABB half-extent the resident region covers around the camera (for logging / framing).
pub fn region_half_extent_m(cfg: &StreamingConfig) -> f32 {
    cfg.residency_radius_bricks as f32 * BRICK_WORLD_SIZE
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

    /// LOD rings: a brick at the camera is LOD0; crossing each ring threshold bumps the LOD; far bricks
    /// clamp at MAX_LOD.
    #[test]
    fn lod_rings_by_distance() {
        let cfg = StreamingConfig { lod_ring_bricks: [6, 12, 18], ..Default::default() };
        let cam = IVec3::new(0, 0, 0);
        assert_eq!(brick_lod(IVec3::new(0, 0, 0), cam, &cfg), 0);
        assert_eq!(brick_lod(IVec3::new(5, 0, 0), cam, &cfg), 0); // just inside ring 0
        assert_eq!(brick_lod(IVec3::new(6, 0, 0), cam, &cfg), 1); // hits first threshold
        assert_eq!(brick_lod(IVec3::new(0, 11, 0), cam, &cfg), 1);
        assert_eq!(brick_lod(IVec3::new(0, 0, 12), cam, &cfg), 2);
        assert_eq!(brick_lod(IVec3::new(18, 0, 0), cam, &cfg), 3);
        assert_eq!(brick_lod(IVec3::new(999, 0, 0), cam, &cfg), MAX_LOD); // clamps
        // Chebyshev metric: a diagonal brick uses the max axis distance.
        assert_eq!(brick_lod(IVec3::new(6, 6, 0), cam, &cfg), 1);
    }

    /// The desired region is a cube of `(2r+1)³` bricks centred on the camera brick, each tagged with its
    /// ring LOD.
    #[test]
    fn desired_region_is_a_cube() {
        let cfg = StreamingConfig { residency_radius_bricks: 3, lod_ring_bricks: [2, 3, 4], max_resident_bricks: 100_000, ..Default::default() };
        let cam = IVec3::new(10, -5, 7);
        let d = desired_residency(cam, &cfg);
        assert_eq!(d.len(), (2 * 3 + 1usize).pow(3));
        assert_eq!(d[&cam], 0);
        // A corner brick (distance 3) is past the lod_ring threshold [2,3,4] → exceeds 2 and 3 → LOD2.
        assert_eq!(d[&(cam + IVec3::new(3, 3, 3))], 2);
        // Outside the region is absent.
        assert!(!d.contains_key(&(cam + IVec3::new(4, 0, 0))));
    }

    /// The resident cap drops the farthest bricks so the desired set never exceeds `max_resident_bricks`.
    #[test]
    fn resident_cap_drops_farthest() {
        let cfg = StreamingConfig { residency_radius_bricks: 4, max_resident_bricks: 10, ..Default::default() };
        let cam = IVec3::ZERO;
        let d = desired_residency(cam, &cfg);
        assert_eq!(d.len(), 10, "capped to max_resident_bricks");
        // The camera brick (nearest) is always kept.
        assert!(d.contains_key(&cam));
    }

    /// Residency reconciliation: a simulated camera move enters new bricks (enqueued, then voxelized into
    /// resident) and drops exited ones; empty (sky) bricks are skipped; the keep-old invariant holds (the
    /// set isn't dirty until a revealing batch lands).
    #[test]
    fn residency_updates_as_camera_moves() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        // Small region near the surface so most bricks are non-empty terrain.
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let surf_brick_y = (surf / BRICK_WORLD_SIZE).floor() as i32;
        let cfg = StreamingConfig { residency_radius_bricks: 2, lod_ring_bricks: [1, 2, 3], max_resident_bricks: 10_000, max_bricks_per_frame: 1000 };

        let mut mgr = ResidencyManager::new();
        let cam0 = IVec3::new(0, surf_brick_y, 0);
        mgr.update(cam0, &cfg);
        assert!(mgr.pending() > 0, "entering a fresh region enqueues work");
        assert!(!mgr.is_dirty(), "no bricks voxelized yet → not dirty (keep-old)");

        let n0 = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        assert_eq!(n0, (2 * 2 + 1i32).pow(3) as usize, "voxelized the whole 5³ region in one drain");
        assert!(mgr.is_dirty(), "voxelizing real terrain bricks reveals new geometry → dirty");
        assert!(mgr.take_dirty());
        let resident0 = mgr.resident_count();
        assert!(resident0 > 0 && resident0 <= 125, "some non-empty bricks resident, ≤ region size");

        // Move the camera +5 bricks in X (region fully shifts). New bricks enter, far ones drop.
        let cam1 = cam0 + IVec3::new(5, 0, 0);
        let dropped = mgr.update(cam1, &cfg);
        assert!(dropped > 0, "moving away drops the bricks left behind");
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        // After the move every resident brick lies within the new region.
        for e in mgr.resident_entries() {
            assert!(cheby(e.coord, cam1) <= cfg.residency_radius_bricks, "resident bricks stay in-region");
        }
    }

    /// The per-frame cap bounds work: a large fresh region drains at most `max_bricks_per_frame` per call,
    /// carrying the rest in the queue across calls until empty.
    #[test]
    fn carry_queue_caps_per_frame_work() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let surf_brick_y = (surf / BRICK_WORLD_SIZE).floor() as i32;
        let cfg = StreamingConfig { residency_radius_bricks: 4, lod_ring_bricks: [2, 3, 4], max_resident_bricks: 10_000, max_bricks_per_frame: 50 };

        let mut mgr = ResidencyManager::new();
        let cam = IVec3::new(0, surf_brick_y, 0);
        mgr.update(cam, &cfg);
        let total = (2 * 4 + 1i32).pow(3) as usize; // 729
        assert_eq!(mgr.pending(), total);

        // Each drain does at most 50; it takes ceil(729/50)=15 drains to clear the queue.
        let mut drains = 0;
        while mgr.pending() > 0 {
            let n = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
            assert!(n <= 50, "never exceeds the per-frame cap");
            drains += 1;
            assert!(drains <= 20, "must terminate");
        }
        assert_eq!(drains, total.div_ceil(50));
    }

    /// LOD change re-queues a brick: as a brick crosses into a coarser ring (camera moves so the brick's
    /// Chebyshev distance grows), `update` enqueues it for re-voxelization at the new LOD.
    #[test]
    fn lod_change_requeues_brick() {
        let layer = test_layer();
        let lib = test_library();
        let reg = registry();
        let surf = layer.sample_world(0.0, 0.0, SEED).height;
        let surf_brick_y = (surf / BRICK_WORLD_SIZE).floor() as i32;
        let cfg = StreamingConfig { residency_radius_bricks: 6, lod_ring_bricks: [2, 4, 6], max_resident_bricks: 10_000, max_bricks_per_frame: 10_000 };

        let mut mgr = ResidencyManager::new();
        let cam0 = IVec3::new(0, surf_brick_y, 0);
        mgr.update(cam0, &cfg);
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        mgr.take_dirty();

        // Pick a resident brick at LOD0 near the camera, then move the camera so its distance jumps a ring.
        let probe = IVec3::new(0, surf_brick_y, 1); // distance 1 from cam0 → LOD0
        if mgr.resident_entries().iter().any(|e| e.coord == probe) {
            let lod_before = mgr.resident_entries().into_iter().find(|e| e.coord == probe).unwrap().lod;
            assert_eq!(lod_before, 0);
            // Move camera away so `probe` is now ≥ ring threshold 2 (distance 3).
            let cam1 = probe + IVec3::new(3, 0, 0);
            mgr.update(cam1, &cfg);
            assert!(mgr.pending() > 0, "the LOD-changed brick is re-queued");
            mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
            let lod_after = mgr.resident_entries().into_iter().find(|e| e.coord == probe).map(|e| e.lod);
            assert_eq!(lod_after, Some(1), "the brick is now stored at the coarser ring's LOD");
        }
    }
}
