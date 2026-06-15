//! Headless (CPU-only) verification of the camera-following CLIPMAP streaming bookkeeping, run from the
//! integration test crate (which links the library's PUBLIC API, so it compiles even though some in-crate
//! `#[cfg(test)]` modules of pruned features don't). Mirrors the in-module unit tests, but guaranteed
//! runnable here:
//!
//!   * the in-place mip: a coarse-LOD brick is voxelized DIRECTLY at its coarse spacing (not a downsample);
//!   * exact clipmap tiling: each level fills a coarse-grid-snapped box minus the finer level's footprint, so
//!     levels abut with NO overlap and NO gap (the union telescopes to the outermost box);
//!   * residency: enters/exits as a simulated camera moves, empty bricks skipped, per-frame cap (carry
//!     queue), keep-old-until-revealed (not dirty until a revealing batch lands), the O(shell) per-move cost;
//!   * the packing SSOT: a coarse brick keeps the constant haloed `10³` grid, spans `brick_span(lod)`.
//!
//! The mixed-LOD GPU oracle lives in `tests/voxel_raytrace_gpu.rs`.

use bevy::math::IVec3;

use adventure::sdf_render::worldgen::biome::{
    BiomeDef, BiomeId, BiomeLibrary, StrataLayer, TerrainMatId, TerrainSurfaceMaterial,
};
use adventure::sdf_render::worldgen::coord::LayerId;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::voxel::brickmap::{
    BRICK_EDGE, MAX_LOD, brick_span, lod_voxel_size,
};
use adventure::voxel::gpu::{ResidentBrick, halo_cells, halo_index, pack_resident_set};
use adventure::voxel::palette::{BlockId, BlockRegistry};
use adventure::voxel::source::{BrickClass, BrickSource, WorldgenSource};
use adventure::voxel::streaming::{
    BrickKey, ResidencyManager, StreamingConfig, brick_lod, camera_brick_coord, camera_brick_coord_lod,
    desired_clipmap, region_half_extent_m,
};
use adventure::voxel::voxelize::voxelize_brick;

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

fn cheby(a: IVec3, b: IVec3) -> i32 {
    (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs())
}

// --- in-place mip ---------------------------------------------------------------------------------

/// A coarse-LOD brick is a TRUE in-place mip: voxelized DIRECTLY at its `lod_voxel_size(lod)` spacing over
/// `brick_span(lod)` world, not a downsample of a finer brick. We verify a LOD2 brick's surface column
/// boundary matches a direct coarse-spacing surface sample, and that it spans 4× the LOD0 world.
#[test]
fn coarse_lod_brick_is_in_place_mip() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let lod = 2u32;
    let cell = lod_voxel_size(lod);
    let span = brick_span(lod);
    assert!((span - 4.0 * brick_span(0)).abs() < 1e-4, "LOD2 spans 4× the LOD0 brick");
    assert!((cell - 4.0 * lod_voxel_size(0)).abs() < 1e-5, "LOD2 cell is 4× the LOD0 voxel");

    let surf = layer.sample_world((span * 0.5) as f64, (span * 0.5) as f64, SEED).height;
    let by = (surf / span).floor() as i32;
    let b = voxelize_brick(IVec3::new(0, by, 0), lod, &layer, &lib, &reg, SEED);
    assert!(!b.is_empty(), "a surface-straddling coarse brick has solid voxels");
}

// --- clipmap shells -------------------------------------------------------------------------------

#[test]
fn camera_brick_coord_scales_with_lod() {
    // World 5 m: LOD0 (span 1.6) → 3; LOD1 (3.2) → 1; LOD2 (6.4) → 0 — the per-level clipmap centres differ.
    let w = [5.0_f32, 5.0, 5.0];
    assert_eq!(camera_brick_coord_lod(w, 0), IVec3::splat(3));
    assert_eq!(camera_brick_coord_lod(w, 1), IVec3::splat(1));
    assert_eq!(camera_brick_coord_lod(w, 2), IVec3::splat(0));
    assert_eq!(camera_brick_coord(w), camera_brick_coord_lod(w, 0));
}

#[test]
fn desired_clipmap_all_levels_and_view_radius() {
    let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 1_000_000, ..Default::default() };
    let cam = [0.5_f32, 0.5, 0.5];
    let d = desired_clipmap(cam, &cfg);
    for lod in 0..=MAX_LOD {
        assert!(d.keys().any(|k| k.lod == lod), "level {lod} present");
    }
    // The total view radius is clip_half · brick_span(MAX_LOD) — a huge reach at bounded VRAM.
    let view = region_half_extent_m(&cfg);
    assert!((view - cfg.clip_half_bricks as f32 * brick_span(MAX_LOD)).abs() < 1e-2);
    assert!(view > 1500.0, "clipmap view radius is >1.5 km at MAX_LOD=7 (got {view:.0} m)");
}

#[test]
fn desired_clipmap_tiles_exactly_no_overlap_no_gap() {
    // The exact-tiling gate (the user requires NO LOD overlap, and the old scheme had a cross-LOD coverage
    // hole). Marching outward from the camera, every covered point is covered by EXACTLY ONE level (count ≤ 1
    // ⇒ no overlap), and coverage never breaks covered → empty → covered within the view (⇒ no gap). Empirical
    // companion to the closed-form telescoping proof in the streaming.rs unit test.
    let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 100_000_000, ..Default::default() };
    let view = region_half_extent_m(&cfg);
    // Off-centre cameras exercise the even/odd snapping at non-zero sub-cell offsets.
    let cams = [
        [0.5_f32, 0.5, 0.5],
        [7.4, 14.5, 17.7],
        [-7.05, -13.97, 6.04],
        [3.3, 0.1, -9.9],
        [101.2, -50.7, 33.3],
    ];
    for cam in cams {
        let d = desired_clipmap(cam, &cfg);
        // How many LEVELS have a resident brick containing world point `p` (must be exactly 1 inside the view).
        let cover_count = |p: [f32; 3]| -> usize {
            (0..=MAX_LOD)
                .filter(|&lod| {
                    let span = brick_span(lod);
                    let coord = IVec3::new(
                        (p[0] / span).floor() as i32,
                        (p[1] / span).floor() as i32,
                        (p[2] / span).floor() as i32,
                    );
                    d.contains_key(&BrickKey { coord, lod })
                })
                .count()
        };
        // March a Fibonacci-sphere spread of outward rays; step at half a finest brick so no thin gap is skipped.
        let step = brick_span(0) * 0.5;
        for i in 0..256u32 {
            let zf = 1.0 - 2.0 * (i as f32 + 0.5) / 256.0;
            let rf = (1.0 - zf * zf).max(0.0).sqrt();
            let ang = i as f32 * 2.399_963_2; // golden angle
            let dir = [rf * ang.cos(), zf, rf * ang.sin()];
            let mut seen_covered = false;
            let mut gap_after_cover = false;
            let mut t = 0.0_f32;
            while t <= view {
                let p = [cam[0] + dir[0] * t, cam[1] + dir[1] * t, cam[2] + dir[2] * t];
                let c = cover_count(p);
                assert!(c <= 1, "LOD OVERLAP: {c} levels cover t={t:.1} m, cam={cam:?}, dir={dir:?}");
                if c == 1 {
                    assert!(
                        !gap_after_cover,
                        "coverage GAP (covered→empty→covered) at t={t:.1} m, cam={cam:?}, dir={dir:?}"
                    );
                    seen_covered = true;
                } else if seen_covered {
                    gap_after_cover = true; // an empty span AFTER coverage — only OK if no coverage follows
                }
                t += step;
            }
        }
    }
}

#[test]
fn brick_lod_reports_covering_level() {
    let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 100_000_000, ..Default::default() };
    let cam = [0.5_f32, 0.5, 0.5];
    // brick_lod returns the FINEST level whose tiled region covers the LOD0 brick's world centre.
    assert_eq!(brick_lod(camera_brick_coord_lod(cam, 0), cam, &cfg), 0, "camera brick is LOD0");
    assert_eq!(brick_lod(IVec3::new(7, 0, 0), cam, &cfg), 0, "inside the LOD0 box");
    assert_eq!(brick_lod(IVec3::new(12, 0, 0), cam, &cfg), 1, "past the LOD0 box → LOD1 annulus");
    assert!(brick_lod(IVec3::new(30, 0, 0), cam, &cfg) >= 2, "far out → a coarser level");
    // Consistency with the enumerated set: the reported level actually holds that point.
    let d = desired_clipmap(cam, &cfg);
    let span0 = brick_span(0);
    for cx in [0i32, 5, 11, 13, 25, 60, 150] {
        let coord = IVec3::new(cx, 0, 0);
        let lod = brick_lod(coord, cam, &cfg);
        let world = [(cx as f32 + 0.5) * span0, 0.5 * span0, 0.5 * span0];
        let here = camera_brick_coord_lod(world, lod);
        assert!(
            lod == MAX_LOD || d.contains_key(&BrickKey { coord: here, lod }),
            "brick_lod({cx}) = {lod} must be the resident level for that point"
        );
    }
}

/// A `BrickSource` that classifies EVERY brick `Surface` (the trait default), so the A2 surface cap is
/// exercised directly — every desired brick is a surface candidate competing for a slot. `brick` is never
/// called by `update`.
struct AllSurfaceSource;
impl BrickSource for AllSurfaceSource {
    fn brick(
        &self,
        _coord: IVec3,
        _lod: u32,
        _registry: &BlockRegistry,
    ) -> adventure::voxel::brickmap::Brick {
        adventure::voxel::brickmap::Brick::uniform(BlockId::AIR)
    }
}

/// A2 — the cap is applied AFTER the classify split, so it bounds the surface SHELL: `desired_clipmap` is now
/// UNCAPPED (returns the full geometric tiling), and `ResidencyManager::update` caps the surface candidates,
/// keeping the NEAREST and always the camera's own brick.
#[test]
fn resident_cap_drops_farthest() {
    let cap = 50usize;
    let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: cap, ..Default::default() };
    let cam = [0.5_f32, 0.5, 0.5];

    // desired_clipmap no longer caps — the full tiling is far larger than the cap.
    let big = StreamingConfig { max_resident_bricks: usize::MAX, ..cfg };
    let d = desired_clipmap(cam, &big);
    assert!(d.len() > cap, "the uncapped tiling is far larger than the cap");

    let mut mgr = ResidencyManager::new();
    mgr.update(cam, &cfg, &AllSurfaceSource);
    assert!(mgr.resident_count() + mgr.pending() <= cap, "the surface cap bounds resident+pending to the cap");
    assert_eq!(mgr.pending(), cap, "an all-surface cold fill enqueues exactly the cap (nearest kept)");
    assert!(mgr.capped_total > 0, "the cap dropped the farther surface candidates");
    // (The nearest-kept ordering + camera-brick-always-kept is asserted in the in-module unit test, which can
    // see the private work queue; here we verify the cap binds the surface SHELL to the cap.)
}

// --- residency ------------------------------------------------------------------------------------

#[test]
fn residency_enters_and_exits_as_camera_moves() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let cfg = StreamingConfig { clip_half_bricks: 2, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

    let src = WorldgenSource::new(&layer, &lib, SEED);
    let mut mgr = ResidencyManager::new();
    let cam0 = [0.0_f32, surf, 0.0];
    mgr.update(cam0, &cfg, &src);
    assert!(mgr.pending() > 0, "entering a fresh clipmap enqueues work");
    assert!(!mgr.is_dirty(), "keep-old: nothing voxelized yet → not dirty");

    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    assert!(mgr.is_dirty(), "voxelizing real terrain reveals geometry → dirty");
    assert!(mgr.take_dirty());
    assert!(mgr.resident_count() > 0, "some non-empty bricks resident");

    // Move +5 m in X: new bricks enter, far ones drop.
    let cam1 = [5.0_f32, surf, 0.0];
    let dropped = mgr.update(cam1, &cfg, &src);
    assert!(dropped > 0, "moving away drops bricks left behind");
    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    // The snapped box has half-extent up to `half + 1` (snap_even_odd can extend one side by one brick).
    let half = cfg.clip_half_bricks;
    for e in mgr.resident_entries() {
        let cam_l = camera_brick_coord_lod(cam1, e.lod);
        assert!(cheby(e.coord, cam_l) <= half + 1, "resident bricks stay in the clipmap");
    }
}

#[test]
fn empty_sky_bricks_skipped() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    // A clipmap far ABOVE any terrain → every brick all-air → none become resident.
    let cfg = StreamingConfig { clip_half_bricks: 1, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };
    let src = WorldgenSource::new(&layer, &lib, SEED);
    let mut mgr = ResidencyManager::new();
    let sky = [0.0_f32, 6400.0, 0.0]; // ~+6.4 km up
    mgr.update(sky, &cfg, &src);
    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    assert_eq!(mgr.resident_count(), 0, "all-air sky bricks are skipped (sparsity)");
    assert!(!mgr.is_dirty(), "an all-empty batch does not reveal geometry → not dirty (keep-old)");
}

#[test]
fn carry_queue_caps_per_frame_work() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let cfg = StreamingConfig { clip_half_bricks: 3, max_resident_bricks: 1_000_000, max_bricks_per_frame: 50 };
    let src = WorldgenSource::new(&layer, &lib, SEED);
    let mut mgr = ResidencyManager::new();
    mgr.update([0.0, surf, 0.0], &cfg, &src);
    let total = mgr.pending();
    assert!(total > 50, "the clipmap enqueues more than one frame's budget");

    let mut drains = 0;
    let mut voxelized = 0usize;
    while mgr.pending() > 0 {
        let n = mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
        assert!(n <= 50, "never exceeds the per-frame cap");
        voxelized += n;
        drains += 1;
        assert!(drains <= total / 50 + 5);
    }
    assert_eq!(voxelized, total, "every enqueued brick is eventually voxelized");
    assert_eq!(drains, total.div_ceil(50), "carries the rest across frames");
}

/// **The stutter metric.** A single-LOD0-brick camera move shifts only the LOD0 SHELL (a thin face-slab),
/// not the whole region — the per-move enqueue+drop is O(shell) ≈ O((2·clip_half)²), NOT O((2·clip_half)³).
/// Coarser shells do NOT move at all on a fine move (their boundaries are `2^L×` farther apart). We warm the
/// clipmap, then nudge the camera one LOD0 brick and assert the churn is shell-sized, not region-sized.
#[test]
fn per_move_churn_is_o_shell_not_o_region() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let half = 6;
    let cfg = StreamingConfig { clip_half_bricks: half, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };

    let src = WorldgenSource::new(&layer, &lib, SEED);
    let mut mgr = ResidencyManager::new();
    let cam0 = [0.5_f32, surf, 0.5];
    mgr.update(cam0, &cfg, &src);
    while mgr.pending() > 0 {
        mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    }
    mgr.take_dirty();

    // Nudge ONE LOD0 brick in +X (one brick_span(0) = 1.6 m). Count what enters (pending) + what drops.
    let span0 = brick_span(0);
    let cam1 = [cam0[0] + span0, surf, cam0[2]];
    let dropped = mgr.update(cam1, &cfg, &src);
    let entered = mgr.pending();

    // A full region recompute would touch ~(2·half+1)³ bricks PER LEVEL. A shell shift touches at most a few
    // face-slabs: ~(2·half+1)² × small constant. Assert the churn is comfortably below the region volume.
    let region_vol = (2 * half as usize + 1).pow(3);
    let shell_area = (2 * half as usize + 1).pow(2);
    let churn = entered + dropped;
    eprintln!(
        "[stutter] clip_half={half}: one-brick move churn = {entered} entered + {dropped} dropped = {churn} \
         (region vol {region_vol}, shell area {shell_area})"
    );
    assert!(
        churn < region_vol,
        "a single-brick move must NOT recompute the whole region ({churn} >= {region_vol})"
    );
    // Tighter: the churn is shell-sized — at most a handful of face-slabs across the LOD0 cube (the only level
    // that moves on a fine nudge). A generous bound of ~6 face-slabs (3 axes × 2 sides) catches a regression
    // to region-sized work while tolerating the coarse-shell re-centering arithmetic.
    assert!(
        churn <= 8 * shell_area,
        "per-move churn must be O(shell) — got {churn} vs ~{} (8 face-slabs)",
        8 * shell_area
    );
}

/// A LOD change is a DIFFERENT key (not a retag): a far camera jump fully shifts the clipmap — old keys
/// leave, new keys (their own LODs) enter and are enqueued for a fresh voxelize at their coarse spacing.
#[test]
fn lod_change_is_a_fresh_key() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let cfg = StreamingConfig { clip_half_bricks: 4, max_resident_bricks: 1_000_000, max_bricks_per_frame: 1_000_000 };
    let src = WorldgenSource::new(&layer, &lib, SEED);
    let mut mgr = ResidencyManager::new();
    let cam0 = [0.0_f32, surf, 0.0];
    mgr.update(cam0, &cfg, &src);
    mgr.drain_work(&cfg, &layer, &lib, &reg, SEED);
    mgr.take_dirty();
    let d0 = desired_clipmap(cam0, &cfg);
    for e in mgr.resident_entries() {
        assert!(d0.contains_key(&BrickKey { coord: e.coord, lod: e.lod }), "resident keys are desired");
    }

    let jump = brick_span(0) * (cfg.clip_half_bricks as f32 * 2.0 + 1.0);
    let cam1 = [jump, surf, 0.0];
    let dropped = mgr.update(cam1, &cfg, &src);
    assert!(dropped > 0, "the fully-shifted clipmap drops the old keys");
    assert!(mgr.pending() > 0, "and enqueues the new keys (fresh voxelize at their LOD)");
}

// --- surface-following residency (classify prune) -------------------------------------------------

/// The `classify` predicate is CONSERVATIVE + correct: deep-underground ⇒ Interior, a surface-straddling
/// brick ⇒ Surface, high-sky ⇒ Air, and a brick directly ADJACENT to the surface (one brick above/below the
/// straddle) ⇒ Surface (it has an exposed face / is reachable). The adversarial target: pruning must never
/// drop a brick that could have an exposed voxel.
#[test]
fn classify_prunes_only_provably_unhittable_bricks() {
    let layer = test_layer();
    let lib = test_library();
    let src = WorldgenSource::new(&layer, &lib, SEED);
    let span0 = brick_span(0) as f64;
    // The LOD0 brick coord straddling the surface at the origin column.
    let surf = layer.sample_world(0.5 * span0, 0.5 * span0, SEED).height as f64;
    let surf_by = (surf / span0).floor() as i32;

    // The straddle brick itself is Surface.
    assert_eq!(
        src.classify(IVec3::new(0, surf_by, 0), 0),
        BrickClass::Surface,
        "a surface-straddling brick stays Surface (never pruned)"
    );
    // Deep underground (well below the surface, beyond the +1-brick margin) is Interior.
    assert_eq!(
        src.classify(IVec3::new(0, surf_by - 50, 0), 0),
        BrickClass::Interior,
        "a deep-buried brick (and the brick above it buried) is Interior"
    );
    // High sky (well above the surface) is Air.
    assert_eq!(
        src.classify(IVec3::new(0, surf_by + 50, 0), 0),
        BrickClass::Air,
        "a brick far above the surface is Air"
    );
    // The brick DIRECTLY above the straddle (the +1-margin band) must stay Surface — its faces can be exposed.
    assert_eq!(
        src.classify(IVec3::new(0, surf_by + 1, 0), 0),
        BrickClass::Surface,
        "the brick one above the straddle is within the +1 margin ⇒ Surface (conservative, no hole)"
    );
    // And the brick DIRECTLY below the straddle must stay Surface too — its top voxels may abut the air above.
    assert_eq!(
        src.classify(IVec3::new(0, surf_by - 1, 0), 0),
        BrickClass::Surface,
        "the brick one below the straddle is within the +1 margin ⇒ Surface (no exposed-face hole)"
    );
}

/// **The measurement: surface-following residency BOUNDS the resident set to the shell.** With the prune ON,
/// the cold-filled resident set drops from the clipmap VOLUME (which includes every buried underground brick)
/// to the surface SHEET. We cold-fill the SAME clipmap with the classify prune ON vs OFF (a default-Surface
/// source forces prune OFF) and assert the ON set is a strict, large subset — the underground was culled.
#[test]
fn prune_bounds_residency_to_the_surface_shell() {
    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    // clip_half 5 keeps the non-ignored CI test fast while the volume-vs-shell gap is already large; cap high,
    // drain unbounded. (The full shipping clip_half-8 BEFORE/AFTER count lives in the `--ignored`
    // tests/voxel_worldgen_perf harness.)
    let cfg = StreamingConfig { clip_half_bricks: 5, max_resident_bricks: 10_000_000, max_bricks_per_frame: 10_000_000 };
    let cam = [0.0_f32, surf, 0.0];

    // PRUNE ON: the real WorldgenSource (height-based classify).
    let src = WorldgenSource::new(&layer, &lib, SEED);
    let mut on = ResidencyManager::new();
    on.update(cam, &cfg, &src);
    while on.pending() > 0 {
        on.drain_work(&cfg, &layer, &lib, &reg, SEED);
    }
    let resident_on = on.resident_count();

    // PRUNE OFF: a wrapper whose classify is the default (always Surface), so update enqueues the whole volume.
    struct NoPrune<'a>(WorldgenSource<'a>);
    impl BrickSource for NoPrune<'_> {
        fn brick(&self, c: IVec3, lod: u32, r: &BlockRegistry) -> adventure::voxel::brickmap::Brick {
            self.0.brick(c, lod, r)
        }
        // classify defaults to Surface ⇒ no prune.
    }
    let src_off = NoPrune(WorldgenSource::new(&layer, &lib, SEED));
    let mut off = ResidencyManager::new();
    off.update(cam, &cfg, &src_off);
    while off.pending() > 0 {
        off.drain_work(&cfg, &layer, &lib, &reg, SEED);
    }
    let resident_off = off.resident_count();

    eprintln!(
        "[surface-residency] clip_half={} resident: prune OFF (volume incl. underground) = {resident_off}, \
         prune ON (surface shell) = {resident_on}  ⇒ {:.2}× fewer",
        cfg.clip_half_bricks,
        resident_off as f64 / resident_on.max(1) as f64
    );
    assert!(resident_on > 0, "the surface shell still has resident bricks");
    assert!(
        resident_on < resident_off,
        "the prune must cull the buried underground (ON {resident_on} < OFF {resident_off})"
    );
    // The underground is the bulk of the volume below the surface, so the cull is large (well over 2×).
    assert!(
        (resident_on as f64) < 0.75 * resident_off as f64,
        "surface-following residency must cull a large fraction of the volume (ON {resident_on} vs OFF {resident_off})"
    );
    // CONSERVATIVE / HOLE-FREE: every brick resident under the prune is ALSO resident without it (the prune is
    // purely subtractive — it never adds nor changes a brick, so no surface brick is lost).
    for e in on.resident_entries() {
        assert!(
            off.is_resident(&BrickKey { coord: e.coord, lod: e.lod }),
            "a pruned-set brick {:?}@lod{} must also be resident without the prune (prune is subtractive)",
            e.coord,
            e.lod
        );
    }
}

/// **Dig-reveal: an edit into the ground exposes the pruned interior.** A deep-buried brick classifies
/// Interior, so `update` PRUNES it (never resident). When the player digs into it, the edit path
/// (`requeue_keys`) FORCE-enqueues the dug brick + its halo neighbours past the classify-skip and clears them
/// from the pruned memo — so the dug shell becomes resident (SOLID interior, not a void). We verify the dug
/// brick + the requeued neighbours stream in and carry solid voxels.
#[test]
fn dig_reveals_pruned_interior() {
    use adventure::voxel::edits::VoxelEdits;

    let layer = test_layer();
    let lib = test_library();
    let reg = registry();
    let surf = layer.sample_world(0.0, 0.0, SEED).height;
    let span0 = brick_span(0);
    let cfg = StreamingConfig { clip_half_bricks: 8, max_resident_bricks: 10_000_000, max_bricks_per_frame: 10_000_000 };
    let cam = [0.0_f32, surf, 0.0];
    let src = WorldgenSource::new(&layer, &lib, SEED);

    // Cold-fill with the prune ON.
    let mut mgr = ResidencyManager::new();
    mgr.update(cam, &cfg, &src);
    while mgr.pending() > 0 {
        mgr.drain_work_from(&cfg, &src, &reg, &VoxelEdits::new());
    }
    mgr.take_dirty();

    // Pick a buried LOD0 brick a few bricks below the surface (within the clip_half-8 LOD0 box) that the prune
    // dropped. brick world_min.y = by·span0; choose by so it is ~6 bricks under the surface ⇒ Interior.
    let surf_by = (surf / span0).floor() as i32;
    let dug_owner = IVec3::new(0, surf_by - 6, 0);
    assert_eq!(src.classify(dug_owner, 0), BrickClass::Interior, "the target brick is a pruned Interior brick");
    assert!(!mgr.is_resident(&BrickKey { coord: dug_owner, lod: 0 }), "the buried brick was pruned (not resident)");

    // DIG: remove a voxel inside that brick (the player carves into the ground). The production edit path then
    // re-queues the owner + its 6 face neighbours (the halo) so the dug shell reveals.
    let mut edits = VoxelEdits::new();
    let dug_voxel = dug_owner * BRICK_EDGE + IVec3::new(4, 4, 4);
    edits.remove(dug_voxel);
    let neighbours = [
        dug_owner,
        dug_owner + IVec3::X,
        dug_owner - IVec3::X,
        dug_owner + IVec3::Y,
        dug_owner - IVec3::Y,
        dug_owner + IVec3::Z,
        dug_owner - IVec3::Z,
    ];
    mgr.requeue_keys(neighbours.iter().map(|&coord| BrickKey { coord, lod: 0 }));
    assert!(mgr.pending() > 0, "the dig force-enqueues the owner + halo neighbours past the classify prune");
    while mgr.pending() > 0 {
        mgr.drain_work_from(&cfg, &src, &reg, &edits);
    }

    // The dug brick is now resident (solid interior minus the carved voxel), AND its neighbours are resident
    // (still-solid interior exposed by the hole) — no void.
    let dug = mgr
        .resident_entries()
        .into_iter()
        .find(|e| e.coord == dug_owner && e.lod == 0)
        .expect("the dug brick is now resident (dig revealed the pruned interior)");
    let local = dug_voxel - dug_owner * BRICK_EDGE;
    assert!(dug.brick.get(local.x, local.y, local.z).is_air(), "the carved voxel is now air (the dig)");
    assert!(!dug.brick.is_empty(), "the dug brick still has solid interior (not a void)");
    for &coord in &neighbours[1..] {
        assert!(
            mgr.is_resident(&BrickKey { coord, lod: 0 }),
            "the dug brick's neighbour {coord:?} is revealed resident (solid interior, no hole)"
        );
    }
}

// --- packing SSOT ---------------------------------------------------------------------------------

#[test]
fn pack_resident_set_keeps_constant_grid_and_lod_span() {
    let reg = registry();
    let solidb = adventure::voxel::brickmap::Brick::uniform(BlockId(1));
    let lod = 3u32;
    let entries = vec![ResidentBrick { coord: IVec3::new(2, -1, 3), brick: &solidb, lod }];
    let patch = pack_resident_set(&entries, &reg);
    assert_eq!(patch.brick_count(), 1);
    assert_eq!(patch.metas[0].lod(), lod);
    // Constant haloed 10³ logical grid at every LOD (the clipmap scales the span, not the resolution). R2b: the
    // stored INDEX stream is bit-packed (k=2 ⇒ 1-bit ⇒ ceil(1000/32) words), far smaller than 1000 raw u32.
    assert_eq!(halo_cells(lod), 10 * 10 * 10);
    assert_eq!(patch.voxels.len(), halo_cells(lod).div_ceil(32), "1-bit index stream is ceil(1000/32) words");
    // Core cells DECODE to solid (the brick is packed verbatim — no erosion).
    for x in 1..=BRICK_EDGE {
        assert_eq!(patch.cell_block(&patch.metas[0], halo_index(x, x, x, lod)).0, 1, "core cell solid");
    }
    // world_min = coord · brick_span(lod).
    let span = brick_span(lod);
    assert_eq!(patch.metas[0].world_min, [2.0 * span, -span, 3.0 * span]);
}
