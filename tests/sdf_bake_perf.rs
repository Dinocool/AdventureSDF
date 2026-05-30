//! Performance + correctness harness for the SDF clipmap bake path.
//!
//! Drives the **pure atlas API** (no GPU, no Bevy App) so it is deterministic and
//! fast to iterate. It mirrors what `schedule_bakes` + `apply_bakes` do to the atlas
//! on a camera move (entered-shell bake, exited-shell evict) but runs synchronously
//! and inline, so scheduler logic is exercised without the task pool.
//!
//! All scenarios are `#[ignore]` so a normal `cargo test` stays fast. Run them with:
//!
//! ```sh
//! cargo test --release --test sdf_bake_perf -- --ignored --nocapture
//! ```
//!
//! Each scenario prints (brick count, entered/exited shell size, wall-time) and
//! asserts the incremental result is byte-identical to a from-scratch `full_bake`.

use std::collections::HashSet;
use std::time::Instant;

use bevy::math::bounding::Aabb3d;
use bevy::prelude::*;

use adventure::sdf_render::atlas::{ring_window_coords, coord_in_window, BrickKey, SdfAtlas};
use adventure::sdf_render::bvh::Bvh;
use adventure::sdf_render::edits::{edit_world_aabb, CsgKind, ResolvedEdit, SdfOp, SdfPrimitive};
use adventure::sdf_render::SdfGridConfig;

// --- Harness ----------------------------------------------------------------------

/// Owns the atlas + a snapshot of the edits/BVH, and exposes the same entered/exited
/// shell diff the real scheduler runs, executed synchronously per camera step. This
/// is the single definition of "what a camera move costs" the scenarios measure.
struct BakeHarness {
    config: SdfGridConfig,
    edits: Vec<ResolvedEdit>,
    bvh: Bvh,
    atlas: SdfAtlas,
    cam: Vec3,
}

/// Stats from one incremental recenter step.
#[derive(Default, Debug)]
struct StepStats {
    /// Brick coords that entered a ring window this step (baked or culled-empty).
    entered: usize,
    /// Brick coords that exited a ring window this step (evicted).
    exited: usize,
    /// Resident bricks after the step.
    resident: usize,
    /// Wall-time of the incremental step.
    nanos: u128,
}

impl BakeHarness {
    fn new(config: SdfGridConfig, edits: Vec<ResolvedEdit>, cam0: Vec3) -> Self {
        let bvh = build_bvh(&edits, &config);
        let mut atlas = SdfAtlas::default();
        atlas.full_bake(&edits, &bvh, &config, &adventure::sdf_render::height::HeightField::default(), cam0);
        Self { config, edits, bvh, atlas, cam: cam0 }
    }

    /// Incrementally recenter to `new_cam`: bake bricks that entered any ring window,
    /// evict bricks that exited. Same diff `schedule_bakes`/`apply_bakes` enqueue and
    /// apply (eager eviction). Returns per-step stats.
    fn recenter(&mut self, new_cam: Vec3) -> StepStats {
        let mut stats = StepStats::default();
        let t0 = Instant::now();
        for lod in 0..self.config.lod_count {
            let old_origin = self.config.ring_origin(self.cam, lod);
            let new_origin = self.config.ring_origin(new_cam, lod);
            if old_origin == new_origin {
                continue;
            }
            // Entered → bake (or remove if empty space).
            for coord in ring_window_coords(&self.config, new_origin) {
                if !coord_in_window(&self.config, coord, old_origin) {
                    stats.entered += 1;
                    let key = BrickKey::new(lod, coord);
                    match SdfAtlas::bake_brick(key, &self.edits, &self.bvh, &self.config, &adventure::sdf_render::height::HeightField::default()) {
                        Some(b) => self.atlas.insert_brick(key, b),
                        None => {
                            self.atlas.remove_brick(&key);
                        }
                    }
                }
            }
            // Exited → evict.
            for coord in ring_window_coords(&self.config, old_origin) {
                if !coord_in_window(&self.config, coord, new_origin) {
                    stats.exited += 1;
                    self.atlas.remove_brick(&BrickKey::new(lod, coord));
                }
            }
        }
        self.cam = new_cam;
        stats.nanos = t0.elapsed().as_nanos();
        stats.resident = self.atlas.bricks.len();
        stats
    }

    /// Resident brick key set (for full_bake-equivalence assertions).
    fn brick_set(&self) -> HashSet<BrickKey> {
        self.atlas.bricks.keys().copied().collect()
    }

    /// Time a from-scratch `full_bake` at the current camera and return (set, nanos).
    fn reference_full_bake(&self) -> (HashSet<BrickKey>, u128) {
        let mut reference = SdfAtlas::default();
        let t0 = Instant::now();
        reference.full_bake(&self.edits, &self.bvh, &self.config, &adventure::sdf_render::height::HeightField::default(), self.cam);
        let nanos = t0.elapsed().as_nanos();
        (reference.bricks.keys().copied().collect(), nanos)
    }

    /// Assert the incremental atlas matches a from-scratch full_bake at the current
    /// camera, both in resident set and per-brick distance data.
    fn assert_matches_full_bake(&self, label: &str) {
        let mut reference = SdfAtlas::default();
        reference.full_bake(&self.edits, &self.bvh, &self.config, &adventure::sdf_render::height::HeightField::default(), self.cam);
        let inc = self.brick_set();
        let refk: HashSet<BrickKey> = reference.bricks.keys().copied().collect();
        assert_eq!(inc, refk, "{label}: incremental brick set diverged from full_bake");
        for (key, rb) in &reference.bricks {
            assert_eq!(self.atlas.bricks[key].dist, rb.dist, "{label}: dist mismatch at {key:?}");
        }
    }
}

// --- Edit builders ----------------------------------------------------------------

fn build_bvh(edits: &[ResolvedEdit], _config: &SdfGridConfig) -> Bvh {
    let aabbs: Vec<Aabb3d> = edits
        .iter()
        .map(|e| edit_world_aabb(&e.prim, &e.transform, e.op.smoothing))
        .collect();
    Bvh::build(&aabbs)
}

fn box_edit(pos: Vec3, half: f32, mat: u16) -> ResolvedEdit {
    ResolvedEdit {
        prim: SdfPrimitive::Box { half_extents: Vec3::splat(half) },
        transform: Transform::from_translation(pos),
        op: SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        material_id: mat,
    }
}

/// A long terrain-ish row of boxes spread along X so camera traverse crosses real
/// surface at several LODs.
fn terrain_row(count: i32, spacing: f32, half: f32) -> Vec<ResolvedEdit> {
    (-count..=count)
        .map(|i| box_edit(Vec3::new(i as f32 * spacing, 0.0, 0.0), half, (i.rem_euclid(3)) as u16))
        .collect()
}

/// Small config so scenarios run fast while still crossing multiple LOD boundaries.
fn perf_config() -> SdfGridConfig {
    SdfGridConfig { lod_count: 4, ring_bricks: 8, ..Default::default() }
}

fn report(scenario: &str, line: &str) {
    println!("[sdf-perf] {scenario}: {line}");
}

// --- S1: Static idle --------------------------------------------------------------

#[test]
#[ignore = "perf scenario; run with --ignored --nocapture"]
fn s1_static_idle_does_no_work() {
    let cfg = perf_config();
    let edits = terrain_row(6, 1.5, 0.4);
    let mut h = BakeHarness::new(cfg, edits, Vec3::ZERO);

    // Re-applying the same camera position must enter/exit zero bricks.
    let s = h.recenter(Vec3::ZERO);
    report("S1", &format!("entered={} exited={} resident={} {}ns", s.entered, s.exited, s.resident, s.nanos));
    assert_eq!(s.entered, 0, "idle frame baked bricks");
    assert_eq!(s.exited, 0, "idle frame evicted bricks");
}

// --- S2: Slow pan (sub-brick steps) -----------------------------------------------

#[test]
#[ignore = "perf scenario; run with --ignored --nocapture"]
fn s2_slow_pan_minimal_shell() {
    let cfg = perf_config();
    let edits = terrain_row(8, 1.5, 0.4);
    let mut h = BakeHarness::new(cfg, edits, Vec3::ZERO);

    // Tiny steps; most should produce a zero or one-brick shift at LOD 0 only.
    let mut total_entered = 0;
    for step in 1..=20 {
        let s = h.recenter(Vec3::new(step as f32 * 0.05, 0.0, 0.0));
        total_entered += s.entered;
    }
    report("S2", &format!("total_entered_over_20_steps={total_entered}"));
    h.assert_matches_full_bake("S2");
}

// --- S3: Steady traverse (shell, not volume) --------------------------------------

#[test]
#[ignore = "perf scenario; run with --ignored --nocapture"]
fn s3_steady_traverse_cost_is_shell_not_volume() {
    let cfg = perf_config();
    let r = cfg.ring_bricks as usize;
    let volume = r * r * r; // candidates per LOD ring
    let edits = terrain_row(40, 1.5, 0.4);
    let mut h = BakeHarness::new(cfg, edits, Vec3::ZERO);

    let (_refset, full_nanos) = h.reference_full_bake();

    let mut max_entered = 0;
    let mut max_step_nanos = 0u128;
    let mut cam_x = 0.0f32;
    for _ in 0..40 {
        cam_x += 0.4; // ~ a brick width at LOD 0
        let s = h.recenter(Vec3::new(cam_x, 0.0, 0.0));
        max_entered = max_entered.max(s.entered);
        max_step_nanos = max_step_nanos.max(s.nanos);
    }
    report(
        "S3",
        &format!(
            "max_entered_per_step={max_entered} (ring_volume_per_lod={volume}) max_step={max_step_nanos}ns full_bake={full_nanos}ns",
        ),
    );
    h.assert_matches_full_bake("S3");

    // A single-axis step must expose far less than a full ring volume: the swept shell
    // is O(face) per shifted LOD, never the O(volume) a full bake touches.
    assert!(
        max_entered < volume,
        "traverse step baked {max_entered} bricks (>= ring volume {volume}); shell diff not working"
    );
}

// --- S4: Fast teleport ------------------------------------------------------------

#[test]
#[ignore = "perf scenario; run with --ignored --nocapture"]
fn s4_fast_teleport_converges_to_full_bake() {
    let cfg = perf_config();
    let edits = terrain_row(60, 1.5, 0.4);
    let mut h = BakeHarness::new(cfg, edits, Vec3::ZERO);

    // Jump far past most geometry, then recenter in one incremental step.
    let s = h.recenter(Vec3::new(50.0, 0.0, 0.0));
    report("S4", &format!("teleport entered={} exited={} resident={} {}ns", s.entered, s.exited, s.resident, s.nanos));
    h.assert_matches_full_bake("S4 after teleport");

    // And back to the origin — no stale bricks may survive the round trip.
    let s2 = h.recenter(Vec3::ZERO);
    report("S4", &format!("return entered={} exited={} resident={}", s2.entered, s2.exited, s2.resident));
    h.assert_matches_full_bake("S4 after return");
}

// --- S5: Edit-set unchanged, camera moving (resident integrity) -------------------

#[test]
#[ignore = "perf scenario; run with --ignored --nocapture"]
fn s5_camera_move_keeps_resident_set_exact() {
    let cfg = perf_config();
    let edits = terrain_row(30, 1.2, 0.5);
    let mut h = BakeHarness::new(cfg, edits, Vec3::ZERO);

    // Walk a winding path (all three axes) and verify exactness at every stop.
    let path = [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 1.5, 0.0),
        Vec3::new(2.0, 1.5, 3.0),
        Vec3::new(-4.0, 1.5, 3.0),
        Vec3::new(-4.0, -2.0, -2.0),
        Vec3::ZERO,
    ];
    for (i, p) in path.iter().enumerate() {
        h.recenter(*p);
        h.assert_matches_full_bake(&format!("S5 stop {i}"));
    }
    report("S5", &format!("walked {} stops, all exact", path.len()));
}

// --- S6: LOD scaling curve --------------------------------------------------------

#[test]
#[ignore = "perf scenario; run with --ignored --nocapture"]
fn s6_lod_scaling_curve() {
    let edits = terrain_row(60, 1.5, 0.4);
    for lod_count in [1u32, 3, 5, 8] {
        for ring_bricks in [8u32, 16] {
            let cfg = SdfGridConfig { lod_count, ring_bricks, ..Default::default() };
            let mut h = BakeHarness::new(cfg, edits.clone(), Vec3::ZERO);
            let (_refset, full_nanos) = h.reference_full_bake();

            // One representative traverse step.
            let mut max_step = 0u128;
            let mut cam_x = 0.0;
            for _ in 0..8 {
                cam_x += 0.4;
                let s = h.recenter(Vec3::new(cam_x, 0.0, 0.0));
                max_step = max_step.max(s.nanos);
            }
            report(
                "S6",
                &format!("lod_count={lod_count} ring_bricks={ring_bricks} resident={} full_bake={full_nanos}ns max_step={max_step}ns", h.atlas.bricks.len()),
            );
            h.assert_matches_full_bake(&format!("S6 lod={lod_count} ring={ring_bricks}"));
        }
    }
}
