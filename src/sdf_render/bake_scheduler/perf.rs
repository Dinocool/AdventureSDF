//! SDF bake **performance harness** for the stress scene.
//!
//! Two questions drive SDF bake tuning, and this rig measures both against the REAL production
//! topology code (`emit_gpu_bakes` — no ECS App, no GPU/window needed, because
//! the CPU half is what steals main-thread frame time):
//!
//!   1. **Time-to-bake** — from an empty atlas, how long until the whole stress field is resident?
//!      Reported as frames-to-settle, total main-thread CPU time, and the worst single-frame hitch.
//!   2. **Frame-rate impact** — once settled, as the camera flies through the field one step per
//!      frame, how much main-thread time does each frame's recenter + bake cost? Reported as a
//!      distribution (mean / p50 / p99 / max) — the **max is the visible hitch** we want to shrink.
//!
//! It complements [`super::tests::lod_recenter_cost_walk`] (the recenter-cost rig): this drives the
//! production `emit_gpu_bakes` — bounded batch, PARALLEL classify, apply ≤ soft-budget/frame, with
//! over-budget Keeps carried in `ready` — exactly as the running app does, one call per frame.
//!
//! Run it (it is `#[ignore]`, like every perf rig here):
//! ```sh
//! cargo test -p adventure --release sdf_render::bake_scheduler::perf -- --ignored --nocapture
//! ```
//! It prints a `BAKE-PERF` report and writes `.soul/bake_perf.json` so a before/after change can be
//! diffed numerically (re-run after a tuning change and compare the two JSON blobs).

use super::*;
use crate::sdf_render::edits::{CsgKind, ResolvedEdit};

/// Map the shared `tower_field` stress geometry to `ResolvedEdit`s EXACTLY as the runtime
/// `TowerSpawner` does (`Union`, no smoothing; roles → distinct material ids — see
/// `stress::expand_tower_spawners`), so the bake cost measured here is the real scene's.
fn stress_edits() -> Vec<ResolvedEdit> {
    use tower_field::TowerRole;
    tower_field::tower_field_edits(&tower_field::TowerFieldParams::default())
        .into_iter()
        .map(|(_order, transform, prim, role)| {
            let mat = match role {
                TowerRole::Ground => 0u16,
                TowerRole::Cube => 1,
                TowerRole::Cap => 2,
            };
            ResolvedEdit::new(prim, transform, SdfOp { kind: CsgKind::Union, smoothing: 0.0 }, mat)
        })
        .collect()
}

/// Per-frame main-thread bake samples gathered over a scenario, plus the derived summary. Times are
/// microseconds of MAIN-THREAD work (the whole bake — gather + parallel classify + apply — runs on
/// the calling thread per frame; the parallel classify blocks it only while the compute pool works).
#[derive(Default)]
struct Samples {
    /// Main-thread `emit_gpu_bakes` time per frame (µs).
    dispatch_us: Vec<u128>,
    /// `recenter_step` time per frame (µs) — 0 for the cold-bake (no camera motion).
    recenter_us: Vec<u128>,
    /// GPU bake jobs emitted per frame (the GPU-dispatch + atlas-copy load proxy).
    jobs: Vec<usize>,
}

impl Samples {
    fn push(&mut self, recenter_us: u128, dispatch_us: u128, jobs: usize) {
        self.recenter_us.push(recenter_us);
        self.dispatch_us.push(dispatch_us);
        self.jobs.push(jobs);
    }
}

/// `p`-percentile (0..=100) of `xs` by value. `xs` is sorted in place. Empty → 0.
fn pct(xs: &mut [u128], p: u32) -> u128 {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    // Nearest-rank: ceil(p/100 * n) - 1, clamped into range.
    let n = xs.len();
    let rank = ((p as usize * n).div_ceil(100)).clamp(1, n) - 1;
    xs[rank]
}

fn sum(xs: &[u128]) -> u128 {
    xs.iter().copied().sum()
}

fn max(xs: &[u128]) -> u128 {
    xs.iter().copied().max().unwrap_or(0)
}

/// Per-frame main-thread bake total = recenter + dispatch (what a single real frame would pay).
fn frame_totals(s: &Samples) -> Vec<u128> {
    s.recenter_us
        .iter()
        .zip(&s.dispatch_us)
        .map(|(r, d)| r + d)
        .collect()
}

/// One scenario's printed + JSON summary line. Holds the headline numbers only (the raw samples
/// stay in `Samples`).
struct Summary {
    label: &'static str,
    frames: usize,
    /// Frames that emitted ≥1 job (a budget-spent frame can bake 0 while it drains carried work).
    productive_frames: usize,
    total_ms: f64,
    mean_us: u128,
    p50_us: u128,
    p99_us: u128,
    max_us: u128,
    jobs_total: usize,
    jobs_max: usize,
}

fn summarize(label: &'static str, s: &Samples) -> Summary {
    let mut totals = frame_totals(s);
    let frames = totals.len();
    let total = sum(&totals);
    let mean = if frames == 0 { 0 } else { total / frames as u128 };
    Summary {
        label,
        frames,
        productive_frames: s.jobs.iter().filter(|&&j| j > 0).count(),
        total_ms: total as f64 / 1000.0,
        mean_us: mean,
        p50_us: pct(&mut totals, 50),
        p99_us: pct(&mut totals, 99),
        max_us: max(&frame_totals(s)),
        jobs_total: s.jobs.iter().sum(),
        jobs_max: s.jobs.iter().copied().max().unwrap_or(0),
    }
}

impl Summary {
    fn print(&self) {
        eprintln!(
            "BAKE-PERF [{}]: frames={} (productive={}) main_thread_total={:.2}ms | per-frame us: mean={} p50={} p99={} MAX={} | jobs total={} max/frame={}",
            self.label,
            self.frames,
            self.productive_frames,
            self.total_ms,
            self.mean_us,
            self.p50_us,
            self.p99_us,
            self.max_us,
            self.jobs_total,
            self.jobs_max,
        );
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"label\":\"{}\",\"frames\":{},\"productive_frames\":{},\"main_thread_total_ms\":{:.3},\"per_frame_us\":{{\"mean\":{},\"p50\":{},\"p99\":{},\"max\":{}}},\"jobs\":{{\"total\":{},\"max_per_frame\":{}}}}}",
            self.label, self.frames, self.productive_frames, self.total_ms, self.mean_us, self.p50_us, self.p99_us, self.max_us, self.jobs_total, self.jobs_max,
        )
    }
}

/// SCENARIO A — cold bake: from an empty atlas, recenter once at a fixed camera, then drain to
/// convergence, timing every frame. Drives the production `emit_gpu_bakes` (bounded batch, PARALLEL
/// classify, apply ≤ soft-budget/frame, over-budget Keeps carried in `ready`) — each call is exactly
/// one frame, so this is "total CPU work to bake the scene from cold + the worst per-frame hitch".
/// Leaves `atlas`/`sched` settled at `cam` for the flythrough.
fn cold_bake(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    cfg: &SdfGridConfig,
    cam: Vec3,
    dbg: &mut crate::sdf_render::BakedBrickDebug,
) -> Samples {
    let mut s = Samples::default();
    let t_recenter = std::time::Instant::now();
    recenter_step(sched, atlas, cfg, cam);
    let first_recenter_us = t_recenter.elapsed().as_micros();

    let mut gpu = PendingGpuBakes::default();
    let mut guard = 0u32;
    let mut first = true;
    loop {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        let t = std::time::Instant::now();
        emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, dbg, 0.0);
        let us = t.elapsed().as_micros();
        // Attribute the one-time window recenter to the first frame (recenter + emit share a frame
        // in the real `schedule_bakes` body too).
        s.push(if first { first_recenter_us } else { 0 }, us, gpu.jobs.len());
        first = false;
        guard += 1;
        assert!(guard < 50_000, "cold bake did not converge");
        if sched.pending.is_empty() && sched.ready.is_empty() {
            break;
        }
    }
    s
}

/// SCENARIO C — small edit: from the settled atlas, nudge ONE cube edit and re-settle, measuring the
/// main-thread cost. The "small edits must stay extremely quick" guard: only the moved cube's own
/// footprint should re-bake (the content-hash skip keeps everything else free), so it must settle in
/// a frame or two at sub-millisecond cost regardless of scene size. Mirrors `schedule_bakes` step 1
/// (rebuild edits+BVH, bump edit_gen, dirty the old∪new footprint within each LOD window).
fn small_edit(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    cfg: &SdfGridConfig,
    cam: Vec3,
    base_edits: &[ResolvedEdit],
    move_index: usize,
    dbg: &mut crate::sdf_render::BakedBrickDebug,
) -> Samples {
    let old = base_edits[move_index].clone();
    let mut moved_tf = old.transform;
    moved_tf.translation += Vec3::new(0.3, 0.0, 0.0);
    let new = ResolvedEdit::new(old.prim.clone(), moved_tf, old.op, old.material_id);
    let mut new_edits = base_edits.to_vec();
    new_edits[move_index] = new.clone();
    sched.edits = std::sync::Arc::new(new_edits);
    sched.bvh = std::sync::Arc::new(build_bvh(&sched.edits));
    sched.edit_gen = sched.edit_gen.wrapping_add(1);

    // Surface-pruned dirty over the moved edit's old∪new position (mirror schedule_bakes step 1).
    dirty_moving_edit(&mut sched.pending, &old, &new, cfg, cam);

    // Re-settle via the sync path (a small footprint never crosses the async threshold), timing each.
    let mut s = Samples::default();
    let mut gpu = PendingGpuBakes::default();
    let mut guard = 0u32;
    loop {
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        let t = std::time::Instant::now();
        emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, dbg, 0.0);
        s.push(0, t.elapsed().as_micros(), gpu.jobs.len());
        guard += 1;
        assert!(guard < 10_000, "small edit did not converge");
        if sched.pending.is_empty() && sched.ready.is_empty() {
            break;
        }
    }
    s
}

/// Total set bits across a `pending` map = the candidate BRICKS this frame's bake will gather/classify
/// (an empty chunk's bits become evictions, not classifies, but those are a small fraction). The
/// headline number Option 4 (brick-level dirtying) drives down for a small-edit drag.
fn pending_dirty_bricks(sched: &BakeScheduler) -> usize {
    sched.pending.values().map(|m| m.count_ones() as usize).sum()
}

/// SCENARIO D — continuous drag: the real "fps drop while moving a small sphere" case the trace
/// showed. From the settled atlas, move ONE edit ~0.3 u/frame for `steps` frames and run ONE bake
/// frame each step WITHOUT settling between — so the moved edit's old∪new footprint is re-dirtied and
/// re-classified every frame, exactly as a live drag does (vs `small_edit`, which nudges once and
/// settles). Mirrors `schedule_bakes` step-1 incremental path: bump `edit_gen`, refit the moved BVH
/// leaf, dirty old∪new, reconcile the carry queue. Returns the per-frame samples AND the dirty-brick
/// count per frame (the candidate volume — pre-Option-4 this was ~thousands for a tiny sphere).
fn continuous_drag(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    cfg: &SdfGridConfig,
    cam: Vec3,
    move_index: usize,
    steps: usize,
    dbg: &mut crate::sdf_render::BakedBrickDebug,
) -> (Samples, Vec<usize>) {
    let mut s = Samples::default();
    let mut dirty_bricks = Vec::with_capacity(steps);
    let mut gpu = PendingGpuBakes::default();
    for i in 1..=steps {
        // Move the edit a little (sub-voxel at coarse LODs, supra-voxel at LOD0 — the realistic drag).
        let old = sched.edits[move_index].clone();
        let mut tf = old.transform;
        tf.translation += Vec3::new(0.3, 0.0, 0.0);
        let new_aabb = edits::edit_world_aabb(&old.prim, &tf, old.op.smoothing);
        let resolved = ResolvedEdit::new(old.prim.clone(), tf, old.op, old.material_id);
        dirty_moving_edit(&mut sched.pending, &old, &resolved, cfg, cam);
        std::sync::Arc::make_mut(&mut sched.edits)[move_index] = resolved;
        std::sync::Arc::make_mut(&mut sched.bvh).refit_edit(move_index as u32, new_aabb);
        sched.edit_gen = sched.edit_gen.wrapping_add(1);
        invalidate_ready_on_edit_change(sched, cfg, false);
        // Suppress the unused warning on the loop counter while keeping the 1-based intent.
        let _ = i;

        dirty_bricks.push(pending_dirty_bricks(sched));
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        let t = std::time::Instant::now();
        emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, dbg, 0.0);
        s.push(0, t.elapsed().as_micros(), gpu.jobs.len());
    }
    (s, dirty_bricks)
}

/// SCENARIO B — flythrough: from the settled state, step the camera `steps` × `step_m` along +X,
/// one `recenter_step` + one `emit_gpu_bakes` per frame (NO drain — real frames do exactly one),
/// timing each. The MAX per-frame total is the streaming hitch we want to minimize; jobs/frame is
/// the GPU load. A few snap-boundary frames enqueue a whole coarse shell — those are the spikes.
fn flythrough(
    atlas: &mut SdfAtlas,
    sched: &mut BakeScheduler,
    cfg: &SdfGridConfig,
    start: Vec3,
    step_m: f32,
    steps: usize,
    dbg: &mut crate::sdf_render::BakedBrickDebug,
) -> Samples {
    let mut s = Samples::default();
    let mut gpu = PendingGpuBakes::default();
    for i in 1..=steps {
        let cam = start + Vec3::new(i as f32 * step_m, 0.0, 0.0);
        let t_r = std::time::Instant::now();
        recenter_step(sched, atlas, cfg, cam);
        let recenter_us = t_r.elapsed().as_micros();
        gpu.jobs.clear();
        gpu.edits.clear();
        atlas.gpu_baked_tiles.clear();
        let t = std::time::Instant::now();
        emit_gpu_bakes(atlas, sched, &mut gpu, cfg, cam, dbg, 0.0);
        s.push(recenter_us, t.elapsed().as_micros(), gpu.jobs.len());
    }
    s
}

/// Append a JSON line `{config, edits, resident, scenarios:[...]}` to `.soul/bake_perf.json`
/// (best-effort — a write failure just warns; it must never fail the rig). Each run overwrites, so
/// the file always holds the latest baseline; copy it aside before a tuning change to diff.
fn write_json(edit_count: usize, resident: usize, summaries: &[Summary]) {
    let scenarios = summaries.iter().map(Summary::to_json).collect::<Vec<_>>().join(",");
    let body = format!(
        "{{\"edits\":{},\"resident_bricks\":{},\"knobs\":{{\"soft_bake_budget\":{},\"classify_refill_chunks\":{},\"gpu_job_cap\":{}}},\"scenarios\":[{}]}}\n",
        edit_count, resident, SOFT_BAKE_BUDGET, CLASSIFY_REFILL_CHUNKS, GPU_BAKE_JOB_CAP, scenarios,
    );
    if let Err(e) = std::fs::create_dir_all(".soul").and_then(|()| std::fs::write(".soul/bake_perf.json", &body)) {
        eprintln!("BAKE-PERF: could not write .soul/bake_perf.json: {e}");
    } else {
        eprintln!("BAKE-PERF: wrote .soul/bake_perf.json");
    }
}

#[test]
#[ignore = "perf measurement rig; run explicitly with --release --ignored --nocapture"]
fn bake_perf_stress_scene() {
    let cfg = SdfGridConfig::default(); // production: 8 LODs, default ring/snap.
    let edits = stress_edits();
    eprintln!(
        "BAKE-PERF: stress scene = {} edits | config: {} LODs, ring_bricks={}, snap={} | knobs: soft_budget={} refill_chunks={} gpu_cap={}",
        edits.len(),
        cfg.lod_count,
        cfg.ring_bricks,
        cfg.recenter_snap_chunks,
        SOFT_BAKE_BUDGET,
        CLASSIFY_REFILL_CHUNKS,
        GPU_BAKE_JOB_CAP,
    );

    let mut dbg = crate::sdf_render::BakedBrickDebug::default(); // disabled — no debug-marker overhead

    // A representative play viewpoint over the field (matches the recenter-walk rig's camera).
    let cam0 = Vec3::new(0.0, 5.0, 10.0);

    // Cold bake from empty (total CPU + frames-to-settle + worst per-frame hitch).
    let mut atlas = SdfAtlas::default();
    let mut sched = primed_sched(&edits);
    let cold = cold_bake(&mut atlas, &mut sched, &cfg, cam0, &mut dbg);
    let resident = atlas.bricks.len();
    eprintln!("BAKE-PERF: cold bake settled — {resident} resident bricks");

    // Material-tile reclamation histogram: only MULTI-material bricks own a material atlas tile, so
    // this quantifies the VRAM win (single-material bricks store 0 material bytes). dist = R16Snorm
    // (2 B/voxel), mat = Rgba16Snorm (8 B/voxel), 512 voxels/brick.
    let multi = atlas.mat_tiles.len();
    let single = resident.saturating_sub(multi);
    let voxels = 512u64;
    let dist_mb = resident as u64 * voxels * 2 / (1 << 20);
    let mat_now_mb = multi as u64 * voxels * 8 / (1 << 20);
    let mat_old_mb = resident as u64 * voxels * 8 / (1 << 20); // pre-reclamation (every brick)
    let pct = if resident > 0 { single * 100 / resident } else { 0 };
    eprintln!(
        "BAKE-PERF [material-reclaim]: {single}/{resident} bricks single-material ({pct}%), {multi} multi \
         | material VRAM {mat_old_mb} MB -> {mat_now_mb} MB (dist {dist_mb} MB unchanged)"
    );

    // SMALL EDIT from the settled state at cam0 — nudge a cube near the camera and re-settle.
    let move_index = edits
        .iter()
        .position(|e| matches!(e.prim, SdfPrimitive::Box { .. }) && e.transform.translation.distance(cam0) < 60.0)
        .unwrap_or(1);
    let small = small_edit(&mut atlas, &mut sched, &cfg, cam0, &edits, move_index, &mut dbg);

    // CONTINUOUS DRAG (the "fps drop while moving a small sphere" case) from the settled state at
    // cam0 — 60 frames, ~0.3 u/frame, re-classifying the footprint each frame (no settle between).
    let (drag, drag_bricks) = continuous_drag(&mut atlas, &mut sched, &cfg, cam0, move_index, 60, &mut dbg);
    let drag_bricks_mean = if drag_bricks.is_empty() { 0 } else { drag_bricks.iter().sum::<usize>() / drag_bricks.len() };
    let drag_bricks_max = drag_bricks.iter().copied().max().unwrap_or(0);
    // Drain back to a clean settled state so the flythrough starts from the same point as the baseline.
    {
        let mut gpu = PendingGpuBakes::default();
        let mut g = 0u32;
        while !(sched.pending.is_empty() && sched.ready.is_empty()) {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam0, &mut dbg, 0.0);
            g += 1;
            assert!(g < 50_000, "post-drag settle did not converge");
        }
    }

    // FLYTHROUGH (streaming frame-impact) continues from the settled state.
    let fly = flythrough(&mut atlas, &mut sched, &cfg, cam0, 1.5, 120, &mut dbg);

    // MASSIVE-SPHERE DRAG (the limit test): swap the dragged edit for a window-spanning sphere,
    // settle it, then drag it 60 frames. The worst case for footprint dirtying — surface-pruning
    // (`dirty_moving_edit`) must keep per-frame cost O(surface shell), NOT O(solid volume): a naive
    // solid-AABB footprint would dirty ~the whole window every frame (the 25 ms hitch in the trace).
    let (mdrag, mbricks) = {
        let big = ResolvedEdit::new(
            SdfPrimitive::Sphere { radius: 40.0 },
            Transform::from_translation(cam0 + Vec3::new(0.0, 0.0, -20.0)),
            sched.edits[move_index].op,
            sched.edits[move_index].material_id,
        );
        let big_aabb = edits::edit_world_aabb(&big.prim, &big.transform, 0.0);
        let old = sched.edits[move_index].clone();
        dirty_moving_edit(&mut sched.pending, &old, &big, &cfg, cam0);
        std::sync::Arc::make_mut(&mut sched.edits)[move_index] = big;
        std::sync::Arc::make_mut(&mut sched.bvh).refit_edit(move_index as u32, big_aabb);
        sched.edit_gen = sched.edit_gen.wrapping_add(1);
        invalidate_ready_on_edit_change(&mut sched, &cfg, false);
        let mut gpu = PendingGpuBakes::default();
        let mut g = 0u32;
        while !(sched.pending.is_empty() && sched.ready.is_empty()) {
            gpu.jobs.clear();
            gpu.edits.clear();
            atlas.gpu_baked_tiles.clear();
            emit_gpu_bakes(&mut atlas, &mut sched, &mut gpu, &cfg, cam0, &mut dbg, 0.0);
            g += 1;
            assert!(g < 200_000, "massive-sphere settle did not converge");
        }
        continuous_drag(&mut atlas, &mut sched, &cfg, cam0, move_index, 60, &mut dbg)
    };
    let mbricks_mean = if mbricks.is_empty() { 0 } else { mbricks.iter().sum::<usize>() / mbricks.len() };
    let mbricks_max = mbricks.iter().copied().max().unwrap_or(0);

    eprintln!(
        "BAKE-PERF [continuous-drag]: dirty bricks/frame mean={drag_bricks_mean} max={drag_bricks_max} (candidates classified per drag frame)"
    );
    eprintln!(
        "BAKE-PERF [massive-drag]: dirty bricks/frame mean={mbricks_mean} max={mbricks_max} (radius-40 window-spanning sphere — surface-prune target)"
    );
    let summaries = [
        summarize("cold-bake", &cold),
        summarize("small-edit", &small),
        summarize("continuous-drag", &drag),
        summarize("massive-drag", &mdrag),
        summarize("flythrough", &fly),
    ];
    for sm in &summaries {
        sm.print();
    }
    write_json(edits.len(), resident, &summaries);
}
