//! **Bistro-interior FPS benchmark harness** (dev-only, editor-gated, fully env-driven).
//!
//! The MEASUREMENT GATE for the voxel-RT perf goal "get the Bistro-interior view to 165 FPS". It is purely
//! additive — installed only when `ADVENTURE_BENCH_BISTRO=1` (see `main.rs`) — and changes NOTHING about the
//! renderer. It measures; it does not optimize.
//!
//! Wiring (all env-driven, so a single launch is fully deterministic + self-terminating):
//! * `ADVENTURE_BENCH_BISTRO=1` — (a) boots the GALLERY scene, which with this env set loads **Bistro ALONE at
//!   origin** (see [`crate::voxel::gallery::bistro_bench_placements`] + `raytrace::stream_voxel_rt_residency`);
//!   (b) installs the camera-pin / FPS-report / screenshot systems here.
//! * `ADVENTURE_CAM="tx,ty,tz,dist,yaw,pitch"` — pins a FIXED interior view: target `(tx,ty,tz)`, orbit
//!   `distance`, `yaw`/`pitch` (radians). The eye is the orbit eye `target + dir·distance`; with `pitch≈0` the
//!   eye sits at target height looking horizontally. Pinned EVERY frame straight onto the [`SdfCamera`]
//!   transform (independent of the editor's input-gated `orbit_camera`), so no input is needed and it never
//!   drifts.
//! * `ADVENTURE_EXIT_AFTER_SECS=N` — the run length (handled in `main.rs`); the report + screenshot fire
//!   relative to this deadline.
//!
//! At exit the harness logs `BENCH RESULT: bistro-interior avg_fps=<X> frame_time_ms=<Y> (over <n> frames)`,
//! averaging the smoothed `FrameTimeDiagnosticsPlugin::FPS` over ONLY the steady-state window (the last ~5 s
//! before exit — after the scene has streamed in + settled), and saves the final frame to
//! `D:/tmp_test/bistro_bench.png` for a visual check that the view is dense interior geometry.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

use crate::sdf_render::SdfCamera;

/// Where the steady-state averaging window begins, measured as seconds BEFORE the `ADVENTURE_EXIT_AFTER_SECS`
/// exit. The scene streams in over the first few seconds; we exclude that and average only the settled tail.
const STEADY_WINDOW_SECS: f32 = 5.0;

/// Seconds before exit to trigger the final-frame screenshot (a couple of frames of slack so the PNG write +
/// readback complete before the app quits).
const SCREENSHOT_LEAD_SECS: f32 = 1.0;

/// Output path for the final-frame screenshot (matches the harness spec).
const SCREENSHOT_PATH: &str = "D:/tmp_test/bistro_bench.png";

/// Parsed `ADVENTURE_CAM` view + the exit deadline, plus the running FPS accumulator for the steady-state
/// window. Inserted as a resource so the per-frame systems share it.
#[derive(Resource, Default)]
struct BistroBench {
    /// `ADVENTURE_EXIT_AFTER_SECS`, if set — the run deadline the report/screenshot key off of.
    exit_at: Option<f32>,
    /// `ADVENTURE_CAM` as `(eye, look_at)` world points — pinned directly onto the SdfCamera transform each
    /// frame (immune to the editor plugin's runtime orbit reset, which overrode the old shared-resource pin).
    cam: Option<(Vec3, Vec3)>,
    /// Sum of smoothed-FPS samples taken inside the steady-state window.
    fps_sum: f64,
    /// Number of steady-state samples accumulated (the `<n> frames` in the report line).
    fps_samples: u64,
    /// Whether the screenshot has been requested already (fire-once latch).
    shot_fired: bool,
    /// Whether the final `BENCH RESULT:` line has been logged already (fire-once latch).
    reported: bool,
    /// **G-c.4** — the MAX frame-time (ms) seen inside the steady-state window (the HITCH metric: a classify
    /// freeze on a brick crossing shows up here as a 100+ ms spike that the avg-FPS hides). 0 ⇒ no samples yet.
    max_frame_ms: f64,
    /// **G-c.4** — the scripted MOVING-camera fly-through (`ADVENTURE_CAM_PATH="x0,y0,z0,lx,ly,lz; …"` of ≥2
    /// keyframe eye+look_at points). The camera lerps across them over the run so it CROSSES brick boundaries
    /// (the only way to surface the classify freeze — a static pin can't). `None` ⇒ use the fixed `ADVENTURE_CAM`.
    cam_path: Option<Vec<(Vec3, Vec3)>>,
}

/// Install the Bistro bench: boot the gallery (⇒ Bistro-alone via the env), pin the camera from
/// `ADVENTURE_CAM`, and add the per-frame sampler / screenshot / report systems. Editor build only.
pub fn install_bistro_bench(app: &mut App) {
    // Boot the bench scene. Default GALLERY (⇒ Bistro alone at origin, the streamed `.vxo` path). `ADVENTURE_
    // BENCH_SCENE=sponza` instead boots the IN-RAM `.vox` Sponza — the scene whose occupancy + core store the
    // GPU front end CAN bind today (the streamed `.vxo` per-region paging is the remaining piece). Used to
    // validate the live GPU-driven drive end-to-end (no classify freeze + convergence) on the in-RAM path.
    let scene = match std::env::var("ADVENTURE_BENCH_SCENE").ok().as_deref() {
        Some("sponza") => {
            info!("bench: ADVENTURE_BENCH_SCENE=sponza — booting the IN-RAM .vox Sponza (GPU-residency-bindable)");
            crate::voxel::VoxelScene::Sponza
        }
        _ => crate::voxel::VoxelScene::Gallery,
    };
    app.insert_resource(scene);

    // **G-c.4** — `ADVENTURE_GPU_RESIDENCY=1` flips on the GPU-driven readback-free residency front end (the
    // A/B toggle). When set, the residency DECISION + pack + AABB-fill + BLAS rebuild all run on the GPU each
    // frame, eliminating the CPU `vox_residency_classify` freeze (the whole point of the A/B). Inserted as the
    // initial `VoxelRtToggle` so it is on from the first streamed frame. Unset ⇒ the CPU path (the A side).
    if std::env::var("ADVENTURE_GPU_RESIDENCY").is_ok() {
        info!("bench: ADVENTURE_GPU_RESIDENCY set — GPU-driven residency front end ON (readback-free)");
        app.insert_resource(crate::voxel::raytrace::VoxelRtToggle {
            enabled: true,
            gpu_pack: false,
            gpu_residency: true,
        });
    } else {
        // A-side baseline: the pure CPU residency + CPU pack path. Set EXPLICITLY (not via the live default,
        // which is now `gpu_pack: true`) so the A/B comparison measures CPU-pack vs the GPU residency front end.
        app.insert_resource(crate::voxel::raytrace::VoxelRtToggle {
            enabled: true,
            gpu_pack: false,
            gpu_residency: false,
        });
    }

    // Turn on the lightweight CPU+GPU span instrumentation so the render world's per-pass GPU timestamp
    // read-back (`instrument::record_gpu`) populates — `bench_diag` logs a non-draining peek each tick.
    crate::instrument::set_enabled(true);

    // ADVENTURE_CLIP_HALF=N overrides the clipmap half-extent (bricks). The default 160 ⇒ a 64 m LOD0 ring,
    // which over a dense scene like Bistro means ~300k resident bricks (0.05 m detail 64 m out is invisible) and
    // perpetual streaming churn. A smaller ring converges far faster + cheaper. Inserted before Startup so
    // `init_voxel_rt_streaming` picks it up as the cfg override (the SSOT streaming knob).
    // ADVENTURE_STREAM_BUDGET=N overrides `max_bricks_per_frame` (default 256). A HIGH value (e.g. 30000) loads
    // the whole clipmap in a few frames — one big initial hitch, then a FULLY-RESIDENT, STABLE scene so capture
    // frames measure the converged steady state (NOT a mid-stream frame with a streaming hitch). Essential for
    // valid perf A/B at high brick counts.
    let clip = std::env::var("ADVENTURE_CLIP_HALF").ok().and_then(|s| s.trim().parse::<i32>().ok());
    let budget = std::env::var("ADVENTURE_STREAM_BUDGET").ok().and_then(|s| s.trim().parse::<usize>().ok());
    if clip.is_some() || budget.is_some() {
        let mut cfg = crate::voxel::streaming::StreamingConfig::default();
        if let Some(n) = clip { cfg.clip_half_bricks = n; }
        if let Some(b) = budget { cfg.max_bricks_per_frame = b; }
        info!(
            "bench: streaming override → clip_half_bricks={} max_bricks_per_frame={}",
            cfg.clip_half_bricks, cfg.max_bricks_per_frame
        );
        app.insert_resource(cfg);
    }

    let exit_at = std::env::var("ADVENTURE_EXIT_AFTER_SECS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok());
    if exit_at.is_none() {
        warn!(
            "bench: ADVENTURE_BENCH_BISTRO set but ADVENTURE_EXIT_AFTER_SECS is not — the run won't self-exit \
             and no BENCH RESULT will be logged. Set e.g. ADVENTURE_EXIT_AFTER_SECS=20."
        );
    }

    // **G-c.4** — the MOVING-camera fly-through takes priority over the fixed pin: a static pin can't surface the
    // classify freeze (no brick crossings). `ADVENTURE_CAM_PATH` is `;`-separated keyframes, each six floats
    // `ex,ey,ez,lx,ly,lz` (eye + look_at); the camera lerps across them over the whole run.
    let cam_path = parse_adventure_cam_path();
    let cam = parse_adventure_cam();
    if let Some(path) = &cam_path {
        info!("bench: ADVENTURE_CAM_PATH set — {} keyframe(s), MOVING fly-through (crosses brick boundaries)", path.len());
        app.add_systems(Update, pin_bench_camera);
    } else if let Some((eye, look)) = cam {
        info!(
            "bench: ADVENTURE_CAM eye=({:.1},{:.1},{:.1}) look_at=({:.1},{:.1},{:.1}) (STATIC pin)",
            eye.x, eye.y, eye.z, look.x, look.y, look.z
        );
        app.add_systems(Update, pin_bench_camera);
    } else {
        warn!(
            "bench: neither ADVENTURE_CAM_PATH nor ADVENTURE_CAM set — using the default orbit view, which likely \
             is NOT inside Bistro (and won't move, so it can't surface a classify freeze)."
        );
    }
    app.insert_resource(BistroBench { exit_at, cam, cam_path, ..default() });

    app.add_systems(Update, (sample_fps, fire_screenshot, report_at_exit, bench_diag));
    // ADVENTURE_DEBUG_VIEW=N forces the shader debug-view selector each frame (0=lit, 1=normals, 2=depth,
    // 3=albedo, 4=AO, 5=GI-only, 6=face-orient, 7=LOD). Albedo/normals bypass lighting — a raw geometry probe.
    // ADVENTURE_GI_RAYS=N forces the ReSTIR initial-candidate count (the p1 cost driver). Either triggers the
    // per-frame lighting-override system.
    if std::env::var("ADVENTURE_DEBUG_VIEW").is_ok()
        || std::env::var("ADVENTURE_GI_RAYS").is_ok()
        || std::env::var("ADVENTURE_WC").is_ok()
    {
        app.add_systems(Update, force_lighting_overrides);
    }
}

/// Ground-truth diagnostic: every ~3 s log the actual SdfCamera world position + forward and the
/// VoxelRtPatch generation (0 ⇒ nothing has streamed/packed yet). Tells us camera-in-empty-space vs
/// geometry-not-streaming when the view is blank.
fn bench_diag(
    time: Res<Time>,
    mut last: Local<f32>,
    cam: Query<&GlobalTransform, With<SdfCamera>>,
    patch: Option<Res<crate::voxel::raytrace::VoxelRtPatch>>,
    streaming: Option<Res<crate::voxel::raytrace::VoxelRtStreaming>>,
) {
    let now = time.elapsed_secs();
    if now - *last < 3.0 {
        return;
    }
    *last = now;
    let (pos, fwd) = match cam.iter().next() {
        Some(g) => (g.translation(), g.forward().as_vec3()),
        None => (Vec3::splat(-999.0), Vec3::ZERO),
    };
    let pgen = patch.as_ref().map(|p| p.generation).unwrap_or(u64::MAX);
    let resident = streaming.as_ref().map(|s| s.manager().resident_count()).unwrap_or(usize::MAX);
    let aabb = streaming.as_ref().and_then(|s| s.manager().resident_world_aabb());
    let aabb_str = match aabb {
        Some((lo, hi)) => format!(
            "geom_aabb=[{:.1},{:.1},{:.1}]..[{:.1},{:.1},{:.1}]",
            lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
        ),
        None => "geom_aabb=NONE".to_string(),
    };
    info!(
        "bench-diag t={:.1} cam=({:.1},{:.1},{:.1}) fwd=({:.2},{:.2},{:.2}) patch_gen={} resident_bricks={} {}",
        now, pos.x, pos.y, pos.z, fwd.x, fwd.y, fwd.z, pgen, resident, aabb_str
    );
    // Per-pass GPU times (ms), populated by the render-world timestamp read-back. Sorted desc so the dominant
    // pass is first. Only the world-cache passes are timestamped today; the rest show up once instrumented.
    let mut gpu: Vec<(String, f32)> = crate::instrument::peek_gpu().into_iter().collect();
    if !gpu.is_empty() {
        gpu.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total: f32 = gpu.iter().map(|(_, ms)| *ms).sum();
        let parts: Vec<String> = gpu.iter().map(|(k, ms)| format!("{k}={ms:.2}")).collect();
        info!("bench-gpu t={:.1} sum={total:.2}ms | {}", now, parts.join(" "));
    }
}

/// Parse `ADVENTURE_CAM="ex,ey,ez,lx,ly,lz"` into an `(eye, look_at)` pair. Returns `None` if unset or the six
/// comma-separated floats don't parse.
fn parse_adventure_cam() -> Option<(Vec3, Vec3)> {
    let raw = std::env::var("ADVENTURE_CAM").ok()?;
    let v: Vec<f32> = raw.split(',').filter_map(|s| s.trim().parse::<f32>().ok()).collect();
    if v.len() != 6 {
        return None;
    }
    Some((Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
}

/// **G-c.4** — parse `ADVENTURE_CAM_PATH="ex,ey,ez,lx,ly,lz; …"` into a list of `(eye, look_at)` keyframes (each
/// segment six comma-separated floats, segments `;`-separated). Returns `None` if unset or fewer than 2 valid
/// keyframes (a single keyframe can't move). The fly-through lerps across these over the run so the camera
/// CROSSES brick boundaries — the only way to exercise enter/drop + surface the CPU classify freeze.
fn parse_adventure_cam_path() -> Option<Vec<(Vec3, Vec3)>> {
    let raw = std::env::var("ADVENTURE_CAM_PATH").ok()?;
    let frames: Vec<(Vec3, Vec3)> = raw
        .split(';')
        .filter_map(|seg| {
            let v: Vec<f32> = seg.split(',').filter_map(|s| s.trim().parse::<f32>().ok()).collect();
            (v.len() == 6).then(|| (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
        })
        .collect();
    (frames.len() >= 2).then_some(frames)
}

/// Force lighting-uniform overrides from env each frame (overrides the editor/preset values), so the bench can
/// sweep knobs without rebuilding presets: `ADVENTURE_DEBUG_VIEW` (the debug-view selector — albedo/normals
/// bypass lighting) and `ADVENTURE_WC` (world-cache on/off). (The old `ADVENTURE_GI_RAYS` knob is gone — the
/// ReSTIR initial-candidate count is always 1, built up by temporal + spatial reuse.)
fn force_lighting_overrides(
    mut lighting: ResMut<crate::voxel::raytrace::VoxelRtLighting>,
    mut wc: ResMut<crate::voxel::raytrace::WorldCacheSettings>,
) {
    if let Ok(v) = std::env::var("ADVENTURE_DEBUG_VIEW").unwrap_or_default().trim().parse::<u32>() {
        lighting.data.debug_view = v;
    }
    if let Ok(v) = std::env::var("ADVENTURE_WC").unwrap_or_default().trim().parse::<u32>() {
        wc.data.use_world_cache = v;
    }
}

/// Pin the [`SdfCamera`] transform to the bench [`SdfOrbitCamera`] EVERY frame. The editor's `orbit_camera`
/// only runs while the pointer is over the viewport (input-gated), so during a headless auto-run it wouldn't
/// apply the view — we write it directly here so the interior view is deterministic + drift-free.
fn pin_bench_camera(time: Res<Time>, bench: Res<BistroBench>, mut cam: Query<&mut Transform, With<SdfCamera>>) {
    // **G-c.4** — a MOVING fly-through (cam_path) takes priority: lerp the eye+look_at across the keyframes over
    // the whole run so the camera continuously crosses brick boundaries (exercises enter/drop). A static `cam`
    // pin is the fallback.
    let (eye, look) = if let Some(path) = &bench.cam_path {
        let dur = bench.exit_at.unwrap_or(20.0).max(0.001);
        // Normalized [0,1) progress through the run, mapped onto the (n-1) segments; lerp within the segment.
        let prog = (time.elapsed_secs() / dur).clamp(0.0, 0.99999);
        let segs = (path.len() - 1) as f32;
        let f = prog * segs;
        let i = (f.floor() as usize).min(path.len() - 2);
        let local = f - i as f32;
        let (e0, l0) = path[i];
        let (e1, l1) = path[i + 1];
        (e0.lerp(e1, local), l0.lerp(l1, local))
    } else if let Some((eye, look)) = bench.cam {
        (eye, look)
    } else {
        return;
    };
    let view = Transform::from_translation(eye).looking_at(look, Vec3::Y);
    for mut t in &mut cam {
        *t = view;
    }
}

/// Accumulate the smoothed FPS once per frame, but ONLY inside the steady-state window (the last
/// [`STEADY_WINDOW_SECS`] before the exit deadline) so the initial streaming-in frames are excluded.
fn sample_fps(time: Res<Time>, diagnostics: Res<DiagnosticsStore>, mut bench: ResMut<BistroBench>) {
    let Some(exit_at) = bench.exit_at else { return };
    let now = time.elapsed_secs();
    if now < exit_at - STEADY_WINDOW_SECS {
        return; // still streaming in / settling — not steady state yet.
    }
    if let Some(fps) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
    {
        bench.fps_sum += fps;
        bench.fps_samples += 1;
    }
    // **G-c.4** — the HITCH metric: the MAX raw (un-smoothed) frame-time inside the steady-state window. A
    // classify freeze on a brick crossing shows here as a 100+ ms spike that the smoothed avg-FPS hides; the
    // GPU-driven path must keep this bounded (no 100+ ms hitch).
    if let Some(ft_ms) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.value())
        .filter(|&ft| ft > bench.max_frame_ms)
    {
        bench.max_frame_ms = ft_ms;
    }
}

/// Request a single screenshot of the final frame a moment before exit, saved to [`SCREENSHOT_PATH`].
fn fire_screenshot(time: Res<Time>, mut bench: ResMut<BistroBench>, mut commands: Commands) {
    let Some(exit_at) = bench.exit_at else { return };
    if bench.shot_fired || time.elapsed_secs() < exit_at - SCREENSHOT_LEAD_SECS {
        return;
    }
    bench.shot_fired = true;
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(SCREENSHOT_PATH));
    info!("bench: requested final-frame screenshot → {SCREENSHOT_PATH}");
}

/// Log the `BENCH RESULT:` line once, right at the exit deadline (averaging the steady-state samples).
fn report_at_exit(time: Res<Time>, mut bench: ResMut<BistroBench>) {
    let Some(exit_at) = bench.exit_at else { return };
    if bench.reported || time.elapsed_secs() < exit_at {
        return;
    }
    bench.reported = true;
    if bench.fps_samples == 0 {
        warn!("BENCH RESULT: bistro-interior NO STEADY-STATE SAMPLES (run too short? raise ADVENTURE_EXIT_AFTER_SECS)");
        return;
    }
    let avg_fps = bench.fps_sum / bench.fps_samples as f64;
    let frame_time_ms = if avg_fps > 0.0 { 1000.0 / avg_fps } else { f64::INFINITY };
    info!(
        "BENCH RESULT: bistro-interior avg_fps={:.1} frame_time_ms={:.3} max_frame_ms={:.1} (over {} frames)",
        avg_fps, frame_time_ms, bench.max_frame_ms, bench.fps_samples
    );
}
