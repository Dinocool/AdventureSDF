//! **The "boil-meter" — a headless QUANTIFICATION of GI temporal variance (boil).**
//!
//! "Boil" is residual temporal variance in the GI that never converges: on a STATIC camera the GI output
//! still shimmers frame-to-frame. The ReSTIR estimator is reference-correct (see `voxel_restir_gi_gpu`); this
//! rig measures the *unconverged variance* that the cache + ReSTIR reuse leave in the per-frame GI estimate —
//! the input variance DLSS-RR must then denoise (and that it can't fully remove if it's too high, per the
//! DLSS-RR research in `docs/GI_BOIL_PLAN.md`).
//!
//! Method: boot the full `VoxelRtPlugin` on the static Cornell box, force **debug view 5 (GI-only)** — which
//! writes the RAW per-frame ReSTIR GI estimate to the output BEFORE any temporal-accumulation blend (see
//! `restir_p2`/`restir_dlss_p2`: the `debug_view != 0` branch returns before the history mix), so we read the
//! true reservoir variance with nothing masking it. Hold the camera still, warm up `WARMUP` frames (cache +
//! reservoirs converge), then collect `MEASURE` distinct frames and compute the **per-pixel temporal
//! coefficient of variation** `CoV = stddev_t(luma) / mean_t(luma)` over the lit GI pixels; report the mean
//! and 95th-percentile CoV as the **boil score** (lower = less boil).
//!
//! This is an INSTRUMENT, not a pass/fail correctness gate: it prints the score so before/after of each boil
//! fix is a NUMBER. It only asserts the harness actually measured something (lit, finite, enough pixels) so a
//! broken rig can't silently report "0 boil". Skips cleanly without an `EXPERIMENTAL_RAY_QUERY` adapter.
//!
//! Run it (numbers go to stderr):
//! ```sh
//! TMP=D:\tmp_test TEMP=D:\tmp_test cargo test --test voxel_gi_boil_gpu -- --nocapture
//! ```

use bevy::prelude::*;

use adventure::voxel::VoxelScene;
use adventure::voxel::cornell::{interior_center_world, interior_extent_world};
use adventure::voxel::raytrace::{
    RestirSettings, SPONZA_VOX_PATH, VoxelRtLighting, VoxelRtToggle, WorldCacheSettings,
};

mod common;
use common::HeadlessRender;

const W: u32 = 192;
const H: u32 = 192;
/// Frames to discard while the world cache + reservoirs converge on the static scene.
const WARMUP: usize = 90;
/// Distinct GI frames to measure the temporal variance over.
const MEASURE: usize = 24;

/// Per-pixel temporal-variance summary over the measured frames.
struct BoilStats {
    /// Mean over lit pixels of `stddev_t(luma)/mean_t(luma)` — the headline boil score.
    mean_cov: f32,
    /// 95th-percentile per-pixel CoV — captures the worst-shimmering pixels.
    p95_cov: f32,
    /// Number of lit GI pixels that contributed (sanity: enough surface to be meaningful).
    lit_pixels: usize,
    /// Mean GI luma over lit pixels (sanity: the GI buffer is actually lit, not black).
    mean_luma: f32,
}

/// Compute the per-pixel temporal CoV of luma across `frames` (row-padded RGBA8). Only pixels whose temporal
/// MEAN luma is in `[lo, hi]` (lit but not saturated) contribute — a near-black pixel has a meaningless CoV
/// (tiny mean → divide-by-noise) and a clipped pixel has artificially zero variance.
fn boil_stats(frames: &[Vec<u8>], padded_row: usize, w: usize, h: usize, lo: f32, hi: f32) -> BoilStats {
    let luma = |p: &[u8]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
    let n = frames.len() as f32;
    let mut covs: Vec<f32> = Vec::new();
    let mut lit_luma_sum = 0.0f32;
    for y in 0..h {
        for x in 0..w {
            let off = y * padded_row + x * 4;
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for f in frames {
                let l = luma(&f[off..off + 4]);
                sum += l;
                sum_sq += l * l;
            }
            let mean = sum / n;
            if mean < lo || mean > hi {
                continue; // not a meaningfully-lit GI pixel
            }
            let var = (sum_sq / n - mean * mean).max(0.0);
            covs.push(var.sqrt() / mean);
            lit_luma_sum += mean;
        }
    }
    if covs.is_empty() {
        return BoilStats { mean_cov: 0.0, p95_cov: 0.0, lit_pixels: 0, mean_luma: 0.0 };
    }
    let lit = covs.len();
    let mean_cov = covs.iter().sum::<f32>() / lit as f32;
    let mut sorted = covs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = sorted[((lit as f32 * 0.95) as usize).min(lit - 1)];
    BoilStats { mean_cov, p95_cov: p95, lit_pixels: lit, mean_luma: lit_luma_sum / lit as f32 }
}

/// Force GI-only debug view (raw per-frame reservoir estimate, before any temporal-accum mask). Re-asserted
/// every frame so a scene-load lighting preset can't quietly clear it.
fn set_gi_only(hr: &mut HeadlessRender) {
    hr.app.world_mut().resource_mut::<VoxelRtLighting>().data.debug_view = 5;
}

/// **Blotch metric** — the LOW-FREQUENCY temporal variance the per-pixel `boil_stats` misses. Spatially
/// downsample each frame into `block`×`block` averages (a low-pass that erases per-pixel grain but keeps
/// cell-sized "blotches"), then the temporal CoV of those coarse cells over a LONG window = how much the
/// low-frequency GI structure SHIFTS frame-to-frame. This is the "blotchy patches that slowly shift" boil DLSS-RR
/// passes through (RR removes high-freq grain, not a wandering low-freq mean). Returns the mean coarse-cell CoV.
fn blotch_cov(frames: &[Vec<u8>], padded_row: usize, w: usize, h: usize, block: usize, lo: f32) -> f32 {
    let luma = |p: &[u8]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
    let n = frames.len() as f32;
    let (cw, ch) = (w / block, h / block);
    let mut covs: Vec<f32> = Vec::new();
    for cy in 0..ch {
        for cx in 0..cw {
            // Temporal series of this coarse cell's spatial-mean luma.
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for f in frames {
                let mut bs = 0.0f32;
                for y in (cy * block)..(cy * block + block) {
                    for x in (cx * block)..(cx * block + block) {
                        bs += luma(&f[y * padded_row + x * 4..y * padded_row + x * 4 + 4]);
                    }
                }
                let bm = bs / (block * block) as f32;
                sum += bm;
                sum_sq += bm * bm;
            }
            let mean = sum / n;
            if mean < lo {
                continue;
            }
            let var = (sum_sq / n - mean * mean).max(0.0);
            covs.push(var.sqrt() / mean);
        }
    }
    if covs.is_empty() {
        return 0.0;
    }
    covs.iter().sum::<f32>() / covs.len() as f32
}

/// Sun-lit Cornell (emitter off, sun angled in) — the Sponza-like sun/sky GI regime where the user sees boil.
/// Re-applied every frame.
fn apply_sunlit(hr: &mut HeadlessRender) {
    let mut l = hr.app.world_mut().resource_mut::<VoxelRtLighting>();
    l.data.sun_direction = Vec3::new(0.0, -0.45, 1.0).normalize().into();
    l.data.sun_intensity = 3.0;
    l.data.sun_color = [1.0, 0.97, 0.92];
    l.data.emissive_strength = 0.0;
    l.data.ambient_color = [0.0, 0.0, 0.0];
    l.data.debug_view = 5;
}

/// Render `warmup` + collect `measure` distinct sun-lit GI-only frames at the CURRENT settings.
fn collect_sunlit(hr: &mut HeadlessRender, warmup: usize, measure: usize) -> Vec<Vec<u8>> {
    for _ in 0..warmup {
        apply_sunlit(hr);
        hr.app.update();
    }
    let padded_row = hr.padded_row();
    let need = padded_row * hr.h as usize;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..(measure * 12) {
        apply_sunlit(hr);
        hr.app.update();
        if let Some(b) = hr.latest.0.lock().unwrap().clone()
            && b.len() >= need
            && last.as_ref() != Some(&b)
        {
            last = Some(b.clone());
            frames.push(b);
        }
        if frames.len() >= measure {
            break;
        }
    }
    frames
}

/// **Sponza blotch repro** — the REAL boil the user sees, headless. Streams the actual Sponza `.vox` and frames
/// the exact captured viewpoint (eye/look-at logged live via F9), then measures the low-freq blotch (`blotch_cov`)
/// over a long window for several configs. DLSS is off in headless, but the boil is in the GI-only reservoir
/// resolve (debug 5) which is RR-INDEPENDENT (the user confirmed it boils in the debug view), so this reproduces
/// it. Sponza STREAMS via the clipmap, so a long warmup is needed for the viewpoint's bricks to fill in.
/// Skips cleanly without a ray-query device or if the `.vox` asset is absent.
#[test]
fn gi_sponza_blotch() {
    if !std::path::Path::new(SPONZA_VOX_PATH).exists() {
        eprintln!("no {SPONZA_VOX_PATH} — skipping gi_sponza_blotch");
        return;
    }
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping gi_sponza_blotch");
        return;
    };
    hr.app.insert_resource(VoxelScene::Sponza);
    // Captured live (F9) at the worst-boil viewpoint.
    hr.spawn_camera(Vec3::new(2.566, 3.498, -0.647), Vec3::new(7.532, 2.944, -0.468), "Sponza Boil Camera");
    hr.finalize();

    // Long warmup so the clipmap streams the viewpoint's bricks in (Sponza is not fully resident like Cornell),
    // then the cache + reservoirs converge. debug_view 5 every frame (don't touch the scene's own sun/sky preset).
    let pr = hr.padded_row();
    let report = |hr: &mut HeadlessRender, label: &str| {
        for _ in 0..280 {
            set_gi_only(hr);
            hr.app.update();
        }
        let need = pr * hr.h as usize;
        let mut frames: Vec<Vec<u8>> = Vec::new();
        let mut last: Option<Vec<u8>> = None;
        for _ in 0..(80 * 12) {
            set_gi_only(hr);
            hr.app.update();
            if let Some(b) = hr.latest.0.lock().unwrap().clone()
                && b.len() >= need
                && last.as_ref() != Some(&b)
            {
                last = Some(b.clone());
                frames.push(b);
            }
            if frames.len() >= 80 {
                break;
            }
        }
        let s = boil_stats(&frames, pr, W as usize, H as usize, 4.0, 250.0);
        let blotch = blotch_cov(&frames, pr, W as usize, H as usize, 16, 4.0);
        eprintln!(
            "[SPONZA {label:24}] frames={} lit_px={} luma={:.1} fine_CoV={:.4} blotch_CoV={blotch:.4}",
            frames.len(), s.lit_pixels, s.mean_luma, s.mean_cov,
        );
    };

    // Screen-probe GI (P1/P2) vs the M4 per-pixel reference. Validate: probe luma ≈ M4 luma (energy correct, not
    // biased) + blotch. Temporal OFF here (P1/P2) — single-frame probe variance ≥ M1; the win is the SH low-pass
    // + (P3) temporal. M4 = the boil-free reference (~0.036).
    // Half-res ReSTIR GI vs the full-res reference (both M4). Half-res traces ¼ the GI bounces; the full-res
    // shade reservoir-resolve-gathers. Target: blotch ≤ full-res at the reduced trace cost.
    let set = |hr: &mut HeadlessRender, half: bool, m: u32| {
        *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
        let mut r = hr.app.world_mut().resource_mut::<RestirSettings>();
        r.gi_half_res = half;
        r.gi_initial_samples = m;
    };
    set(&mut hr, false, 4);
    report(&mut hr, "full-res M4 (reference)");
    set(&mut hr, true, 4);
    report(&mut hr, "HALF-res M4");
    set(&mut hr, true, 8);
    report(&mut hr, "HALF-res M8");
    set(&mut hr, false, 1);
    report(&mut hr, "full-res M1");
}

/// **Probe SPATIAL diagnostic** — the aggregate CoV/luma metric is blind to spatial correctness (a flat/wrong GI
/// can have low variance + right average). Prints a coarse region-luma grid of the GI-only image for the per-
/// pixel ReSTIR reference vs the probe GI, so we can SEE whether the probe matches the reference's spatial
/// pattern or is flat/wrong. Run: `... cargo test --test voxel_gi_boil_gpu gi_probe_spatial_diag -- --nocapture`.
#[test]
fn gi_probe_spatial_diag() {
    if !std::path::Path::new(SPONZA_VOX_PATH).exists() {
        eprintln!("no {SPONZA_VOX_PATH} — skipping");
        return;
    }
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping");
        return;
    };
    hr.app.insert_resource(VoxelScene::Sponza);
    hr.spawn_camera(Vec3::new(2.566, 3.498, -0.647), Vec3::new(7.532, 2.944, -0.468), "diag");
    hr.finalize();
    let pr = hr.padded_row();
    let need = pr * hr.h as usize;
    let grab = |hr: &mut HeadlessRender| -> Vec<u8> {
        let mut last = vec![0u8; need];
        for _ in 0..420 {
            set_gi_only(hr);
            hr.app.update();
            if let Some(b) = hr.latest.0.lock().unwrap().clone()
                && b.len() >= need
            {
                last = b;
            }
        }
        last
    };
    let grid = |hr: &HeadlessRender, b: &[u8]| -> String {
        let (gw, gh) = (8usize, 6usize);
        let mut out = String::new();
        for gy in 0..gh {
            for gx in 0..gw {
                let x0 = gx * W as usize / gw;
                let x1 = (gx + 1) * W as usize / gw;
                let y0 = gy * H as usize / gh;
                let y1 = (gy + 1) * H as usize / gh;
                let (r, g, bl) = hr.region_mean(b, x0, x1, y0, y1);
                let l = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
                out += &format!("{l:6.1}");
            }
            out += "\n";
        }
        out
    };
    *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
    let restir = grab(&mut hr);
    eprintln!("=== FULL-RES RESTIR GI region luma (reference) ===\n{}", grid(&hr, &restir));
    {
        let mut r = hr.app.world_mut().resource_mut::<RestirSettings>();
        r.gi_half_res = true;
    }
    let half = grab(&mut hr);
    eprintln!("=== HALF-RES GI region luma (must MATCH reference — sharp) ===\n{}", grid(&hr, &half));
}

/// **Blotch sweep** — measures BOTH the fine per-pixel grain (`boil_stats`) AND the low-freq blotch
/// (`blotch_cov`) over a LONG window, for spatial reuse ON vs OFF, to settle whether GI spatial reuse is what
/// turns per-pixel grain into the slow low-freq blotch DLSS-RR can't clean. Prints; the headline is the blotch
/// column. Run: `TMP=D:\tmp_test TEMP=D:\tmp_test cargo test --test voxel_gi_boil_gpu gi_blotch_sweep -- --nocapture`.
#[test]
fn gi_blotch_sweep() {
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping gi_blotch_sweep");
        return;
    };
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);
    hr.app.insert_resource(VoxelScene::Cornell);
    hr.spawn_camera(cam_pos, target, "Blotch Camera");
    hr.finalize();

    let long = 80usize; // long window to catch the slow blotch shift
    let pr = hr.padded_row();
    let report = |hr: &mut HeadlessRender, label: &str| {
        let frames = collect_sunlit(hr, 90, long);
        let fine = boil_stats(&frames, pr, W as usize, H as usize, 4.0, 250.0).mean_cov;
        let blotch = blotch_cov(&frames, pr, W as usize, H as usize, 16, 4.0); // 16px blocks (low-pass)
        eprintln!("[BLOTCH {label:22}] frames={} fine_CoV={fine:.4} blotch_CoV={blotch:.4}", frames.len());
    };

    *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
    report(&mut hr, "default (spatial 4)");

    hr.app.world_mut().resource_mut::<RestirSettings>().spatial_samples = 0;
    report(&mut hr, "spatial OFF (temporal-only)");

    *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
    hr.app.world_mut().resource_mut::<RestirSettings>().confidence_cap = 24.0;
    report(&mut hr, "cap 24 (spatial 4)");

    // ENERGY CHECK: vary M (gi_initial_samples) on this LOW-variance converging scene and report the per-pixel
    // temporal-mean LUMA. The M-candidate RIS merge is the canonical unbiased Solari merge, so the luma must stay
    // ~M-stable here (any drift with M would flag an over-count bug). Contrast with Sponza, where luma RISES with M
    // because M=1 under-converges the bright bounce directions — that rise is accuracy, not bias.
    let report_luma = |hr: &mut HeadlessRender, label: &str| {
        let frames = collect_sunlit(hr, 90, long);
        let s = boil_stats(&frames, pr, W as usize, H as usize, 4.0, 250.0);
        eprintln!("[BLOTCH-M {label:20}] frames={} luma={:.2} blotch_CoV={:.4}", frames.len(), s.mean_luma, blotch_cov(&frames, pr, W as usize, H as usize, 16, 4.0));
    };
    for m in [1u32, 4, 8] {
        *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
        hr.app.world_mut().resource_mut::<RestirSettings>().gi_initial_samples = m;
        report_luma(&mut hr, &format!("M{m}"));
    }
}

/// Pump `warmup` frames (let the cache + reservoirs converge to the CURRENT config), then collect up to
/// `measure` distinct GI-only frames and return their boil stats. Used by both the baseline and the
/// attribution sweep (which mutates `RestirSettings`/`WorldCacheSettings` between calls and re-warms).
fn warm_and_measure(hr: &mut HeadlessRender, warmup: usize, measure: usize) -> BoilStats {
    for _ in 0..warmup {
        set_gi_only(hr);
        hr.app.update();
    }
    let padded_row = hr.padded_row();
    let need = padded_row * hr.h as usize;
    let (w, h) = (hr.w as usize, hr.h as usize);
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..(measure * 12) {
        set_gi_only(hr);
        hr.app.update();
        if let Some(b) = hr.latest.0.lock().unwrap().clone()
            && b.len() >= need
            && last.as_ref() != Some(&b)
        {
            last = Some(b.clone());
            frames.push(b);
        }
        if frames.len() >= measure {
            break;
        }
    }
    boil_stats(&frames, padded_row, w, h, 4.0, 250.0)
}

#[test]
fn gi_boil_meter_cornell() {
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping gi_boil_meter_cornell");
        return;
    };

    // Frame the open front of the static Cornell box (same framing as the cornell-colours oracle).
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);

    hr.app.insert_resource(VoxelScene::Cornell);
    assert!(hr.app.world().resource::<VoxelRtToggle>().enabled, "HW-RT must default ON");
    hr.spawn_camera(cam_pos, target, "Boil Meter Camera");
    hr.finalize();

    set_gi_only(&mut hr);

    // Warm up: let the world cache + reservoirs converge on the static box.
    for _ in 0..WARMUP {
        set_gi_only(&mut hr);
        hr.app.update();
    }

    // Collect distinct GI-only frames.
    let padded_row = hr.padded_row();
    let need = padded_row * H as usize;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..(MEASURE * 12) {
        set_gi_only(&mut hr);
        hr.app.update();
        if let Some(b) = hr.latest.0.lock().unwrap().clone()
            && b.len() >= need
            && last.as_ref() != Some(&b)
        {
            last = Some(b.clone());
            frames.push(b);
        }
        if frames.len() >= MEASURE {
            break;
        }
    }

    assert!(frames.len() >= MEASURE / 2, "only collected {} GI frames — readback stalled", frames.len());

    // Measure boil over the box interior region (avoid the black border outside the box). Lit band [4, 250] in
    // 8-bit luma: above the near-black floor (meaningless CoV) and below saturation (artificially zero CoV).
    let stats = boil_stats(&frames, padded_row, W as usize, H as usize, 4.0, 250.0);
    eprintln!(
        "[BOIL-METER cornell GI-only] frames={} lit_pixels={} mean_luma={:.1} | mean_CoV={:.4} p95_CoV={:.4}",
        frames.len(),
        stats.lit_pixels,
        stats.mean_luma,
        stats.mean_cov,
        stats.p95_cov,
    );

    // Sanity (the rig actually measured a lit GI signal).
    assert!(stats.lit_pixels > (W * H / 20) as usize, "too few lit GI pixels ({}) — rig misframed/dark", stats.lit_pixels);
    assert!(stats.mean_luma > 4.0, "GI buffer too dark (mean luma {:.1}) — GI-only view not lit", stats.mean_luma);
    assert!(stats.mean_cov.is_finite() && stats.p95_cov.is_finite(), "non-finite boil score");
    // REGRESSION GUARD: the tuned config (Stage 1 + LD-over-time + cap5/radius12) measures mean_CoV ≈ 0.21 here;
    // a GROSS boil regression (e.g. world cache disabled measured ≈0.36) must fail. Generous ceiling (≈40 %
    // headroom over the measured value + run-to-run ~±2 % noise) so it guards without flaking. Tighten if the
    // boil is reduced further (e.g. once ReSTIR DI lands).
    assert!(stats.mean_cov < 0.30, "GI boil regressed: mean_CoV {:.4} exceeds the 0.30 ceiling", stats.mean_cov);
}

/// **Sun-lit boil baseline** — the OTHER boil axis. Cornell with the emissive ceiling OFF and the SUN angled
/// in through the open front, so the box is lit ONLY by the sun + its indirect bounces. This exercises the
/// **fresh-`direct_lighting`-at-the-GI-bounce** variance (Sponza's regime — no emitters): each single LD bounce
/// lands on a sunlit-or-shadowed patch, and that hard sun-shadow term is recomputed fresh and frozen into the
/// candidate radiance → boil that the cache (which today stores INDIRECT only) does not damp. This is the
/// before-number for Stage 1 (fold the bounce-hit direct into the cache). Prints; asserts only lit+finite.
#[test]
fn gi_boil_meter_cornell_sunlit() {
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping gi_boil_meter_cornell_sunlit");
        return;
    };
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);
    hr.app.insert_resource(VoxelScene::Cornell);
    hr.spawn_camera(cam_pos, target, "Boil Sunlit Camera");
    hr.finalize();

    // Sun angled DOWN and INTO the box through the open −Z front; emitter OFF; ambient ~0 so GI is the only
    // fill in shadow. Re-applied every frame (set_gi_only handles debug_view; lighting set here in warm loop).
    let apply_sunlit = |hr: &mut HeadlessRender| {
        let mut l = hr.app.world_mut().resource_mut::<VoxelRtLighting>();
        l.data.sun_direction = Vec3::new(0.0, -0.45, 1.0).normalize().into();
        l.data.sun_intensity = 3.0;
        l.data.sun_color = [1.0, 0.97, 0.92];
        l.data.emissive_strength = 0.0; // kill the Cornell ceiling emitter — sun-only scene
        l.data.ambient_color = [0.0, 0.0, 0.0];
        l.data.debug_view = 5;
    };
    for _ in 0..WARMUP {
        apply_sunlit(&mut hr);
        hr.app.update();
    }
    let padded_row = hr.padded_row();
    let need = padded_row * H as usize;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..(MEASURE * 12) {
        apply_sunlit(&mut hr);
        hr.app.update();
        if let Some(b) = hr.latest.0.lock().unwrap().clone()
            && b.len() >= need
            && last.as_ref() != Some(&b)
        {
            last = Some(b.clone());
            frames.push(b);
        }
        if frames.len() >= MEASURE {
            break;
        }
    }
    assert!(frames.len() >= MEASURE / 2, "only {} sunlit GI frames", frames.len());
    let s = boil_stats(&frames, padded_row, W as usize, H as usize, 4.0, 250.0);
    eprintln!(
        "[BOIL-METER cornell SUN-LIT GI-only] frames={} lit_pixels={} mean_luma={:.1} | mean_CoV={:.4} p95_CoV={:.4}",
        frames.len(), s.lit_pixels, s.mean_luma, s.mean_cov, s.p95_cov,
    );
    assert!(s.lit_pixels > (W * H / 40) as usize, "too few lit sunlit-GI pixels ({})", s.lit_pixels);
    assert!(s.mean_cov.is_finite() && s.p95_cov.is_finite(), "non-finite sunlit boil score");
    // REGRESSION GUARD (sun-lit axis ≈ 0.19 with the tuned config): same generous 0.30 ceiling as the emitter meter.
    assert!(s.mean_cov < 0.30, "sun-lit GI boil regressed: mean_CoV {:.4} exceeds the 0.30 ceiling", s.mean_cov);
}

/// **Resolution-dependence check for the spatial radius.** The boil-meter runs at 192²; the live render runs
/// much larger, and a spatial radius in PIXELS covers a different world/angular extent at different resolutions
/// — so a radius that minimises boil at 192² may not transfer. This re-runs the radius sweep at 384² (4× the
/// pixels). If the optimal *pixel* radius roughly DOUBLES with the 2× linear resolution, the radius should be
/// expressed as a FRACTION of resolution (ReSTIR-GI paper: ~10% of image dim, adaptive); if the optimum is the
/// same px at both, a fixed px default is fine. Prints both curves; informs the default, not a pass/fail gate.
#[test]
fn gi_boil_radius_resolution_check() {
    let Some(mut hr) = HeadlessRender::new(384, 384) else {
        eprintln!("no ray-query device — skipping gi_boil_radius_resolution_check");
        return;
    };
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);
    hr.app.insert_resource(VoxelScene::Cornell);
    hr.spawn_camera(cam_pos, target, "Boil ResCheck Camera");
    hr.finalize();

    for radius in [10.0f32, 16.0, 20.0, 32.0, 48.0] {
        *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
        {
            let mut r = hr.app.world_mut().resource_mut::<RestirSettings>();
            r.confidence_cap = 5.0;
            r.spatial_radius = radius;
        }
        let s = warm_and_measure(&mut hr, 70, MEASURE);
        eprintln!("[RESCHECK 384 cap5 radius{radius:>4.0}] mean_CoV={:.4} p95={:.4} luma={:.1}", s.mean_cov, s.p95_cov, s.mean_luma);
        assert!(s.mean_cov.is_finite() && s.lit_pixels > (384 * 384 / 20) as usize, "rescheck r{radius}: bad measure");
    }
}

/// **GI 4.0 — ReSTIR DI proof.** Measures the DI-ONLY buffer (debug view 8 = the direct-emitter reservoir
/// estimate) on the emitter-lit Cornell box. Proves three things at once: (1) the scene actually has an
/// emissive-voxel light list (DI-only luma > 0 ⇒ `voxel_lights` populated + DI resolves it), (2) DI delivers
/// the emitter as a LOW-VARIANCE estimate (its temporal CoV is far below the ~0.21 raw-GI boil — the whole
/// point: emitter light no longer found by hit-or-miss random bounces), and (3) it doesn't NaN/crash. Prints
/// the numbers; the boil win is the low DI CoV vs the GI boil.
#[test]
fn gi_di_emitter_direct_is_low_variance() {
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping gi_di_emitter_direct_is_low_variance");
        return;
    };
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);
    hr.app.insert_resource(VoxelScene::Cornell);
    hr.spawn_camera(cam_pos, target, "DI Proof Camera");
    hr.finalize();

    let set_di_view = |hr: &mut HeadlessRender| {
        hr.app.world_mut().resource_mut::<VoxelRtLighting>().data.debug_view = 8; // DI-only
    };
    for _ in 0..WARMUP {
        set_di_view(&mut hr);
        hr.app.update();
    }
    let padded_row = hr.padded_row();
    let need = padded_row * H as usize;
    let mut frames: Vec<Vec<u8>> = Vec::new();
    let mut last: Option<Vec<u8>> = None;
    for _ in 0..(MEASURE * 12) {
        set_di_view(&mut hr);
        hr.app.update();
        if let Some(b) = hr.latest.0.lock().unwrap().clone()
            && b.len() >= need
            && last.as_ref() != Some(&b)
        {
            last = Some(b.clone());
            frames.push(b);
        }
        if frames.len() >= MEASURE {
            break;
        }
    }
    assert!(frames.len() >= MEASURE / 2, "only {} DI frames", frames.len());
    let s = boil_stats(&frames, padded_row, W as usize, H as usize, 4.0, 250.0);
    eprintln!(
        "[DI-ONLY cornell] frames={} lit_pixels={} mean_luma={:.1} | mean_CoV={:.4} p95_CoV={:.4}",
        frames.len(), s.lit_pixels, s.mean_luma, s.mean_cov, s.p95_cov,
    );
    // Cornell's emissive ceiling must produce a light list that DI resolves to visible direct light.
    assert!(s.lit_pixels > (W * H / 40) as usize && s.mean_luma > 4.0,
        "DI produced no emitter light (lit_pixels={}, luma={:.1}) — Cornell light list empty or DI broken?",
        s.lit_pixels, s.mean_luma);
    // The DI estimate is LOW variance — well below the ~0.21 raw-GI boil. (Generous 0.15 ceiling vs noise.)
    assert!(s.mean_cov.is_finite() && s.mean_cov < 0.15,
        "DI direct-emitter should be low-variance (got mean_CoV {:.4}); the whole point is to beat emitter-via-bounce", s.mean_cov);
}

/// **Attribution sweep** — the diagnosis instrument. Within ONE app boot, measure the boil score under
/// several configs (mutating the runtime knobs + re-warming between each) to ATTRIBUTE the variance to a
/// source. Prints a table; asserts only that each config measured a lit signal. Reads the dominant lever off
/// the printed numbers (see docs/GI_BOIL_PLAN.md §diagnosis):
///   - world-cache OFF spiking CoV ⇒ the cache is the main damper (improving WHAT it captures is high value).
///   - spatial/confidence changes barely moving CoV ⇒ the residual is candidate-radiance variance (the
///     fresh-direct-at-bounce hypothesis), not reuse strength.
#[test]
fn gi_boil_attribution_sweep() {
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping gi_boil_attribution_sweep");
        return;
    };
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);
    hr.app.insert_resource(VoxelScene::Cornell);
    hr.spawn_camera(cam_pos, target, "Boil Sweep Camera");
    hr.finalize();

    // ISOLATED sweep: each point sets ONE knob from the default baseline, re-warms long enough for the cache +
    // reservoirs to re-converge, measures, then RESETS to default before the next (no cumulative confound).
    let def = RestirSettings::default();
    let reset = |hr: &mut HeadlessRender| {
        *hr.app.world_mut().resource_mut::<RestirSettings>() = RestirSettings::default();
        hr.app.world_mut().resource_mut::<WorldCacheSettings>().data.max_temporal_samples = 32.0;
        hr.app.world_mut().resource_mut::<WorldCacheSettings>().data.use_world_cache = 1;
    };
    reset(&mut hr);
    let base = warm_and_measure(&mut hr, WARMUP, MEASURE);
    eprintln!("[SWEEP base (cap{} sp{}/{})  ] mean_CoV={:.4} p95={:.4} luma={:.1}",
        def.confidence_cap, def.spatial_samples, def.spatial_radius, base.mean_cov, base.p95_cov, base.mean_luma);

    let run = |hr: &mut HeadlessRender, label: &str, f: &dyn Fn(&mut HeadlessRender)| {
        reset(hr);
        f(hr);
        let s = warm_and_measure(hr, 70, MEASURE);
        eprintln!("[SWEEP {label:24}] mean_CoV={:.4} p95={:.4} luma={:.1}", s.mean_cov, s.p95_cov, s.mean_luma);
        assert!(s.mean_cov.is_finite() && s.lit_pixels > (W * H / 20) as usize, "sweep {label}: bad measure");
    };

    run(&mut hr, "cap 3", &|hr| hr.app.world_mut().resource_mut::<RestirSettings>().confidence_cap = 3.0);
    run(&mut hr, "cap 5", &|hr| hr.app.world_mut().resource_mut::<RestirSettings>().confidence_cap = 5.0);
    run(&mut hr, "cap 6", &|hr| hr.app.world_mut().resource_mut::<RestirSettings>().confidence_cap = 6.0);
    run(&mut hr, "cap5 radius12", &|hr| {
        let mut r = hr.app.world_mut().resource_mut::<RestirSettings>();
        r.confidence_cap = 5.0;
        r.spatial_radius = 12.0;
    });
    run(&mut hr, "cap5 radius10 sp6", &|hr| {
        let mut r = hr.app.world_mut().resource_mut::<RestirSettings>();
        r.confidence_cap = 5.0;
        r.spatial_radius = 10.0;
        r.spatial_samples = 6;
    });
}
