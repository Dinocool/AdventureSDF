//! **The temporal-accumulation (denoise) ORACLE for the static Cornell scene.**
//!
//! DEFECT under test: the single-bounce GI is estimated per-frame from a handful of hashed bounce rays, so an
//! un-accumulated frame SPARKLES (high per-pixel temporal noise). The fix adds a persistent history texture
//! that runs an exponential/cumulative mean across frames (reset on camera move), so with a STILL camera the
//! image converges to a clean average and the frame-to-frame noise collapses.
//!
//! This rig boots the real [`VoxelRtPlugin`] on the (explicitly-selected) Cornell scene with a STILL camera,
//! pumps frames
//! through the actual render path, and captures a time series of read-back frames. It then measures the
//! per-pixel TEMPORAL standard deviation over an EARLY window (just after accumulation starts — still noisy)
//! vs a LATE window (after many accumulated frames — converged). With accumulation working, the late window's
//! temporal noise is MUCH lower than the early window's: a still camera's displayed pixels barely change once
//! the running mean has settled. Skips cleanly without a ray-query adapter.

use bevy::prelude::*;

use adventure::voxel::VoxelScene;
use adventure::voxel::cornell::{interior_center_world, interior_extent_world};
use adventure::voxel::raytrace::VoxelRtPatch;

mod common;
use common::HeadlessRender;

const W: u32 = 192;
const H: u32 = 192;

/// Mean per-pixel temporal standard deviation (luma) across a window of consecutive frames, over the lit
/// interior region of the frame. Higher = noisier (more sparkle frame-to-frame).
fn temporal_noise(frames: &[Vec<u8>], padded_row: usize) -> f32 {
    let w = W as usize;
    let h = H as usize;
    let x0 = w / 4;
    let x1 = w * 3 / 4;
    let y0 = h / 4;
    let y1 = h * 3 / 4;
    let mut sum_std = 0.0f64;
    let mut n_px = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            // Luma of this pixel across the window.
            let mut vals = [0.0f64; 64];
            let k = frames.len().min(64);
            for (i, f) in frames.iter().take(k).enumerate() {
                let row = &f[y * padded_row..];
                let r = row[x * 4] as f64;
                let g = row[x * 4 + 1] as f64;
                let b = row[x * 4 + 2] as f64;
                vals[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            }
            let mean = vals[..k].iter().sum::<f64>() / k as f64;
            let var = vals[..k].iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / k as f64;
            sum_std += var.sqrt();
            n_px += 1.0;
        }
    }
    (sum_std / n_px) as f32
}

#[test]
fn temporal_accumulation_reduces_gi_noise() {
    // Boots the shared headless render app (#134 DLSS fix lives in the harness); skips cleanly without a
    // ray-query device.
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping temporal_accumulation_reduces_gi_noise");
        return;
    };

    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);

    // This rig renders the static CORNELL box. The engine now boots into the large streamed Worldgen scene by
    // default (Phase 2.6); select Cornell explicitly (it is the `V`-toggle correctness anchor).
    hr.app.insert_resource(VoxelScene::Cornell);
    hr.app.insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.0)));

    // A perfectly STILL camera — the accumulator must converge.
    hr.spawn_camera(cam_pos, target, "Temporal Cornell Camera");
    hr.finalize();

    let padded_row = hr.padded_row();

    // Pump frames and snapshot each distinct read-back frame. The readback is a few frames deep + async, so
    // we dedup identical consecutive snapshots and collect a long sequence to span the accumulation ramp.
    let frames = hr.collect_distinct_frames(240, 80);

    let patch = hr.app.world().resource::<VoxelRtPatch>();
    assert!(!patch.upload.is_empty(), "the static Cornell brick set must be non-empty");
    assert!(
        frames.len() >= 30,
        "need a run of distinct frames to measure temporal noise (got {})",
        frames.len()
    );

    // Find the first meaningfully-lit frame index (the box has rendered) so the early window is real signal,
    // not warmup black.
    let luma_mean = |f: &[u8]| -> f32 {
        let w = W as usize;
        let h = H as usize;
        let (mut s, mut n) = (0.0f32, 0.0f32);
        for y in (h / 4)..(h * 3 / 4) {
            for x in (w / 4)..(w * 3 / 4) {
                let row = &f[y * padded_row..];
                s += 0.2126 * row[x * 4] as f32 + 0.7152 * row[x * 4 + 1] as f32 + 0.0722 * row[x * 4 + 2] as f32;
                n += 1.0;
            }
        }
        s / n
    };
    let first_lit = frames.iter().position(|f| luma_mean(f) > 10.0).expect("box must light up");
    // Need enough frames after first_lit for both windows.
    assert!(
        frames.len() - first_lit >= 24,
        "not enough lit frames after warmup ({} from idx {})",
        frames.len(),
        first_lit
    );

    // EARLY window: the first several lit frames (accumulator just reset / few samples → noisy).
    // LATE window: the final several frames (many samples accumulated → converged).
    let early = &frames[first_lit..first_lit + 8];
    let late = &frames[frames.len() - 8..];

    let early_noise = temporal_noise(early, padded_row);
    let late_noise = temporal_noise(late, padded_row);
    eprintln!(
        "temporal noise: early(window {}..{})={early_noise:.3}  late(last 8)={late_noise:.3}  ratio={:.2}",
        first_lit,
        first_lit + 8,
        early_noise / late_noise.max(1e-3)
    );

    // The accumulated (late) frames must be MUCH less noisy frame-to-frame than the early ones. A still
    // camera with a working running-mean accumulator settles: consecutive late frames are nearly identical.
    assert!(
        late_noise < 0.5 * early_noise,
        "temporal accumulation did not reduce noise: late {late_noise:.3} vs early {early_noise:.3} \
         (expected late < 0.5×early)"
    );
    // And the late frames are genuinely settled (low absolute temporal noise), not merely less-than-early.
    assert!(late_noise < 2.0, "late frames still noisy (temporal std {late_noise:.3}) — not converged");
}
