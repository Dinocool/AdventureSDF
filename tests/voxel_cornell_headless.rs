//! **The on-screen correctness ORACLE for the static Cornell-box scene + its GI.**
//!
//! `voxel_gi_gpu.rs` proves the GI math (colour bleed / emissive / shadow-fill) on single rays in
//! isolation, and `voxel_render_headless.rs` proves the streaming-worldgen composite reaches the screen.
//! This rig closes the gap for the CORNELL scene specifically: it boots a headless Bevy `App` with the real
//! [`VoxelRtPlugin`], selects [`VoxelScene::Cornell`] explicitly (the engine now boots into the large
//! streamed Worldgen scene; Cornell is the `V`-toggle correctness anchor), frames a camera on the
//! OPEN front of the static box, renders to an offscreen image, reads it back, and asserts the box looks
//! right AND that single-bounce GI colour-bleed is visible:
//!
//!   * the LEFT-wall side of the box reads GREENish and the RIGHT-wall side REDish (the camera looks +Z into
//!     the box from the open −Z front, so world −X RED maps to screen-right and world +X GREEN to
//!     screen-left — see the region derivation below), the back/floor are whitish/neutral, and the ceiling
//!     light region is BRIGHT;
//!   * COLOUR BLEED: a white floor/back patch NEAR the red wall has a higher R/G ratio than one far from it
//!     (red bleed), and likewise green on the other side — proving Stage-4b GI on the Cornell box.
//!
//! Skips cleanly (no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter, like the other
//! GPU rigs.

use bevy::prelude::*;

use adventure::voxel::VoxelScene;
use adventure::voxel::cornell::{interior_center_world, interior_extent_world};
use adventure::voxel::raytrace::{VoxelRtPatch, VoxelRtToggle};

mod common;
use common::HeadlessRender;

/// Offscreen render-target dimensions. Small + square keeps the readback cheap and deterministic.
const W: u32 = 256;
const H: u32 = 256;

#[test]
fn headless_cornell_colours_and_bleed() {
    // Boots the shared headless render app (#134 DLSS fix lives in the harness); skips cleanly without a
    // ray-query device.
    let Some(mut hr) = HeadlessRender::new(W, H) else {
        eprintln!("no ray-query device — skipping headless_cornell_colours_and_bleed");
        return;
    };

    // Frame the OPEN front (−Z) of the static box, looking +Z so the box fills the view. The camera sits
    // back along −Z from the interior centre by ~1.1× the interior extent (close enough that the walls fill
    // most of the frame). Deterministic transform.
    let [cx, cy, cz] = interior_center_world();
    let extent = interior_extent_world();
    // Look roughly level into the box from just outside the open front. A small X offset avoids the
    // axis-aligned `looking_at` degeneracy (a pure +Z forward with +Y up yields a NaN view basis → black
    // frame), and aiming slightly ABOVE the centre lifts the emissive ceiling panel into clear view.
    let target = Vec3::new(cx, cy + extent * 0.12, cz);
    let cam_pos = Vec3::new(cx + extent * 0.06, cy, cz - extent * 1.15);

    // This rig validates the static CORNELL scene. The engine now boots into the large streamed Worldgen
    // scene by default (Phase 2.6 — the primary GI showcase); Cornell stays reachable via the `V` toggle and
    // is the correctness anchor. Select it explicitly here (it is no longer the boot default).
    hr.app.insert_resource(VoxelScene::Cornell);
    assert!(hr.app.world().resource::<VoxelRtToggle>().enabled, "HW-RT must default ON");
    // Validate the colour-BLEED showcase on the GI-INDIRECT path: disable GI 4.0 screen-space ReSTIR DI so the
    // large emissive ceiling's strong WHITE DIRECT light doesn't swamp the subtle indirect wall tint on the
    // floor (DI is correct + default-on for the live engine; the bleed is a property of the indirect transport,
    // which this test isolates). All other assertions (wall colours, brightness, seams) hold either way.
    {
        let mut s = hr.app.world_mut().resource_mut::<adventure::voxel::raytrace::RestirSettings>();
        s.di_enabled = false;
    }
    hr.app.insert_resource(ClearColor(Color::srgb(0.9, 0.0, 0.9))); // garish magenta — must NOT survive.

    hr.spawn_camera(cam_pos, target, "Headless Cornell Camera");
    hr.finalize();

    let unpadded_row = (W * 4) as usize;
    let padded_row = hr.padded_row();

    // The Cornell box is static + tiny: it packs on the first streaming tick, the BLAS/TLAS builds once, then
    // the composite + readback run. The GPU readback pipeline is a few frames deep and lands ASYNCHRONOUSLY,
    // so rather than a fixed frame count (whose tail can latch a stale warmup readback) we PUMP frames until
    // the latest read-back frame is meaningfully LIT (a non-trivial mean luma — the box rendered), capped at
    // a generous budget. This is robust to readback latency without weakening any colour assertion below.
    let bytes = hr.pump_until_lit(120, 10.0);
    let lit = !bytes.is_empty() && hr.centre_mean_luma(&bytes) > 10.0;

    // The static patch is resident (the box voxelized).
    let patch = hr.app.world().resource::<VoxelRtPatch>();
    assert!(!patch.upload.is_empty(), "the static Cornell brick set must be non-empty");
    assert!(lit, "the Cornell box never rendered a lit frame within the frame budget");
    assert!(bytes.len() >= padded_row * H as usize, "readback too small");

    let w = W as usize;
    let h = H as usize;
    // The shared row-padded readback helpers (same signatures as the old in-file ones).
    use common::{px, region_mean};

    // --- Region means ---------------------------------------------------------------------------------
    // Camera looks +Z (up +Y). With Bevy's right-handed `looking_at`, the camera's local +X (screen-right)
    // maps to world −X, so the RED left wall (−X) is on the SCREEN-RIGHT and the GREEN right wall (+X) is on
    // the SCREEN-LEFT. (The test prints both means; if a future convention change flips this, the print makes
    // the failure obvious.)
    let left = region_mean(&bytes, padded_row, w / 16, w / 4, h * 3 / 8, h * 5 / 8); // green wall side
    let right = region_mean(&bytes, padded_row, w * 3 / 4, w * 15 / 16, h * 3 / 8, h * 5 / 8); // red wall side
    let ceiling = region_mean(&bytes, padded_row, w * 3 / 8, w * 5 / 8, 0, h / 8); // ceiling light
    let centre = region_mean(&bytes, padded_row, w * 7 / 16, w * 9 / 16, h * 7 / 16, h * 9 / 16); // back wall
    let floor = region_mean(&bytes, padded_row, w * 3 / 8, w * 5 / 8, h * 13 / 16, h * 15 / 16); // floor

    let luma = |c: (f32, f32, f32)| 0.2126 * c.0 + 0.7152 * c.1 + 0.0722 * c.2;
    eprintln!(
        "cornell regions: left(green-wall)={left:?} right(red-wall)={right:?} ceiling={ceiling:?} \
         centre(back)={centre:?} floor={floor:?}; lumas L={:.1} R={:.1} ceil={:.1} back={:.1} floor={:.1}",
        luma(left), luma(right), luma(ceiling), luma(centre), luma(floor)
    );

    // 1. Left side reads GREENish (G dominates R and B).
    assert!(
        left.1 > left.0 && left.1 > left.2,
        "left box wall must be green-dominant (G>R,B), got {left:?}"
    );
    // 2. Right side reads REDish (R dominates G and B).
    assert!(
        right.0 > right.1 && right.0 > right.2,
        "right box wall must be red-dominant (R>G,B), got {right:?}"
    );
    // 3. Back wall + floor are whitish/neutral (no single channel dominates strongly).
    for (name, c) in [("back", centre), ("floor", floor)] {
        let mx = c.0.max(c.1).max(c.2);
        let mn = c.0.min(c.1).min(c.2);
        assert!(mx > 12.0, "{name} must be lit, not black: {c:?}");
        assert!(mn > 0.45 * mx, "{name} must be ~neutral (no strong tint), got {c:?}");
    }
    // 4. The ceiling light region is BRIGHT (the emissive panel) — clearly brighter than the floor.
    assert!(
        luma(ceiling) > luma(floor) && luma(ceiling) > 60.0,
        "ceiling light must be the brightest region (got ceil luma {:.1}, floor {:.1})",
        luma(ceiling),
        luma(floor)
    );

    // --- Colour bleed (the GI showcase) ---------------------------------------------------------------
    // Colour bleed is an INDIRECT light-transport property (ceiling→red wall→floor), so it is asserted on the
    // GI-indirect path — this test runs with **ReSTIR DI disabled** (see the `RestirSettings` override above).
    // With DI ON, the large emissive ceiling lights the floor with strong uniform WHITE DIRECT light (correct
    // for a big area light — ground-truth Cornell also has the direct dominate), which swamps the subtle wall
    // tint on the brightly-lit floor/back-wall to near-neutral; the bleed is still computed (the walls read
    // their colour) but sub-threshold for a strict ratio test. Isolating the indirect path keeps this a sharp
    // GI showcase. A white floor patch NEAR the red wall (screen-right) reads REDDER than one near the green.
    let floor_y0 = h * 21 / 32;
    let floor_y1 = h * 13 / 16;
    let near_red = region_mean(&bytes, padded_row, w * 11 / 16, w * 13 / 16, floor_y0, floor_y1); // toward red wall
    let near_green = region_mean(&bytes, padded_row, w * 3 / 16, w * 5 / 16, floor_y0, floor_y1); // toward green wall

    let rg = |c: (f32, f32, f32)| c.0 / c.1.max(1.0);
    let gr = |c: (f32, f32, f32)| c.1 / c.0.max(1.0);
    eprintln!(
        "cornell bleed: floor near-red={near_red:?} (R/G={:.3})  near-green={near_green:?} (G/R={:.3})",
        rg(near_red),
        gr(near_green)
    );
    // Red bleed: the floor by the red wall is redder than the floor by the green wall.
    assert!(
        rg(near_red) > rg(near_green),
        "floor near the RED wall must have a higher R/G ratio than floor near the green wall \
         (red colour bleed): near-red R/G={:.3} vs near-green R/G={:.3}",
        rg(near_red),
        rg(near_green)
    );
    // Green bleed: the floor by the green wall is greener than the floor by the red wall.
    assert!(
        gr(near_green) > gr(near_red),
        "floor near the GREEN wall must have a higher G/R ratio than floor near the red wall \
         (green colour bleed): near-green G/R={:.3} vs near-red G/R={:.3}",
        gr(near_green),
        gr(near_red)
    );

    // --- Sanity: the frame is a real render, not the clear colour --------------------------------------
    let mut distinct: std::collections::HashSet<[u8; 3]> = std::collections::HashSet::new();
    let mut clear_magenta = 0usize;
    let mut total = 0usize;
    let mut lumas: Vec<f32> = Vec::new();
    for y in 0..h {
        let row = &bytes[y * padded_row..y * padded_row + unpadded_row];
        for x in 0..w {
            let p = &row[x * 4..x * 4 + 4];
            let (r, g, b) = (p[0], p[1], p[2]);
            distinct.insert([r, g, b]);
            total += 1;
            if r > 180 && g < 80 && b > 180 {
                clear_magenta += 1;
            }
            lumas.push(0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32);
        }
    }
    let (lmin, lmax) = lumas.iter().fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
    eprintln!(
        "cornell frame: {} distinct colours, clear_magenta_frac={:.3}, luma range [{:.1}, {:.1}]",
        distinct.len(),
        clear_magenta as f32 / total as f32,
        lmin,
        lmax
    );
    assert!(distinct.len() > 8, "frame is ~uniform ({} colours) — composite likely never ran", distinct.len());
    assert!(
        (clear_magenta as f32 / total as f32) < 0.05,
        "too much magenta clear survived — the box did not cover the view"
    );
    // Sensible luma range: not all-black, not all-saturated; a real lit scene spans a band.
    assert!(lmax > 60.0, "frame too dark — nothing lit (max luma {lmax:.1})");
    assert!((lmax - lmin) > 25.0, "luma range too flat ({:.1}) — scene not lit/shaded", lmax - lmin);

    // --- NO BLACK BRICK SEAMS on the back wall (the DEFECT-1 on-screen oracle) ----------------------------
    // The back wall fills the centre of the frame and spans many bricks, so its brick boundaries used to show
    // as BLACK vertical lines (rays slipping between abutting bricks → a miss → near-black). Scan a band of
    // the lit back wall and compute each COLUMN's mean luma; a seam manifests as a column that is far darker
    // than its lit neighbours. Assert no such dark-line column exists. (Band chosen well inboard of the floor
    // boxes / walls so we sample plain back wall.)
    let bx0 = w * 5 / 16;
    let bx1 = w * 11 / 16;
    let by0 = h * 3 / 8;
    let by1 = h * 5 / 8;
    let col_luma: Vec<f32> = (bx0..bx1)
        .map(|x| {
            let (mut s, mut nn) = (0.0f32, 0.0f32);
            for y in by0..by1 {
                let (r, g, bl) = px(&bytes, padded_row, x, y);
                s += 0.2126 * r + 0.7152 * g + 0.0722 * bl;
                nn += 1.0;
            }
            s / nn
        })
        .collect();
    let band_mean = col_luma.iter().sum::<f32>() / col_luma.len() as f32;
    // The band must be genuinely lit (so a "no dark column" pass is meaningful, not vacuous on a black band).
    assert!(band_mean > 20.0, "back-wall band not lit enough to test seams (mean luma {band_mean:.1})");
    // A seam line = a column dipping far below the band (and below its immediate neighbours). Flag any column
    // under HALF the band mean whose neighbours are well-lit — that's a black brick-boundary line.
    let mut seam_cols = 0usize;
    let mut darkest = f32::MAX;
    for i in 1..col_luma.len() - 1 {
        let c = col_luma[i];
        darkest = darkest.min(c);
        let nbr = col_luma[i - 1].max(col_luma[i + 1]);
        if c < 0.5 * band_mean && nbr > 0.75 * band_mean {
            seam_cols += 1;
        }
    }
    eprintln!(
        "cornell back-wall seam scan: band_mean={band_mean:.1} darkest_col={darkest:.1} seam_cols={seam_cols}"
    );
    assert_eq!(
        seam_cols, 0,
        "back wall has {seam_cols} black-line seam column(s) (a dark column amid lit neighbours = brick seam)"
    );
}
