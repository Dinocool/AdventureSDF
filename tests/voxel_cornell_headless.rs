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

use std::sync::{Arc, Mutex};

use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages, WgpuFeatures};
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use adventure::sdf_render::SdfCamera;
use adventure::voxel::VoxelScene;
use adventure::voxel::cornell::{interior_center_world, interior_extent_world};
use adventure::voxel::raytrace::{VoxelRtPatch, VoxelRtPlugin, VoxelRtToggle};

mod common;

/// Offscreen render-target dimensions. Small + square keeps the readback cheap and deterministic.
const W: u32 = 256;
const H: u32 = 256;

/// CPU-side latest readback of the render target (raw `Rgba8UnormSrgb` bytes, row-padded by the GPU copy).
#[derive(Resource, Clone)]
struct LatestFrame(Arc<Mutex<Option<Vec<u8>>>>);

/// wgpu settings enabling AABB-BLAS `ray_query` — the same feature `main.rs` requests.
fn rt_wgpu_settings() -> WgpuSettings {
    WgpuSettings { features: WgpuFeatures::EXPERIMENTAL_RAY_QUERY, ..default() }
}

/// One read-back RGB pixel at `(x, y)`.
fn px(bytes: &[u8], padded_row: usize, x: usize, y: usize) -> (f32, f32, f32) {
    let row = &bytes[y * padded_row..];
    (row[x * 4] as f32, row[x * 4 + 1] as f32, row[x * 4 + 2] as f32)
}

/// Average RGB over a rectangular screen region `[x0,x1) × [y0,y1)` (returns linear-ish 0..255 means).
fn region_mean(bytes: &[u8], padded_row: usize, x0: usize, x1: usize, y0: usize, y1: usize) -> (f32, f32, f32) {
    let (mut r, mut g, mut b, mut n) = (0.0, 0.0, 0.0, 0.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let (pr, pg, pb) = px(bytes, padded_row, x, y);
            r += pr;
            g += pg;
            b += pb;
            n += 1.0;
        }
    }
    (r / n, g / n, b / n)
}

#[test]
fn headless_cornell_colours_and_bleed() {
    if common::headless_ray_query_device().is_none() {
        eprintln!("no ray-query device — skipping headless_cornell_colours_and_bleed");
        return;
    }

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

    let latest = LatestFrame(Arc::new(Mutex::new(None)));

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(rt_wgpu_settings())),
                ..default()
            }),
    );
    app.add_plugins(VoxelRtPlugin);

    // This rig validates the static CORNELL scene. The engine now boots into the large streamed Worldgen
    // scene by default (Phase 2.6 — the primary GI showcase); Cornell stays reachable via the `V` toggle and
    // is the correctness anchor. Select it explicitly here (it is no longer the boot default).
    app.insert_resource(VoxelScene::Cornell);
    assert!(app.world().resource::<VoxelRtToggle>().enabled, "HW-RT must default ON");

    app.insert_resource(latest.clone());
    app.insert_resource(ClearColor(Color::srgb(0.9, 0.0, 0.9))); // garish magenta — must NOT survive.

    let image_handle = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = Image::new_target_texture(W, H, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        images.add(image)
    };

    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(image_handle.clone().into()),
        bevy::camera::Hdr,
        Msaa::Off,
        Transform::from_translation(cam_pos).looking_at(target, Vec3::Y),
        SdfCamera,
        Name::new("Headless Cornell Camera"),
    ));

    let sink = latest.0.clone();
    app.world_mut()
        .spawn(Readback::texture(image_handle.clone()))
        .observe(move |event: On<ReadbackComplete>| {
            *sink.lock().unwrap() = Some(event.data.clone());
        });

    app.finish();
    app.cleanup();

    let unpadded_row = (W * 4) as usize;
    let padded_row = bevy::render::renderer::RenderDevice::align_copy_bytes_per_row(unpadded_row);

    // The Cornell box is static + tiny: it packs on the first streaming tick, the BLAS/TLAS builds once, then
    // the composite + readback run. The GPU readback pipeline is a few frames deep and lands ASYNCHRONOUSLY,
    // so rather than a fixed frame count (whose tail can latch a stale warmup readback) we PUMP frames until
    // the latest read-back frame is meaningfully LIT (a non-trivial mean luma — the box rendered), capped at
    // a generous budget. This is robust to readback latency without weakening any colour assertion below.
    let mut bytes = Vec::new();
    let mut lit = false;
    for _ in 0..120 {
        app.update();
        if let Some(b) = latest.0.lock().unwrap().clone()
            && b.len() >= padded_row * H as usize
        {
            // Mean luma over the centre of the frame — non-trivial once the box has actually rendered.
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for y in (H as usize / 4)..(H as usize * 3 / 4) {
                for x in (W as usize / 4)..(W as usize * 3 / 4) {
                    let (r, g, bl) = px(&b, padded_row, x, y);
                    sum += 0.2126 * r + 0.7152 * g + 0.0722 * bl;
                    n += 1.0;
                }
            }
            if sum / n > 10.0 {
                bytes = b;
                lit = true;
                break;
            }
        }
    }

    // The static patch is resident (the box voxelized).
    let patch = app.world().resource::<VoxelRtPatch>();
    assert!(!patch.patch.is_empty(), "the static Cornell brick set must be non-empty");
    assert!(lit, "the Cornell box never rendered a lit frame within the frame budget");
    assert!(bytes.len() >= padded_row * H as usize, "readback too small");

    let w = W as usize;
    let h = H as usize;

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
    // A white floor patch NEAR the red wall (screen-right) must be REDDER (higher R/G) than a white floor
    // patch NEAR the green wall (screen-left). Sample two floor strips just inboard of each wall, at the
    // SAME height band, so only the nearby wall's colour bleed differs.
    let floor_y0 = h * 21 / 32;
    let floor_y1 = h * 13 / 16;
    // Hug each wall (near the screen edges) where the wall's bounce dominates — maximising the bleed signal.
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
