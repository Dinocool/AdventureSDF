//! Editor "Render / GI" dock panel: live sliders for the raymarch lighting + GI uniforms
//! (knobs-as-uniforms — every value is a runtime uniform the shader reads) plus a **debug-view**
//! selector (normals / depth / albedo / AO / GI-only / face-orientation) for diagnosing the renderer.
//!
//! It mutates the [`VoxelRtLighting`] resource directly; the change is extracted to the render world +
//! uploaded to the WGSL `LightingUniform` next frame, so tweaks are live.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::voxel::raytrace::{
    LightingUniformData, SkyUniformData, VoxelRtLighting, VoxelRtSky, VoxelRtToggle,
};

/// Labels for `LightingUniformData.debug_view` (index == the u32 value the shader branches on).
const DEBUG_LABELS: [&str; 10] = [
    "Lit (normal)",
    "Normals",
    "Depth",
    "Albedo",
    "Ambient occlusion",
    "GI only",
    "Face orient (red = BACK face)",
    "LOD (ring colour)",
    "DI only (emitter direct)",
    "Motion vectors (DLSS, px)",
];

/// The panel body. Registered via `editor::panels::register_panel`.
pub fn render_gi_panel(world: &mut World, ui: &mut egui::Ui) {
    // The live scene decides which preset the Reset button restores (Sponza is the default boot scene).
    let scene = world.get_resource::<crate::voxel::VoxelScene>().copied().unwrap_or_default();
    // Scope the `VoxelRtLighting` borrow so it ends before the DLSS section reads other world resources.
    {
        let Some(mut lighting) = world.get_resource_mut::<VoxelRtLighting>() else {
            ui.label("voxel renderer not active");
            return;
        };
        let d = &mut lighting.data;

        ui.label(egui::RichText::new("Debug view").strong());
        let cur = (d.debug_view as usize).min(DEBUG_LABELS.len() - 1);
        egui::ComboBox::from_id_salt("voxel_debug_view")
            .selected_text(DEBUG_LABELS[cur])
            .show_ui(ui, |ui| {
                for (i, label) in DEBUG_LABELS.iter().enumerate() {
                    ui.selectable_value(&mut d.debug_view, i as u32, *label);
                }
            });
        ui.label(
            egui::RichText::new(
                "Face orient: green = front, RED = back-face hit (show-through bug)",
            )
            .weak()
            .size(11.0),
        );

        ui.separator();
        ui.label(egui::RichText::new("Sun").strong());
        ui.add(egui::Slider::new(&mut d.sun_intensity, 0.0..=5.0).text("intensity"));
        let mut dir = Vec3::from_array(d.sun_direction);
        let mut dir_changed = false;
        dir_changed |= ui
            .add(egui::Slider::new(&mut dir.x, -1.0..=1.0).text("dir x"))
            .changed();
        dir_changed |= ui
            .add(egui::Slider::new(&mut dir.y, -1.0..=1.0).text("dir y"))
            .changed();
        dir_changed |= ui
            .add(egui::Slider::new(&mut dir.z, -1.0..=1.0).text("dir z"))
            .changed();
        if dir_changed {
            let n = dir.normalize_or_zero();
            if n != Vec3::ZERO {
                d.sun_direction = n.into();
            }
        }
        color_row(ui, "color", &mut d.sun_color);

        ui.separator();
        ui.label(egui::RichText::new("Ambient / AO").strong());
        color_row(ui, "ambient", &mut d.ambient_color);
        ui.add(egui::Slider::new(&mut d.ao_radius, 0.0..=3.0).text("AO radius (m)"));
        ui.add(egui::Slider::new(&mut d.ao_samples, 0..=8).text("AO samples"));
        // Clamp below one voxel (0.2 m): a shadow_bias >= one voxel near a thin wall pulls the shadow-ray
        // origin past a floor and can disarm the occlusion backstop, leaking light through thin geometry.
        ui.add(egui::Slider::new(&mut d.shadow_bias, 0.0..=0.15).text("shadow bias"));

        ui.separator();
        ui.label(egui::RichText::new("Global illumination").strong());
        // No "GI rays / px" slider: the ReSTIR initial-candidate count is always 1 (the effective sample count
        // is built by temporal + spatial reservoir reuse). Set GI intensity to 0 to disable GI.
        ui.add(egui::Slider::new(&mut d.gi_intensity, 0.0..=4.0).text("GI intensity"));
        ui.add(egui::Slider::new(&mut d.gi_bounce_dist, 1.0..=64.0).text("bounce dist (m)"));
        ui.add(egui::Slider::new(&mut d.emissive_strength, 0.0..=16.0).text("emissive strength"));

        ui.separator();
        if ui.button(format!("Reset to {scene:?} defaults")).clicked() {
            let keep = d.debug_view;
            // Scene-aware: restore the preset for the LIVE scene, not always Cornell. The Gallery (a row of
            // baked scenes) shares Sponza's open-sky GI preset so the comparison reads as geometry, not lighting.
            *d = match scene {
                crate::voxel::VoxelScene::Sponza | crate::voxel::VoxelScene::Gallery => {
                    LightingUniformData::sponza()
                }
                crate::voxel::VoxelScene::Worldgen => LightingUniformData::worldgen(),
                crate::voxel::VoxelScene::Cornell => LightingUniformData::cornell(),
            };
            d.debug_view = keep;
        }
    } // `lighting` borrow released here

    // Procedural sky / environment knobs (the `Sky` UBO, group 1 binding 11 — knobs-as-uniforms). Drives the
    // primary-miss sky gradient + sun disk AND how strongly a GI bounce that escapes to sky lights the scene.
    {
        ui.separator();
        ui.label(egui::RichText::new("Sky / environment").strong());
        if let Some(mut sky) = world.get_resource_mut::<VoxelRtSky>() {
            let s = &mut sky.data;
            ui.add(egui::Slider::new(&mut s.intensity, 0.0..=4.0).text("sky intensity"));
            ui.add(
                egui::Slider::new(&mut s.gi_sky_intensity, 0.0..=4.0).text("GI sky intensity"),
            );
            color_row(ui, "horizon", &mut s.horizon_color);
            color_row(ui, "zenith", &mut s.zenith_color);
            color_row(ui, "ground", &mut s.ground_color);
            color_row(ui, "sun tint", &mut s.sun_tint);
            ui.add(egui::Slider::new(&mut s.sun_size, 0.0..=0.3).text("sun size (rad)"));
            ui.label(
                egui::RichText::new(
                    "GI sky intensity = how strongly a bounce that escapes to open sky lights the scene",
                )
                .weak()
                .size(11.0),
            );
            if ui.button("Reset sky defaults").clicked() {
                *s = SkyUniformData::default();
            }
        }
    }

    // ReSTIR GI controls: the A/B `gi_mode` toggle (ReSTIR vs legacy gather_gi) + the reservoir knobs.
    {
        ui.separator();
        ui.label(egui::RichText::new("ReSTIR GI").strong());
        if let Some(mut s) = world.get_resource_mut::<crate::voxel::raytrace::RestirSettings>() {
            ui.checkbox(&mut s.restir, "ReSTIR GI (off = legacy gather_gi)");
            let on = s.restir;
            ui.add_enabled_ui(on, |ui| {
                ui.add(egui::Slider::new(&mut s.spatial_samples, 0..=8).text("spatial search taps"));
                ui.add(egui::Slider::new(&mut s.spatial_radius, 1.0..=48.0).text("spatial radius (px)"));
                ui.add(egui::Slider::new(&mut s.confidence_cap, 1.0..=32.0).text("history cap (frames)"));
                // 0 = uncapped (pure Solari); >0 caps the dissimilarity view-distance → absolute tangent reject
                // beyond it, closing far thin-wall GI leaks. Raise toward off if it adds boil on slopes/terrain.
                ui.add(egui::Slider::new(&mut s.gi_dissim_cap_dist, 0.0..=80.0).text("thin-wall reject cap dist (m, 0=off)"));
            });
            // Half-resolution ReSTIR GI: trace GI at render_res/2 (¼ the bounce traces), full-res reservoir-
            // resolve gather. SHARP (re-resolved per full-res normal) but ~2× boilier pre-DLSS-RR (¼ the samples);
            // a perf/quality trade that leans on RR to clean it.
            ui.checkbox(&mut s.gi_half_res, "Half-res GI (¼ traces; sharp but boilier — needs DLSS-RR)");
            // Screen-space radiance probes (Lumen-style): downsampled SH GI — kills the boil at a fraction of
            // the per-pixel trace cost. Replaces the per-pixel ReSTIR diffuse gather when on. SHELVED (flat).
            ui.checkbox(&mut s.screen_probes, "Screen-probe GI (Lumen-style — SHELVED, flat)");
            ui.add_enabled_ui(on && s.screen_probes, |ui| {
                ui.add(egui::Slider::new(&mut s.probe_size, 4..=32).text("probe spacing (px)"));
                ui.add(egui::Slider::new(&mut s.probe_oct_res, 4..=16).text("probe directions √N (oct res)"));
                ui.checkbox(&mut s.probe_temporal, "probe temporal accumulation (light)");
            });
            // GI 4.0: screen-space ReSTIR DI (emissive-voxel direct light) — the emitter-boil fix.
            ui.checkbox(&mut s.di_enabled, "ReSTIR DI (emissive-voxel direct light)");
            ui.add_enabled_ui(on && s.di_enabled, |ui| {
                ui.add(egui::Slider::new(&mut s.di_initial_samples, 1..=32).text("DI initial RIS candidates"));
                ui.add(egui::Slider::new(&mut s.di_confidence_cap, 1.0..=40.0).text("DI history cap (frames)"));
            });
            ui.label(
                egui::RichText::new(
                    "search taps = disk samples tried to find ONE valid neighbour/frame (not accumulated); \
                     history cap trades lag vs stability",
                )
                .weak()
                .size(11.0),
            );
        }

        // Phase 2.2 A/B gate: the world-space radiance cache feeds the ReSTIR initial reservoir (default on).
        // Off = the FRESH single-bounce path (no cache query → the cache stays idle, like Phase 2.1).
        if let Some(mut wc) = world.get_resource_mut::<crate::voxel::raytrace::WorldCacheSettings>() {
            let mut on = wc.data.use_world_cache != 0;
            if ui
                .checkbox(&mut on, "World-cache GI (off = fresh single-bounce reservoir)")
                .changed()
            {
                wc.data.use_world_cache = u32::from(on);
            }
            ui.label(
                egui::RichText::new(
                    "on = the initial reservoir reads the pre-accumulated world cache (lower boil, multi-bounce \
                     below); off = a fresh bounce trace each frame (A/B comparison)",
                )
                .weak()
                .size(11.0),
            );

            // Phase 2.3 A/B gate: the cache UPDATE pass feeds the cache forward at each bounce hit (cells
            // querying cells ⇒ multi-bounce fill light), stabilised by the temporal blend. Only meaningful
            // when the world cache is live; an edit never clears the cache either way (adapt-not-reset).
            let cache_live = wc.data.use_world_cache != 0;
            ui.add_enabled_ui(cache_live, |ui| {
                let mut mb = wc.data.gi_multibounce != 0;
                if ui.checkbox(&mut mb, "Multi-bounce (cache feeds itself)").changed() {
                    wc.data.gi_multibounce = u32::from(mb);
                }
            });
            ui.label(
                egui::RichText::new(
                    "on = each cache cell adds albedo·cache(hit) on its bounce ⇒ feed-forward multi-bounce \
                     fill light (open-world / shadow GI); off = single-bounce (direct+emissive/sky only)",
                )
                .weak()
                .size(11.0),
            );

            // Phase 2.5 A/B gate: NEE (direct emissive-voxel light sampling, MIS-balanced) in the cache update.
            // The principled firefly/variance fix — emitters are sampled DIRECTLY (a shadow ray per cell) instead
            // of only being found by the random bounce. Off = the pre-2.5 bounce-only path (higher boil). An edit
            // never clears the cache either way (adapt-not-reset).
            let mut nee = wc.data.nee_enabled != 0;
            if ui.checkbox(&mut nee, "NEE (direct emissive-voxel light sampling + MIS)").changed() {
                wc.data.nee_enabled = u32::from(nee);
            }
            ui.add_enabled_ui(nee, |ui| {
                ui.add(
                    egui::Slider::new(&mut wc.data.nee_samples, 1..=8).text("NEE shadow rays / cell / frame"),
                );
            });
            ui.label(
                egui::RichText::new(
                    "on = sample emissive voxels DIRECTLY (importance-sampled light list + shadow ray, MIS-combined \
                     with the bounce ⇒ no double-count) ⇒ far lower emitter variance (the principled clamp \
                     replacement); off = emitters found only by the random bounce (pre-2.5, more boil)",
                )
                .weak()
                .size(11.0),
            );
        }
    }

    // World-cache TUNING (Phase 2.4 knobs-as-uniforms): per-tunable sliders on `WorldCacheUniformData`. These
    // mutate `WorldCacheSettings.data` (already extracted + uploaded to the WGSL `WorldCacheUniform` each frame),
    // so tweaks are live. Defaults are byte-identical to the Solari-tuned values the GPU convergence/energy tests
    // assert, so leaving the sliders alone changes nothing. Collapsed by default to keep the panel tidy.
    {
        ui.separator();
        egui::CollapsingHeader::new(egui::RichText::new("World Cache").strong())
            .default_open(false)
            .show(ui, |ui| {
                if let Some(mut wc) =
                    world.get_resource_mut::<crate::voxel::raytrace::WorldCacheSettings>()
                {
                    let d = &mut wc.data;
                    // Cell sizing / LOD. `cell_base_size` = the LOD-0 cell edge (m); `lod_scale` = how fast cells
                    // grow with camera distance (bigger ⇒ slower growth ⇒ finer cells further out, more cells).
                    ui.label(egui::RichText::new("Cell sizing").strong());
                    ui.add(
                        egui::Slider::new(&mut d.cell_base_size, 0.02..=2.0)
                            .text("cell base size (m)"),
                    );
                    ui.add(egui::Slider::new(&mut d.lod_scale, 1.0..=64.0).text("distance-LOD scale"));

                    // Update-pass GI ray reach + temporal behaviour.
                    ui.label(egui::RichText::new("GI / temporal").strong());
                    ui.add(
                        egui::Slider::new(&mut d.gi_ray_distance, 1.0..=200.0)
                            .text("GI ray distance (m)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut d.cell_lifetime, 1..=120)
                            .text("cell lifetime (frames)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut d.max_temporal_samples, 1.0..=128.0)
                            .text("max temporal samples"),
                    );
                    ui.label(
                        egui::RichText::new(
                            "lifetime = frames a cell survives un-queried before decay clears it; \
                             temporal samples cap = smoother but laggier",
                        )
                        .weak()
                        .size(11.0),
                    );

                    // STOCHASTIC per-frame active-cell soft cap (Solari's 40000, the default). 0 = UNLIMITED;
                    // > 0 keeps each cell with Bernoulli probability cap/active_count (lower steady GPU cost).
                    ui.separator();
                    ui.label(egui::RichText::new("Per-frame active-cell cap").strong());
                    ui.add(
                        egui::Slider::new(&mut d.max_active_cells_per_frame, 0..=131_072)
                            .text("avg active cells / frame (0 = unlimited)"),
                    );
                    ui.label(
                        egui::RichText::new(
                            "0 = unlimited (every active cell updates each frame). > 0 = stochastic soft cap \
                             (Solari 40000, the default): each cell updates with probability cap/active_count, so \
                             ~N cells refresh per frame — a random subset the temporal blend integrates to the \
                             same converged GI. Lower, steadier GPU cost; skipped cells keep last radiance \
                             (never cleared). No starvation (equal per-frame chance for every cell).",
                        )
                        .weak()
                        .size(11.0),
                    );

                    ui.separator();
                    if ui.button("Reset world-cache defaults").clicked() {
                        // Preserve the A/B gates the ReSTIR section owns (use_world_cache / gi_multibounce /
                        // nee_enabled / nee_samples) + the render-pass-stamped fields; reset only the TUNABLE
                        // sliders to their defaults.
                        let keep_use = d.use_world_cache;
                        let keep_mb = d.gi_multibounce;
                        let keep_nee = d.nee_enabled;
                        let keep_nee_n = d.nee_samples;
                        let def = crate::voxel::raytrace::WorldCacheUniformData::default();
                        d.cell_base_size = def.cell_base_size;
                        d.lod_scale = def.lod_scale;
                        d.gi_ray_distance = def.gi_ray_distance;
                        d.cell_lifetime = def.cell_lifetime;
                        d.max_temporal_samples = def.max_temporal_samples;
                        d.max_active_cells_per_frame = def.max_active_cells_per_frame;
                        d.use_world_cache = keep_use;
                        d.gi_multibounce = keep_mb;
                        d.nee_enabled = keep_nee;
                        d.nee_samples = keep_nee_n;
                    }
                }
            });
    }

    // Phase G G-wire — the GPU-pack A/B toggle. OFF by default; flip it ON to A/B the LIVE flag-ON production path
    // (the streamed re-pack drives the GPU classify → readback → GPU pack/AABB → fill-then-build BLAS instead of the
    // all-CPU `update` + `apply_delta`). Byte- and render-identical to the CPU path (the parity + render-identity
    // gates). The user confirms the live win by toggling this ON in-editor; the default stays OFF.
    {
        ui.separator();
        ui.label(egui::RichText::new("Streaming / brick pack").strong());
        if let Some(mut toggle) = world.get_resource_mut::<VoxelRtToggle>() {
            ui.checkbox(&mut toggle.gpu_pack, "GPU pack (off = CPU pack + apply_delta)");
            ui.label(
                egui::RichText::new(
                    "GPU pack: the streamed re-pack runs on the GPU (classify → pack/AABB → fill-then-build) \
                     instead of the CPU. Render-identical; moves vox_pack_update + vox_blas_delta off the CPU.",
                )
                .weak()
                .size(11.0),
            );
        }
    }

    // DLSS Ray Reconstruction controls (only when built with `--features dlss`). RR denoises + upscales the
    // noisy GI; toggle it and pick the quality preset live.
    #[cfg(feature = "dlss")]
    {
        ui.separator();
        ui.label(egui::RichText::new("DLSS Ray Reconstruction").strong());
        let supported = world
            .get_resource::<bevy::anti_alias::dlss::DlssRayReconstructionSupported>()
            .is_some();
        if !supported {
            ui.label(egui::RichText::new("not supported on this GPU/driver").weak());
        } else if let Some(mut s) = world.get_resource_mut::<crate::voxel::raytrace::DlssSettings>()
        {
            ui.checkbox(&mut s.enabled, "RR enabled (denoise + upscale)");
            let enabled = s.enabled;
            ui.add_enabled_ui(enabled, |ui| {
                use bevy::anti_alias::dlss::DlssPerfQualityMode as M;
                let modes = [
                    (M::Auto, "Auto"),
                    (M::Dlaa, "DLAA — native res, best quality"),
                    (M::Quality, "Quality"),
                    (M::Balanced, "Balanced"),
                    (M::Performance, "Performance"),
                    (M::UltraPerformance, "Ultra Performance"),
                ];
                let cur = modes
                    .iter()
                    .find(|(m, _)| *m == s.mode)
                    .map(|(_, l)| *l)
                    .unwrap_or("Auto");
                egui::ComboBox::from_id_salt("dlss_mode")
                    .selected_text(cur)
                    .show_ui(ui, |ui| {
                        for (m, label) in modes {
                            ui.selectable_value(&mut s.mode, m, label);
                        }
                    });
            });
            ui.label(
                egui::RichText::new("DLAA = denoise at native res (cleanest); lower presets render smaller + upscale")
                    .weak()
                    .size(11.0),
            );
        }
    }
}

/// A label + an RGB colour swatch editor on one row.
fn color_row(ui: &mut egui::Ui, label: &str, c: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.color_edit_button_rgb(c);
    });
}
