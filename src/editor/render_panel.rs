//! Editor "Render / GI" dock panel: live sliders for the raymarch lighting + GI uniforms
//! (knobs-as-uniforms — every value is a runtime uniform the shader reads) plus a **debug-view**
//! selector (normals / depth / albedo / AO / GI-only / face-orientation) for diagnosing the renderer.
//!
//! It mutates the [`VoxelRtLighting`] resource directly; the change is extracted to the render world +
//! uploaded to the WGSL `LightingUniform` next frame, so tweaks are live.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::voxel::raytrace::{LightingUniformData, SkyUniformData, VoxelRtLighting, VoxelRtSky};

/// Labels for `LightingUniformData.debug_view` (index == the u32 value the shader branches on).
const DEBUG_LABELS: [&str; 8] = [
    "Lit (normal)",
    "Normals",
    "Depth",
    "Albedo",
    "Ambient occlusion",
    "GI only",
    "Face orient (red = BACK face)",
    "LOD (ring colour)",
];

/// The panel body. Registered via `editor::panels::register_panel`.
pub fn render_gi_panel(world: &mut World, ui: &mut egui::Ui) {
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
        ui.add(egui::Slider::new(&mut d.gi_rays, 0..=32).text("GI rays / px"));
        ui.add(egui::Slider::new(&mut d.gi_intensity, 0.0..=4.0).text("GI intensity"));
        ui.add(egui::Slider::new(&mut d.gi_bounce_dist, 1.0..=64.0).text("bounce dist (m)"));
        ui.add(egui::Slider::new(&mut d.emissive_strength, 0.0..=16.0).text("emissive strength"));

        ui.separator();
        if ui.button("Reset to Cornell defaults").clicked() {
            let keep = d.debug_view;
            *d = LightingUniformData::cornell();
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
                     in 2.3); off = a fresh bounce trace each frame (A/B comparison)",
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
