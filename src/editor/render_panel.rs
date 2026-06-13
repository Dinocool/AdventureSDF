//! Editor "Render / GI" dock panel: live sliders for the raymarch lighting + GI uniforms
//! (knobs-as-uniforms — every value is a runtime uniform the shader reads) plus a **debug-view**
//! selector (normals / depth / albedo / AO / GI-only / face-orientation) for diagnosing the renderer.
//!
//! It mutates the [`VoxelRtLighting`] resource directly; the change is extracted to the render world +
//! uploaded to the WGSL `LightingUniform` next frame, so tweaks are live.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::voxel::raytrace::{LightingUniformData, VoxelRtLighting};

/// Labels for `LightingUniformData.debug_view` (index == the u32 value the shader branches on).
const DEBUG_LABELS: [&str; 7] = [
    "Lit (normal)",
    "Normals",
    "Depth",
    "Albedo",
    "Ambient occlusion",
    "GI only",
    "Face orient (red = BACK face)",
];

/// The panel body. Registered via `editor::panels::register_panel`.
pub fn render_gi_panel(world: &mut World, ui: &mut egui::Ui) {
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
        egui::RichText::new("Face orient: green = front, RED = back-face hit (show-through bug)")
            .weak()
            .size(11.0),
    );

    ui.separator();
    ui.label(egui::RichText::new("Sun").strong());
    ui.add(egui::Slider::new(&mut d.sun_intensity, 0.0..=5.0).text("intensity"));
    let mut dir = Vec3::from_array(d.sun_direction);
    let mut dir_changed = false;
    dir_changed |= ui.add(egui::Slider::new(&mut dir.x, -1.0..=1.0).text("dir x")).changed();
    dir_changed |= ui.add(egui::Slider::new(&mut dir.y, -1.0..=1.0).text("dir y")).changed();
    dir_changed |= ui.add(egui::Slider::new(&mut dir.z, -1.0..=1.0).text("dir z")).changed();
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
    ui.add(egui::Slider::new(&mut d.shadow_bias, 0.0..=0.2).text("shadow bias"));

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
}

/// A label + an RGB colour swatch editor on one row.
fn color_row(ui: &mut egui::Ui, label: &str, c: &mut [f32; 3]) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.color_edit_button_rgb(c);
    });
}
