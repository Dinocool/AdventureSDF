//! Bottom status bar (jackdaw `status_bar`): scene stats at a glance.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::scene_manager::SceneEntity;
use crate::sdf_render::SdfVolume;
use crate::sdf_render::atlas::SdfAtlas;

use super::profiling::ShaderProfilingData;

/// Render the bottom status strip: entity/volume counts, atlas bake state, and the perf
/// readout (FPS / frame time).
pub fn status_bar_ui(world: &mut World, ctx: &egui::Context) {
    let scene_entities = world
        .query_filtered::<(), With<SceneEntity>>()
        .iter(world)
        .count();
    let volumes = world
        .query_filtered::<(), With<SdfVolume>>()
        .iter(world)
        .count();
    let (bricks, dirty) = world
        .get_resource::<SdfAtlas>()
        .map(|a| (a.bricks.len(), a.rebake_all || !a.gpu_baked_tiles.is_empty()))
        .unwrap_or((0, false));
    let perf = world
        .get_resource::<ShaderProfilingData>()
        .map(|p| (p.fps_smoothed, p.frame_time_ms))
        .unwrap_or((0.0, 0.0));

    egui::TopBottomPanel::bottom("editor_status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label(format!("Entities: {scene_entities}"));
            ui.separator();
            ui.label(format!("SDF volumes: {volumes}"));
            ui.separator();
            ui.label(format!("Bricks: {bricks}"));
            ui.separator();
            let (color, text) = if dirty {
                (egui::Color32::YELLOW, "BAKING")
            } else {
                (egui::Color32::GREEN, "BAKED")
            };
            ui.colored_label(color, text);

            // Perf readout, right-aligned so it sits at the far end of the bar.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{:.2} ms", perf.1));
                ui.separator();
                ui.label(format!("{:.1} FPS", perf.0));
            });
        });
    });
}
