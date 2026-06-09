//! Bottom status bar (jackdaw `status_bar`): scene stats at a glance.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::scene_manager::SceneEntity;
use crate::sdf_render::SdfVolume;
use crate::sdf_render::atlas::SdfAtlas;
use crate::sdf_render::mesh_bake::MeshBakeStats;

use super::config::EditorConfig;
use super::profiling::ShaderProfilingData;

/// Cached scene entity / SDF-volume counts for the status bar. Filled by [`update_scene_stats`]
/// in `Update` (parallelizable) so the egui pass doesn't count the whole world every frame
/// inside its exclusive-`World` critical section (perf roadmap E4).
#[derive(Resource, Default)]
pub struct EditorSceneStats {
    pub scene_entities: usize,
    pub volumes: usize,
}

/// Recount scene entities + SDF volumes into [`EditorSceneStats`]. A normal (parallel) system,
/// not part of the serial egui pass; the status bar reads the cached resource.
pub fn update_scene_stats(
    scene_entities: Query<(), With<SceneEntity>>,
    volumes: Query<(), With<SdfVolume>>,
    mut stats: ResMut<EditorSceneStats>,
) {
    stats.scene_entities = scene_entities.iter().count();
    stats.volumes = volumes.iter().count();
}

/// Register the status-bar stats resource + its updater.
pub struct StatusBarPlugin;

impl Plugin for StatusBarPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorSceneStats>().add_systems(
            Update,
            update_scene_stats.run_if(|c: Res<EditorConfig>| c.enabled),
        );
    }
}

/// Render the bottom status strip: entity/volume counts, atlas bake state, and the perf
/// readout (FPS / frame time).
pub fn status_bar_ui(world: &mut World, ctx: &egui::Context) {
    let (scene_entities, volumes) = world
        .get_resource::<EditorSceneStats>()
        .map(|s| (s.scene_entities, s.volumes))
        .unwrap_or((0, 0));
    let bricks = world.get_resource::<SdfAtlas>().map(|a| a.bricks.len()).unwrap_or(0);
    // The mesh bake is the renderer; the editor "BAKING" light reflects ITS pending work, not the
    // gated-off GPU SDF atlas (whose dirty flags stay set when the GPU bake never runs).
    let pending = world.get_resource::<MeshBakeStats>().map(|s| s.pending).unwrap_or(0);
    let dirty = pending > 0;
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
                (egui::Color32::YELLOW, format!("BAKING ({pending})"))
            } else {
                (egui::Color32::GREEN, "BAKED".to_string())
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
