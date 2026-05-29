use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;

use super::config::EditorConfig;

#[derive(Resource, Default)]
pub struct ShaderProfilingData {
    pub fps_smoothed: f64,
    pub frame_time_ms: f64,
    pub frame_count: u64,
}

pub struct ProfilingPlugin;

impl Plugin for ProfilingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderProfilingData>()
            .add_systems(Update, collect_profiling_data);
    }
}

fn collect_profiling_data(
    mut data: ResMut<ShaderProfilingData>,
    diagnostics: Res<DiagnosticsStore>,
) {
    data.frame_count += 1;

    if let Some(fps_diagnostic) =
        diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS)
        && let Some(smoothed) = fps_diagnostic.smoothed()
    {
        data.fps_smoothed = smoothed;
    }

    if let Some(ft_diagnostic) =
        diagnostics.get(&bevy::diagnostic::FrameTimeDiagnosticsPlugin::FRAME_TIME)
        && let Some(smoothed) = ft_diagnostic.smoothed()
    {
        data.frame_time_ms = smoothed;
    }
}

pub fn profiling_ui(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    let config = world.resource::<EditorConfig>();
    if !config.enabled {
        return;
    }

    let data = world.resource::<ShaderProfilingData>();

    ui.label(format!("{:.1} FPS", data.fps_smoothed));
    ui.label(format!("{:.2} ms", data.frame_time_ms));
    ui.label(format!("Frame {}", data.frame_count));
}
