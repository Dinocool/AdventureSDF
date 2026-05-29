use std::time::Instant;

use bevy::prelude::*;

use super::config::DebugToolkitConfig;

#[derive(Resource, Default)]
pub struct ShaderHotReloadState {
    pub reload_count: u32,
    pub last_reload: Option<Instant>,
    pub debounce_ms: u64,
}

pub struct HotReloadPlugin;

impl Plugin for HotReloadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderHotReloadState>()
            .add_systems(Update, detect_shader_changes);
    }
}

fn detect_shader_changes(
    mut events: MessageReader<AssetEvent<Shader>>,
    mut state: ResMut<ShaderHotReloadState>,
    config: Res<DebugToolkitConfig>,
) {
    if !config.enabled || !config.hot_reload_enabled {
        return;
    }

    let mut got_event = false;
    for _ in events.read() {
        got_event = true;
    }

    if !got_event {
        return;
    }

    if let Some(last) = state.last_reload
        && last.elapsed().as_millis() < state.debounce_ms as u128
    {
        return;
    }

    state.reload_count += 1;
    state.last_reload = Some(Instant::now());
}

pub fn hot_reload_ui(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    if !world.resource::<DebugToolkitConfig>().enabled {
        return;
    }

    let state = world.resource::<ShaderHotReloadState>();
    ui.label(format!("Reloads: {}", state.reload_count));
    match state.last_reload {
        Some(t) => ui.label(format!("Last: {:.1}s ago", t.elapsed().as_secs_f32())),
        None => ui.label("Last: never"),
    };
}
