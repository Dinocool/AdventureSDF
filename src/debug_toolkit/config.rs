use bevy::prelude::*;

#[derive(Resource, Reflect, Clone)]
#[reflect(Resource)]
pub struct DebugToolkitConfig {
    pub enabled: bool,
    pub hot_reload_enabled: bool,
}

impl Default for DebugToolkitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hot_reload_enabled: true,
        }
    }
}

pub struct DebugToolkitConfigPlugin;

impl Plugin for DebugToolkitConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugToolkitConfig>()
            .register_type::<DebugToolkitConfig>();
    }
}
