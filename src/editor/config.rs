use bevy::prelude::*;

#[derive(Resource, Reflect, Clone)]
#[reflect(Resource)]
pub struct EditorConfig {
    pub enabled: bool,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

pub struct EditorConfigPlugin;

impl Plugin for EditorConfigPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorConfig>()
            .register_type::<EditorConfig>();
    }
}
