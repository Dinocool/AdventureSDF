use std::any::Any;
use std::collections::HashMap;

use bevy::prelude::*;

use super::config::EditorConfig;

type BoxedUiFn = Box<dyn Fn(&mut bevy_egui::egui::Ui, &mut dyn Any) + Send + Sync>;
type BoxedReaderFn = Box<dyn Fn(&World) -> Option<Box<dyn Any>> + Send + Sync>;

struct UniformEntry {
    read: BoxedReaderFn,
    ui: BoxedUiFn,
}

#[derive(Resource, Default)]
pub struct UniformInspectorRegistry {
    entries: HashMap<String, UniformEntry>,
}

impl UniformInspectorRegistry {
    pub fn register<F>(&mut self, label: &str, read: F, ui: BoxedUiFn)
    where
        F: Fn(&World) -> Option<Box<dyn Any>> + Send + Sync + 'static,
    {
        self.entries.insert(
            label.to_string(),
            UniformEntry {
                read: Box::new(read),
                ui,
            },
        );
    }
}

pub struct UniformInspectorPlugin;

impl Plugin for UniformInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UniformInspectorRegistry>();
    }
}

pub fn uniforms_ui(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    let config = world.resource::<EditorConfig>();
    if !config.enabled {
        return;
    }

    let labels: Vec<String> = {
        let registry = world.resource::<UniformInspectorRegistry>();
        registry.entries.keys().cloned().collect()
    };

    if labels.is_empty() {
        ui.label("No uniforms registered");
        return;
    }

    for label in &labels {
        ui.collapsing(label.as_str(), |ui| {
            let has_entry = {
                let registry = world.resource::<UniformInspectorRegistry>();
                registry.entries.contains_key(label)
            };

            if !has_entry {
                ui.label("Entry not found");
                return;
            }

            let data = {
                let registry = world.resource::<UniformInspectorRegistry>();
                let entry = registry.entries.get(label).unwrap();
                (entry.read)(world)
            };

            let Some(mut data) = data else {
                ui.label("No data available");
                return;
            };

            let registry = world.resource::<UniformInspectorRegistry>();
            let entry = registry.entries.get(label).unwrap();
            (entry.ui)(ui, data.as_mut());
        });
    }
}
