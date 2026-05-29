use std::collections::HashMap;

use bevy::prelude::*;

use super::config::DebugToolkitConfig;

#[derive(Clone, Debug)]
pub enum DebugModeKind {
    /// Radio-button group: only one active at a time within the named group.
    Exclusive { group: String },
    /// Independent toggle checkbox.
    Toggle,
}

#[derive(Clone)]
pub struct ShaderDebugMode {
    pub id: String,
    pub label: String,
    pub shader_define: String,
    pub kind: DebugModeKind,
    pub description: String,
}

#[derive(Resource, Default)]
pub struct ShaderDebugRegistry {
    modes: Vec<ShaderDebugMode>,
}

impl ShaderDebugRegistry {
    pub fn register(&mut self, mode: ShaderDebugMode) {
        self.modes.push(mode);
    }

    pub fn modes(&self) -> &[ShaderDebugMode] {
        &self.modes
    }
}

#[derive(Resource, Default)]
pub struct ShaderDebugState {
    active: HashMap<String, bool>,
}

impl ShaderDebugState {
    pub fn is_active(&self, mode_id: &str) -> bool {
        self.active.get(mode_id).copied().unwrap_or(false)
    }

    pub fn set(&mut self, mode_id: &str, enabled: bool) {
        self.active.insert(mode_id.to_string(), enabled);
    }

    pub fn active_defines_for_prefix(
        &self,
        registry: &ShaderDebugRegistry,
        prefix: &str,
    ) -> Vec<String> {
        let mut defines: Vec<String> = registry
            .modes()
            .iter()
            .filter(|m| m.id.starts_with(prefix) && self.is_active(&m.id))
            .map(|m| m.shader_define.clone())
            .collect();
        defines.sort();
        defines
    }
}

pub struct ShaderDebugRegistryPlugin;

impl Plugin for ShaderDebugRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderDebugRegistry>()
            .init_resource::<ShaderDebugState>();
    }
}

/// Snapshot of registry data, cloned once per frame to avoid holding world borrows across UI calls.
struct GroupedModes {
    exclusive: Vec<(String, Vec<ShaderDebugMode>)>,
    toggles: Vec<ShaderDebugMode>,
}

fn snapshot_modes(registry: &ShaderDebugRegistry) -> GroupedModes {
    let mut exclusive: Vec<(String, Vec<ShaderDebugMode>)> = Vec::new();
    let mut toggles = Vec::new();

    for mode in registry.modes().iter().cloned() {
        match &mode.kind {
            DebugModeKind::Exclusive { group } => {
                if let Some(entry) = exclusive.iter_mut().find(|(g, _)| g == group) {
                    entry.1.push(mode);
                } else {
                    exclusive.push((group.clone(), vec![mode]));
                }
            }
            DebugModeKind::Toggle => toggles.push(mode),
        }
    }

    GroupedModes { exclusive, toggles }
}

pub fn debug_modes_ui(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    let config = world.resource::<DebugToolkitConfig>();
    if !config.enabled {
        return;
    }

    // Clone mode data so we can drop the registry borrow
    let grouped = {
        let registry = world.resource::<ShaderDebugRegistry>();
        snapshot_modes(&registry)
    };

    // Render exclusive groups
    for (group_name, modes) in &grouped.exclusive {
        ui.heading(group_name);
        ui.horizontal(|ui| {
            for mode in modes {
                let active = world.resource::<ShaderDebugState>().is_active(&mode.id);
                if ui.selectable_label(active, &mode.label).clicked() {
                    let mut state = world.resource_mut::<ShaderDebugState>();
                    for m in modes {
                        state.set(&m.id, false);
                    }
                    state.set(&mode.id, true);
                }
            }

            let any_active = {
                let state = world.resource::<ShaderDebugState>();
                modes.iter().any(|m| state.is_active(&m.id))
            };
            if any_active {
                if ui.selectable_label(false, "Off").clicked() {
                    let mut state = world.resource_mut::<ShaderDebugState>();
                    for m in modes {
                        state.set(&m.id, false);
                    }
                }
            }
        });

        let state = world.resource::<ShaderDebugState>();
        if let Some(active) = modes.iter().find(|m| state.is_active(&m.id)) {
            ui.label(&active.description);
        } else {
            ui.label("No overlay");
        }
    }

    if !grouped.toggles.is_empty() {
        ui.separator();
        ui.heading("Toggles");
        for mode in &grouped.toggles {
            let active = world.resource::<ShaderDebugState>().is_active(&mode.id);
            let mut toggled = active;
            if ui.checkbox(&mut toggled, &mode.label).changed() {
                world
                    .resource_mut::<ShaderDebugState>()
                    .set(&mode.id, toggled);
            }
        }
    }
}
