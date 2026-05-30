use std::collections::HashMap;

use bevy::prelude::*;

use super::config::EditorConfig;

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

/// Prettify an exclusive-group id for display: strip a leading `sdf_` and
/// capitalise (`sdf_overlay` -> `Overlay`).
fn pretty_group(name: &str) -> String {
    let n = name.strip_prefix("sdf_").unwrap_or(name);
    let mut chars = n.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

pub fn debug_modes_ui(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use bevy_egui::egui;

    let config = world.resource::<EditorConfig>();
    if !config.enabled {
        return;
    }

    // Clone mode data so we can drop the registry borrow
    let grouped = {
        let registry = world.resource::<ShaderDebugRegistry>();
        snapshot_modes(registry)
    };

    // Each exclusive group is a single-select dropdown: "Off" plus one entry per mode.
    for (group_name, modes) in &grouped.exclusive {
        let active = {
            let state = world.resource::<ShaderDebugState>();
            modes.iter().find(|m| state.is_active(&m.id)).cloned()
        };
        let selected_text = active
            .as_ref()
            .map(|m| m.label.clone())
            .unwrap_or_else(|| "Off".to_string());

        // `Some(None)` = pick Off; `Some(Some(id))` = activate that mode.
        let mut pick: Option<Option<String>> = None;
        ui.horizontal(|ui| {
            ui.label(pretty_group(group_name));
            egui::ComboBox::from_id_salt(group_name)
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(active.is_none(), "Off").clicked() {
                        pick = Some(None);
                    }
                    for mode in modes {
                        let is_on = active.as_ref().is_some_and(|a| a.id == mode.id);
                        if ui
                            .selectable_label(is_on, &mode.label)
                            .on_hover_text(&mode.description)
                            .clicked()
                        {
                            pick = Some(Some(mode.id.clone()));
                        }
                    }
                });
        });
        if let Some(choice) = pick {
            let mut state = world.resource_mut::<ShaderDebugState>();
            for m in modes {
                state.set(&m.id, false);
            }
            if let Some(id) = choice {
                state.set(&id, true);
            }
        }
        if let Some(active) = &active {
            ui.label(&active.description);
        }
    }

    // Diagnostic toggles: independent checkboxes, tucked away (default-collapsed) so
    // they don't crowd the common overlay/raymarch controls.
    if !grouped.toggles.is_empty() {
        egui::CollapsingHeader::new("Diagnostics")
            .default_open(false)
            .show(ui, |ui| {
                for mode in &grouped.toggles {
                    let active = world.resource::<ShaderDebugState>().is_active(&mode.id);
                    let mut toggled = active;
                    if ui
                        .checkbox(&mut toggled, &mode.label)
                        .on_hover_text(&mode.description)
                        .changed()
                    {
                        world
                            .resource_mut::<ShaderDebugState>()
                            .set(&mode.id, toggled);
                    }
                }
            });
    }
}
