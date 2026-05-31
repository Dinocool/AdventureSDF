//! Resource Inspector (Godot-style): edit material *resources* and browse texture
//! resources. Two internal tabs:
//! - **Materials**: every registered [`MaterialAsset`] — edit base_color,
//!   blend_softness, and the 5 per-map texture references; Save writes the RON
//!   resource to `assets/materials/`.
//! - **Textures**: texture variants discovered from the on-disk manifests, with
//!   read-only import info.
//!
//! Edits go through `Assets::<MaterialAsset>::get_mut`, which fires
//! `AssetEvent::Modified` → `assets::compile` rebuilds the registry → the GPU table
//! re-uploads via change detection. So material edits are live.

use bevy::prelude::*;
use bevy_egui::egui;

use crate::assets::MaterialAssetTable;

/// Which inner tab the Resource Inspector shows.
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy)]
pub enum ResourceTab {
    #[default]
    Materials,
    Textures,
}

/// Transient panel UI state (selection + the save-name field).
#[derive(Resource, Default)]
pub struct ResourceInspectorState {
    tab: ResourceTab,
    /// Selected material: index into [`MaterialAssetTable::handles`].
    selected: Option<usize>,
    save_name: String,
}

/// The panel render entry point (registered via `register_panel`).
pub fn resource_inspector_ui(world: &mut World, ui: &mut egui::Ui) {
    let tab = world.resource::<ResourceInspectorState>().tab;
    ui.horizontal(|ui| {
        let mut sel = tab;
        ui.selectable_value(&mut sel, ResourceTab::Materials, "Materials");
        ui.selectable_value(&mut sel, ResourceTab::Textures, "Textures");
        if sel != tab {
            world.resource_mut::<ResourceInspectorState>().tab = sel;
        }
    });
    ui.separator();

    match tab {
        ResourceTab::Materials => materials_tab(world, ui),
        ResourceTab::Textures => textures_tab(world, ui),
    }
}

fn materials_tab(world: &mut World, ui: &mut egui::Ui) {
    // Snapshot the registered material count + selection.
    let count = world.resource::<MaterialAssetTable>().handles.len();
    let selected = world.resource::<ResourceInspectorState>().selected;

    // Left: list of materials (id 0 = fallback, skip it as non-editable).
    ui.label(format!("{} materials", count.saturating_sub(1)));
    let mut clicked: Option<usize> = None;
    egui::ScrollArea::vertical()
        .max_height(120.0)
        .show(ui, |ui| {
            for id in 1..count {
                let is_sel = selected == Some(id);
                if ui
                    .selectable_label(is_sel, format!("Material {id}"))
                    .clicked()
                {
                    clicked = Some(id);
                }
            }
        });
    if let Some(id) = clicked {
        world.resource_mut::<ResourceInspectorState>().selected = Some(id);
    }

    let Some(id) = world.resource::<ResourceInspectorState>().selected else {
        ui.weak("Select a material to edit.");
        return;
    };
    if id >= count {
        return;
    }

    ui.separator();

    // Shared editor: interactive preview + field editors (single source of truth).
    let handle = world.resource::<MaterialAssetTable>().handles[id].clone();
    crate::editor::material_editor::material_editor_ui(world, &handle, ui);

    // Save controls.
    ui.separator();
    let mut name = world.resource::<ResourceInspectorState>().save_name.clone();
    ui.horizontal(|ui| {
        ui.label("Save as:");
        ui.text_edit_singleline(&mut name);
    });
    if name != world.resource::<ResourceInspectorState>().save_name {
        world.resource_mut::<ResourceInspectorState>().save_name = name.clone();
    }
    if ui.button("Save").clicked() && !name.is_empty() {
        crate::editor::material_editor::save_material(world, &handle, &name);
    }
}

fn textures_tab(_world: &mut World, ui: &mut egui::Ui) {
    let variants = crate::editor::material_editor::discover_variants();
    ui.label(format!("{} texture variants", variants.len()));
    ui.weak("Import: 1024², BC7 (read-only)");
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        let mut last_slug = String::new();
        for v in &variants {
            if v.slug != last_slug {
                ui.strong(&v.slug);
                last_slug = v.slug.clone();
            }
            ui.label(format!("  {} — {}", v.dir, v.display_name));
        }
    });
}
