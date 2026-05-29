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

use crate::assets::{Asset, MaterialAsset, MaterialAssetTable, TexRef};
use crate::sdf_render::edits::MATERIAL_TEX_MAPS;
use crate::sdf_render::textures::{LibraryVariant, TEXTURE_ROOT, read_manifest};

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

/// Map names for the per-map texture-reference editors.
const MAP_LABELS: [&str; MATERIAL_TEX_MAPS] = ["Diffuse", "Normal", "MRA", "Height", "Edge"];

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

    // Discover available (slug, dir) variants for the map pickers.
    let variants = discover_variants();

    // Edit the selected material's asset in place (fires Modified -> recompile).
    let handle = world.resource::<MaterialAssetTable>().handles[id].clone();
    let mut changed = false;
    {
        let mut assets = world.resource_mut::<Assets<MaterialAsset>>();
        let Some(asset) = assets.get_mut(&handle) else {
            ui.weak("Material asset still loading…");
            return;
        };

        let mut rgb = [
            asset.base_color[0],
            asset.base_color[1],
            asset.base_color[2],
        ];
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            asset.base_color[0] = rgb[0];
            asset.base_color[1] = rgb[1];
            asset.base_color[2] = rgb[2];
            changed = true;
        }
        if ui
            .add(egui::Slider::new(&mut asset.blend_softness, 0.0..=1.0).text("Blend softness"))
            .changed()
        {
            changed = true;
        }

        ui.separator();
        ui.label("Texture maps");
        for (m, label) in MAP_LABELS.iter().enumerate() {
            let current = asset.maps[m]
                .as_ref()
                .map(|t| format!("{}/{}", t.slug, t.dir))
                .unwrap_or_else(|| "(none)".to_string());
            egui::ComboBox::from_label(*label)
                .selected_text(current)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(asset.maps[m].is_none(), "(none)")
                        .clicked()
                    {
                        asset.maps[m] = None;
                        changed = true;
                    }
                    for v in &variants {
                        let sel = asset.maps[m]
                            .as_ref()
                            .is_some_and(|t| t.slug == v.slug && t.dir == v.dir);
                        if ui
                            .selectable_label(sel, format!("{}/{}", v.slug, v.dir))
                            .clicked()
                        {
                            asset.maps[m] = Some(TexRef {
                                slug: v.slug.clone(),
                                dir: v.dir.clone(),
                            });
                            changed = true;
                        }
                    }
                });
        }
    }
    let _ = changed; // get_mut already marked the asset modified.

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
        let path = std::path::PathBuf::from(format!("assets/materials/{name}.material.ron"));
        let assets = world.resource::<Assets<MaterialAsset>>();
        if let Some(asset) = assets.get(&handle) {
            match asset.save(&path) {
                Ok(()) => info!("saved material to {}", path.display()),
                Err(e) => error!("material save failed: {e}"),
            }
        }
    }
}

fn textures_tab(_world: &mut World, ui: &mut egui::Ui) {
    let variants = discover_variants();
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

/// Discover all texture variants by scanning `assets/textures/<slug>/material.ron`.
/// Demand-driven loading still happens via materials; this is purely for the editor
/// pickers/listing (what textures *could* be referenced).
fn discover_variants() -> Vec<LibraryVariant> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(TEXTURE_ROOT) else {
        return out;
    };
    let mut slugs: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter(|e| e.path().join("material.ron").is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    slugs.sort();
    for slug in slugs {
        out.extend(read_manifest(&slug));
    }
    out
}
