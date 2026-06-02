//! Asset inspector: renders the inspector view for a selected asset file. Modular —
//! each asset kind supplies an [`AssetInspector`] (mirrors the assets-browser
//! `ThumbnailProvider` pattern); files with no registered inspector show read-only
//! metadata.

use std::path::Path;

use bevy::prelude::*;
use bevy_egui::egui;

use crate::editor::assets_browser::thumbnail::{ensure_image_texture, ImageTexture};
use crate::editor::import_settings::{ColorSpace, ImageFilter, TextureImportSettings, WrapMode};

/// Per-asset-kind inspector. `matches` claims a path (usually by extension); `ui`
/// draws the editor with exclusive `World` access.
pub trait AssetInspector: Send + Sync + 'static {
    fn matches(&self, path: &Path) -> bool;
    fn ui(&self, world: &mut World, path: &Path, ui: &mut egui::Ui);
}

/// Ordered registry of asset inspectors. First match wins; otherwise generic metadata.
#[derive(Resource, Default)]
pub struct AssetInspectorRegistry {
    inspectors: Vec<Box<dyn AssetInspector>>,
}

impl AssetInspectorRegistry {
    pub fn register(&mut self, inspector: impl AssetInspector) {
        self.inspectors.push(Box::new(inspector));
    }
}

/// Wires the asset-inspector registry (seeded with the built-in texture/material/PBR inspectors)
/// plus the per-asset [`ImportSettingsEdits`] buffer. Was inline in `EditorPlugin::build`, seeding
/// a registry defined here.
pub struct AssetInspectorPlugin;

impl Plugin for AssetInspectorPlugin {
    fn build(&self, app: &mut App) {
        let mut reg = AssetInspectorRegistry::default();
        reg.register(TextureAssetInspector);
        reg.register(MaterialAssetInspector);
        reg.register(PbrTextureAssetInspector);
        app.init_resource::<ImportSettingsEdits>().insert_resource(reg);
    }
}

/// Render the inspector for `path`. Dispatches to the first matching [`AssetInspector`],
/// else shows generic file metadata.
pub fn asset_inspector_ui(world: &mut World, path: &Path, ui: &mut egui::Ui) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned());
    ui.heading(name);
    ui.separator();

    // Take the registry out so inspectors get exclusive `&mut World` (restored after).
    let handled = crate::editor::fs_util::with_registry(
        world,
        |world, registry: &AssetInspectorRegistry| {
            for inspector in &registry.inspectors {
                if inspector.matches(path) {
                    inspector.ui(world, path, ui);
                    return true;
                }
            }
            false
        },
    );

    if !handled {
        generic_file_info(path, ui);
    }
}

/// Read-only metadata fallback for assets with no dedicated inspector.
fn generic_file_info(path: &Path, ui: &mut egui::Ui) {
    ui.label(format!("Path: {}", path.display()));
    if let Ok(meta) = std::fs::metadata(path) {
        ui.label(format!("Size: {} bytes", meta.len()));
    }
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        ui.label(format!("Type: .{ext}"));
    }
}

// === Texture inspector ===============================================================

/// In-progress import-settings edits, keyed by image path. Holds the working copy
/// between frames (the on-disk sidecar is only written on Save).
#[derive(Resource, Default)]
pub struct ImportSettingsEdits {
    map: std::collections::HashMap<std::path::PathBuf, TextureImportSettings>,
}

/// Inspector for raster image assets: preview + read-only info + editable import
/// settings persisted to a `<file>.import.ron` sidecar.
pub struct TextureAssetInspector;

impl AssetInspector for TextureAssetInspector {
    fn matches(&self, path: &Path) -> bool {
        crate::editor::fs_util::is_image_file(path)
    }

    fn ui(&self, world: &mut World, path: &Path, ui: &mut egui::Ui) {
        // Preview thumbnail (reuses the assets-browser image cache).
        let (dims, ready) = match ensure_image_texture(world, path) {
            ImageTexture::Ready { tex_id, handle } => {
                let dims = world
                    .resource::<Assets<Image>>()
                    .get(&handle)
                    .map(|img| img.texture_descriptor.size);
                ui.image(egui::load::SizedTexture::new(tex_id, egui::vec2(160.0, 160.0)));
                (dims, true)
            }
            ImageTexture::Loading => {
                ui.weak("Loading preview…");
                (None, false)
            }
            ImageTexture::Invalid => {
                ui.weak("Not a loadable asset path.");
                (None, false)
            }
        };
        let _ = ready;

        // Read-only info.
        if let Some(size) = dims {
            ui.label(format!("Dimensions: {}×{}", size.width, size.height));
        }
        if let Ok(meta) = std::fs::metadata(path) {
            ui.label(format!("Size: {} KB", meta.len() / 1024));
        }

        ui.separator();
        ui.strong("Import");

        // Load the working copy on first edit of this path (from the sidecar/defaults).
        // Single entry lookup (no separate contains-then-index) so the read can't race a
        // concurrent edit out from under us.
        let key = path.to_path_buf();
        let mut settings = world
            .resource_mut::<ImportSettingsEdits>()
            .map
            .entry(key.clone())
            .or_insert_with(|| TextureImportSettings::load_for(path))
            .clone();

        egui::ComboBox::from_label("Filter")
            .selected_text(format!("{:?}", settings.filter))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut settings.filter, ImageFilter::Linear, "Linear");
                ui.selectable_value(&mut settings.filter, ImageFilter::Nearest, "Nearest");
            });
        egui::ComboBox::from_label("Color space")
            .selected_text(format!("{:?}", settings.color_space))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut settings.color_space, ColorSpace::Srgb, "sRGB");
                ui.selectable_value(&mut settings.color_space, ColorSpace::Linear, "Linear");
            });
        egui::ComboBox::from_label("Wrap")
            .selected_text(format!("{:?}", settings.wrap))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut settings.wrap, WrapMode::Repeat, "Repeat");
                ui.selectable_value(&mut settings.wrap, WrapMode::Clamp, "Clamp");
            });

        // Store working edits back.
        world
            .resource_mut::<ImportSettingsEdits>()
            .map
            .insert(key.clone(), settings.clone());

        ui.separator();
        if ui.button("Save import settings").clicked() {
            match settings.save_for(path) {
                Ok(()) => info!("wrote import settings for {}", path.display()),
                Err(e) => error!("import settings save failed: {e}"),
            }
        }
    }
}

// === Material inspector ==============================================================

/// Inspector for `*.material.ron` assets: the shared interactive material editor
/// (preview + fields) plus a Save back to the file.
pub struct MaterialAssetInspector;

impl AssetInspector for MaterialAssetInspector {
    fn matches(&self, path: &Path) -> bool {
        crate::editor::fs_util::is_material_ron(path)
    }

    fn ui(&self, world: &mut World, path: &Path, ui: &mut egui::Ui) {
        use crate::editor::material_editor;

        let Some(handle) = material_editor::handle_for_path(world, path) else {
            ui.weak("Could not resolve material asset.");
            return;
        };

        material_editor::material_editor_ui(world, &handle, ui);

        ui.separator();
        if ui.button("Save").clicked() {
            // Derive the on-disk name from the file stem (`x.material.ron` → `x`).
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.trim_end_matches(".material.ron").to_string())
                .unwrap_or_default();
            if !name.is_empty() {
                material_editor::save_material(world, &handle, &name);
            }
        }
    }
}

// === PBR texture (bundle) inspector =================================================

/// Inspector for `*.pbrtex.ron` bundles: the 7 role file pickers + live preview + Save.
pub struct PbrTextureAssetInspector;

impl AssetInspector for PbrTextureAssetInspector {
    fn matches(&self, path: &Path) -> bool {
        path.to_string_lossy().to_lowercase().ends_with(".pbrtex.ron")
    }

    fn ui(&self, world: &mut World, path: &Path, ui: &mut egui::Ui) {
        use crate::editor::material_editor;

        let Some(handle) = material_editor::pbrtex_handle_for_path(world, path) else {
            ui.weak("Could not resolve PBR texture.");
            return;
        };
        material_editor::pbr_texture_editor_ui(world, &handle, ui);

        ui.separator();
        if ui.button("Save").clicked()
            && let Some(rel) = crate::editor::fs_util::relative_to_assets(path)
        {
            material_editor::save_pbr_texture(world, &handle, &rel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_inspector_matches_image_extensions() {
        let t = TextureAssetInspector;
        assert!(t.matches(Path::new("a/b/diffuse.png")));
        assert!(t.matches(Path::new("X.JPG")));
        assert!(!t.matches(Path::new("a.material.ron")));
        assert!(!t.matches(Path::new("a.txt")));
    }

    #[test]
    fn material_inspector_matches_material_ron_only() {
        let m = MaterialAssetInspector;
        assert!(m.matches(Path::new("materials/sand.material.ron")));
        assert!(!m.matches(Path::new("scene.ron"))); // plain .ron is not a material
        assert!(!m.matches(Path::new("a.png")));
    }
}
