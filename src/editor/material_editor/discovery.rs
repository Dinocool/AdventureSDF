//! Filesystem discovery for the material/PBR-texture editors: turn `.material.ron`,
//! `.pbrtex.ron`, and image files on disk into [`PickerEntry`]s, and enumerate the
//! texture-library variants.

use crate::editor::assets_browser::{ASSETS_ROOT, TileThumb};
use crate::editor::fs_util::{is_image_file, walk_files};
use crate::editor::resource_picker::PickerEntry;
use crate::sdf_render::textures::{LibraryVariant, TEXTURE_ROOT, read_manifest};

/// A `.pbrtex.ron` bundle path → `PickerEntry` (key/label = the path; thumbnail = the
/// bundle file itself, so the assets-browser provider renders its sphere).
pub(super) fn pbrtex_entry(path: &std::path::Path) -> PickerEntry {
    let label = path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".pbrtex.ron").to_string())
        .unwrap_or_default();
    PickerEntry {
        key: path.to_string_lossy().into_owned(),
        label,
        // Working-dir path so the assets-browser thumbnail provider can resolve it.
        thumb: TileThumb::Path(std::path::Path::new(ASSETS_ROOT).join(path)),
    }
}

/// Picker entries for every `*.pbrtex.ron` under `assets/` (recursive).
pub(super) fn pbrtex_picker_entries() -> Vec<PickerEntry> {
    let root = std::path::Path::new(ASSETS_ROOT);
    let mut out: Vec<_> = walk_files(root)
        .into_iter()
        .filter(|p| p.to_string_lossy().to_lowercase().ends_with(".pbrtex.ron"))
        .filter_map(|p| p.strip_prefix(root).ok().map(pbrtex_entry))
        .collect();
    out.sort_by_key(|e| e.label.to_lowercase());
    out
}

/// An image file (path relative to `assets/`) → `PickerEntry` (thumbnail = the image).
pub(super) fn image_file_entry(path: &std::path::Path) -> PickerEntry {
    PickerEntry {
        key: path.to_string_lossy().into_owned(),
        label: crate::editor::fs_util::file_name_str(path),
        thumb: TileThumb::Path(std::path::Path::new(ASSETS_ROOT).join(path)),
    }
}

/// Picker entries for every image file under `assets/textures/` (recursive). Used by the
/// per-role override pickers — any supported image format.
pub(super) fn image_file_picker_entries() -> Vec<PickerEntry> {
    let root = std::path::Path::new(ASSETS_ROOT);
    let tex_root = root.join("textures");
    let mut out: Vec<_> = walk_files(&tex_root)
        .into_iter()
        .filter(|p| is_image_file(p))
        .filter_map(|p| p.strip_prefix(root).ok().map(image_file_entry))
        .collect();
    out.sort_by_key(|e| e.label.to_lowercase());
    out
}

/// List every `assets/materials/*.material.ron` as a [`PickerEntry`] (key = working-dir
/// path string, label = file stem, thumb = the file path so the registry renders its
/// sphere). For the material resource picker.
pub fn material_picker_entries() -> Vec<PickerEntry> {
    let dir = std::path::Path::new(ASSETS_ROOT).join("materials");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<PickerEntry> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| crate::editor::fs_util::is_material_ron(p))
        .map(|p| {
            let label = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.trim_end_matches(".material.ron").to_string())
                .unwrap_or_default();
            PickerEntry {
                key: p.to_string_lossy().into_owned(),
                label,
                thumb: TileThumb::Path(p),
            }
        })
        .collect();
    out.sort_by_key(|e| e.label.to_lowercase());
    out
}

/// Discover all texture variants by scanning `assets/textures/<slug>/material.ron`.
pub fn discover_variants() -> Vec<LibraryVariant> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn pbrtex_entry_key_is_path_label_is_stem() {
        let e = pbrtex_entry(Path::new("pbrtextures/cobble_stone_3.pbrtex.ron"));
        assert_eq!(e.key, "pbrtextures/cobble_stone_3.pbrtex.ron");
        assert_eq!(e.label, "cobble_stone_3");
    }

    #[test]
    fn image_file_entry_key_is_path_label_is_name() {
        let e = image_file_entry(Path::new("textures/sand/1/height.png"));
        assert_eq!(e.key, "textures/sand/1/height.png");
        assert_eq!(e.label, "height.png");
    }
}
