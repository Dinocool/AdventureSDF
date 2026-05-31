//! Project Files dock (jackdaw `project_files` / Godot FileSystem dock): a
//! read-only tree of the project's `assets/` directory. Bottom-left in the layout.
//!
//! Material/texture *previews* are Phase 2 (the material agent's territory); this
//! is just a file tree for now.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::egui;

use super::assets_browser::{ASSETS_ROOT, AssetsBrowserState};

/// Render the assets file tree. Stateless: re-reads the directory each frame
/// (cheap for a small tree; revisit with a cached/watched model if it grows).
/// Clicking a folder header syncs the Assets browser tab to that folder.
pub fn project_files_ui(world: &mut World, ui: &mut egui::Ui) {
    let root = Path::new(ASSETS_ROOT);
    if !root.is_dir() {
        ui.weak(format!("No `{ASSETS_ROOT}/` directory found."));
        return;
    }
    let mut nav_to: Option<PathBuf> = None;
    egui::ScrollArea::vertical().show(ui, |ui| {
        dir_tree(ui, root, &mut nav_to);
    });
    if let Some(rel) = nav_to {
        world.resource_mut::<AssetsBrowserState>().current = rel;
    }
}

/// Recursively render a directory as collapsing headers; files as labels. A folder
/// header's body-open click also reports its root-relative path via `nav_to`.
fn dir_tree(ui: &mut egui::Ui, dir: &Path, nav_to: &mut Option<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    // Collect + sort: directories first, then files, each alphabetically.
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else {
            files.push(path);
        }
    }
    dirs.sort();
    files.sort();

    for d in dirs {
        let name = d
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let header = egui::CollapsingHeader::new(format!("\u{1F4C1} {name}"))
            .id_salt(&d)
            .show(ui, |ui| {
                dir_tree(ui, &d, nav_to);
            });
        // Clicking the header label syncs the Assets browser to this folder.
        if header.header_response.clicked()
            && let Ok(rel) = d.strip_prefix(ASSETS_ROOT)
        {
            *nav_to = Some(rel.to_path_buf());
        }
    }
    for f in files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        ui.label(format!("  {name}"));
    }
}
