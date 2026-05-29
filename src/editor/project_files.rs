//! Project Files dock (jackdaw `project_files` / Godot FileSystem dock): a
//! read-only tree of the project's `assets/` directory. Bottom-left in the layout.
//!
//! Material/texture *previews* are Phase 2 (the material agent's territory); this
//! is just a file tree for now.

use bevy::prelude::*;
use bevy_egui::egui;

/// Root the file tree walks. Relative to the working dir (the worktree), matching
/// how Bevy's `AssetServer` resolves `assets/`.
const ASSETS_ROOT: &str = "assets";

/// Render the assets file tree. Stateless: re-reads the directory each frame
/// (cheap for a small tree; revisit with a cached/watched model if it grows).
pub fn project_files_ui(_world: &mut World, ui: &mut egui::Ui) {
    let root = std::path::Path::new(ASSETS_ROOT);
    if !root.is_dir() {
        ui.weak(format!("No `{ASSETS_ROOT}/` directory found."));
        return;
    }
    egui::ScrollArea::vertical().show(ui, |ui| {
        dir_tree(ui, root);
    });
}

/// Recursively render a directory as collapsing headers; files as labels.
fn dir_tree(ui: &mut egui::Ui, dir: &std::path::Path) {
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
        egui::CollapsingHeader::new(format!("\u{1F4C1} {name}"))
            .id_salt(&d)
            .show(ui, |ui| {
                dir_tree(ui, &d);
            });
    }
    for f in files {
        let name = f
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        ui.label(format!("  {name}"));
    }
}
