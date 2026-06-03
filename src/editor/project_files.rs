//! Project Files dock (jackdaw `project_files` / Godot FileSystem dock): a
//! read-only tree of the project's `assets/` directory. Bottom-left in the layout.
//!
//! Material/texture *previews* are Phase 2 (the material agent's territory); this
//! is just a file tree for now.

use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_egui::egui;

use super::assets_browser::{ASSETS_ROOT, AssetsBrowserState};

/// Render the assets file tree. Reads directory listings through `read_sorted_cached`
/// (5s TTL), so re-rendering each frame doesn't hit the disk every frame.
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
    // Directories first, then files, each alphabetically. Cached (5s TTL) so the recursive
    // tree walk doesn't `read_dir` every frame inside the egui pass (perf roadmap E1).
    let (dirs, files) = crate::editor::fs_util::read_sorted_cached(dir);

    for d in dirs {
        let name = crate::editor::fs_util::file_name_str(&d);
        let header =
            egui::CollapsingHeader::new(format!("{} {name}", egui_phosphor::regular::FOLDER))
            .id_salt(&d)
            .show(ui, |ui| {
                dir_tree(ui, &d, nav_to);
            });
        // Clicking the header label syncs the Assets browser to this folder.
        if header.header_response.clicked()
            && let Some(rel) = crate::editor::fs_util::relative_to_assets(&d)
        {
            *nav_to = Some(rel);
        }
    }
    for f in files {
        let name = crate::editor::fs_util::file_name_str(&f);
        ui.label(format!("  {name}"));
    }
}
