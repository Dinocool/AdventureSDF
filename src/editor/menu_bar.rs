//! Top menu bar (jackdaw `TopLevelMenu`). File actions don't perform scene I/O
//! directly — they set [`EditorRequests`] flags that the `soul_scene` save/load
//! systems (Part B) drain. This keeps the editor shell decoupled from the scene
//! backend (and compiling without it).

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_egui::egui;

/// One-shot editor commands raised by the menu bar / keybinds, drained by the
/// systems that own each action (scene I/O lives in `soul_scene`).
#[derive(Resource, Default)]
pub struct EditorRequests {
    pub new_scene: bool,
    pub save: bool,
    pub save_as: Option<PathBuf>,
    pub open: Option<PathBuf>,
}

/// Default path the editor saves to until "Save As" picks another. Kept on the
/// requests resource's sibling so the status bar / save system can show it.
#[derive(Resource)]
pub struct CurrentScenePath(pub PathBuf);

impl Default for CurrentScenePath {
    fn default() -> Self {
        // The editor loads the gallery as its default scene, so File→Save targets it.
        Self(PathBuf::from(crate::sdf_render::DEFAULT_SCENE_PATH))
    }
}

pub fn menu_bar_ui(world: &mut World, ctx: &egui::Context) {
    use super::dock::{EditorDockState, EditorTab};
    use super::layout::{self, LayoutsDialog};
    use super::panels::{DebugPanelRegistry, DockSide};
    use super::scene_browser::{OpenSceneDialog, SaveSceneDialog};

    let current = world.resource::<CurrentScenePath>().0.clone();
    let mut req_new = false;
    let mut req_save = false;
    let mut open_browser = false;
    let mut save_as_browser = false;
    let mut panel_toggles: Vec<(EditorTab, DockSide, bool)> = Vec::new();
    let mut restore_default = false;
    let mut open_layouts = false;

    // Edit-menu state (enable/disable) + one-shot click flags, dispatched after the egui pass.
    let (can_undo, can_redo, has_clip, has_sel) = {
        use crate::editor::history::{EditHistories, EditorClipboard};
        let h = world.resource::<EditHistories>();
        (
            h.can_undo(),
            h.can_redo(),
            world.resource::<EditorClipboard>().content.is_some(),
            world.resource::<crate::sdf_render::SdfSelection>().entity.is_some(),
        )
    };
    let mut e_undo = false;
    let mut e_redo = false;
    let mut e_copy = false;
    let mut e_cut = false;
    let mut e_paste = false;
    let mut e_delete = false;

    egui::TopBottomPanel::top("editor_menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Scene").clicked() {
                    req_new = true;
                    ui.close();
                }
                if ui.button("Open…").clicked() {
                    // Open the file browser, starting in the current scene's directory.
                    open_browser = true;
                    ui.close();
                }
                if ui.button("Save").clicked() {
                    req_save = true;
                    ui.close();
                }
                if ui.button("Save As…").clicked() {
                    save_as_browser = true;
                    ui.close();
                }
                ui.separator();
                ui.weak(current.display().to_string());
            });

            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(can_undo, egui::Button::new("Undo  Ctrl+Z")).clicked() {
                    e_undo = true;
                    ui.close();
                }
                if ui.add_enabled(can_redo, egui::Button::new("Redo  Ctrl+Y")).clicked() {
                    e_redo = true;
                    ui.close();
                }
                ui.separator();
                if ui.add_enabled(has_sel, egui::Button::new("Cut  Ctrl+X")).clicked() {
                    e_cut = true;
                    ui.close();
                }
                if ui.add_enabled(has_sel, egui::Button::new("Copy  Ctrl+C")).clicked() {
                    e_copy = true;
                    ui.close();
                }
                if ui.add_enabled(has_clip, egui::Button::new("Paste  Ctrl+V")).clicked() {
                    e_paste = true;
                    ui.close();
                }
                if ui.add_enabled(has_sel, egui::Button::new("Delete  Del")).clicked() {
                    e_delete = true;
                    ui.close();
                }
            });

            ui.menu_button("View", |ui| {
                ui.label("Panels");
                ui.separator();
                let panels = layout::toggleable_panels(world.resource::<DebugPanelRegistry>());
                for (tab, title, side) in panels {
                    let mut present =
                        layout::panel_present(world.resource::<EditorDockState>(), &tab);
                    if ui.checkbox(&mut present, title).changed() {
                        panel_toggles.push((tab, side, present));
                    }
                }
            });

            ui.menu_button("Layout", |ui| {
                if ui.button("Restore Default Layout").clicked() {
                    restore_default = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Manage Layouts\u{2026}").clicked() {
                    open_layouts = true;
                    ui.close();
                }
            });
        });
    });

    // Dispatch Edit-menu actions (each needs exclusive `&mut World`, so run them after the pass).
    if e_undo {
        crate::editor::history::undo(world);
    }
    if e_redo {
        crate::editor::history::redo(world);
    }
    if e_copy {
        crate::editor::history::copy(world);
    }
    if e_cut {
        crate::editor::history::cut(world);
    }
    if e_paste {
        crate::editor::history::paste(world);
    }
    if e_delete {
        crate::editor::history::delete_selected(world);
    }

    for (tab, side, present) in panel_toggles {
        layout::set_panel_present(world, tab, side, present);
    }
    if restore_default {
        let registry = world
            .remove_resource::<DebugPanelRegistry>()
            .unwrap_or_default();
        layout::restore_default(world, &registry);
        world.insert_resource(registry);
    }
    if open_layouts {
        world.resource_mut::<LayoutsDialog>().open = true;
    }

    if req_new || req_save {
        let mut requests = world.resource_mut::<EditorRequests>();
        requests.new_scene |= req_new;
        requests.save |= req_save;
    }
    if open_browser {
        let start = current.parent().map(PathBuf::from).unwrap_or_default();
        world.resource_mut::<OpenSceneDialog>().show_at(&start);
    }
    if save_as_browser {
        world.resource_mut::<SaveSceneDialog>().show_for(&current);
    }
}
