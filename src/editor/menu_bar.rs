//! Top menu bar (jackdaw `TopLevelMenu`). File actions don't perform scene I/O
//! directly — they set [`EditorRequests`] flags that the `soul_scene` save/load
//! systems (Part B) drain. This keeps the editor shell decoupled from the scene
//! backend (and compiling without it).

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_egui::egui;

use super::config::EditorConfig;

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
        Self(PathBuf::from("assets/scenes/untitled.scene"))
    }
}

pub fn menu_bar_ui(world: &mut World, ctx: &egui::Context) {
    let current = world.resource::<CurrentScenePath>().0.clone();
    let mut req_new = false;
    let mut req_save = false;
    let mut req_open: Option<PathBuf> = None;

    egui::TopBottomPanel::top("editor_menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Scene").clicked() {
                    req_new = true;
                    ui.close();
                }
                if ui.button("Open…").clicked() {
                    // Until a file dialog is wired, reopen the current path.
                    req_open = Some(current.clone());
                    ui.close();
                }
                if ui.button("Save").clicked() {
                    req_save = true;
                    ui.close();
                }
                ui.separator();
                ui.weak(current.display().to_string());
            });

            ui.menu_button("View", |ui| {
                let mut enabled = world.resource::<EditorConfig>().enabled;
                if ui.checkbox(&mut enabled, "Editor panels").changed() {
                    world.resource_mut::<EditorConfig>().enabled = enabled;
                }
            });
        });
    });

    if req_new || req_save || req_open.is_some() {
        let mut requests = world.resource_mut::<EditorRequests>();
        requests.new_scene |= req_new;
        requests.save |= req_save;
        if req_open.is_some() {
            requests.open = req_open;
        }
    }
}
