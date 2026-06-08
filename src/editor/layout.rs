//! Dock layout persistence + panel visibility. Layouts capture the *panel arrangement* and
//! where the center "scene box" sits — but not which scenes are open (those are session
//! state). The live layout auto-persists on exit and restores on launch; named layouts live
//! under the editor-context dir and are managed from the Layouts modal.
//!
//! Serialization collapses the scene box to a single [`EditorTab::NoScene`] placeholder
//! (scene ids are session-specific); applying a layout re-injects the live scenes in its
//! place, so loading a layout only moves the scene box, never the scenes.

use std::path::PathBuf;

use bevy::prelude::*;
use bevy_egui::egui;
use egui_dock::{DockState, SurfaceIndex, TabIndex};

use super::dock::{add_panel_tab, is_center_tab, EditorDockState, EditorTab};
use super::panels::{DebugPanelRegistry, DockSide};
use super::scene_tabs;

/// Working-dir-relative root for all editor-side context (layouts, persisted UI state).
const EDITOR_DIR: &str = ".soul";
/// File the live layout auto-persists to (within [`EDITOR_DIR`]).
const CURRENT_LAYOUT_FILE: &str = "layout.ron";
/// Subdir (within [`EDITOR_DIR`]) holding named layouts as `<name>.ron`.
const LAYOUTS_SUBDIR: &str = "layouts";

fn editor_dir() -> PathBuf {
    PathBuf::from(EDITOR_DIR)
}
fn current_layout_path() -> PathBuf {
    editor_dir().join(CURRENT_LAYOUT_FILE)
}
fn layouts_dir() -> PathBuf {
    editor_dir().join(LAYOUTS_SUBDIR)
}
fn named_layout_path(name: &str) -> PathBuf {
    layouts_dir().join(format!("{name}.ron"))
}

// --- Serialize / apply -------------------------------------------------------------------

/// Serialize the dock's panel arrangement to RON, with the scene box collapsed to a single
/// `NoScene` placeholder (so the layout is scene-agnostic).
pub fn serialize_layout(dock: &EditorDockState) -> Option<String> {
    let mut state = dock.state.clone();
    set_scene_box_tabs(&mut state, vec![EditorTab::NoScene], 0);
    ron::ser::to_string_pretty(&state, ron::ser::PrettyConfig::default()).ok()
}

/// Parse a layout RON and install it, re-injecting the live scenes into its scene box.
/// Returns whether it applied.
pub fn apply_layout(world: &mut World, ron: &str) -> bool {
    let Ok(mut state) = ron::from_str::<DockState<EditorTab>>(ron) else {
        warn!("layout: failed to parse");
        return false;
    };
    inject_live_scenes(world, &mut state);
    world.resource_mut::<EditorDockState>().state = state;
    true
}

/// Replace whichever main-surface leaf holds the scene box with `tabs` (active = `active`).
fn set_scene_box_tabs(state: &mut DockState<EditorTab>, tabs: Vec<EditorTab>, active: usize) {
    for node in state.main_surface_mut().iter_mut() {
        if let Some(leaf) = node.get_leaf_mut()
            && leaf.tabs.iter().any(is_center_tab)
        {
            leaf.tabs = tabs.clone();
            leaf.active = TabIndex(active.min(leaf.tabs.len().saturating_sub(1)));
        }
    }
}

/// Put the currently-open scenes (or the empty placeholder) into `state`'s scene box.
fn inject_live_scenes(world: &World, state: &mut DockState<EditorTab>) {
    let (ids, active) = scene_tabs::scene_tab_ids(world);
    let active_idx = active
        .and_then(|a| ids.iter().position(|id| *id == a))
        .unwrap_or(0);
    let tabs: Vec<EditorTab> = if ids.is_empty() {
        vec![EditorTab::NoScene]
    } else {
        ids.into_iter().map(EditorTab::Scene).collect()
    };
    set_scene_box_tabs(state, tabs, active_idx);
}

/// Rebuild the default arrangement, keeping the live scenes in the center.
pub fn restore_default(world: &mut World, registry: &DebugPanelRegistry) {
    let mut state = EditorDockState::build(registry).state;
    inject_live_scenes(world, &mut state);
    world.resource_mut::<EditorDockState>().state = state;
}

// --- Named layout storage ----------------------------------------------------------------

/// Names of all saved layouts (alphabetical, case-insensitive).
pub fn list_layouts() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(layouts_dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|x| x.to_str()) == Some("ron"))
                .then(|| p.file_stem()?.to_str().map(str::to_owned))
                .flatten()
        })
        .collect();
    names.sort_by_key(|n| n.to_lowercase());
    names
}

fn save_named(name: &str, dock: &EditorDockState) -> bool {
    let Some(ron) = serialize_layout(dock) else {
        return false;
    };
    let _ = std::fs::create_dir_all(layouts_dir());
    std::fs::write(named_layout_path(name), ron).is_ok()
}

fn delete_named(name: &str) {
    let _ = std::fs::remove_file(named_layout_path(name));
}

fn load_named(world: &mut World, name: &str) -> bool {
    match std::fs::read_to_string(named_layout_path(name)) {
        Ok(ron) => apply_layout(world, &ron),
        Err(_) => false,
    }
}

// --- Auto-persist ------------------------------------------------------------------------

/// Write the live layout to the auto-persist file. Called on exit.
fn save_current_layout(dock: &EditorDockState) {
    if let Some(ron) = serialize_layout(dock) {
        let _ = std::fs::create_dir_all(editor_dir());
        let _ = std::fs::write(current_layout_path(), ron);
    }
}

/// Restore the auto-persisted layout if present (called once at startup, after the default
/// dock is built). Silently no-ops when there's no saved file.
pub fn load_current_layout(world: &mut World) {
    if let Ok(ron) = std::fs::read_to_string(current_layout_path()) {
        apply_layout(world, &ron);
    }
    inject_new_panels(world);
}

/// Path to the persisted set of registered-panel ids we've shown before.
fn known_panels_path() -> PathBuf {
    editor_dir().join("known_panels.ron")
}

/// Surface registered panels that are GENUINELY NEW — added in a newer build and never shown before —
/// so they appear once even behind a stale persisted layout, WITHOUT re-opening panels the user has
/// deliberately closed (those stay in the known set). Seeds the known set on a fresh install. Called
/// once at startup after the default dock build / persisted-layout apply.
fn inject_new_panels(world: &mut World) {
    let known: std::collections::HashSet<String> = std::fs::read_to_string(known_panels_path())
        .ok()
        .and_then(|s| ron::from_str(&s).ok())
        .unwrap_or_default();
    let (all_ids, new_panels): (Vec<String>, Vec<(EditorTab, DockSide)>) = {
        let registry = world.resource::<DebugPanelRegistry>();
        let mut all_ids = Vec::new();
        let mut new_panels = Vec::new();
        for side in [DockSide::Left, DockSide::Right, DockSide::Bottom] {
            for p in registry.panels_for(side) {
                all_ids.push(p.id.clone());
                if !known.contains(&p.id) {
                    new_panels.push((EditorTab::Registered(p.id.clone()), side));
                }
            }
        }
        (all_ids, new_panels)
    };
    for (tab, side) in new_panels {
        if !panel_present(world.resource::<EditorDockState>(), &tab) {
            set_panel_present(world, tab, side, true);
        }
    }
    let _ = std::fs::create_dir_all(editor_dir());
    if let Ok(s) = ron::to_string(&all_ids) {
        let _ = std::fs::write(known_panels_path(), s);
    }
}

/// System: persist the live layout when the app is exiting.
pub fn save_layout_on_exit(mut exit: MessageReader<AppExit>, dock: Option<Res<EditorDockState>>) {
    if exit.read().next().is_none() {
        return;
    }
    if let Some(dock) = dock {
        save_current_layout(&dock);
    }
}

// --- Panel enable/disable ----------------------------------------------------------------

/// Every toggleable panel `(tab, title, home side)` in View-menu order. The center scene
/// tabs are deliberately excluded — they're not panels.
pub fn toggleable_panels(registry: &DebugPanelRegistry) -> Vec<(EditorTab, String, DockSide)> {
    let mut panels = vec![
        (EditorTab::Hierarchy, "Scene".to_string(), DockSide::Left),
        (EditorTab::Inspector, "Inspector".to_string(), DockSide::Right),
        (EditorTab::ProjectFiles, "Project Files".to_string(), DockSide::Left),
        (EditorTab::AssetsDrawer, "Assets".to_string(), DockSide::Bottom),
    ];
    for side in [DockSide::Left, DockSide::Right, DockSide::Bottom] {
        for p in registry.panels_for(side) {
            panels.push((EditorTab::Registered(p.id.clone()), p.title.clone(), side));
        }
    }
    panels
}

/// Remembers where a hidden panel last lived (the tabs it shared a leaf with), so re-showing
/// it returns it to that group rather than its default side.
#[derive(Resource, Default)]
pub struct PanelRestore {
    siblings: Vec<(EditorTab, Vec<EditorTab>)>,
}

impl PanelRestore {
    fn remember(&mut self, tab: EditorTab, siblings: Vec<EditorTab>) {
        self.siblings.retain(|(t, _)| *t != tab);
        self.siblings.push((tab, siblings));
    }
    fn siblings_of(&self, tab: &EditorTab) -> Option<Vec<EditorTab>> {
        self.siblings
            .iter()
            .find(|(t, _)| t == tab)
            .map(|(_, s)| s.clone())
    }
}

/// Whether `tab` is currently shown in the dock.
pub fn panel_present(dock: &EditorDockState, tab: &EditorTab) -> bool {
    dock.state.find_main_surface_tab(tab).is_some()
}

/// The other tabs sharing `tab`'s leaf (its neighbours), for remembering where it lived.
fn sibling_tabs(dock: &EditorDockState, tab: &EditorTab) -> Option<Vec<EditorTab>> {
    let (node, _) = dock.state.find_main_surface_tab(tab)?;
    let leaf = dock.state.main_surface()[node].get_leaf()?;
    Some(leaf.tabs.iter().filter(|t| *t != tab).cloned().collect())
}

/// Show or hide `tab`. Hiding records its neighbours; showing restores it next to a surviving
/// neighbour (its previous location), falling back to its home `side`.
pub fn set_panel_present(world: &mut World, tab: EditorTab, side: DockSide, present: bool) {
    if !present {
        if let Some(siblings) = sibling_tabs(world.resource::<EditorDockState>(), &tab) {
            world.resource_mut::<PanelRestore>().remember(tab.clone(), siblings);
        }
        let mut dock = world.resource_mut::<EditorDockState>();
        if let Some((node, idx)) = dock.state.find_main_surface_tab(&tab) {
            dock.state.remove_tab((SurfaceIndex::main(), node, idx));
        }
        return;
    }

    // Re-show: prefer the leaf of a remembered neighbour that's still open.
    let anchor = world.resource::<PanelRestore>().siblings_of(&tab).and_then(|sibs| {
        let dock = world.resource::<EditorDockState>();
        sibs.into_iter()
            .find(|s| dock.state.find_main_surface_tab(s).is_some())
    });
    let mut dock = world.resource_mut::<EditorDockState>();
    if dock.state.find_main_surface_tab(&tab).is_some() {
        return; // already shown
    }
    if let Some(anchor) = anchor
        && let Some((node, _)) = dock.state.find_main_surface_tab(&anchor)
    {
        dock.state.main_surface_mut()[node].append_tab(tab);
        return;
    }
    add_panel_tab(&mut dock, tab, side);
}

// --- Layouts modal -----------------------------------------------------------------------

/// State for the "Layouts" manager window.
#[derive(Resource, Default)]
pub struct LayoutsDialog {
    pub open: bool,
    name_input: String,
}

/// Render the Layouts manager: name + Save current, and a list of saved layouts to load or
/// delete. No-op while closed. Reads/writes [`EditorDockState`], so call before the dock is
/// taken out of the world.
pub fn layouts_ui(world: &mut World, ctx: &egui::Context) {
    if !world.resource::<LayoutsDialog>().open {
        return;
    }

    let mut name = world.resource::<LayoutsDialog>().name_input.clone();
    let names = list_layouts();
    let mut keep_open = true;
    let mut do_save = false;
    let mut load: Option<String> = None;
    let mut delete: Option<String> = None;

    egui::Window::new("Layouts")
        .id(egui::Id::new("layouts_dialog"))
        .collapsible(false)
        .resizable(true)
        .default_size([320.0, 360.0])
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .open(&mut keep_open)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.add(
                    egui::TextEdit::singleline(&mut name)
                        .hint_text("layout name")
                        .desired_width(f32::INFINITY),
                );
            });
            if ui
                .add_enabled(
                    !name.trim().is_empty(),
                    egui::Button::new("Save current layout"),
                )
                .clicked()
            {
                do_save = true;
            }
            ui.separator();
            ui.label("Saved layouts:");
            egui::ScrollArea::vertical().show(ui, |ui| {
                if names.is_empty() {
                    ui.weak("(none yet)");
                }
                for n in &names {
                    ui.horizontal(|ui| {
                        if ui.button("Load").clicked() {
                            load = Some(n.clone());
                        }
                        if ui
                            .small_button(egui_phosphor::regular::TRASH)
                            .on_hover_text("Delete")
                            .clicked()
                        {
                            delete = Some(n.clone());
                        }
                        ui.label(n);
                    });
                }
            });
        });

    if do_save {
        let trimmed = name.trim().to_string();
        let saved = {
            let dock = world.resource::<EditorDockState>();
            save_named(&trimmed, dock)
        };
        if saved {
            world
                .resource_mut::<super::notifications::Notifications>()
                .success(format!("Saved layout \u{201C}{trimmed}\u{201D}"));
            name.clear();
        }
    }
    if let Some(n) = load
        && load_named(world, &n)
    {
        world
            .resource_mut::<super::notifications::Notifications>()
            .info(format!("Loaded layout \u{201C}{n}\u{201D}"));
    }
    if let Some(n) = delete {
        delete_named(&n);
    }

    let mut dialog = world.resource_mut::<LayoutsDialog>();
    dialog.name_input = name;
    if !keep_open {
        dialog.open = false;
    }
}
