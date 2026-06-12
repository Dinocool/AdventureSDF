//! Minimal egui_dock editor shell (voxel-RT rebuild).
//!
//! The old soul-engine dock hosted the SDF-editor toolchain — scene tabs, hierarchy,
//! inspector, gizmo/picking, material editor, worldgen graph — all of which were pruned in
//! the voxel-RT rebuild. This is a slimmed shell that keeps only what compiles cleanly against
//! the surviving crate: a dockable/tabbed `egui_dock` host that renders every panel contributed
//! via [`super::panels::register_panel`] (Performance, diagnostics, …) plus the status bar.
//!
//! Voxel-specific panels (a viewport, a voxel inspector, …) will be added back here later as the
//! voxel-RT engine grows; the registry extension API ([`register_panel`](super::panels::register_panel))
//! is preserved so they slot in without touching this host.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::egui::{LayerId, Ui, UiBuilder};
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};

use super::config::EditorConfig;
use super::panels::{DebugPanelRegistry, DockSide};

/// A tab in the editor dock. Built-in shell tabs have their own variants; contributed
/// debug/tool panels come through `Registered`.
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EditorTab {
    /// Center placeholder tab — the empty viewport region. Replaced by a real voxel viewport
    /// once that lands; for now it just shows a short hint.
    Viewport,
    /// A panel from [`DebugPanelRegistry`], keyed by its stable id.
    Registered(String),
}

/// Editor dock state.
#[derive(Resource)]
pub struct EditorDockState {
    pub state: DockState<EditorTab>,
}

impl EditorDockState {
    /// Build the initial layout from the registered panels: left-dock panels on the left,
    /// right-dock panels on the right, bottom-dock panels under the center, center-dock panels
    /// as tabs next to the viewport placeholder.
    pub(crate) fn build(registry: &DebugPanelRegistry) -> Self {
        let mut state = DockState::new(vec![EditorTab::Viewport]);
        let surface = state.main_surface_mut();
        let mut center = NodeIndex::root();

        let left_tabs: Vec<EditorTab> = registry
            .ids_for(DockSide::Left)
            .into_iter()
            .map(EditorTab::Registered)
            .collect();
        if !left_tabs.is_empty() {
            let [new_center, _left] = surface.split_left(center, 0.20, left_tabs);
            center = new_center;
        }

        let right_tabs: Vec<EditorTab> = registry
            .ids_for(DockSide::Right)
            .into_iter()
            .map(EditorTab::Registered)
            .collect();
        if !right_tabs.is_empty() {
            let [new_center, _right] = surface.split_right(center, 0.78, right_tabs);
            center = new_center;
        }

        let bottom_tabs: Vec<EditorTab> = registry
            .ids_for(DockSide::Bottom)
            .into_iter()
            .map(EditorTab::Registered)
            .collect();
        let viewport_leaf = if !bottom_tabs.is_empty() {
            let [viewport_leaf, _bottom] = surface.split_below(center, 0.72, bottom_tabs);
            viewport_leaf
        } else {
            center
        };

        // Center-dock panels become tabs in the viewport leaf, next to the placeholder.
        for id in registry.ids_for(DockSide::Center) {
            surface[viewport_leaf].append_tab(EditorTab::Registered(id));
        }

        Self { state }
    }
}

/// Bridges `egui_dock` tab rendering to the registry. Borrows `&mut World` so panel render
/// closures (which take `&mut World`) can run; the registry is taken out of the world for the
/// duration so the closures get exclusive access.
struct EditorTabViewer<'w> {
    world: &'w mut World,
    registry: &'w DebugPanelRegistry,
}

impl TabViewer for EditorTabViewer<'_> {
    type Tab = EditorTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            EditorTab::Viewport => "Viewport".into(),
            EditorTab::Registered(id) => self.registry.title_for(id).into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let _cpu = crate::instrument::span(match tab {
            EditorTab::Viewport => "ui: viewport",
            EditorTab::Registered(id) => {
                crate::instrument::intern(&format!("ui: {}", self.registry.title_for(id)))
            }
        });
        match tab {
            EditorTab::Viewport => {
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Voxel-RT viewport")
                            .weak()
                            .size(16.0),
                    );
                });
            }
            EditorTab::Registered(id) => {
                if let Some(render) = self.registry.panel_by_id(id).map(|p| &p.render) {
                    egui::ScrollArea::both().show(ui, |ui| {
                        render(self.world, ui);
                    });
                } else {
                    ui.weak(format!("(panel '{id}' not registered)"));
                }
            }
        }
    }

    fn closeable(&mut self, tab: &mut Self::Tab) -> bool {
        // The viewport placeholder stays; panels can be closed.
        !matches!(tab, EditorTab::Viewport)
    }

    fn clear_background(&self, tab: &Self::Tab) -> bool {
        // Paint the placeholder; a future real viewport would let the 3D camera show through.
        !matches!(tab, EditorTab::Viewport)
    }
}

/// Initialise the dock layout from the registered panels, once, after all plugins have
/// registered their panels.
pub fn init_dock_state(world: &mut World) {
    let registry = world
        .remove_resource::<DebugPanelRegistry>()
        .unwrap_or_default();
    let dock = EditorDockState::build(&registry);
    world.insert_resource(registry);
    world.insert_resource(dock);
}

/// Render the editor dock each frame (status bar + central DockArea).
pub fn show_editor_dock(world: &mut World) {
    if !world.resource::<EditorConfig>().enabled {
        return;
    }

    let Ok(egui_ctx) = world
        .query_filtered::<&mut bevy_egui::EguiContext, With<bevy_egui::PrimaryEguiContext>>()
        .single_mut(world)
    else {
        return;
    };
    let ctx = egui_ctx.into_inner().get_mut().clone();

    // bevy_egui 0.40 / egui 0.34 idiom: build ONE root viewport `Ui` covering the window on the
    // background layer, then draw every panel + the dock into it with `show_inside`. (The old
    // top-level `Panel::show(ctx, …)` / `DockArea::show(ctx, …)` entrypoints are deprecated.)
    let mut viewport_ui = Ui::new(
        ctx.clone(),
        "editor_viewport".into(),
        UiBuilder::new()
            .layer_id(LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    bevy::log::info_span!("editor_status_bar").in_scope(|| {
        let _cpu = crate::instrument::span("ui: status bar");
        super::status_bar::status_bar_ui(world, &mut viewport_ui)
    });

    // Take the registry and dock state out so the tab closures get exclusive `&mut World`.
    // Both are restored before returning.
    let registry = world
        .remove_resource::<DebugPanelRegistry>()
        .unwrap_or_default();
    let mut dock = match world.remove_resource::<EditorDockState>() {
        Some(d) => d,
        None => {
            world.insert_resource(registry);
            return;
        }
    };

    {
        let _dock_span = bevy::log::info_span!("editor_dockarea").entered();
        let mut viewer = EditorTabViewer {
            world,
            registry: &registry,
        };
        // egui 0.34: `Context::style()` → `global_style()`.
        DockArea::new(&mut dock.state)
            .style(Style::from_egui(ctx.global_style().as_ref()))
            .show_inside(&mut viewport_ui, &mut viewer);
    }

    world.insert_resource(dock);
    world.insert_resource(registry);
}

/// Wires the dock shell: disables egui's blanket input absorption, installs the Phosphor icon
/// font, builds the dock layout after Startup (so every plugin's panels are registered), and
/// renders the dock each frame.
pub struct DockPlugin;

impl Plugin for DockPlugin {
    fn build(&self, app: &mut App) {
        // Don't clear bevy input when egui has focus — the (future) viewport tab is itself an
        // egui surface, so the absorber would kill 3D interaction over it.
        app.world_mut()
            .resource_mut::<bevy_egui::EguiGlobalSettings>()
            .enable_absorb_bevy_input_system = false;

        // Install the Phosphor icon font into the primary egui context once it exists (only
        // resolves in PreUpdate's InitContexts of the first frame, after PostStartup), build the
        // dock layout after Startup, then render the dock each frame.
        app.add_systems(
            PreUpdate,
            install_phosphor_font
                .after(bevy_egui::EguiPreUpdateSet::InitContexts)
                .before(bevy_egui::EguiPreUpdateSet::BeginPass),
        )
        .add_systems(PostStartup, init_dock_state)
        .add_systems(bevy_egui::EguiPrimaryContextPass, show_editor_dock);
    }
}

/// Merge the Phosphor icon font into the primary egui context's fonts, once.
fn install_phosphor_font(
    mut contexts: Query<&mut bevy_egui::EguiContext, With<bevy_egui::PrimaryEguiContext>>,
    mut installed: Local<bool>,
) {
    if *installed {
        return;
    }
    let Ok(mut egui_ctx) = contexts.single_mut() else {
        return; // context not created yet — retry next frame
    };
    let ctx = egui_ctx.get_mut();
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
    *installed = true;
}
