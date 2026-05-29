//! egui_dock-driven editor layout (the soul-engine shell, modelled on the Bevy
//! Editor Figma / jackdaw `layout`). Replaces the old fixed-`SidePanel` dock.
//!
//! The host owns an [`EditorDockState`] (an `egui_dock::DockState<EditorTab>`).
//! Content panels are *not* hard-coded here: every panel contributed via
//! [`super::panels::register_panel`] becomes an [`EditorTab::Registered`] tab, so
//! the registry extension API is preserved unchanged — only the layout host moved
//! from collapsing-sections-in-SidePanel to dockable/tabbed `egui_dock`.

use bevy::prelude::*;
use bevy_egui::egui;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};

use super::config::EditorConfig;
use super::panels::{DebugPanelRegistry, DockSide};

/// A tab in the editor dock. `Viewport` is the center 3D region (the SDF view);
/// the rest are content panels. Built-in shell tabs have their own variants;
/// contributed debug/tool panels come through `Registered`.
#[derive(Clone, PartialEq, Eq)]
pub enum EditorTab {
    /// Center 3D scene view. Its on-screen rect is fed back to the SDF camera so
    /// the raymarch only fills the viewport region, not the whole window.
    Viewport,
    /// Scene-tree panel. Top-left.
    Hierarchy,
    /// Selected-entity component editor (Godot-style). Right.
    Inspector,
    /// Read-only `assets/` file tree. Bottom-left.
    ProjectFiles,
    /// Center-bottom drawer. A stub for now; gains tabs (assets, output, etc.) later.
    AssetsDrawer,
    /// A panel from [`DebugPanelRegistry`], keyed by its stable id.
    Registered(String),
}

/// Editor dock state + the viewport rect/pointer feedback the camera system reads.
#[derive(Resource)]
pub struct EditorDockState {
    pub state: DockState<EditorTab>,
    /// Center viewport rect in egui points, captured each frame. Consumed by
    /// `set_sdf_camera_viewport` to confine the 3D camera.
    pub viewport_rect: egui::Rect,
    /// True when the cursor is inside the viewport tab (gates picking so clicks on
    /// panels don't fall through to the scene).
    pub pointer_in_viewport: bool,
}

impl EditorDockState {
    /// Build the initial layout from the registered panels: Hierarchy + left-dock
    /// panels on the left, right-dock panels on the right, bottom-dock panels in a
    /// strip under the center viewport. Mirrors the Bevy Editor Figma arrangement.
    fn build(registry: &DebugPanelRegistry) -> Self {
        // Godot/jackdaw arrangement. Root holds the single Viewport tab. Each split
        // returns [old_node, new_node]; we thread those indices so the viewport stays
        // one center tab (splitting `root()` repeatedly caused the "two viewports" bug).
        //
        //   ┌──────────┬───────────────────┬───────────┐
        //   │ Hierarchy│      Viewport     │ Inspector │
        //   │          ├───────────────────┤  + right  │
        //   │  Project │   Assets drawer   │   panels  │
        //   │   Files  │  + bottom panels  │           │
        //   └──────────┴───────────────────┴───────────┘
        // egui_dock 0.18 semantics: `split_X(parent, fraction, tabs)` places the NEW
        // `tabs` node on the X side at `fraction` of the parent's size, and returns
        // `[old, new]` where `old` is the inherited content (the viewport). So a 20%
        // left panel = `split_left(.., 0.20, ..)`, and we thread `old` (the viewport)
        // forward as `center`.
        let mut state = DockState::new(vec![EditorTab::Viewport]);
        let surface = state.main_surface_mut();
        let mut center = NodeIndex::root();

        // Left column: Hierarchy (+ Left-dock registered panels), 20% of the window.
        let mut left_tabs = vec![EditorTab::Hierarchy];
        left_tabs.extend(
            registry
                .ids_for(DockSide::Left)
                .into_iter()
                .map(EditorTab::Registered),
        );
        let [new_center, left] = surface.split_left(center, 0.20, left_tabs);
        center = new_center;
        // Stack Project Files under Hierarchy in the SAME left column (Hierarchy
        // keeps the top 60%, Project Files the bottom 40%).
        surface.split_below(left, 0.60, vec![EditorTab::ProjectFiles]);

        // Right column: Inspector (+ Right-dock registered panels), ~22% of the window.
        let mut right_tabs = vec![EditorTab::Inspector];
        right_tabs.extend(
            registry
                .ids_for(DockSide::Right)
                .into_iter()
                .map(EditorTab::Registered),
        );
        let [new_center, _right] = surface.split_right(center, 0.78, right_tabs);
        center = new_center;

        // Center-bottom: Assets drawer (+ Bottom-dock panels); viewport keeps the top 72%.
        let mut bottom_tabs = vec![EditorTab::AssetsDrawer];
        bottom_tabs.extend(
            registry
                .ids_for(DockSide::Bottom)
                .into_iter()
                .map(EditorTab::Registered),
        );
        surface.split_below(center, 0.72, bottom_tabs);

        Self {
            state,
            viewport_rect: egui::Rect::NOTHING,
            pointer_in_viewport: false,
        }
    }
}

/// Bridges `egui_dock` tab rendering to the registry. Borrows `&mut World` so panel
/// render closures (which take `&mut World`) can run; the registry is taken out of
/// the world for the duration so the closures get exclusive access.
struct EditorTabViewer<'w> {
    world: &'w mut World,
    registry: &'w DebugPanelRegistry,
    viewport_rect: &'w mut egui::Rect,
}

impl TabViewer for EditorTabViewer<'_> {
    type Tab = EditorTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab {
            EditorTab::Viewport => "Viewport".into(),
            EditorTab::Hierarchy => "Hierarchy".into(),
            EditorTab::Inspector => "Inspector".into(),
            EditorTab::ProjectFiles => "Project Files".into(),
            EditorTab::AssetsDrawer => "Assets".into(),
            EditorTab::Registered(id) => self.registry.title_for(id).into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            EditorTab::Viewport => {
                // Capture the region the 3D camera should render into. The SDF pass
                // fills this rect; everything else here is just reserved space.
                *self.viewport_rect = ui.clip_rect();
            }
            EditorTab::Hierarchy => {
                super::hierarchy::hierarchy_ui(self.world, ui);
            }
            EditorTab::Inspector => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    super::inspector::inspector_ui(self.world, ui);
                });
            }
            EditorTab::ProjectFiles => {
                super::project_files::project_files_ui(self.world, ui);
            }
            EditorTab::AssetsDrawer => {
                // Stub: gains tabs (asset browser, output log, etc.) in a later pass.
                ui.weak("Assets — coming soon.");
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

    fn clear_background(&self, tab: &Self::Tab) -> bool {
        // Don't paint over the viewport — the 3D camera renders there.
        !matches!(tab, EditorTab::Viewport)
    }
}

/// Initialise the dock layout from the registered panels, once, after all plugins
/// have registered their panels.
pub fn init_dock_state(world: &mut World) {
    let registry = world
        .remove_resource::<DebugPanelRegistry>()
        .unwrap_or_default();
    let dock = EditorDockState::build(&registry);
    world.insert_resource(registry);
    world.insert_resource(dock);
}

/// Render the editor dock each frame (menu bar + status bar + central DockArea).
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

    super::menu_bar::menu_bar_ui(world, &ctx);
    super::status_bar::status_bar_ui(world, &ctx);

    // Take the registry and dock state out so the tab closures get exclusive
    // `&mut World`. Both are restored before returning.
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

    let mut viewport_rect = dock.viewport_rect;
    {
        let mut viewer = EditorTabViewer {
            world,
            registry: &registry,
            viewport_rect: &mut viewport_rect,
        };
        DockArea::new(&mut dock.state)
            .style(Style::from_egui(ctx.style().as_ref()))
            .show(&ctx, &mut viewer);
    }

    dock.viewport_rect = viewport_rect;
    // Allow viewport interaction only while the pointer is inside the viewport tab
    // (and not dragging a dock divider). Gates the SDF orbit/pick systems so clicks
    // on panels don't fall through to the 3D scene.
    let in_viewport = ctx
        .pointer_latest_pos()
        .is_some_and(|p| viewport_rect.contains(p));
    dock.pointer_in_viewport = in_viewport;
    world
        .resource_mut::<crate::sdf_render::ViewportInputAllowed>()
        .0 = in_viewport;

    // NOTE: the SDF camera is left full-window (its `viewport` is NOT confined to the
    // dock rect). bevy_egui auto-attaches the PrimaryEguiContext to that same camera,
    // so shrinking its viewport collapses egui's layout area to a degenerate rect and
    // egui_dock panics with a NaN separator. With a full-window camera + the gizmo's
    // `viewport_rect = None`, the gizmo maps the cursor in full-window coords too, so
    // handles stay aligned with where they're drawn. (A future dedicated full-window
    // UI camera would let us confine the SDF camera to the center region.)

    world.insert_resource(dock);
    world.insert_resource(registry);
}
