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
            EditorTab::Hierarchy => "Scene".into(),
            EditorTab::Inspector => "Inspector".into(),
            EditorTab::ProjectFiles => "Project Files".into(),
            EditorTab::AssetsDrawer => "Assets".into(),
            EditorTab::Registered(id) => self.registry.title_for(id).into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            EditorTab::Viewport => {
                // Toolbar strip across the top of the viewport tab. The remaining area
                // below it is what the 3D camera renders into.
                viewport_toolbar(self.world, ui);
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

/// Toolbar strip rendered across the top of the Viewport tab: camera-mode toggle
/// (Orbit ⇄ FPS), gizmo transform tools (mode + snap), and a view-options dropdown.
/// Drawn with a `TopBottomPanel::top` scoped to the tab's `ui`, so the 3D camera's
/// reserved rect (captured after) sits below it.
fn viewport_toolbar(world: &mut World, ui: &mut egui::Ui) {
    use crate::sdf_render::gizmo::{GizmoModes, GizmoState};
    use crate::sdf_render::{SdfCameraMode, SdfOrbitCamera, WireframeBoundsVisible};

    egui::TopBottomPanel::top("viewport_toolbar")
        .exact_height(28.0)
        .show_inside(ui, |ui| {
            ui.horizontal_centered(|ui| {
                let fps = world.resource::<SdfCameraMode>().fps;

                // Orbit / FPS segmented toggle.
                if ui.selectable_label(!fps, "🛰 Orbit").clicked() && fps {
                    world.resource_mut::<SdfCameraMode>().fps = false;
                }
                if ui.selectable_label(fps, "🎮 FPS").clicked() && !fps {
                    // Seed the free-fly yaw/pitch from the orbit view so toggling in
                    // doesn't snap the camera to a new orientation.
                    // Orbit stores the target→camera offset (view looks along -dir),
                    // while FPS stores the look direction (+dir); they differ by π in
                    // yaw and a negated pitch.
                    let orbit = world.resource::<SdfOrbitCamera>();
                    let (yaw, pitch) = (orbit.yaw, orbit.pitch);
                    let mut mode = world.resource_mut::<SdfCameraMode>();
                    mode.fps = true;
                    mode.yaw = yaw + std::f32::consts::PI;
                    mode.pitch = -pitch;
                }

                ui.separator();

                // Camera-mode instructions, to the left of the gizmo tools.
                if fps {
                    let speed = world.resource::<SdfCameraMode>().speed;
                    ui.label(format!("Fly: RMB look · WASD · Space/Ctrl · {speed:.0} u/s"));
                } else {
                    ui.label("Orbit: MMB rotate · Shift+MMB pan · wheel zoom");
                }

                ui.separator();

                // Gizmo transform tools: mode icon buttons + snap magnet. Disabled in
                // FPS mode (the gizmo only operates under the orbit camera). Icons are
                // Phosphor glyphs (installed via `install_phosphor_font`).
                ui.add_enabled_ui(!fps, |ui| {
                    use egui_phosphor::regular as icon;
                    let cur = world.resource::<GizmoState>().modes;
                    // (mode, icon glyph, tooltip incl. keybind).
                    for (modes, glyph, tip) in [
                        (GizmoModes::TRANSLATE, icon::ARROWS_OUT_CARDINAL, "Move (W)"),
                        (GizmoModes::ROTATE, icon::ARROW_CLOCKWISE, "Rotate (E)"),
                        (GizmoModes::SCALE, icon::ARROWS_OUT, "Scale (R)"),
                        (GizmoModes::all(), icon::CUBE, "All modes (Q)"),
                    ] {
                        if ui
                            .selectable_label(cur == modes, glyph)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            world.resource_mut::<GizmoState>().modes = modes;
                        }
                    }

                    // Snap magnet: sticky toggle (Ctrl-hold still forces snap on too).
                    let sticky = world.resource::<GizmoState>().snap_sticky;
                    if ui
                        .selectable_label(sticky, icon::MAGNET)
                        .on_hover_text("Snap (toggle; or hold Ctrl)")
                        .clicked()
                    {
                        world.resource_mut::<GizmoState>().snap_sticky = !sticky;
                    }

                    // Snap settings dropdown: per-axis snap step sizes.
                    egui::ComboBox::from_id_salt("viewport_snap_settings")
                        .selected_text(format!("{} Snap settings", icon::GEAR))
                        .show_ui(ui, |ui| {
                            let mut g = world.resource_mut::<GizmoState>();
                            ui.add(egui::Slider::new(&mut g.snap_move, 0.0..=2.0).text("Move"));
                            ui.add(
                                egui::Slider::new(
                                    &mut g.snap_angle,
                                    0.0..=std::f32::consts::FRAC_PI_2,
                                )
                                .text("Rotate (rad)"),
                            );
                            ui.add(egui::Slider::new(&mut g.snap_scale, 0.0..=1.0).text("Scale"));
                        });
                });

                // Blender-style view-options dropdown, right-aligned. Display toggles.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::ComboBox::from_id_salt("viewport_view_options")
                        .selected_text("View")
                        .show_ui(ui, |ui| {
                            let mut wf = world.resource::<WireframeBoundsVisible>().0;
                            if ui.checkbox(&mut wf, "Bounds wireframe").changed() {
                                world.resource_mut::<WireframeBoundsVisible>().0 = wf;
                            }
                        });
                });
            });
        });
}
