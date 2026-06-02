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
use egui_dock::tab_viewer::OnCloseResponse;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};

use super::config::EditorConfig;
use super::panels::{DebugPanelRegistry, DockSide};
use super::scene_tabs::{self, INITIAL_SCENE_ID, OpenScenes, SceneId};

/// A tab in the editor dock. `Scene(id)` is a center 3D scene view (one per open scene,
/// named after the file); the rest are content panels. Built-in shell tabs have their own
/// variants; contributed debug/tool panels come through `Registered`.
///
/// Serializable so the dock layout can be saved/restored. Saved layouts collapse the scene
/// box to a single `NoScene` placeholder (scene ids are session-specific); applying a layout
/// re-injects the live scenes — see [`super::layout`].
#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EditorTab {
    /// Center 3D scene view for the open scene `id`. Only the active scene renders; the
    /// tab's on-screen rect is fed back to the SDF camera so the raymarch only fills the
    /// viewport region. Switching scene tabs swaps which scene is live in the world.
    Scene(SceneId),
    /// Empty-state placeholder shown in the center when every scene has been closed. Not
    /// closeable; replaced by a `Scene` tab as soon as one is opened.
    NoScene,
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
    /// Reused by `layout::restore_default`, hence `pub(crate)`.
    pub(crate) fn build(registry: &DebugPanelRegistry) -> Self {
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
        let mut state = DockState::new(vec![EditorTab::Scene(INITIAL_SCENE_ID)]);
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
            EditorTab::Scene(id) => self.world.resource::<OpenScenes>().tab_title(*id).into(),
            // Blank label: the empty-state center has no real tab, just room for the prompt.
            EditorTab::NoScene => "".into(),
            EditorTab::Hierarchy => "Scene".into(),
            EditorTab::Inspector => "Inspector".into(),
            EditorTab::ProjectFiles => "Project Files".into(),
            EditorTab::AssetsDrawer => "Assets".into(),
            EditorTab::Registered(id) => self.registry.title_for(id).into(),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        // Per-panel span so a chrome trace attributes the egui-pass cost panel-by-panel
        // (perf roadmap E4 — measure before cutting).
        let _span = match tab {
            EditorTab::Scene(_) => bevy::log::info_span!("panel_viewport"),
            EditorTab::NoScene => bevy::log::info_span!("panel_noscene"),
            EditorTab::Hierarchy => bevy::log::info_span!("panel_hierarchy"),
            EditorTab::Inspector => bevy::log::info_span!("panel_inspector"),
            EditorTab::ProjectFiles => bevy::log::info_span!("panel_project_files"),
            EditorTab::AssetsDrawer => bevy::log::info_span!("panel_assets"),
            EditorTab::Registered(_) => bevy::log::info_span!("panel_registered"),
        }
        .entered();
        match tab {
            EditorTab::Scene(id) => {
                // Only the active scene tab's `ui()` runs (egui_dock renders one tab per
                // leaf), so recording it here tells the swap logic which scene is visible.
                self.world.resource_mut::<OpenScenes>().rendered = Some(*id);
                // Toolbar strip across the top of the viewport tab. The remaining area
                // below it is what the 3D camera renders into.
                super::viewport_toolbar::viewport_toolbar(self.world, ui);
                // Capture the region the 3D camera should render into. The SDF pass
                // fills this rect; everything else here is just reserved space.
                let rect = ui.clip_rect();
                *self.viewport_rect = rect;
                viewport_material_drop(self.world, ui, rect);
            }
            EditorTab::NoScene => {
                // Empty-state center: no live viewport here, just the centered prompt. Clear
                // the viewport rect so pointer-in-viewport (camera input) reads false.
                *self.viewport_rect = egui::Rect::NOTHING;
                empty_state_message(ui);
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
                super::assets_browser::assets_browser_ui(self.world, ui);
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
        // The empty-state placeholder has no close button; everything else does.
        !matches!(tab, EditorTab::NoScene)
    }

    fn on_close(&mut self, tab: &mut Self::Tab) -> OnCloseResponse {
        match tab {
            // Route scene-tab closes through the document manager (it handles the
            // unsaved-changes prompt and the last-scene → empty-state transition); ignore
            // the close here, the manager removes the tab once confirmed.
            EditorTab::Scene(id) => {
                self.world.resource_mut::<OpenScenes>().close_request = Some(*id);
                OnCloseResponse::Ignore
            }
            _ => OnCloseResponse::Close,
        }
    }

    fn clear_background(&self, tab: &Self::Tab) -> bool {
        // Don't paint over the active scene's region — the 3D camera renders there. The
        // placeholder IS painted (the world is empty, so there's nothing to see through).
        !matches!(tab, EditorTab::Scene(_))
    }
}

/// Centered "no scene open" prompt, shown in the empty state (no tabs, no buttons).
fn empty_state_message(ui: &mut egui::Ui) {
    ui.centered_and_justified(|ui| {
        ui.label(
            egui::RichText::new("No scene open\nUse File \u{25B8} Open to load a scene")
                .weak()
                .size(16.0),
        );
    });
}

/// Accept a material dropped onto the 3D viewport: ray-pick the SDF volume under the cursor
/// and set its material. Mirrors the inspector drop, but resolves the target entity via CPU
/// picking instead of an explicit selection.
fn viewport_material_drop(world: &mut World, ui: &mut egui::Ui, rect: egui::Rect) {
    use crate::editor::assets_browser::MaterialDrag;

    let resp = ui.interact(
        rect,
        ui.id().with("viewport_material_drop"),
        egui::Sense::hover(),
    );

    // Highlight the viewport while a material drag hovers it.
    if egui::DragAndDrop::payload::<MaterialDrag>(ui.ctx()).is_some() && resp.contains_pointer() {
        ui.painter().rect_stroke(
            rect.shrink(1.0),
            0.0,
            ui.visuals().selection.stroke,
            egui::StrokeKind::Inside,
        );
    }

    if let Some(drag) = resp.dnd_release_payload::<MaterialDrag>() {
        // Cursor in window-logical points — the SDF camera is full-window, so this matches
        // what `sdf_picking` reads from `window.cursor_position()`.
        if let Some(p) = ui.ctx().pointer_interact_pos() {
            let cursor = bevy::math::Vec2::new(p.x, p.y);
            if let Some(entity) = crate::sdf_render::pick_sdf_volume(world, cursor) {
                crate::sdf_render::debug::set_entity_material(world, entity, &drag.0);
            }
        }
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
    // Restore the auto-persisted layout from the last session, if any (keeps the live scenes
    // in the center; only the panel arrangement is restored).
    super::layout::load_current_layout(world);
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

    // Section spans so a chrome trace breaks the editor egui pass into its parts.
    bevy::log::info_span!("editor_menu_bar").in_scope(|| super::menu_bar::menu_bar_ui(world, &ctx));
    super::scene_browser::open_scene_dialog_ui(world, &ctx);
    super::scene_browser::save_scene_dialog_ui(world, &ctx);
    bevy::log::info_span!("editor_status_bar")
        .in_scope(|| super::status_bar::status_bar_ui(world, &ctx));
    super::layout::layouts_ui(world, &ctx);
    super::notifications::notifications_ui(world, &ctx);

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

    // Multi-scene tab orchestration. Drain File-menu requests (which may add/remove scene
    // tabs) and refresh the active scene's dirty flag BEFORE rendering, so titles and the
    // tab set are current this frame.
    let type_registry = world.resource::<AppTypeRegistry>().clone();
    scene_tabs::drain_requests(world, &mut dock, &type_registry);
    // Throttled, but on the frame it runs this serializes the whole scene — span it.
    bevy::log::info_span!("editor_dirty_check")
        .in_scope(|| scene_tabs::refresh_active_dirty(world, &type_registry));

    let mut viewport_rect = dock.viewport_rect;
    {
        let _dock_span = bevy::log::info_span!("editor_dockarea").entered();
        let mut viewer = EditorTabViewer {
            world,
            registry: &registry,
            viewport_rect: &mut viewport_rect,
        };
        DockArea::new(&mut dock.state)
            .style(Style::from_egui(ctx.style().as_ref()))
            .show(&ctx, &mut viewer);
    }

    // After render: swap to a scene tab the user clicked, and process any close request
    // (which may pop the unsaved-changes prompt).
    scene_tabs::handle_activation(world, &mut dock, &type_registry);
    scene_tabs::handle_close(world, &mut dock, &type_registry, &ctx);

    dock.viewport_rect = viewport_rect;
    // Allow viewport interaction only while the pointer is inside the viewport tab AND not
    // over a *floating* egui layer (a Window/popup such as the Create Node dialog). We test
    // the layer order rather than `wants_pointer_input()`: the dock panels live on the
    // Background layer, so `wants_pointer_input()` is true whenever the pointer hovers the
    // viewport tab with no button down — which dropped wheel-zoom events (orbit/pan use a
    // held button, so they were unaffected). A floating window is Order::Middle or above.
    let over_floating = ctx.pointer_latest_pos().is_some_and(|p| {
        ctx.layer_id_at(p)
            .is_some_and(|layer| layer.order > egui::Order::Background)
    });
    let in_viewport = ctx
        .pointer_latest_pos()
        .is_some_and(|p| viewport_rect.contains(p))
        && !over_floating;
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

/// Wires the dock shell: the egui input-absorption poke, the Phosphor icon font, the one-shot
/// dock-layout build, and the per-frame dock render. Was inline in `EditorPlugin::build`; added
/// LAST (after every plugin has registered its panels, which `init_dock_state` consumes).
pub struct DockPlugin;

impl Plugin for DockPlugin {
    fn build(&self, app: &mut App) {
        // Do NOT use egui's blanket input absorption: egui_dock's central Viewport tab is itself an
        // egui surface, so the absorber would clear mouse input even when the cursor is over the 3D
        // region — killing viewport interaction. Instead the SDF orbit/pick/gizmo systems gate on
        // `ViewportInputAllowed`, which `show_editor_dock` sets from the pointer-in-viewport test.
        app.world_mut()
            .resource_mut::<bevy_egui::EguiGlobalSettings>()
            .enable_absorb_bevy_input_system = false;

        // Install the Phosphor icon font once the egui context exists, build the dock layout after
        // Startup (so every plugin's panels are registered), then render the dock each frame.
        app.add_systems(
            PostStartup,
            install_phosphor_font.after(bevy_egui::EguiStartupSet::InitContexts),
        )
        .add_systems(PostStartup, init_dock_state)
        .add_systems(bevy_egui::EguiPrimaryContextPass, show_editor_dock);
    }
}

/// Merge the Phosphor icon font into the primary egui context's fonts, once at startup.
/// `add_to_fonts` inserts it into the Proportional family alongside egui's built-ins, so
/// icon glyphs (`egui_phosphor::regular::*`) render inline with normal toolbar text.
fn install_phosphor_font(world: &mut World) {
    let Ok(mut egui_ctx) = world
        .query_filtered::<&mut bevy_egui::EguiContext, With<bevy_egui::PrimaryEguiContext>>()
        .single_mut(world)
    else {
        return;
    };
    let ctx = egui_ctx.get_mut();
    let mut fonts = bevy_egui::egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}
