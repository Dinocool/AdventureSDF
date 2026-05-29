//! soul-engine editor harness. A jackdaw-modelled `egui_dock` shell (menu bar,
//! status bar, hierarchy, inspector, dockable/tabbed panels) hosting the SDF
//! content tools. Feature-gated behind `editor`.

use bevy::prelude::*;

pub mod config;
pub mod dock;
pub mod hierarchy;
pub mod hot_reload;
pub mod inspector;
pub mod keybinds;
pub mod menu_bar;
pub mod panels;
pub mod profiling;
pub mod project_files;
pub mod registry;
pub mod resource_inspector;
pub mod status_bar;
pub mod uniform_inspector;

use panels::DockSide;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            bevy_egui::EguiPlugin::default(),
            bevy_inspector_egui::DefaultInspectorConfigPlugin,
            config::EditorConfigPlugin,
            panels::DebugPanelRegistryPlugin,
            registry::ShaderDebugRegistryPlugin,
            uniform_inspector::UniformInspectorPlugin,
            profiling::ProfilingPlugin,
            hot_reload::HotReloadPlugin,
        ))
        .init_resource::<menu_bar::EditorRequests>()
        .init_resource::<menu_bar::CurrentScenePath>()
        .init_resource::<inspector::InspectorOverrides>();

        // Do NOT use egui's blanket input absorption: egui_dock's central Viewport
        // tab is itself an egui surface, so the absorber would clear mouse input even
        // when the cursor is over the 3D region — killing all viewport interaction.
        // Instead we gate the SDF orbit/pick/gizmo systems on `ViewportInputAllowed`,
        // which the dock sets from the pointer-in-viewport test each frame (see
        // `dock::show_editor_dock`). egui widgets still receive their own events.
        app.world_mut()
            .resource_mut::<bevy_egui::EguiGlobalSettings>()
            .enable_absorb_bevy_input_system = false;

        // Framework-level panels (registered into the dock by stable id).
        panels::register_panel(
            app,
            "core/profiling",
            "Perf",
            DockSide::Bottom,
            0,
            profiling::profiling_ui,
        );
        panels::register_panel(
            app,
            "core/hot_reload",
            "Hot Reload",
            DockSide::Bottom,
            10,
            hot_reload::hot_reload_ui,
        );

        keybinds::plugin(app);

        // A small viewport-ops panel so gizmo mode + snapping are visible/settable
        // in the UI, not only via keybinds.
        panels::register_panel(
            app,
            "core/viewport_ops",
            "Viewport",
            DockSide::Left,
            1,
            viewport_ops_ui,
        );

        // Resource Inspector (Godot-style): edit material resources + browse textures.
        app.init_resource::<resource_inspector::ResourceInspectorState>();
        panels::register_panel(
            app,
            "core/resources",
            "Resources",
            DockSide::Left,
            2,
            resource_inspector::resource_inspector_ui,
        );

        // Build the dock layout once, after `Startup` (so every plugin — including
        // the SDF debug plugin — has registered its panels), then render each frame.
        app.add_systems(PostStartup, dock::init_dock_state)
            .add_systems(bevy_egui::EguiPrimaryContextPass, dock::show_editor_dock);
    }
}

/// Panel: gizmo mode buttons + snap settings, bound to the transform-gizmo
/// [`GizmoOptions`].
fn viewport_ops_ui(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use transform_gizmo_bevy::{GizmoMode, GizmoOptions};

    let mut options = world.resource_mut::<GizmoOptions>();
    ui.label("Gizmo mode");
    ui.horizontal(|ui| {
        for (modes, label) in [
            (GizmoMode::all_translate(), "Move (W)"),
            (GizmoMode::all_rotate(), "Rotate (E)"),
            (GizmoMode::all_scale(), "Scale (R)"),
            (GizmoMode::all(), "All (Q)"),
        ] {
            if ui
                .selectable_label(options.gizmo_modes == modes, label)
                .clicked()
            {
                options.gizmo_modes = modes;
            }
        }
    });
    ui.separator();
    ui.checkbox(&mut options.snapping, "Snap (hold Ctrl)");
    ui.add(bevy_egui::egui::Slider::new(&mut options.snap_distance, 0.0..=2.0).text("Move step"));
    ui.add(
        bevy_egui::egui::Slider::new(&mut options.snap_angle, 0.0..=std::f32::consts::FRAC_PI_2)
            .text("Rotate step (rad)"),
    );
    ui.add(bevy_egui::egui::Slider::new(&mut options.snap_scale, 0.0..=1.0).text("Scale step"));
}
