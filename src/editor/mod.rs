//! soul-engine editor harness — slimmed for the voxel-RT rebuild.
//!
//! The original editor was a full SDF-content toolchain (scene tabs, hierarchy, inspector,
//! gizmo/picking, material editor, worldgen graph). All of that was coupled to the SDF render
//! path / mesh-bake / worldgen-editor modules that were PRUNED in the voxel-RT rebuild, so it's
//! been stripped to a minimal `egui_dock` shell hosting the panels that still compile cleanly:
//! a Performance/diagnostics panel, the chrome-trace capture toggle, and a status bar. The
//! pruned panel files are kept on disk (un-`mod`-declared) so they can be ported back as
//! voxel-specific tools land.
//!
//! `bevy-inspector-egui` was DROPPED (no Bevy-0.19 release) along with the inspector panel.
//!
//! Feature-gated behind `editor`.

use bevy::prelude::*;

pub mod chrome_trace;
pub mod config;
pub mod dock;
#[cfg(feature = "shader-debug")]
pub mod nsight_capture;
pub mod panels;
pub mod profiling;
pub mod registry;
#[cfg(feature = "renderdoc")]
pub mod renderdoc_capture;
pub mod status_bar;
pub mod viewport;

use panels::DockSide;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            // bevy_egui 0.40 (egui 0.34): `EguiPlugin::default()` (was a unit struct on 0.39).
            bevy_egui::EguiPlugin::default(),
            config::EditorConfigPlugin,
            panels::DebugPanelRegistryPlugin,
            registry::ShaderDebugRegistryPlugin,
            profiling::ProfilingPlugin,
            status_bar::StatusBarPlugin,
            // The 3D viewport: retargets the SdfCamera into an offscreen image shown in the Viewport tab.
            viewport::ViewportPlugin,
            // Added LAST (after every plugin has registered its panels, which the dock's
            // `init_dock_state` consumes at PostStartup).
            dock::DockPlugin,
        ));

        // F11 acknowledges an Nsight GPU-Trace capture; only meaningful in `shader-debug` builds.
        #[cfg(feature = "shader-debug")]
        app.add_plugins(nsight_capture::NsightCapturePlugin);

        // In-app RenderDoc capture (F7), `renderdoc` feature only.
        #[cfg(feature = "renderdoc")]
        app.add_plugins(renderdoc_capture::RenderDocCapturePlugin);

        // The dedicated Performance dock panel: a per-frame compute breakdown (stacked GPU + CPU
        // contributors), the chrome-trace capture toggle, and a system-memory breakdown.
        panels::register_panel(
            app,
            "core/performance",
            "Performance",
            DockSide::Bottom,
            0,
            profiling::performance_panel,
        );

        // F6 toggles chrome-trace capture (global).
        app.add_systems(Update, chrome_trace::toggle_on_f6);
    }
}
