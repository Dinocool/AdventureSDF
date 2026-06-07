//! soul-engine editor harness. A jackdaw-modelled `egui_dock` shell (menu bar,
//! status bar, hierarchy, inspector, dockable/tabbed panels) hosting the SDF
//! content tools. Feature-gated behind `editor`.

use bevy::prelude::*;

pub mod asset_inspector;
pub mod assets_browser;
pub mod chrome_trace;
pub mod config;
pub mod dock;
pub mod fs_util;
pub mod hierarchy;
pub mod history;
pub mod import_settings;
pub mod inspector;
pub mod keybinds;
pub mod layout;
pub mod material_editor;
pub mod material_preview;
pub mod menu_bar;
pub mod notifications;
#[cfg(feature = "shader-debug")]
pub mod nsight_capture;
pub mod panels;
pub mod profiling;
pub mod project_files;
pub mod registry;
#[cfg(feature = "renderdoc")]
pub mod renderdoc_capture;
pub mod resource_picker;
pub mod scene_browser;
pub mod scene_tabs;
pub mod selection;
pub mod status_bar;
pub mod transform_editor;
pub mod uniform_inspector;
pub mod viewport_toolbar;

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
            // Per-concern editor sub-plugins — each owns the registry/state it seeds (was inline
            // here). Added after `EguiPlugin` so `DockPlugin` can poke `EguiGlobalSettings`; the
            // material preview rig feeds the inspector's thumbnails.
            material_preview::MaterialPreviewPlugin,
            assets_browser::ThumbnailRegistryPlugin,
            asset_inspector::AssetInspectorPlugin,
            selection::SelectionPlugin,
            keybinds::KeybindsPlugin,
            status_bar::StatusBarPlugin,
            hierarchy::HierarchyPlugin,
            dock::DockPlugin,
        ))
        .add_plugins(history::EditHistoryPlugin)
        .init_resource::<menu_bar::EditorRequests>()
        .init_resource::<menu_bar::CurrentScenePath>()
        .init_resource::<scene_browser::OpenSceneDialog>()
        .init_resource::<scene_browser::SaveSceneDialog>()
        .init_resource::<scene_tabs::OpenScenes>()
        // Keep the active tab's unsaved-changes `*` marker live via Bevy change-detection —
        // never a periodic full-scene serialize (that stuttered ~380ms on large scenes).
        .add_systems(Update, scene_tabs::mark_scene_dirty)
        .init_resource::<notifications::Notifications>()
        .init_resource::<layout::LayoutsDialog>()
        .init_resource::<layout::PanelRestore>()
        .add_systems(Last, layout::save_layout_on_exit)
        .init_resource::<inspector::InspectorOverrides>()
        .register_type::<import_settings::ImageFilter>()
        .register_type::<import_settings::ColorSpace>()
        .register_type::<import_settings::WrapMode>()
        .register_type::<import_settings::TextureImportSettings>();

        // In-app RenderDoc capture (F7) only exists with the `renderdoc` feature.
        #[cfg(feature = "renderdoc")]
        app.add_plugins(renderdoc_capture::RenderDocCapturePlugin);

        // F11 acknowledges an Nsight GPU-Trace capture (the trace itself is armed by
        // `run-worktree.ps1`'s profiling launch via `--start-after-hotkey`). Only meaningful
        // in `shader-debug` builds, which is when the editor is run under Nsight.
        #[cfg(feature = "shader-debug")]
        app.add_plugins(nsight_capture::NsightCapturePlugin);

        // Custom euler-angle Transform editor (replaces the generic Quat-xyzw UI).
        inspector::register_component_editor::<Transform>(app, transform_editor::transform_editor);

        // Compact FPS / frame-time readout lives in the bottom status bar
        // (`status_bar::status_bar_ui`); the full Performance tab (readout + shared FPS /
        // frame-time graph) is a dedicated bottom dock panel.
        panels::register_panel(
            app,
            "core/performance",
            "Performance",
            DockSide::Bottom,
            0,
            profiling::performance_panel,
        );

        // F6 toggles chrome-trace capture (global; RenderDoc is F7 behind the `renderdoc` feature).
        app.add_systems(Update, chrome_trace::toggle_on_f6);
    }
}
