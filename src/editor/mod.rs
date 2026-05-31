//! soul-engine editor harness. A jackdaw-modelled `egui_dock` shell (menu bar,
//! status bar, hierarchy, inspector, dockable/tabbed panels) hosting the SDF
//! content tools. Feature-gated behind `editor`.

use bevy::prelude::*;

pub mod asset_inspector;
pub mod assets_browser;
pub mod config;
pub mod dock;
pub mod hierarchy;
pub mod import_settings;
pub mod inspector;
pub mod keybinds;
pub mod material_editor;
pub mod material_preview;
pub mod menu_bar;
pub mod panels;
pub mod profiling;
pub mod project_files;
pub mod registry;
pub mod resource_inspector;
pub mod resource_picker;
pub mod selection;
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
        ))
        .init_resource::<menu_bar::EditorRequests>()
        .init_resource::<menu_bar::CurrentScenePath>()
        .init_resource::<inspector::InspectorOverrides>();

        // Assets browser: navigation state + modular thumbnail providers + the
        // offscreen render rig that fills material/image thumbnails.
        app.add_plugins(assets_browser::ThumbnailRenderPlugin)
            .init_resource::<assets_browser::AssetsBrowserState>();
        {
            let mut registry = assets_browser::ThumbnailRegistry::default();
            registry.register(assets_browser::ImageThumbnailProvider);
            registry.register(assets_browser::MaterialThumbnailProvider);
            app.insert_resource(registry);
        }

        // Unified selection: the Inspector follows whichever of {entity, asset} was
        // selected last. `sync_selection` keeps it in step with the entity-side
        // `SdfSelection`.
        app.add_plugins(material_preview::MaterialPreviewPlugin)
            .init_resource::<selection::EditorSelection>()
            .init_resource::<asset_inspector::ImportSettingsEdits>()
            .add_systems(Update, selection::sync_selection);
        {
            let mut reg = asset_inspector::AssetInspectorRegistry::default();
            reg.register(asset_inspector::TextureAssetInspector);
            reg.register(asset_inspector::MaterialAssetInspector);
            reg.register(asset_inspector::PbrTextureAssetInspector);
            app.insert_resource(reg);
        }

        // Do NOT use egui's blanket input absorption: egui_dock's central Viewport
        // tab is itself an egui surface, so the absorber would clear mouse input even
        // when the cursor is over the 3D region — killing all viewport interaction.
        // Instead we gate the SDF orbit/pick/gizmo systems on `ViewportInputAllowed`,
        // which the dock sets from the pointer-in-viewport test each frame (see
        // `dock::show_editor_dock`). egui widgets still receive their own events.
        app.world_mut()
            .resource_mut::<bevy_egui::EguiGlobalSettings>()
            .enable_absorb_bevy_input_system = false;

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

        keybinds::plugin(app);

        // Gizmo transform tools (mode + snap) now live in the viewport toolbar
        // (see `dock::viewport_toolbar`), so there's no separate Transform panel.

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
        // Install the Phosphor icon font once, after the egui context exists, so the
        // toolbar can use icon glyphs (see `dock::viewport_toolbar`).
        app.add_systems(
            PostStartup,
            install_phosphor_font.after(bevy_egui::EguiStartupSet::InitContexts),
        )
        .add_systems(PostStartup, dock::init_dock_state)
        .add_systems(bevy_egui::EguiPrimaryContextPass, dock::show_editor_dock);
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
