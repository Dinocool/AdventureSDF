use bevy::prelude::*;

pub mod config;
pub mod hot_reload;
pub mod panels;
pub mod profiling;
pub mod registry;
pub mod uniform_inspector;

use panels::{DebugPanelRegistry, DockSide};

pub struct DebugToolkitPlugin;

impl Plugin for DebugToolkitPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            bevy_egui::EguiPlugin::default(),
            bevy_inspector_egui::DefaultInspectorConfigPlugin,
            config::DebugToolkitConfigPlugin,
            panels::DebugPanelRegistryPlugin,
            registry::ShaderDebugRegistryPlugin,
            uniform_inspector::UniformInspectorPlugin,
            profiling::ProfilingPlugin,
            hot_reload::HotReloadPlugin,
        ));

        // Let egui absorb pointer/keyboard input upstream: when the cursor is over
        // a panel (or a widget has focus), `absorb_bevy_input_system` (PreUpdate)
        // resets `ButtonInput<MouseButton>` and clears the mouse messages BEFORE
        // our `Update` viewport systems read them. So clicks/scrolls on the debug
        // UI never reach picking/orbit/gizmo — no per-system guards needed. This is
        // the same interception layer the game UI will rely on. Off by default.
        app.world_mut()
            .resource_mut::<bevy_egui::EguiGlobalSettings>()
            .enable_absorb_bevy_input_system = true;

        // Framework-level panels.
        panels::register_panel(
            app,
            "core/world_inspector",
            "World Inspector",
            DockSide::Right,
            0,
            |world, ui| {
                bevy_inspector_egui::bevy_inspector::ui_for_world(world, ui);
            },
        );
        panels::register_panel(
            app,
            "core/uniforms",
            "Shader Uniforms",
            DockSide::Right,
            10,
            |world, ui| uniform_inspector::uniforms_ui(world, ui),
        );
        panels::register_panel(
            app,
            "core/profiling",
            "Perf",
            DockSide::Bottom,
            0,
            |world, ui| profiling::profiling_ui(world, ui),
        );
        panels::register_panel(
            app,
            "core/hot_reload",
            "Hot Reload",
            DockSide::Bottom,
            10,
            |world, ui| hot_reload::hot_reload_ui(world, ui),
        );

        app.add_systems(bevy_egui::EguiPrimaryContextPass, dock_layout);
    }
}

fn dock_layout(world: &mut World) {
    if !world.resource::<config::DebugToolkitConfig>().enabled {
        return;
    }

    let Ok(egui_ctx) = world
        .query_filtered::<&mut bevy_egui::EguiContext, With<bevy_egui::PrimaryEguiContext>>()
        .single_mut(world)
    else {
        return;
    };
    let ctx = egui_ctx.into_inner().get_mut().clone();

    // Take the registry out so panel closures get exclusive `&mut World`.
    let registry = world
        .remove_resource::<DebugPanelRegistry>()
        .unwrap_or_default();

    render_side(
        &ctx,
        world,
        &registry,
        DockSide::Left,
        "debug_toolkit_left",
        280.0,
    );
    render_side(
        &ctx,
        world,
        &registry,
        DockSide::Right,
        "debug_toolkit_right",
        320.0,
    );
    render_bottom(&ctx, world, &registry, "debug_toolkit_bottom", 220.0);

    world.insert_resource(registry);
}

fn render_side(
    ctx: &bevy_egui::egui::Context,
    world: &mut World,
    registry: &DebugPanelRegistry,
    side: DockSide,
    id: &str,
    width: f32,
) {
    let panels = registry.panels_for(side);
    if panels.is_empty() {
        return;
    }
    let builder = match side {
        DockSide::Left => bevy_egui::egui::SidePanel::left(id.to_string()),
        _ => bevy_egui::egui::SidePanel::right(id.to_string()),
    };
    builder
        .default_width(width)
        .resizable(true)
        .show(ctx, |ui| {
            bevy_egui::egui::ScrollArea::vertical().show(ui, |ui| {
                for panel in &panels {
                    ui.collapsing(panel.title.as_str(), |ui| {
                        (panel.render)(world, ui);
                    });
                }
            });
        });
}

fn render_bottom(
    ctx: &bevy_egui::egui::Context,
    world: &mut World,
    registry: &DebugPanelRegistry,
    id: &str,
    height: f32,
) {
    let panels = registry.panels_for(DockSide::Bottom);
    if panels.is_empty() {
        return;
    }
    bevy_egui::egui::TopBottomPanel::bottom(id.to_string())
        .default_height(height)
        .resizable(true)
        .show(ctx, |ui| {
            bevy_egui::egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal_top(|ui| {
                    for panel in &panels {
                        ui.vertical(|ui| {
                            ui.set_min_width(260.0);
                            ui.heading(panel.title.as_str());
                            (panel.render)(world, ui);
                        });
                        ui.separator();
                    }
                });
            });
        });
}
