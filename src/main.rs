use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy::remote::RemotePlugin;
use bevy::window::WindowResolution;
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_rapier3d::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Adventure".into(),
            resolution: WindowResolution::new(1920, 1080),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
    .add_plugins(RemotePlugin::default())
    .add_plugins(BrpExtrasPlugin)
    .add_plugins(WireframePlugin::default())
    .add_plugins(adventure::scene_manager::SceneManagerPlugin)
    .add_plugins(adventure::sdf_render::SdfScenePlugin)
    .add_plugins(adventure::sdf_render::render::SdfRenderPlugin)
    .add_plugins(adventure::camera::CameraPlugin)
    .add_plugins(adventure::player::PlayerPlugin)
    .add_plugins(adventure::world::WorldPlugin)
    .add_plugins(adventure::ui::UiPlugin)
    .add_plugins(adventure::combat::CombatPlugin)
    .add_plugins(adventure::inventory::InventoryPlugin)
    .add_plugins(adventure::networking::NetworkingPlugin)
    .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)));

    #[cfg(feature = "debug_toolkit")]
    {
        app.add_plugins(adventure::debug_toolkit::DebugToolkitPlugin);
    }

    app.run();
}
