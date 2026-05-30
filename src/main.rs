use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy::remote::RemotePlugin;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::WindowResolution;
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_rapier3d::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Adventure".into(),
                    resolution: WindowResolution::new(1920, 1080),
                    ..default()
                }),
                ..default()
            })
            // Enable BC texture compression so the SDF PBR atlases can upload as
            // BC7 (~⅙ the VRAM of RGBA8). Desktop Vulkan/DX12/Metal support BC
            // universally; device init fails loudly if a backend somehow lacks it.
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    // BC7 texture compression (~1/6 the VRAM of RGBA8) + 16-bit-norm texture
                    // formats. TEXTURE_FORMAT_16BIT_NORM is required for the R16Snorm /
                    // Rgba16Snorm SDF distance atlases AND the 3D R16Snorm distance-clipmap
                    // volume — without it those `create_texture` calls fail validation.
                    features: WgpuFeatures::TEXTURE_COMPRESSION_BC
                        | WgpuFeatures::TEXTURE_FORMAT_16BIT_NORM,
                    ..default()
                }),
                ..default()
            }),
    )
    .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
    .add_plugins(RemotePlugin::default())
    .add_plugins(BrpExtrasPlugin)
    .add_plugins(WireframePlugin::default())
    .add_plugins(adventure::scene_manager::SceneManagerPlugin)
    .add_plugins(adventure::soul_scene::SoulScenePlugin)
    .add_plugins(adventure::assets::AssetsPlugin)
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

    #[cfg(feature = "editor")]
    {
        app.add_plugins(adventure::editor::EditorPlugin);
    }

    app.run();
}
