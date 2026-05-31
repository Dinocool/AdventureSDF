use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy::remote::RemotePlugin;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::WindowResolution;
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_rapier3d::prelude::*;

/// Each editor run writes a `trace-<timestamp>.json` (bevy/trace_chrome) into the CWD, and
/// these grow to tens of GB apiece. Bevy's LogPlugin has no retention hook, so prune here —
/// BEFORE DefaultPlugins creates this run's file — keeping the 2 newest so that, once the new
/// trace starts, at most 3 exist. Sorted by name: the timestamp suffix is monotone, so
/// lexical order == chronological order.
#[cfg(feature = "editor")]
fn prune_old_traces(keep: usize) {
    let mut traces: Vec<std::path::PathBuf> = match std::fs::read_dir(".") {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("trace-") && n.ends_with(".json"))
            })
            .collect(),
        Err(_) => return,
    };
    if traces.len() <= keep {
        return;
    }
    traces.sort();
    for old in &traces[..traces.len() - keep] {
        let _ = std::fs::remove_file(old);
    }
}

fn main() {
    #[cfg(feature = "editor")]
    prune_old_traces(2);

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
    .add_plugins(adventure::node::NodePlugin)
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
