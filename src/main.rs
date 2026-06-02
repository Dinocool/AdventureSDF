use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
#[cfg(feature = "editor")]
use bevy::log::LogPlugin;
use bevy::remote::RemotePlugin;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::WindowResolution;
use bevy_brp_extras::BrpExtrasPlugin;
use bevy_rapier3d::prelude::*;

/// Each editor run creates a `trace-<timestamp>.json` (our `editor::chrome_trace` layer) in
/// the CWD; a captured one can grow to tens of GB. Our chrome layer has no retention hook, so
/// prune here — BEFORE DefaultPlugins creates this run's file — keeping the 2 newest so that,
/// once the new trace starts, at most 3 exist. Sorted by name: the timestamp suffix is
/// monotone, so lexical order == chronological order.
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

/// Preload `renderdoc.dll` so RenderDoc's graphics hook installs BEFORE wgpu creates the
/// device inside `DefaultPlugins`. The `renderdoc` crate only searches `$PATH`, and the
/// installer doesn't put its dir there, so we `LoadLibrary` the dll from its standard
/// install location explicitly. Once loaded, `RenderDoc::new()` (in the editor's capture
/// plugin) finds the already-resident module and F7 can trigger captures with no external
/// launcher. Leaked on purpose: the hook must live for the whole process.
///
/// NOTE: incompatible with `--features fast` (Bevy `dynamic_linking`) — RenderDoc can't
/// hook a dynamically-linked Bevy, so capture with `--no-default-features --features editor`.
#[cfg(feature = "renderdoc")]
fn load_renderdoc() {
    if cfg!(feature = "fast") {
        warn!(
            "RenderDoc capture: `fast` (dynamic_linking) is on — captures will likely fail. \
             Run with `--no-default-features --features editor` to capture."
        );
    }
    // Standard Windows install path; the dll sits in the RenderDoc program dir.
    const CANDIDATES: [&str; 2] = [
        r"C:\Program Files\RenderDoc\renderdoc.dll",
        "renderdoc.dll", // fallback: PATH / CWD
    ];
    for path in CANDIDATES {
        // SAFETY: loading a known system DLL; we intentionally leak the handle so the
        // graphics hook persists for the lifetime of the process.
        match unsafe { libloading::Library::new(path) } {
            Ok(lib) => {
                std::mem::forget(lib);
                info!("RenderDoc capture: preloaded {path} (press F7 in editor to capture)");
                return;
            }
            Err(_) => continue,
        }
    }
    warn!(
        "RenderDoc capture: renderdoc.dll not found (looked in Program Files + PATH); \
         F7 capture disabled this run."
    );
}

/// wgpu device settings. BC7 texture compression (~1/6 the VRAM of RGBA8) + 16-bit-norm
/// texture formats — `TEXTURE_FORMAT_16BIT_NORM` is required for the R16Snorm / Rgba16Snorm
/// SDF distance atlases AND the 3D R16Snorm distance-clipmap volume, else those
/// `create_texture` calls fail validation. With `shader-debug`, also turn on wgpu
/// `InstanceFlags::DEBUG` so naga emits `OpLine`/`OpSource` and an Nsight GPU-Trace can
/// correlate sampled cost to WGSL source lines.
fn wgpu_settings() -> WgpuSettings {
    let settings = WgpuSettings {
        features: WgpuFeatures::TEXTURE_COMPRESSION_BC | WgpuFeatures::TEXTURE_FORMAT_16BIT_NORM,
        ..default()
    };
    #[cfg(feature = "shader-debug")]
    let settings = WgpuSettings {
        instance_flags: bevy::render::settings::InstanceFlags::DEBUG,
        ..settings
    };
    settings
}

fn main() {
    #[cfg(feature = "editor")]
    prune_old_traces(2);
    #[cfg(feature = "renderdoc")]
    load_renderdoc();

    let mut app = App::new();

    let default_plugins = DefaultPlugins
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
            render_creation: RenderCreation::Automatic(wgpu_settings()),
            ..default()
        });

    // Editor builds install our runtime-toggleable chrome-trace layer (off by default) via
    // LogPlugin's custom_layer hook. Non-editor builds leave LogPlugin at its default.
    #[cfg(feature = "editor")]
    let default_plugins = default_plugins.set(LogPlugin {
        custom_layer: adventure::editor::chrome_trace::custom_layer,
        ..default()
    });

    app.add_plugins(default_plugins)
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
    .add_plugins(adventure::gizmo_render::GizmoRenderPlugin)
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

    // Headless-capture aid: with `ADVENTURE_EXIT_AFTER_FRAMES=N` set, quit after N rendered
    // frames so a profiler wrapper (Nsight `ngfx`) gets a deterministic, self-terminating run.
    if let Some(limit) = std::env::var("ADVENTURE_EXIT_AFTER_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        app.add_systems(
            Update,
            move |mut count: Local<u64>, mut exit: MessageWriter<AppExit>| {
                *count += 1;
                if *count >= limit {
                    exit.write(AppExit::Success);
                }
            },
        );
    }

    app.run();
}
