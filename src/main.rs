use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
#[cfg(feature = "editor")]
use bevy::log::LogPlugin;
use bevy::render::RenderPlugin;
use bevy::render::render_resource::WgpuFeatures;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::WindowResolution;

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
/// `InstanceFlags::DEBUG` so naga emits `OpLine` (SPIR-V line-number debug info) and an Nsight
/// GPU-Trace can correlate sampled cost to WGSL line NUMBERS. (Bevy/wgpu do NOT emit `OpSource`,
/// so there's no inline source view — map the line numbers to the `.wgsl` by hand. See Cargo.toml.)
fn wgpu_settings() -> WgpuSettings {
    let mut settings = WgpuSettings {
        features: WgpuFeatures::TEXTURE_COMPRESSION_BC
            | WgpuFeatures::TEXTURE_FORMAT_16BIT_NORM
            // Bindless paged SDF atlas: the dist/mat atlases are a `binding_array` of page textures
            // (sdf_render::render::atlas_pages), indexed per-fragment by the brick's page — needs the
            // texture binding array + non-uniform indexing features. Universal on desktop Vulkan/DX12.
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            // Voxel-RT Stage 2: hardware-ray-traced voxel path needs AABB-BLAS `ray_query` (per-brick
            // procedural AABB + in-shader DDA). Bevy 0.19 already enables `ExperimentalFeatures` at device
            // creation, so this flag is all that's required. Universal on desktop Vulkan/DX12 with RT.
            | WgpuFeatures::EXPERIMENTAL_RAY_QUERY,
        ..default()
    };
    // Voxel-RT Phase 2.1: the world-space radiance-cache compute passes bind the scene storage buffers
    // (group 0: metas/voxels/palette = 3) PLUS the dedicated cache bind group (group 3: 11 storage buffers —
    // checksums/life/radiance/geometry/luminance/new_radiance/a/b/active_indices/count/dispatch) in one
    // pipeline layout = 14 storage buffers in a single shader stage. wgpu's default
    // `max_storage_buffers_per_shader_stage` is only 8, so raise it (desktop RTX/Vulkan GPUs support far more).
    // Storage plan R2b added a 4th scene storage buffer (group 0 binding 12, `brick_palettes`); A3 added a 5th
    // (group 0 binding 13, the instance `descriptors`). So the `world_cache_update` pipeline binds 5 scene + 11
    // cache + 2 NEE = 18 storage buffers in one stage — raise the floor to 22 for headroom.
    //
    // G-c.4 paged residency FRONT END: its comprehensive residency bind-group layout
    // (`residency_front_end.rs`) binds 48 storage buffers in ONE compute stage (Pass A–D + the slab allocators +
    // the GPU DenseSlot table + the enter-cap histogram/cut). The live paged drive creates this layout on the
    // render device, so the floor must clear 48 (the headless residency gates request exactly 48 too).
    settings.limits.max_storage_buffers_per_shader_stage =
        settings.limits.max_storage_buffers_per_shader_stage.max(48);
    // The ReSTIR/probe pipeline binds 5 groups (scene/view/reservoir/cache + group 4 screen-probe data); wgpu's
    // default `max_bind_groups` is 4. Desktop GPUs support 8.
    settings.limits.max_bind_groups = settings.limits.max_bind_groups.max(5);
    // The cache decay/compaction passes use `@workgroup_size(1024)` (the prefix-sum scan width). wgpu's
    // default caps invocations-per-workgroup + workgroup_size_x at 256, so raise both to 1024.
    settings.limits.max_compute_invocations_per_workgroup =
        settings.limits.max_compute_invocations_per_workgroup.max(1024);
    settings.limits.max_compute_workgroup_size_x =
        settings.limits.max_compute_workgroup_size_x.max(1024);
    // Editor builds enable GPU timestamp + pipeline-statistics queries so `RenderDiagnosticsPlugin`
    // can measure per-pass GPU time (the Performance panel's "SDF GPU passes" table + the chrome
    // trace). Desktop Vulkan/DX12 support these universally; device init fails loudly otherwise —
    // trim a flag here if a dev GPU lacks one. Off in release/CI builds, so zero runtime overhead.
    #[cfg(feature = "editor")]
    let settings = WgpuSettings {
        features: settings.features
            | WgpuFeatures::TIMESTAMP_QUERY
            | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_PASSES
            | WgpuFeatures::PIPELINE_STATISTICS_QUERY,
        ..settings
    };
    #[cfg(feature = "shader-debug")]
    let settings = WgpuSettings {
        instance_flags: bevy::render::settings::InstanceFlags::DEBUG,
        ..settings
    };
    // DLSS (via `dlss_wgpu`) only works on the Vulkan backend, so pin to Vulkan when built with
    // `--features dlss`. The HW-RT voxel path (AABB-BLAS `ray_query`) is fully supported on Vulkan, so
    // this is safe — it just drops the DX12 fallback that DLSS can't use anyway. We also RAISE
    // `max_storage_textures_per_shader_stage` (wgpu's default is 4): the DLSS-RR `raymarch_dlss` compute
    // writes 6 storage textures in one stage (colour + diffuse/specular albedo + normal/roughness + depth +
    // motion). RTX GPUs support far more; bump to 8.
    #[cfg(feature = "dlss")]
    let settings = WgpuSettings {
        backends: Some(bevy::render::settings::Backends::VULKAN),
        limits: {
            let mut l = settings.limits.clone();
            l.max_storage_textures_per_shader_stage = l.max_storage_textures_per_shader_stage.max(8);
            l
        },
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

    // DLSS Ray Reconstruction (the HW-RT voxel denoiser/upscaler) needs a project UUID registered BEFORE
    // the render device is created (inside DefaultPlugins, where `bevy/dlss` registers `DlssInitPlugin`).
    // Feature-gated — only present when built with `--features dlss` (also needs the NVIDIA DLSS SDK +
    // `DLSS_SDK` env var at build time). A fresh UUID for this project.
    #[cfg(feature = "dlss")]
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
        bevy::asset::uuid::uuid!("b4f1d2c8-3a7e-4d92-9f60-7c5e1a8b3d04"),
    ));

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
            // 0.19: `RenderCreation::Automatic` now takes a `Box<WgpuSettings>`.
            render_creation: RenderCreation::Automatic(Box::new(wgpu_settings())),
            ..default()
        });

    // Editor builds install our runtime-toggleable chrome-trace layer (off by default) via
    // LogPlugin's custom_layer hook. Non-editor builds leave LogPlugin at its default.
    #[cfg(feature = "editor")]
    let default_plugins = default_plugins.set(LogPlugin {
        custom_layer: adventure::editor::chrome_trace::custom_layer,
        ..default()
    });

    // Voxel-RT rebuild: the SDF GPU render path, mesh-bake terrain renderer, SDF editor, and the
    // legacy WoW gameplay modules are being replaced — only the reusable core (scene infra,
    // soul-scene format, node tree, gizmo overlay, free-fly camera + worldgen) is registered here.
    app.add_plugins(default_plugins)
    .add_plugins(WireframePlugin::default())
    .add_plugins(adventure::node::NodePlugin)
    .add_plugins(adventure::scene_manager::SceneManagerPlugin)
    .add_plugins(adventure::soul_scene::SoulScenePlugin)
    .add_plugins(adventure::assets::AssetsPlugin)
    .add_plugins(adventure::sdf_render::SdfScenePlugin)
    // GizmoRenderPlugin dropped: it's the editor gizmo-overlay renderer (its consumers — SDF
    // overlays, node gizmos — were pruned), and its GizmoPipeline::from_world panics requesting
    // the bevy_pbr `MeshPipeline` resource under 0.19's render init order. Re-add with a fix if
    // the voxel editor needs filled gizmos later.
    .add_plugins(adventure::camera::CameraPlugin)
    // Voxel-RT Stage 1: voxelize the worldgen surface into 0.2 m cubes and render a small origin patch.
    .add_plugins(adventure::voxel::VoxelPlugin)
    // Voxel-RT Stage 2: hardware-ray-traced voxel path (per-brick AABB BLAS + in-shader DDA). Additive +
    // toggleable — press R to switch the cubes ↔ the HW-RT view (default OFF, so cubes show first).
    .add_plugins(adventure::voxel::raytrace::VoxelRtPlugin)
    .insert_resource(ClearColor(Color::srgb(0.1, 0.1, 0.15)));

    #[cfg(feature = "editor")]
    {
        // FPS / frame-time diagnostics → DiagnosticsStore, read by the status-bar FPS counter +
        // Performance panel. (Previously supplied transitively by the now-removed BRP plugin; not part
        // of DefaultPlugins, so add it explicitly or the FPS counter reads nothing.)
        app.add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default());
        // Per-pass GPU timing (timestamp queries) → DiagnosticsStore, surfaced in the Performance
        // panel and the chrome trace. Pairs with the timestamp features requested in wgpu_settings.
        app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
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

    // Wall-clock self-exit: `ADVENTURE_EXIT_AFTER_SECS=N` quits N seconds after launch (cleaner than a
    // frame count when fps varies during streaming). Pairs with the trace rig below for a bounded run.
    if let Some(secs) = std::env::var("ADVENTURE_EXIT_AFTER_SECS")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        app.add_systems(Update, move |time: Res<Time>, mut exit: MessageWriter<AppExit>| {
            if time.elapsed_secs() >= secs {
                exit.write(AppExit::Success);
            }
        });
    }

    // Self-contained TRACE RIG: `ADVENTURE_TRACE_RIG=1` starts chrome capture at launch and boots the
    // GALLERY (the streamed merged corpus). Paired with `ADVENTURE_EXIT_AFTER_SECS=N` (clean self-exit ⇒
    // the FlushGuard drops ⇒ the trace flushes to trace-*.json), this gives a profiler-free, self-
    // terminating run that captures the live load + the steady-state HW-RT raymarch — no manual F6/quit.
    #[cfg(feature = "editor")]
    if std::env::var("ADVENTURE_TRACE_RIG").is_ok() {
        adventure::editor::chrome_trace::set_capturing(true);
        app.insert_resource(adventure::voxel::VoxelScene::Gallery);
    }

    // BISTRO FPS BENCH HARNESS (`ADVENTURE_BENCH_BISTRO=1`, editor-only, additive). Measures the steady-state
    // interior raymarch FPS — the gate for the "165 FPS" perf goal. Pairs with `ADVENTURE_EXIT_AFTER_SECS=N`
    // (clean self-exit) + `ADVENTURE_CAM="tx,ty,tz,dist,yaw,pitch"` (a fixed interior view). It (a) boots the
    // GALLERY scene — which, with the env set, loads Bistro ALONE at origin (see raytrace::stream_voxel_rt_
    // residency); (b) pins the camera; (c) averages the smoothed FPS over the last ~5 s before exit + logs a
    // `BENCH RESULT:` line; (d) saves one PNG of the final frame to D:/tmp_test/bistro_bench.png.
    #[cfg(feature = "editor")]
    if std::env::var("ADVENTURE_BENCH_BISTRO").is_ok() {
        adventure::bench::install_bistro_bench(&mut app);
    }

    app.run();
}
