//! **The on-screen correctness ORACLE for the HW-RT voxel composite.**
//!
//! `tests/voxel_raytrace_gpu.rs` proves the GPU `ray_query` DDA core is correct (GPU hit == CPU brickmap
//! ground truth) in ISOLATION — a single ray through a hand-built scene. It does NOT exercise the Bevy-0.19
//! render-system integration (the `Core3d` schedule wiring, the per-view camera ray basis, the composite
//! over the [`ViewTarget`]). That integration is exactly where the "black screen" bug lived.
//!
//! This rig closes that gap WITHOUT a GUI: it boots a HEADLESS Bevy `App` (no window, no winit) with the
//! real [`VoxelRtPlugin`], a `Camera` rendering to an offscreen [`Image`] render target framed onto the
//! voxel patch, runs enough frames for streaming + the BLAS/TLAS build + the render systems, reads the
//! rendered image back from the GPU, and ASSERTS it actually contains voxels (not a uniform clear colour,
//! and a meaningful fraction of clearly-terrain — i.e. non-sky — pixels). If the composite never ran, or ran
//! at the wrong point in the schedule (wiped by the opaque pass clear), or every ray missed, this fails.
//!
//! Skips cleanly (no failure) on a box without an `EXPERIMENTAL_RAY_QUERY` Vulkan adapter, mirroring the
//! other GPU rigs.

use std::sync::{Arc, Mutex};

use bevy::camera::RenderTarget;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{TextureFormat, TextureUsages, WgpuFeatures};
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;

use adventure::sdf_render::SdfCamera;
use adventure::sdf_render::worldgen::layers::erosion::ErosionParams;
use adventure::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use adventure::sdf_render::worldgen::{
    WORLDGEN_SLICE_SEED, WorldBiomeShapes, WorldGraph,
};
use adventure::voxel::VoxelScene;
use adventure::voxel::raytrace::{VoxelRtPatch, VoxelRtPlugin, VoxelRtToggle};
use adventure::voxel::streaming::StreamingConfig;

mod common;

/// Offscreen render-target dimensions. Small + square keeps the readback cheap and deterministic.
const W: u32 = 256;
const H: u32 = 256;

/// The CPU-side latest readback of the render target (raw `Rgba8UnormSrgb` bytes, row-padded by the GPU
/// copy). Filled by the `ReadbackComplete` observer in the render-driven app; read by the test thread after
/// the frames run.
#[derive(Resource, Clone)]
struct LatestFrame(Arc<Mutex<Option<Vec<u8>>>>);

/// wgpu settings enabling AABB-BLAS `ray_query` — the SAME feature the app's `main.rs` requests. Bevy 0.19
/// enables `ExperimentalFeatures` at device creation unconditionally, so this flag is all that's needed.
fn rt_wgpu_settings() -> WgpuSettings {
    WgpuSettings {
        features: WgpuFeatures::EXPERIMENTAL_RAY_QUERY,
        ..default()
    }
}

/// #134 fix — insert the DLSS project UUID the `dlss`-build `DlssInitPlugin` REQUIRES before `DefaultPlugins`.
/// On the default feature set (`dlss` on) `bevy/dlss` registers `DlssInitPlugin`, which panics at device-init
/// time unless a `DlssProjectId` resource is already present (mirrors `src/main.rs`). A no-op without `dlss`.
/// Must be called BEFORE `app.add_plugins(DefaultPlugins…)`.
#[cfg(feature = "dlss")]
fn insert_dlss_project_id(app: &mut App) {
    app.insert_resource(bevy::anti_alias::dlss::DlssProjectId(
        bevy::asset::uuid::uuid!("b4f1d2c8-3a7e-4d92-9f60-7c5e1a8b3d04"),
    ));
}
#[cfg(not(feature = "dlss"))]
fn insert_dlss_project_id(_app: &mut App) {}

/// #134 fix — force the NON-DLSS render path for the headless rigs (dlss build only). DLSS-RR resolves into a
/// swapchain-shaped output the offscreen `Readback` does not capture, so with RR attached the readback reads an
/// all-zero (black) frame and every render-correctness assert fails. Setting `DlssSettings.enabled = false`
/// makes `sync_dlss_camera` never attach the `Dlss` component → the temporal-accumulation fallback writes the
/// offscreen target the readback DOES capture (the SAME path the non-dlss build always uses, so the rig tests
/// the same composite either way). A no-op without `dlss`. Call AFTER `app.add_plugins(VoxelRtPlugin)` so it
/// overrides the plugin's `init_resource` default.
#[cfg(feature = "dlss")]
fn disable_dlss_for_headless(app: &mut App) {
    app.insert_resource(adventure::voxel::raytrace::DlssSettings {
        enabled: false,
        mode: bevy::anti_alias::dlss::DlssPerfQualityMode::Quality,
    });
}
#[cfg(not(feature = "dlss"))]
fn disable_dlss_for_headless(_app: &mut App) {}

/// Pump frames until the latest read-back frame is meaningfully LIT (a non-trivial centre mean luma — the
/// terrain actually rendered), capped at a generous budget, then return the latched bytes. The STREAMING
/// worldgen scene takes a variable number of frames to: stream + voxelize the region around the camera,
/// re-pack + extract the patch, build the BLAS/TLAS, raymarch + composite, and finally land the async readback
/// (a few frames deep). A FIXED frame count is fragile — its tail can latch a stale warm-up (black) readback
/// before the surface is resident + lit (the D1a 0.05 m / wider clipmap made the old fixed-24 budget too small,
/// #134). Pump-until-lit mirrors `voxel_cornell_headless` and is robust to readback latency. Returns the lit
/// bytes, or the last bytes seen (so the caller's asserts produce a meaningful diagnostic on a genuine failure).
fn pump_until_lit(app: &mut App, latest: &LatestFrame) -> Vec<u8> {
    let unpadded_row = (W * 4) as usize;
    let padded_row = bevy::render::renderer::RenderDevice::align_copy_bytes_per_row(unpadded_row);
    let mut last = Vec::new();
    for _ in 0..120 {
        app.update();
        if let Some(b) = latest.0.lock().unwrap().clone()
            && b.len() >= padded_row * H as usize
        {
            // Mean luma over the centre of the frame — non-trivial once the terrain has actually rendered.
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for y in (H as usize / 4)..(H as usize * 3 / 4) {
                for x in (W as usize / 4)..(W as usize * 3 / 4) {
                    let p = &b[y * padded_row + x * 4..y * padded_row + x * 4 + 4];
                    sum += 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
                    n += 1.0;
                }
            }
            last = b;
            if sum / n > 10.0 {
                break;
            }
        }
    }
    last
}

/// Build a `HeightLayer` from the default worldgen resources — the same direct-construction path the voxel
/// modules use — so the test can find the origin surface height to frame the camera.
fn default_layer() -> HeightLayer {
    let height = HeightParams::default();
    let erosion = ErosionParams::default();
    let graph = WorldGraph::default();
    let shapes = WorldBiomeShapes::default();
    adventure::voxel::build_height_layer_pub(&height, &erosion, &graph, &shapes)
}

#[test]
fn headless_render_shows_voxels() {
    // Probe for a ray-query-capable adapter first; skip cleanly if absent (CI box without RT).
    if common::headless_ray_query_device().is_none() {
        eprintln!("no ray-query device — skipping headless_render_shows_voxels");
        return;
    }

    // Camera framing: look at the origin-column surface from a fixed, close distance so the surface bricks
    // sit comfortably INSIDE the streaming residency region (which is centred on the camera). Deterministic
    // — fixed seed, fixed transform.
    let layer = default_layer();
    let surface_y = layer.sample_world(0.0, 0.0, WORLDGEN_SLICE_SEED).height;
    let target = Vec3::new(0.0, surface_y, 0.0);
    // Close in (~9 m), gently above, looking down at the surface so the looked-at terrain sits well INSIDE
    // the streaming residency region (which is centred on the camera) — otherwise the surface at the target
    // is never resident and every ray misses into sky.
    let yaw = 0.7f32;
    let pitch = 0.45f32;
    let distance = 9.0f32;
    let cam_pos = target
        + Vec3::new(
            distance * yaw.cos() * pitch.cos(),
            distance * pitch.sin(),
            distance * yaw.sin() * pitch.cos(),
        );

    let latest = LatestFrame(Arc::new(Mutex::new(None)));

    let mut app = App::new();
    // #134 — with default features (`dlss`), `bevy/dlss`'s `DlssInitPlugin` (inside `DefaultPlugins`) PANICS unless
    // a `DlssProjectId` is present BEFORE it runs. The app's `main.rs` inserts one; the headless rigs must too, or
    // the test dies before a single frame. Insert it BEFORE `DefaultPlugins` (gated to the `dlss` build only).
    insert_dlss_project_id(&mut app);
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(rt_wgpu_settings())),
                ..default()
            }),
    );
    // A tight clipmap so the surface around the camera voxelizes in a handful of frames (the production
    // Default clip_half-160 clipmap / 400k-brick cap would need many frames to drain). Inserted BEFORE Startup so
    // `init_voxel_rt_streaming` picks it up.
    app.insert_resource(StreamingConfig {
        clip_half_bricks: 4,
        max_resident_bricks: 30_000,
        max_bricks_per_frame: 8192,
    });
    // Worldgen sampling resources the voxel streaming reads (the app gets these from SdfScenePlugin; the test
    // inserts the defaults directly to stay minimal).
    app.init_resource::<HeightParams>()
        .init_resource::<ErosionParams>()
        .init_resource::<WorldGraph>()
        .init_resource::<WorldBiomeShapes>();
    app.add_plugins(VoxelRtPlugin);
    disable_dlss_for_headless(&mut app); // #134 — force the readback-capturable non-DLSS path (dlss build only).
    // This rig validates the STREAMING WORLDGEN path, so select it explicitly (the engine now defaults to
    // the static Cornell box). The dedicated `voxel_cornell_headless` rig covers the Cornell scene.
    app.insert_resource(VoxelScene::Worldgen);
    // HW-RT is the default renderer now — assert that, then keep it on for the render.
    assert!(
        app.world().resource::<VoxelRtToggle>().enabled,
        "VoxelRtToggle must default ON (HW-RT is the default renderer)"
    );

    app.insert_resource(latest.clone());
    app.insert_resource(ClearColor(Color::srgb(0.9, 0.0, 0.9))); // a garish magenta — must NOT survive.

    // The offscreen render target image (readable: TEXTURE_BINDING | COPY_SRC | RENDER_ATTACHMENT).
    let image_handle = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = Image::new_target_texture(W, H, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        images.add(image)
    };

    // The camera: offscreen target, SdfCamera (so streaming follows it), Hdr (linear Rgba16Float main
    // texture + tonemapping, matching the app's editor camera), MSAA off.
    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(image_handle.clone().into()),
        bevy::camera::Hdr,
        Msaa::Off,
        Transform::from_translation(cam_pos).looking_at(target, Vec3::Y),
        SdfCamera,
        Name::new("Headless RT Camera"),
    ));

    // Read the render-target image back every frame; stash the latest bytes for the test thread.
    let sink = latest.0.clone();
    app.world_mut()
        .spawn(Readback::texture(image_handle.clone()))
        .observe(move |event: On<ReadbackComplete>| {
            *sink.lock().unwrap() = Some(event.data.clone());
        });

    // Manual update loops must `finish` + `cleanup` the plugins first (this is what `App::run` does): it
    // unpacks the async-created `RenderDevice` into the main world. Without it, the PBR main-world systems
    // (`no_automatic_skin_batching` etc.) panic on a missing `RenderDevice` on the first frame.
    app.finish();
    app.cleanup();

    // Pump frames until the streamed terrain is resident, built, composited, and the async readback lands
    // LIT (robust to readback latency — see `pump_until_lit`; a fixed budget was too small post-D1a, #134).
    let bytes = pump_until_lit(&mut app, &latest);

    // Sanity: the streamed patch actually has resident bricks (the surface near the camera voxelized).
    let patch = app.world().resource::<VoxelRtPatch>();
    assert!(
        !patch.upload.is_empty(),
        "the streamed brick set must be non-empty (surface near the camera voxelized) — got 0 bricks"
    );

    // The readback must have landed at least once.
    assert!(!bytes.is_empty(), "the render target must have been read back at least once");

    // The GPU copy pads each row up to COPY_BYTES_PER_ROW_ALIGNMENT (256). Recover the real pixels per row.
    let unpadded_row = (W * 4) as usize;
    let padded_row = bevy::render::renderer::RenderDevice::align_copy_bytes_per_row(unpadded_row);
    assert!(
        bytes.len() >= padded_row * H as usize,
        "readback too small: {} bytes < {} expected",
        bytes.len(),
        padded_row * H as usize
    );

    // Classify every pixel: the miss path writes a BLUE-DOMINANT sky gradient; voxel HITS write the terrain
    // palette (green / brown / grey — NOT blue-dominant). The clear colour (magenta) is red+blue-dominant. So
    // "terrain" = a pixel whose blue is NOT the largest channel and which isn't near-magenta. This
    // classification survives tonemapping (it only compares relative channel magnitudes).
    let mut total = 0usize;
    let mut terrain = 0usize;
    let mut clear_magenta = 0usize;
    let mut distinct: std::collections::HashSet<[u8; 3]> = std::collections::HashSet::new();
    // Lighting-contrast accumulators: collect the per-pixel luminance of TERRAIN pixels so we can assert the
    // frame is LIT (a spread of brightness from shading/shadow/AO) rather than FLAT (one albedo per block).
    let mut terrain_lumas: Vec<f32> = Vec::new();
    let luma = |r: u8, g: u8, b: u8| 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32;
    for y in 0..H as usize {
        let row = &bytes[y * padded_row..y * padded_row + unpadded_row];
        for x in 0..W as usize {
            let p = &row[x * 4..x * 4 + 4];
            let (r, g, b) = (p[0], p[1], p[2]);
            total += 1;
            distinct.insert([r, g, b]);
            // Near the garish magenta clear (high R, low G, high B)?
            if r > 180 && g < 80 && b > 180 {
                clear_magenta += 1;
            }
            // Terrain = blue is not the dominant channel (sky is blue-dominant), and not magenta-ish.
            let blue_dominant = b >= r && b >= g;
            let magenta_ish = r > 150 && b > 150 && g < r.min(b);
            if !blue_dominant && !magenta_ish {
                terrain += 1;
                terrain_lumas.push(luma(r, g, b));
            }
        }
    }

    let terrain_frac = terrain as f32 / total as f32;

    // Lighting contrast stats over terrain pixels: with FLAT albedo every terrain pixel of a given block had
    // an identical colour (a handful of distinct lumas total); with direct lighting (Lambert N·L, traced
    // hard shadows, AO) the same blocks now span a RANGE of brightness depending on face orientation /
    // occlusion. Measure the spread (max−min) and the standard deviation of terrain luminance.
    let (luma_min, luma_max, luma_std, distinct_lumas) = {
        if terrain_lumas.is_empty() {
            (0.0f32, 0.0f32, 0.0f32, 0usize)
        } else {
            let mut mn = f32::MAX;
            let mut mx = f32::MIN;
            let mut sum = 0.0f32;
            let mut buckets: std::collections::HashSet<i32> = std::collections::HashSet::new();
            for &l in &terrain_lumas {
                mn = mn.min(l);
                mx = mx.max(l);
                sum += l;
                buckets.insert((l / 4.0).round() as i32); // ~4-level luma buckets (out of 255)
            }
            let mean = sum / terrain_lumas.len() as f32;
            let var = terrain_lumas.iter().map(|&l| (l - mean) * (l - mean)).sum::<f32>()
                / terrain_lumas.len() as f32;
            (mn, mx, var.sqrt(), buckets.len())
        }
    };
    // Debug: sample a few pixels so a failure shows WHAT the frame contains.
    let sample = |x: usize, y: usize| {
        let row = &bytes[y * padded_row..];
        [row[x * 4], row[x * 4 + 1], row[x * 4 + 2], row[x * 4 + 3]]
    };
    eprintln!(
        "headless render: {}x{} px, {} distinct colours, terrain_frac={:.3}, clear_magenta_frac={:.3}; \
         lighting: terrain luma min={:.1} max={:.1} std={:.1} distinct_luma_buckets={}; \
         samples TL={:?} C={:?} BL={:?}",
        W,
        H,
        distinct.len(),
        terrain_frac,
        clear_magenta as f32 / total as f32,
        luma_min,
        luma_max,
        luma_std,
        distinct_lumas,
        sample(8, 8),
        sample(W as usize / 2, H as usize / 2),
        sample(8, H as usize - 8),
    );

    // 1. The frame is NOT uniform — the composite produced an image (sky + voxels), not a flat clear. (Flat
    //    palette voxels + a sky gradient yield a modest distinct-colour count; >4 is plenty to rule out the
    //    uniform clear/black failure modes.)
    assert!(distinct.len() > 4, "frame is ~uniform ({} colours) — composite likely never ran", distinct.len());
    // 2. The garish magenta clear colour was almost entirely overwritten — the HW-RT view replaced it.
    assert!(
        (clear_magenta as f32 / total as f32) < 0.02,
        "too much clear colour survived ({:.1}%) — composite did not cover the frame",
        100.0 * clear_magenta as f32 / total as f32
    );
    // 3. A MEANINGFUL fraction of the frame is terrain voxels (not just sky) — proves voxel pixels reached
    //    the screen, i.e. rays hit the streamed bricks AND the composite wrote them through.
    assert!(
        terrain_frac > 0.10,
        "too few terrain (voxel) pixels: {:.1}% — voxels did not reach the screen",
        100.0 * terrain_frac
    );

    // 4. LIGHTING CONTRAST exists — the terrain is LIT, not flat. With the old flat-albedo path every
    //    terrain pixel of a given block was one identical colour (a couple of luma levels total). Direct
    //    lighting (Lambert N·L over varied face orientations + traced hard shadows + AO) now spreads the
    //    SAME blocks across many brightness levels. Assert a meaningful luminance spread AND a meaningful
    //    standard deviation — robust thresholds well clear of a flat frame's ~0 spread.
    assert!(
        (luma_max - luma_min) > 30.0,
        "terrain luminance spread too small ({:.1}) — frame looks flat-shaded, lighting not applied",
        luma_max - luma_min
    );
    assert!(
        luma_std > 3.0,
        "terrain luminance std too small ({:.1}) — too little shading variation across the surface",
        luma_std
    );
    assert!(
        distinct_lumas >= 6,
        "too few distinct terrain brightness levels ({distinct_lumas}) — surface is not being lit"
    );
}

/// **Phase 2.6 — emissive worldgen terrain present + threaded to the GPU palette (CPU-only, no GPU device).**
///
/// The large worldgen scene must (a) actually VOXELIZE emissive terrain (the new lava / crystal materials
/// placed via `surface_rules`) and (b) carry that emissive all the way into the packed GPU palette
/// (`GpuPaletteColor.emissive`), so the shader's `r.emissive` makes those voxels GI light sources. This pins
/// the whole biome → registry → GPU chain WITHOUT needing a ray-query adapter (so it always runs in CI).
///
/// It does NOT depend on the camera/streaming/render path — it drives the SAME deterministic voxelization
/// SSOT (`build_height_layer_pub` + `load_biome_library_pub` + `BlockRegistry::from_biome_library` +
/// `voxel_block_at`/`voxelize_brick` + `pack_brickmap`) the streaming path uses.
#[test]
fn worldgen_voxelizes_emissive_terrain_into_palette() {
    use adventure::voxel::brickmap::{BRICK_EDGE, BrickMap, VOXEL_SIZE, brick_coord_of_voxel};
    use adventure::voxel::gpu::pack_brickmap;
    use adventure::voxel::palette::{BlockId, BlockRegistry};
    use adventure::voxel::voxelize::{voxel_block_at, voxelize_brick};
    use bevy::math::IVec3;

    // The dramatic worldgen layer + the SHIPPED biome library (now carrying the emissive lava/crystal) — the
    // exact direct-construction path `init_voxel_rt_streaming` uses.
    let layer = default_layer();
    let lib = adventure::voxel::load_biome_library_pub();
    let registry = BlockRegistry::from_biome_library(&lib);

    // 1. Biome → registry plumbing: the shipped library MUST contain emissive materials, and they must reach
    //    the registry as emissive blocks (radiance = emissive_color * emissive_intensity).
    let mut emissive_blocks: Vec<(BlockId, [f32; 3])> = Vec::new();
    for i in 0..registry.len() {
        let id = BlockId(i as u16);
        let e = registry.emissive(id);
        if e[0] > 0.0 || e[1] > 0.0 || e[2] > 0.0 {
            emissive_blocks.push((id, e));
        }
    }
    assert!(
        !emissive_blocks.is_empty(),
        "the shipped biome library must define emissive terrain materials (lava/crystal) — the registry has \
         no emissive blocks, so biome → registry emissive plumbing is broken"
    );
    eprintln!("emissive blocks in registry: {emissive_blocks:?}");

    // 2. The worldgen surface MUST actually place an emissive voxel somewhere reachable. Scan a wide XZ grid
    //    (deterministic seed) of surface columns and check the SURFACE voxel's block; emissive lava pools in
    //    deep valley floors + crystal in cold-biome noise patches, so a broad sweep is guaranteed to hit one.
    let seed = WORLDGEN_SLICE_SEED;
    let mut found: Option<(IVec3, BlockId)> = None;
    'scan: for gz in -800..=800 {
        for gx in -800..=800 {
            // World column at a coarse 4 m grid stride (keeps the scan bounded while covering ~±3.2 km — wide
            // enough to reach the deep Plains valley floors where the emissive lava pools, plus the cold-biome
            // crystal patches; the diagnostic sweep shows hundreds of thousands of emissive surface hits here).
            let wx = gx as f64 * 4.0;
            let wz = gz as f64 * 4.0;
            let surf = layer.sample_world(wx, wz, seed).height as f64;
            // The surface voxel (topmost solid): the integer voxel whose centre is just below `surf`.
            let vy = (surf / VOXEL_SIZE as f64 - 0.5).floor() as i32;
            let wv = IVec3::new(
                (wx / VOXEL_SIZE as f64).floor() as i32,
                vy,
                (wz / VOXEL_SIZE as f64).floor() as i32,
            );
            let block = voxel_block_at(wv, &layer, &lib, &registry, seed);
            if !block.is_air() {
                let e = registry.emissive(block);
                if e[0] > 0.0 || e[1] > 0.0 || e[2] > 0.0 {
                    found = Some((wv, block));
                    break 'scan;
                }
            }
        }
    }
    let (emissive_voxel, emissive_id) = found.expect(
        "no emissive surface voxel found over a ~±1.2 km worldgen sweep — emissive lava/crystal placement \
         (surface_rules) never fired in any biome/altitude the scan reached",
    );
    let expected_e = registry.emissive(emissive_id);
    eprintln!("found emissive surface voxel at {emissive_voxel:?} block {emissive_id:?} emissive {expected_e:?}");

    // 3. Registry → GPU plumbing: voxelize the brick containing that emissive voxel, pack it, and assert the
    //    packed palette entry for the emissive block carries the SAME emissive radiance the registry holds.
    let bcoord = brick_coord_of_voxel(emissive_voxel);
    let brick = voxelize_brick(bcoord, 0, &layer, &lib, &registry, seed); // LOD0 brick (world-voxel grid)
    let mut map = BrickMap::new();
    assert!(map.insert(bcoord, brick), "the emissive brick must be non-empty (it contains the surface)");
    let patch = pack_brickmap(&map, &registry);
    let packed = patch.palette[emissive_id.0 as usize].emissive;
    assert_eq!(
        [packed[0], packed[1], packed[2]],
        expected_e,
        "packed GPU palette emissive for the emissive block must equal the registry emissive — registry → \
         GPU plumbing dropped it"
    );
    assert!(
        packed[0] > 0.0 || packed[1] > 0.0 || packed[2] > 0.0,
        "the packed emissive must be non-zero so the shader treats the voxel as a GI light source"
    );

    // Sanity: the brick covers BRICK_EDGE voxels per axis (the emissive voxel is inside it).
    let origin = bcoord * BRICK_EDGE;
    let local = emissive_voxel - origin;
    assert!(
        (0..BRICK_EDGE).contains(&local.x)
            && (0..BRICK_EDGE).contains(&local.y)
            && (0..BRICK_EDGE).contains(&local.z),
        "the emissive voxel {local:?} must sit inside its brick"
    );
}

/// **Phase G Stage G-b — the render PIXEL-IDENTITY gate** (docs/PHASE_G_GALLERY_PLAN.md §"Stage G-b").
///
/// Renders the SAME deterministic worldgen scene twice — once on the CPU path (`gpu_pack = false`: `update` +
/// `apply_delta`, CPU AABB upload) and once on the GPU path (`gpu_pack = true`: `update_gpu` + `apply_gpu_pack`,
/// the GPU `voxel_pack` encode + the GPU `write_aabb` + the fill-then-build BLAS) — and asserts the two readback
/// images MATCH. This proves the acceleration structure built over the GPU-WRITTEN AABBs is correct: if the BLAS
/// built over stale/zero AABBs (a fill-then-build ordering bug — the build executing before the compute fill
/// landed), geometry would be MISSING → huge regions would diverge by large magnitudes → this fails. The parity
/// gate (`voxel_gpu_pack_parity`) already pins the pool/AABB BYTES equal; this pins the rendered RESULT, closing
/// the loop end-to-end (the BLAS topology over those bytes is right).
///
/// **What is compared (and why not raw bytes).** The two runs are separate Bevy `App` instances and the full GI
/// path (ReSTIR temporal accumulation + the per-frame world cache) is NOT bit-reproducible across two independent
/// run sequences — the async readback can even latch a slightly different point in the temporal-convergence
/// sequence, so a BAND of already-terrain pixels can differ by a few LSBs run-to-run (max byte delta ~13, means
/// identical) EVEN WHEN THE GEOMETRY IS IDENTICAL. Raw byte-identity is therefore flaky regardless of
/// correctness. What a fill-then-build bug (the BLAS built over stale/zero GPU AABBs) actually changes is the
/// GEOMETRY SILHOUETTE — WHICH pixels are terrain vs sky — because missing bricks turn terrain pixels into sky
/// (huge silhouette change). A temporal phase only re-shades pixels that are terrain in BOTH frames. So we
/// compare the TERRAIN/SKY MASK (phase-invariant) and assert it matches to within a tiny fraction, plus assert
/// both frames are valid lit renders. This directly tests "the AS over the GPU-written AABBs places the same
/// geometry on screen" while being immune to GI rounding noise.
#[test]
fn gpu_pack_render_pixel_identical_to_cpu() {
    if common::headless_ray_query_device().is_none() {
        eprintln!("no ray-query device — skipping gpu_pack_render_pixel_identical_to_cpu");
        return;
    }
    let cpu_frame = render_worldgen_frame(false);
    let gpu_frame = render_worldgen_frame(true);
    assert_eq!(cpu_frame.len(), gpu_frame.len(), "CPU vs GPU-pack readback length differs");

    // Classify every pixel terrain (blue NOT the dominant channel — the lit terrain palette) vs sky/clear (blue-
    // dominant gradient). Identical classifier to `headless_render_shows_voxels`. Returns (mask, terrain_count).
    let padded_row = bevy::render::renderer::RenderDevice::align_copy_bytes_per_row((W * 4) as usize);
    let unpadded_row = (W * 4) as usize;
    let terrain_mask = |bytes: &[u8]| -> (Vec<bool>, usize) {
        let mut mask = vec![false; (W * H) as usize];
        let mut count = 0usize;
        for y in 0..H as usize {
            let row = &bytes[y * padded_row..y * padded_row + unpadded_row];
            for x in 0..W as usize {
                let p = &row[x * 4..x * 4 + 4];
                let (r, g, b) = (p[0], p[1], p[2]);
                let blue_dominant = b >= r && b >= g;
                let magenta_ish = r > 150 && b > 150 && g < r.min(b);
                let is_terrain = !blue_dominant && !magenta_ish;
                mask[y * W as usize + x] = is_terrain;
                if is_terrain {
                    count += 1;
                }
            }
        }
        (mask, count)
    };
    let (cpu_mask, cpu_terrain) = terrain_mask(&cpu_frame);
    let (gpu_mask, gpu_terrain) = terrain_mask(&gpu_frame);

    let mask_diff = cpu_mask.iter().zip(gpu_mask.iter()).filter(|(a, b)| a != b).count();
    let total_px = (W * H) as usize;
    let mask_diff_frac = mask_diff as f64 / total_px as f64;
    let cpu_frac = cpu_terrain as f64 / total_px as f64;
    let gpu_frac = gpu_terrain as f64 / total_px as f64;
    eprintln!(
        "[gpu-pack-render] terrain silhouette: CPU={:.3} GPU={:.3} ; mask mismatch {mask_diff} px ({:.4}%)",
        cpu_frac,
        gpu_frac,
        100.0 * mask_diff_frac
    );

    // (a) Both frames are valid renders with a meaningful amount of terrain on screen (a black/empty frame, or a
    //     frame whose geometry is entirely missing, would have ~0 terrain — caught here).
    assert!(
        cpu_frac > 0.10 && gpu_frac > 0.10,
        "a frame has too little terrain (CPU={cpu_frac:.3}, GPU={gpu_frac:.3}) — geometry missing on one path"
    );
    // (b) The terrain SILHOUETTE matches: a fill-then-build ordering bug (BLAS over stale/zero AABBs) would drop
    //     bricks → flip a large region of terrain pixels to sky → a large mask mismatch. Phase-invariant (a
    //     temporal-convergence phase only re-shades pixels terrain in BOTH frames, never flips terrain↔sky).
    assert!(
        mask_diff_frac < 0.01,
        "terrain/sky silhouette diverges by {:.3}% of pixels — the AS over the GPU-written AABBs places \
         DIFFERENT geometry than the CPU path (likely a fill-then-build ordering bug: the BLAS built before the \
         GPU AABB fill landed → missing bricks)",
        100.0 * mask_diff_frac
    );
}

/// Boot the headless worldgen render app with `VoxelRtToggle.gpu_pack = gpu_pack`, run a fixed number of frames,
/// and return the latest readback bytes (raw row-padded `Rgba8UnormSrgb`). The CPU and GPU paths share EVERYTHING
/// else (seed, camera, clipmap, frame count) so the only difference is HOW the pool/AABB/BLAS are written — the
/// rendered result must match. Factored from `headless_render_shows_voxels`'s setup.
fn render_worldgen_frame(gpu_pack: bool) -> Vec<u8> {
    let layer = default_layer();
    let surface_y = layer.sample_world(0.0, 0.0, WORLDGEN_SLICE_SEED).height;
    let target = Vec3::new(0.0, surface_y, 0.0);
    let yaw = 0.7f32;
    let pitch = 0.45f32;
    let distance = 9.0f32;
    let cam_pos = target
        + Vec3::new(
            distance * yaw.cos() * pitch.cos(),
            distance * pitch.sin(),
            distance * yaw.sin() * pitch.cos(),
        );

    let latest = LatestFrame(Arc::new(Mutex::new(None)));

    let mut app = App::new();
    // #134 — see `headless_render_shows_voxels`: insert `DlssProjectId` BEFORE `DefaultPlugins` (dlss build only).
    insert_dlss_project_id(&mut app);
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>()
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(rt_wgpu_settings())),
                ..default()
            }),
    );
    app.insert_resource(StreamingConfig {
        clip_half_bricks: 4,
        max_resident_bricks: 30_000,
        max_bricks_per_frame: 8192,
    });
    app.init_resource::<HeightParams>()
        .init_resource::<ErosionParams>()
        .init_resource::<WorldGraph>()
        .init_resource::<WorldBiomeShapes>();
    app.add_plugins(VoxelRtPlugin);
    disable_dlss_for_headless(&mut app); // #134 — force the readback-capturable non-DLSS path (dlss build only).
    app.insert_resource(VoxelScene::Worldgen);
    // The A/B flag under test: flip the gpu_pack path on/off (HW-RT itself stays on for both).
    app.insert_resource(VoxelRtToggle { enabled: true, gpu_pack, ..Default::default() });

    app.insert_resource(latest.clone());
    app.insert_resource(ClearColor(Color::srgb(0.9, 0.0, 0.9)));

    let image_handle = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = Image::new_target_texture(W, H, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        images.add(image)
    };

    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(image_handle.clone().into()),
        bevy::camera::Hdr,
        Msaa::Off,
        Transform::from_translation(cam_pos).looking_at(target, Vec3::Y),
        SdfCamera,
        Name::new("Headless RT Camera"),
    ));

    let sink = latest.0.clone();
    app.world_mut()
        .spawn(Readback::texture(image_handle.clone()))
        .observe(move |event: On<ReadbackComplete>| {
            *sink.lock().unwrap() = Some(event.data.clone());
        });

    app.finish();
    app.cleanup();
    // Pump-until-lit (robust to readback latency — both A/B runs use the SAME budget so the comparison is fair).
    let bytes = pump_until_lit(&mut app, &latest);

    // Sanity: voxels actually voxelized (so the comparison is meaningful, not two empty frames).
    let patch = app.world().resource::<VoxelRtPatch>();
    assert!(
        !patch.upload.is_empty(),
        "the streamed brick set must be non-empty (gpu_pack={gpu_pack}) — got 0 bricks"
    );
    assert!(!bytes.is_empty(), "the render target must have been read back at least once (gpu_pack={gpu_pack})");
    bytes
}
