//! **Stage 2 — hardware-ray-traced voxel render path** (additive + toggleable; the Stage-1 cube path
//! stays intact and is the default view).
//!
//! Wires the `voxel_raytrace.wgsl` compute raymarch into Bevy 0.19's SCHEDULE-BASED render pipeline (the
//! 0.18 render graph was removed): the raymarch is a render-world SYSTEM in the [`Core3d`] schedule,
//! ordered in the [`Core3dSystems::MainPass`] set (after the opaque pass), that runs a compute pass and
//! composites the result into the camera [`ViewTarget`].
//!
//! All GPU objects are created through the RAW wgpu device (`RenderDevice::wgpu_device()`) and stored as
//! raw `wgpu::*` types, mirroring the proven `D:/spike-aabb` AABB-`ray_query` setup and the headless test
//! — Bevy 0.19's `render_resource` wrappers don't cover the acceleration-structure binding path, so we own
//! the wgpu objects directly. The only Bevy render plumbing used is the schedule wiring + `RenderContext`'s
//! raw command encoder + the `ViewTarget`'s main texture.
//!
//! Pipeline:
//!   * Main world: [`build_voxel_rt_patch`] voxelizes the SAME bounded patch as Stage 1 and packs it into
//!     the SSOT GPU layout ([`super::gpu`]); the packed patch + the [`VoxelRtToggle`] resource extract to
//!     the render world.
//!   * Render world ([`RenderStartup`]): [`init_voxel_rt`] builds the raymarch compute pipeline + layouts.
//!   * Render world ([`Render`]/[`RenderSystems::PrepareResources`]): [`prepare_voxel_rt`] uploads the
//!     patch to storage buffers and builds the per-brick AABB BLAS + a brick-instance TLAS ONCE.
//!   * Render world ([`Core3d`]/[`Core3dSystems::MainPass`]): [`voxel_rt_pass`] — when the toggle is ON —
//!     dispatches the raymarch into a per-view output texture and composites it over the [`ViewTarget`].
//!
//! TOGGLE: `VoxelRtToggle` (default OFF). Press **`R`** to flip the Stage-1 cubes ↔ the HW-RT view.

use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::{ExtractedView, ViewTarget};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use wgpu::util::DeviceExt;

#[cfg(feature = "dlss")]
use bevy::anti_alias::dlss::{
    Dlss, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
    ViewDlssRayReconstructionTextures,
};
#[cfg(feature = "dlss")]
use bevy::core_pipeline::prepass::ViewPrepassTextures;

use super::cornell::{build_cornell, build_cornell_with_edits};
use super::edits::{VoxelEdits, VoxelHit, pick_voxel};
use super::gpu::{GpuBrickAabb, GpuBrickPatch, pack_brickmap, pack_resident_set};
use super::palette::{BlockId, BlockRegistry, CornellBlock};
use super::streaming::{ResidencyManager, StreamingConfig, camera_brick_coord, region_half_extent_m};
use super::{VoxelScene, build_height_layer_pub, load_biome_library_pub};
use crate::sdf_render::SdfCamera;
use crate::sdf_render::worldgen::WORLDGEN_SLICE_SEED;
use crate::sdf_render::worldgen::biome::BiomeLibrary;
use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use crate::sdf_render::worldgen::{WorldBiomeShapes, WorldGraph};

/// Runtime toggle: when `true` (the DEFAULT) the HW-RT voxel raymarch composites over the view; when
/// `false` the composite is skipped and the view is just the clear colour (the Stage-1 cubes were dropped —
/// `StandardMaterial`'s bindless PBR shader is broken on the wgpu-trunk fork). Extracted to the render world
/// each frame. Press **`R`** to flip it.
#[derive(Resource, Clone, Copy, Debug, ExtractResource)]
pub struct VoxelRtToggle {
    pub enabled: bool,
}

impl Default for VoxelRtToggle {
    fn default() -> Self {
        // HW-RT is the default (and only) renderer now — on at startup.
        Self { enabled: true }
    }
}

/// The packed, GPU-ready brick patch (the SSOT [`GpuBrickPatch`]) — re-built in the main world whenever the
/// streamed resident set changes, extracted to the render world for upload. `generation` increments on
/// every re-pack so the render world knows to rebuild the BLAS/TLAS (and keeps the OLD one bound until then
/// — keep-old-until-revealed).
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct VoxelRtPatch {
    pub patch: GpuBrickPatch,
    /// Bumped on every re-pack. The render world rebuilds its accel structures when this differs from the
    /// generation it last built for.
    pub generation: u64,
}

/// The main-world streaming state: the worldgen sampling context (built once) + the live
/// [`ResidencyManager`] + config. Drives the camera-following residency each frame. Not extracted — only
/// its packed output ([`VoxelRtPatch`]) crosses to the render world.
#[derive(Resource)]
pub struct VoxelRtStreaming {
    cfg: StreamingConfig,
    manager: ResidencyManager,
    layer: HeightLayer,
    lib: BiomeLibrary,
    registry: BlockRegistry,
    seed: u64,
    /// Last camera brick we reconciled toward, so we only re-`update` when the camera changes bricks.
    last_cam_brick: Option<IVec3>,
    /// The Cornell-box block palette (independent of worldgen) — used to pack the static Cornell patch.
    cornell_registry: BlockRegistry,
    /// Which scene the last packed patch was built for. `None` until the first pack; on a scene switch this
    /// differs from the live [`VoxelScene`], triggering a one-shot re-pack of the new scene.
    packed_scene: Option<VoxelScene>,
    /// The [`VoxelEdits`] generation the last Cornell pack reflected. When the live delta's generation differs
    /// (the user placed/removed a voxel) the static Cornell box is re-baked with the new overlay and re-packed
    /// — making the edit visible next frame. `None` until the first Cornell pack.
    packed_edit_gen: Option<u64>,
}

/// Stage-2 plugin: builds the patch in the main world, registers extraction, and wires the render-world
/// resources + the [`Core3d`] raymarch pass. Added in `main.rs` alongside [`super::VoxelPlugin`].
pub struct VoxelRtPlugin;

impl Plugin for VoxelRtPlugin {
    fn build(&self, app: &mut App) {
        // `VoxelScene` is normally provided by `VoxelPlugin`; init it here too so this plugin works
        // standalone (the headless render tests add only `VoxelRtPlugin`). `init_resource` is idempotent —
        // it never overwrites a value `VoxelPlugin` already inserted.
        app.init_resource::<VoxelScene>()
            .init_resource::<VoxelRtToggle>()
            .init_resource::<VoxelRtPatch>()
            .init_resource::<VoxelRtLighting>()
            .init_resource::<VoxelEdits>()
            .init_resource::<VoxelEditBrush>()
            .add_plugins(ExtractResourcePlugin::<VoxelRtToggle>::default())
            .add_plugins(ExtractResourcePlugin::<VoxelRtPatch>::default())
            .add_plugins(ExtractResourcePlugin::<VoxelRtLighting>::default())
            .add_systems(Startup, init_voxel_rt_streaming)
            // The edit click handler runs BEFORE the residency/re-bake so an edit's delta-generation bump is
            // observed the same frame (the re-bake then bumps the GPU generation → visible next frame).
            .add_systems(
                Update,
                (toggle_voxel_rt_input, voxel_edit_input, stream_voxel_rt_residency).chain(),
            );

        // DLSS-RR (Stage 4c): add the `Dlss<RayReconstruction>` component to the editor/render camera once
        // DLSS-RR is detected as supported on this machine. Its `#[require(...)]` then auto-adds
        // TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr — so bevy_anti_alias's DLSS node
        // runs in `Core3dSystems::EarlyPostProcess` (after our MainPass raymarch). No-op if RR is unsupported.
        #[cfg(feature = "dlss")]
        app.init_resource::<DlssSettings>().add_systems(Update, sync_dlss_camera);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_voxel_rt)
            .add_systems(Render, prepare_voxel_rt.in_set(RenderSystems::PrepareResources))
            // The composite MUST run AFTER the main opaque/transparent passes (so it loads onto the
            // already-cleared, already-rendered view target instead of being wiped by the opaque pass's
            // first-call `LoadOp::Clear`) and BEFORE tonemapping (`Core3dSystems::PostProcess`) so it writes
            // into the HDR main texture that tonemapping then converts to the display format. The
            // `EarlyPostProcess` set sits exactly between `MainPass` and `PostProcess` in the chained
            // `Core3d` schedule — the correct slot for a custom over-the-scene composite.
            .add_systems(Core3d, voxel_rt_pass.in_set(Core3dSystems::EarlyPostProcess));

        // DLSS-RR render-world wiring (Stage 4c). The raymarch produces DLSS-RR's guides, so it must run
        // AFTER the main opaque pass (no `LoadOp::Clear` wipe) but BEFORE bevy_anti_alias's DLSS node — which
        // lives IN `Core3dSystems::EarlyPostProcess`. We can't name that (private) node system, so we carve a
        // dedicated set `VoxelRtDlssSet` strictly between `MainPass` and `EarlyPostProcess` (the chained
        // ordering then guarantees our writes land before the DLSS node reads them). The `voxel_rt_pass`
        // itself stays in `EarlyPostProcess`; under dlss it early-returns and the dlss pass below runs instead.
        #[cfg(feature = "dlss")]
        render_app
            .add_systems(
                Render,
                prepare_voxel_rt_dlss_textures
                    .in_set(RenderSystems::PrepareResources)
                    .after(prepare_voxel_rt),
            )
            .configure_sets(
                Core3d,
                VoxelRtDlssSet
                    .after(Core3dSystems::MainPass)
                    .before(Core3dSystems::EarlyPostProcess),
            )
            .add_systems(Core3d, voxel_rt_dlss_pass.in_set(VoxelRtDlssSet));
    }
}

/// Render-world system set for the DLSS-RR raymarch+resolve pass: carved strictly between the main pass and
/// `EarlyPostProcess` so the guide/colour/depth/motion writes land before bevy_anti_alias's DLSS-RR node
/// (which runs inside `EarlyPostProcess`) consumes them. See the plugin build for the rationale.
#[cfg(feature = "dlss")]
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VoxelRtDlssSet;

/// Main-world startup: build the worldgen sampling context (the SAME direct `HeightLayer` + library Stage 1
/// uses) and an empty [`ResidencyManager`], stored in [`VoxelRtStreaming`]. The resident set is filled
/// lazily by [`stream_voxel_rt_residency`] as the camera position becomes known — no static patch.
fn init_voxel_rt_streaming(
    mut commands: Commands,
    height: Option<Res<HeightParams>>,
    erosion: Option<Res<ErosionParams>>,
    graph: Option<Res<WorldGraph>>,
    biome_shapes: Option<Res<WorldBiomeShapes>>,
    cfg_override: Option<Res<StreamingConfig>>,
) {
    let seed = WORLDGEN_SLICE_SEED;
    // The worldgen sampling resources are only needed by the WORLDGEN scene. Cornell (the default) never
    // touches them, so default any that are absent — the engine boots into Cornell without worldgen wired.
    let height = height.map(|r| *r).unwrap_or_default();
    let erosion = erosion.map(|r| *r).unwrap_or_default();
    let graph = graph.map(|r| (*r).clone()).unwrap_or_default();
    let biome_shapes = biome_shapes.map(|r| (*r).clone()).unwrap_or_default();
    let layer = build_height_layer_pub(&height, &erosion, &graph, &biome_shapes);
    let lib = load_biome_library_pub();
    let registry = BlockRegistry::from_biome_library(&lib);
    // A `StreamingConfig` resource (if inserted before Startup) overrides the default region/LOD/budget — the
    // SSOT knob. The headless render test inserts a tight config so the surface near the camera voxelizes in
    // a few frames; the app leaves it unset and gets the default.
    let cfg = cfg_override.map(|c| *c).unwrap_or_default();
    info!(
        "voxel-RT Stage 3: streaming residency ready (radius {} bricks ≈ {:.0} m, ≤{} bricks/frame, cap {} resident)",
        cfg.residency_radius_bricks,
        region_half_extent_m(&cfg),
        cfg.max_bricks_per_frame,
        cfg.max_resident_bricks,
    );
    commands.insert_resource(VoxelRtStreaming {
        cfg,
        manager: ResidencyManager::new(),
        layer,
        lib,
        registry,
        seed,
        last_cam_brick: None,
        cornell_registry: BlockRegistry::cornell(),
        packed_scene: None,
        packed_edit_gen: None,
    });
}

/// Main-world Update: drive the camera-following residency. Reads the camera world position, reconciles the
/// resident set toward the region around the camera brick ([`ResidencyManager::update`] — cheap, only
/// enqueues), does BOUNDED voxelization ([`ResidencyManager::drain_work`]), and — only when the resident
/// set actually CHANGED ([`ResidencyManager::take_dirty`]) — re-packs the SSOT [`GpuBrickPatch`] and bumps
/// its generation so the render world rebuilds the BLAS/TLAS. Until then the old patch (and the old TLAS)
/// stays valid: keep-old-until-revealed. Runs even when the toggle is off so the set is warm if it flips on
/// (the voxelization is bounded and cheap).
#[allow(clippy::too_many_arguments)]
fn stream_voxel_rt_residency(
    scene: Res<VoxelScene>,
    edits: Res<VoxelEdits>,
    mut streaming: ResMut<VoxelRtStreaming>,
    mut patch_res: ResMut<VoxelRtPatch>,
    mut lighting: ResMut<VoxelRtLighting>,
    cam: Query<&GlobalTransform, With<SdfCamera>>,
) {
    // --- Static Cornell scene: build the resident set once, re-baking ONLY when the edit delta changes. ---
    if scene.is_cornell() {
        // Re-pack on the first Cornell tick AND whenever a build/destroy edit bumps the delta generation. The
        // box is tiny + static, so a full re-bake + re-pack per edit is cheap (a few hundred bricks ≈ sub-ms;
        // see the perf note in the module docs) and trivially correct — every brick (incl. boundary halos) is
        // re-voxelized with the new overlay. If this ever grows, switch to the dirty-brick subset re-bake.
        let scene_new = streaming.packed_scene != Some(VoxelScene::Cornell);
        let edits_changed = streaming.packed_edit_gen != Some(edits.generation());
        if scene_new || edits_changed {
            let map = build_cornell_with_edits(&streaming.cornell_registry, &edits);
            let patch = pack_brickmap(&map, &streaming.cornell_registry);
            let (n, v) = (patch.brick_count(), patch.voxels.len());
            patch_res.patch = patch;
            patch_res.generation = patch_res.generation.wrapping_add(1);
            // Cornell lighting: the box is closed (only the −Z front is open), so the sun can't fill it — the
            // EMISSIVE ceiling panel is the dominant light. Use plenty of GI rays for clear colour bleed, a
            // dim ambient (so the room isn't pitch black before GI converges), and a weak sun angled in
            // through the open front for soft shadow shaping. (Set once; an edit re-pack keeps it.)
            if scene_new {
                lighting.data = LightingUniformData::cornell();
            }
            streaming.packed_scene = Some(VoxelScene::Cornell);
            streaming.packed_edit_gen = Some(edits.generation());
            if scene_new {
                info!("voxel-RT: built STATIC Cornell box — {n} bricks, {v} voxels (no streaming)");
            } else {
                debug!("voxel-RT: re-baked Cornell box for edit gen {} — {n} bricks, {v} voxels", edits.generation());
            }
        }
        return; // static — only re-bakes on an edit, never streams
    }

    // --- Worldgen scene: camera-following streaming residency (the original Stage-3 path). ---
    // On a switch INTO worldgen, drop any Cornell residency so the streamed set rebuilds cleanly.
    if streaming.packed_scene != Some(VoxelScene::Worldgen) {
        streaming.manager = ResidencyManager::new();
        streaming.last_cam_brick = None;
        streaming.packed_scene = Some(VoxelScene::Worldgen);
    }

    let Ok(cam_tf) = cam.single() else {
        return; // camera not spawned yet — try next frame
    };
    let cam_world: [f32; 3] = cam_tf.translation().into();
    let cam_brick = camera_brick_coord(cam_world);

    // Reconcile only when the camera crosses into a new brick (the desired set changes), OR when there is
    // still pending work to drain. This avoids recomputing the (2r+1)³ region every idle frame.
    let cam_changed = streaming.last_cam_brick != Some(cam_brick);
    if cam_changed {
        let dropped = {
            let VoxelRtStreaming { manager, cfg, .. } = &mut *streaming;
            manager.update(cam_brick, cfg)
        };
        streaming.last_cam_brick = Some(cam_brick);
        if dropped > 0 {
            debug!("voxel streaming: dropped {dropped} bricks left behind by camera move");
        }
    } else if streaming.manager.pending() == 0 {
        return; // nothing to do this frame
    }

    // Bounded voxelization of queued bricks.
    let VoxelRtStreaming { manager, cfg, layer, lib, registry, seed, .. } = &mut *streaming;
    manager.drain_work(cfg, layer, lib, registry, *seed);

    // Re-pack only if the resident set changed (keep-old-until-revealed otherwise).
    if manager.take_dirty() {
        let entries = manager.resident_entries();
        let patch = pack_resident_set(&entries, registry);
        let (n, v) = (patch.brick_count(), patch.voxels.len());
        patch_res.patch = patch;
        patch_res.generation = patch_res.generation.wrapping_add(1);
        debug!(
            "voxel-RT: re-packed resident set gen {} — {n} bricks, {v} cells, {} pending",
            patch_res.generation,
            manager.pending()
        );
    }
}

/// Main-world input: press **R** to flip the HW-RT view on/off.
fn toggle_voxel_rt_input(keys: Res<ButtonInput<KeyCode>>, mut toggle: ResMut<VoxelRtToggle>) {
    if keys.just_pressed(KeyCode::KeyR) {
        toggle.enabled = !toggle.enabled;
        info!("voxel-RT view: {}", if toggle.enabled { "ON (HW ray tracing)" } else { "OFF (clear only)" });
    }
}

/// The currently-selected PLACE brush block — the [`BlockId`] a right-click drops into the air voxel adjacent
/// to the hit face (Stage 5 build/destroy editing). The number keys **1–4** pick a Cornell block (white /
/// red / green / light), so the user can build with any palette colour. The default is white. Left-click
/// always REMOVES (ignores the brush). A resource so it's a single SSOT the input system + a future UI read.
#[derive(Resource, Clone, Copy, Debug)]
pub struct VoxelEditBrush {
    /// The block a PLACE (right-click) drops. Always a solid block (never AIR).
    pub block: BlockId,
}

impl Default for VoxelEditBrush {
    fn default() -> Self {
        // White is the most visible default brush against the coloured Cornell walls.
        Self { block: CornellBlock::White.id() }
    }
}

/// Main-world (Update): the build/destroy click handler (Stage 5).
///
/// LEFT-click = REMOVE: CPU-pick the first solid voxel under the cursor and write AIR into the [`VoxelEdits`]
/// delta at its coord (digs it out). RIGHT-click = PLACE: pick the same voxel, then write the current
/// [`VoxelEditBrush`] block into the AIR voxel ADJACENT to the hit FACE (`hit.place_target()`), so a placed
/// block sits ON the surface the user clicked. Number keys **1–4** select the place brush colour.
///
/// The pick DDA-marches the SAME overlaid solidity (base scene ∪ current edits) the GPU traces — for Cornell
/// the base is `build_cornell` (cheap to rebuild) overlaid with the live delta — so a click resolves exactly
/// the voxel on screen. Only acts when the cursor is over the viewport (`ViewportInputAllowed`, which the
/// editor clears over its dock panels) and inside the window. The mutation bumps the delta generation, which
/// [`stream_voxel_rt_residency`] picks up next frame to re-bake + re-pack + bump the GPU generation — so the
/// edit is visible the following frame.
#[allow(clippy::too_many_arguments)]
fn voxel_edit_input(
    scene: Res<VoxelScene>,
    // `ViewportInputAllowed` is owned by `SdfScenePlugin` (the editor sets it false over dock panels). Optional
    // so `VoxelRtPlugin` stays standalone — the headless render tests add only this plugin; absent ⇒ allowed.
    allowed: Option<Res<crate::sdf_render::ViewportInputAllowed>>,
    // The viewport's on-screen rect (set by the editor's Viewport tab); absent ⇒ full-window viewport.
    viewport_rect: Option<Res<crate::sdf_render::EditorViewportRect>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    streaming: Res<VoxelRtStreaming>,
    mut edits: ResMut<VoxelEdits>,
    mut brush: ResMut<VoxelEditBrush>,
    windows: Query<&Window>,
    cam: Query<(&Camera, &GlobalTransform), With<SdfCamera>>,
) {
    // Brush selection (1–4 → the four Cornell blocks). Runs regardless of cursor location so the user can
    // re-arm the brush while the pointer is anywhere.
    for (key, block) in [
        (KeyCode::Digit1, CornellBlock::White.id()),
        (KeyCode::Digit2, CornellBlock::Red.id()),
        (KeyCode::Digit3, CornellBlock::Green.id()),
        (KeyCode::Digit4, CornellBlock::Light.id()),
    ] {
        if keys.just_pressed(key) {
            brush.block = block;
            info!("voxel edit brush → block {}", block.0);
        }
    }

    let remove = mouse.just_pressed(MouseButton::Left);
    let place = mouse.just_pressed(MouseButton::Right);
    if !remove && !place {
        return;
    }
    if allowed.is_some_and(|a| !a.0) {
        return; // pointer is over an editor dock panel — don't edit the scene
    }
    // Editing is wired for the static Cornell scene now (the delta is scene-agnostic; worldgen editing wires
    // in once the base resident map is exposed to the pick). Bail on worldgen so a click there is inert.
    if !scene.is_cornell() {
        return;
    }
    let Ok((camera, cam_xf)) = cam.single() else {
        return; // no active editor camera
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return; // cursor outside the window
    };

    // The viewport rect: the editor's docked image rect, or the whole window in the non-editor build.
    let (vp_min, vp_size) = viewport_rect
        .and_then(|v| v.rect)
        .unwrap_or((Vec2::ZERO, Vec2::new(window.width(), window.height())));

    // Build the camera ray from the cursor (mirrors the raymarch shader's reverse-Z near-plane unprojection,
    // so the CPU pick ray == the GPU primary ray).
    let Some((ro, rd)) = cursor_world_ray(camera, cam_xf, vp_min, vp_size, cursor) else {
        return;
    };

    // The base Cornell map (no edits) overlaid with the live delta is what the renderer traces; the pick
    // consults the SAME overlay (base ∪ edits) so it agrees with the screen. Building the base is cheap for
    // the small static box (a few hundred bricks).
    let base = build_cornell(&streaming.cornell_registry);
    let Some(hit): Option<VoxelHit> = pick_voxel(&base, &edits, ro, rd, 1.0e3) else {
        return; // clicked empty space / sky
    };

    if remove {
        edits.remove(hit.voxel);
        info!("voxel REMOVE at {} (face {})", hit.voxel, hit.normal);
    } else {
        // PLACE the brush into the air voxel adjacent to the hit face.
        let target = hit.place_target();
        edits.place(target, brush.block);
        info!("voxel PLACE block {} at {} (on face {} of {})", brush.block.0, target, hit.normal, hit.voxel);
    }
}

/// Build a world-space camera ray (origin + normalized direction) from a cursor position, matching the
/// raymarch shader's primary-ray generation: unproject the reverse-Z NEAR plane (NDC z = 1) through the
/// camera's clip→view→world chain, so the CPU pick ray is identical to the GPU primary ray. `None` if the
/// viewport size is unavailable or the direction degenerates.
fn cursor_world_ray(
    camera: &Camera,
    cam_xf: &GlobalTransform,
    vp_min: Vec2,
    vp_size: Vec2,
    cursor: Vec2,
) -> Option<(Vec3, Vec3)> {
    let _ = camera.physical_viewport_size()?; // ensure the camera has a live viewport
    // Cursor relative to the viewport rect (the docked image's screen rect in the editor, or the full
    // window otherwise). Outside the rect ⇒ the click isn't on the 3D view.
    let local = cursor - vp_min;
    if vp_size.x <= 0.0 || vp_size.y <= 0.0 {
        return None;
    }
    if local.x < 0.0 || local.y < 0.0 || local.x > vp_size.x || local.y > vp_size.y {
        return None;
    }
    let ndc_x = (2.0 * local.x / vp_size.x) - 1.0;
    let ndc_y = 1.0 - (2.0 * local.y / vp_size.y);
    let world_from_view = cam_xf.to_matrix();
    let view_from_clip = camera.clip_from_view().inverse();
    let ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let view_pos = view_from_clip * ndc;
    let view_pos = view_pos.xyz() / view_pos.w;
    let world_pos = world_from_view.transform_point3(view_pos);
    let origin = cam_xf.translation();
    let dir = (world_pos - origin).normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    Some((origin, dir))
}

/// Editor-tunable DLSS Ray Reconstruction settings (the "Render / GI" panel writes these); the SSOT
/// [`sync_dlss_camera`] applies each frame. `enabled` toggles RR on/off on the camera; `mode` is the
/// quality/upscale preset (Auto / DLAA-native / Quality / Balanced / Performance / UltraPerformance).
#[cfg(feature = "dlss")]
#[derive(Resource, Clone, Copy)]
pub struct DlssSettings {
    pub enabled: bool,
    pub mode: bevy::anti_alias::dlss::DlssPerfQualityMode,
}

#[cfg(feature = "dlss")]
impl Default for DlssSettings {
    fn default() -> Self {
        Self { enabled: true, mode: bevy::anti_alias::dlss::DlssPerfQualityMode::Auto }
    }
}

/// Main-world (Update): reconcile DLSS Ray Reconstruction on the [`SdfCamera`] to match [`DlssSettings`]
/// (gated on RR being supported — the [`DlssRayReconstructionSupported`] resource appears after device init).
/// Adds the [`Dlss`]`<RayReconstruction>` component when enabled (its `#[require(TemporalJitter, MipBias,
/// DepthPrepass, MotionVectorPrepass, Hdr)]` pulls in the rest); removes it (plus TemporalJitter + MipBias, so
/// the non-RR temporal-accum fallback isn't left jittering) when disabled; and updates the quality mode live.
/// No-op forever on machines without RR support.
#[cfg(feature = "dlss")]
fn sync_dlss_camera(
    mut commands: Commands,
    settings: Res<DlssSettings>,
    supported: Option<Res<DlssRayReconstructionSupported>>,
    mut cams: Query<(Entity, Option<&mut Dlss<DlssRayReconstructionFeature>>), With<SdfCamera>>,
) {
    let want = supported.is_some() && settings.enabled;
    for (cam, dlss) in &mut cams {
        match (want, dlss) {
            (true, Some(mut d)) => {
                if d.perf_quality_mode != settings.mode {
                    d.perf_quality_mode = settings.mode;
                }
            }
            (true, None) => {
                commands.entity(cam).insert(Dlss::<DlssRayReconstructionFeature> {
                    perf_quality_mode: settings.mode,
                    reset: false,
                    _phantom_data: core::marker::PhantomData,
                });
                info!("voxel-RT: DLSS-RR enabled on the editor camera ({:?})", settings.mode);
            }
            (false, Some(_)) => {
                commands
                    .entity(cam)
                    .remove::<Dlss<DlssRayReconstructionFeature>>()
                    .remove::<bevy::render::camera::TemporalJitter>()
                    .remove::<bevy::render::camera::MipBias>();
                info!("voxel-RT: DLSS-RR disabled (temporal-accumulation fallback)");
            }
            (false, None) => {}
        }
    }
}

// --- render world (raw wgpu) ------------------------------------------------------------------------

const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The raymarch compute pipeline + bind-group layouts (raw wgpu), built once on the device.
#[derive(Resource)]
struct VoxelRtPipelines {
    /// `group(0)`: TLAS + brick storage buffers (metas, voxels, palette).
    scene_layout: wgpu::BindGroupLayout,
    /// `group(1)`: camera uniform + output storage texture.
    view_layout: wgpu::BindGroupLayout,
    /// `group(2)` (ReSTIR): the two per-pixel reservoir storage buffers (cur/prev) + the restir params uniform.
    /// Shared by the non-DLSS and DLSS ReSTIR entry points.
    reservoir_layout: wgpu::BindGroupLayout,
    /// The `raymarch` compute pipeline (legacy `gather_gi` GI). Built but not dispatched yet — kept for the
    /// R3 `gi_mode` A/B toggle (legacy vs ReSTIR in one build).
    #[allow(dead_code)]
    raymarch: wgpu::ComputePipeline,
    /// The `raymarch_restir` compute pipeline: same primary pass, reservoir (ReSTIR) GI. The live GI path.
    raymarch_restir: wgpu::ComputePipeline,
    /// The composite shader module + its bind-group layout + sampler. The composite render pipeline is
    /// built lazily (and cached) once the live view-target format is known.
    composite_module: wgpu::ShaderModule,
    composite_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    /// DLSS-RR (Stage 4c): the `raymarch_dlss` compute pipeline (writes the full lit colour + the 5 guide
    /// storage textures) + its `group(1)` view layout, and the resolve render pass's bind-group layout
    /// (samples the colour/depth/motion storage textures → view target + prepass depth/motion). The resolve
    /// render pipeline itself is built lazily (format-keyed) in the pass.
    /// Legacy DLSS guide-writing pass (`gather_gi` GI). Built but not dispatched — kept for the R3 A/B toggle.
    #[cfg(feature = "dlss")]
    #[allow(dead_code)]
    raymarch_dlss: wgpu::ComputePipeline,
    /// The `raymarch_dlss_restir` compute pipeline: the DLSS guide-writing pass with reservoir (ReSTIR) GI.
    #[cfg(feature = "dlss")]
    raymarch_dlss_restir: wgpu::ComputePipeline,
    #[cfg(feature = "dlss")]
    dlss_view_layout: wgpu::BindGroupLayout,
    #[cfg(feature = "dlss")]
    dlss_resolve_layout: wgpu::BindGroupLayout,
}

/// GPU scene (rebuilt when the streamed patch generation changes) + per-view output texture. `Option`s let
/// the pass early-return until ready.
#[derive(Resource, Default)]
struct VoxelRtResources {
    /// `group(0)` scene bind group (TLAS + storage buffers). Rebuilt on each new patch generation; the OLD
    /// one stays bound until the new one is fully built (keep-old-until-revealed).
    scene_bind_group: Option<wgpu::BindGroup>,
    /// Keep-alive holders for the GPU objects the bind group / TLAS reference (wgpu uses Arc internally,
    /// but we retain the BLAS + TLAS explicitly so they outlive the bind group deterministically).
    _keep: Option<SceneKeepAlive>,
    brick_count: u32,
    /// The patch generation the current scene bind group was built for. We rebuild when the extracted
    /// patch's generation differs (and only then).
    built_generation: Option<u64>,
    /// Output storage texture (rgba16float) + view + size; reallocated on view resize.
    output: Option<(wgpu::Texture, wgpu::TextureView, UVec2)>,
    /// The TEMPORAL-ACCUMULATION history texture (rgba16float) + view: the previous frame's accumulated
    /// result. Each frame the raymarch blends the new shade into this; after the pass it is refreshed by
    /// copying the output back. Persistent across frames (the accumulator), reallocated only on view resize.
    history: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// ReSTIR per-pixel reservoir storage buffers (a, b) + the size they were allocated for. Ping-ponged by
    /// frame parity: the shader WRITES the current buffer and READS the previous (last frame's). Reallocated
    /// on view resize; the contents are discarded via the `reset` flag (camera move / resize / re-pack), so
    /// no clear is needed. Used by both the non-DLSS and DLSS ReSTIR entry points.
    reservoirs: Option<(wgpu::Buffer, wgpu::Buffer, UVec2)>,
    /// The composite render pipeline, keyed by the view-target format it was built for.
    composite: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    /// Monotonic per-frame counter written into the lighting uniform's `frame_index` so the GI bounce
    /// directions decorrelate across frames (temporal variation feeding the accumulator below).
    frame_index: u32,
    /// TEMPORAL-ACCUMULATION sample count: how many consecutive frames the camera (and scene) has held still.
    /// The blend weight is `1 / accum_samples` (a running mean). RESET to 0 whenever the camera moves or the
    /// scene generation changes, so a moving camera shows the fresh frame (no ghosting) and a still camera
    /// converges. Capped so very long stills still adapt to slow changes (e.g. editor light tweaks).
    accum_samples: u32,
    /// The view-projection matrix the last frame accumulated against. A change (camera moved or projection
    /// changed) resets accumulation. `None` until the first frame.
    prev_view_proj: Option<[[f32; 4]; 4]>,
    /// The scene patch generation the accumulator is valid for; a re-pack (geometry changed) resets it.
    accum_generation: Option<u64>,

    // --- DLSS-RR (Stage 4c) intermediate textures + state (only used under `--features dlss`) ---
    /// The `raymarch_dlss` compute's COLOUR / DEPTH / MOTION storage outputs (the resolve render pass reads
    /// these to fill the view target + the RENDER_ATTACHMENT-only prepass depth/motion textures). The 3
    /// DLSS-RR GUIDE textures (diffuse/specular albedo, normal+roughness) are NOT here — they live in the
    /// `ViewDlssRayReconstructionTextures` component (created in `prepare_voxel_rt_dlss_textures`) and the
    /// compute storage-writes them directly. `(texture, view, size)`; reallocated on a render-resolution change.
    #[cfg(feature = "dlss")]
    dlss_color: Option<(wgpu::Texture, wgpu::TextureView)>,
    #[cfg(feature = "dlss")]
    dlss_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    #[cfg(feature = "dlss")]
    dlss_motion: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Size the dlss intermediate textures were allocated for (the DLSS render resolution).
    #[cfg(feature = "dlss")]
    dlss_size: Option<UVec2>,
    /// The resolve render pipeline, keyed by the (view-target format, motion format) it was built for.
    #[cfg(feature = "dlss")]
    dlss_resolve: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    /// Previous-frame clip_from_world for motion-vector reprojection. `None` on the first dlss frame.
    #[cfg(feature = "dlss")]
    dlss_prev_clip_from_world: Option<[[f32; 4]; 4]>,
    /// DLSS-path ReSTIR reservoirs (cur/prev) + the FULL size they were allocated for (the dispatch indexes
    /// them at the render-resolution stride). Separate from `reservoirs` because the DLSS pass runs on the
    /// DLSS views (the non-DLSS pass filters them out) at the render resolution.
    #[cfg(feature = "dlss")]
    dlss_reservoirs: Option<(wgpu::Buffer, wgpu::Buffer, UVec2)>,
    /// (render_res, clip_from_world, built_generation) at the last DLSS frame — drives the ReSTIR `reset`
    /// (a camera move, a resolution change, or a geometry re-pack invalidates the same-pixel reservoirs).
    #[cfg(feature = "dlss")]
    dlss_restir_prev: Option<(UVec2, [[f32; 4]; 4], Option<u64>)>,
}

/// Holders that must outlive the scene bind group / TLAS for the program's lifetime.
struct SceneKeepAlive {
    _blas: wgpu::Blas,
    _tlas: wgpu::Tlas,
    _buffers: [wgpu::Buffer; 4],
}

/// Camera uniform mirroring the WGSL `CameraUniform` (group 1, binding 0): `world_from_clip` (64) +
/// `cam_pos` (12) + `t_max` (4) + `viewport` (8) + pad (8) = 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniformData {
    world_from_clip: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    t_max: f32,
    viewport: [u32; 2],
    /// Temporal-accumulation blend weight (= 1 / sample_count): the fraction of THIS frame mixed into the
    /// history mean. `1.0` resets (camera moved / first frame); ramps to `1/n` while the camera holds still
    /// so a static view converges to a clean average. Mirrors `CameraUniform.accum_weight` in the shader.
    accum_weight: f32,
    _pad: u32,
}

/// **SSOT for the direct-lighting knobs** (the WGSL `LightingUniform`, group 1 binding 2). All values are
/// runtime UNIFORMS (knobs-as-uniforms mandate) — the GUI/editor can drive any of them; nothing here is a
/// shader const. 80 bytes (std140-safe: each `Vec3` is followed by a scalar to fill its 16-byte slot; the
/// GI knobs form a packed 16-byte row; the final row is `emissive_strength, frame_index, debug_view,
/// gi_firefly_clamp` — exactly 16 bytes, no trailing pad). Mirrored field-for-field by both the WGSL shader
/// and the headless lighting/GI tests, so the lighting layout has exactly one SSOT.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingUniformData {
    /// Normalized direction the SUNLIGHT TRAVELS (points away from the sun). Lambert uses `-sun_direction`.
    pub sun_direction: [f32; 3],
    /// Scalar multiplier on `sun_color`.
    pub sun_intensity: f32,
    /// Linear-RGB sun colour.
    pub sun_color: [f32; 3],
    /// World-metre normal offset for shadow/AO ray origins (avoids self-intersection).
    pub shadow_bias: f32,
    /// Linear-RGB ambient/sky fill colour.
    pub ambient_color: [f32; 3],
    /// AO ray length in world metres.
    pub ao_radius: f32,
    /// Number of AO rays in the hemisphere (0 disables AO → `ao = 1`).
    pub ao_samples: u32,
    // --- GI knobs (single-bounce diffuse GI + emissive lights). All runtime uniforms. ---
    /// Number of cosine-sampled diffuse bounce rays per pixel (0 disables GI). Capped in-shader.
    pub gi_rays: u32,
    /// Scalar multiplier on the accumulated indirect irradiance (artistic GI strength).
    pub gi_intensity: f32,
    /// Max world-metre distance a diffuse bounce ray travels before it counts as a sky/ambient miss.
    pub gi_bounce_dist: f32,
    /// Scalar multiplier on every block's palette emissive (how bright emitters glow + light neighbours).
    pub emissive_strength: f32,
    /// Per-frame counter used to decorrelate the per-pixel bounce-direction hash (temporal variation /
    /// future temporal accumulation). The render path bumps this each frame; tests pass a fixed value.
    pub frame_index: u32,
    /// Debug visualization mode (editor "Render" panel): 0 = lit (normal), 1 = world normals, 2 = depth,
    /// 3 = albedo, 4 = AO, 5 = GI-only, 6 = face-orientation (green front / red BACK-face). Mirrors the
    /// WGSL `LightingUniform.debug_view`.
    pub debug_view: u32,
    /// Max radiance of a single GI bounce sample before it's clamped down (hue-preserving). Caps fireflies /
    /// bright-sample spikes that otherwise boil under temporal denoise. `0.0` disables the clamp (unbiased —
    /// the headless GI tests rely on this). Fills the last std140 slot, so the struct is exactly 80 bytes.
    pub gi_firefly_clamp: f32,
}

impl Default for LightingUniformData {
    /// A sensible noon sun: a slightly-angled warm white key light from above, a dim sky-blue ambient fill,
    /// 4 AO rays over ~1 m. These are the defaults the GUI shows until the user tunes them.
    fn default() -> Self {
        // Noon-ish sun coming down and slightly from -X/-Z.
        let d = Vec3::new(-0.3, -1.0, -0.2).normalize();
        Self {
            sun_direction: d.into(),
            sun_intensity: 1.0,
            sun_color: [1.0, 0.96, 0.9],
            shadow_bias: 0.04,
            ambient_color: [0.10, 0.13, 0.18],
            ao_radius: 1.0,
            ao_samples: 4,
            gi_rays: 8,
            gi_intensity: 1.0,
            gi_bounce_dist: 12.0,
            emissive_strength: 4.0,
            frame_index: 0,
            debug_view: 0,
            gi_firefly_clamp: 0.0, // off by default (unbiased) — the headless GI tests assume no clamp
        }
    }
}

impl LightingUniformData {
    /// Lighting tuned for the static Cornell box: the EMISSIVE ceiling panel is the dominant light (the box
    /// is closed, so the sun can't fill it). High `gi_rays` for clear colour bleed off the red/green walls,
    /// a strong `emissive_strength` so the panel lights the room, a faint neutral ambient so surfaces aren't
    /// fully black before GI converges, and a weak sun angled IN through the open front (−Z) to shape soft
    /// shadows. All runtime uniforms (knobs-as-uniforms) — an editor panel can still override any of them.
    pub fn cornell() -> Self {
        // Sun travels +Z/down: it enters through the open −Z front and grazes the floor + boxes.
        let sun = Vec3::new(0.05, -0.55, 0.83).normalize();
        Self {
            sun_direction: sun.into(),
            sun_intensity: 0.5,
            sun_color: [1.0, 0.98, 0.95],
            shadow_bias: 0.04,
            ambient_color: [0.03, 0.03, 0.035],
            ao_radius: 0.6,
            ao_samples: 4,
            gi_rays: 16,
            gi_intensity: 1.0,
            gi_bounce_dist: 24.0,
            emissive_strength: 6.0,
            frame_index: 0,
            debug_view: 0,
            // The ceiling emitter (emissive ≈ 1 × strength 6) is the brightest legitimate bounce sample;
            // clamp a little above it so grazing/multi-hit fireflies are tamed but the panel itself is intact.
            gi_firefly_clamp: 8.0,
        }
    }
}

/// Runtime lighting resource: the SSOT [`LightingUniformData`] knobs, extracted to the render world each
/// frame and uploaded to the WGSL `LightingUniform`. A future editor panel mutates this; for now it carries
/// the noon-sun defaults. Knobs-as-uniforms: every lighting value the shader reads lives here.
#[derive(Resource, Clone, Copy, Debug, Default, ExtractResource)]
pub struct VoxelRtLighting {
    pub data: LightingUniformData,
}

/// [`RenderStartup`]: build the raymarch compute pipeline + bind-group layouts on the wgpu device (which
/// already has `EXPERIMENTAL_RAY_QUERY`, enabled in `main.rs`). The composite render pipeline is deferred
/// to the pass (format-keyed).
fn init_voxel_rt(mut commands: Commands, render_device: Res<RenderDevice>) {
    let device = render_device.wgpu_device();

    let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_scene_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                count: None,
            },
            storage_ro(1),
            storage_ro(2),
            storage_ro(3),
        ],
    });
    let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_view_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: OUTPUT_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // binding 2: the direct-lighting uniform (sun/ambient/AO knobs), updated per frame.
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: the temporal-accumulation HISTORY texture (previous accumulated frame), sampled.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // binding 4: the history sampler (nearest, non-filtering).
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });

    let raymarch_src =
        std::fs::read_to_string("assets/shaders/voxel_raytrace.wgsl").expect("read voxel_raytrace.wgsl");
    let raymarch_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_raytrace"),
        source: wgpu::ShaderSource::Wgsl(raymarch_src.into()),
    });
    let raymarch_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_raymarch_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout)],
        immediate_size: 0,
    });
    let raymarch = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_raymarch"),
        layout: Some(&raymarch_pl),
        module: &raymarch_module,
        entry_point: Some("raymarch"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ReSTIR group(2): two per-pixel reservoir storage buffers (cur/prev) + the restir params uniform.
    let reservoir_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_reservoir_layout"),
        entries: &[storage_rw(0), storage_rw(1), uniform_buf(2)],
    });
    let restir_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_raymarch_restir_pl"),
        bind_group_layouts: &[Some(&scene_layout), Some(&view_layout), Some(&reservoir_layout)],
        immediate_size: 0,
    });
    let raymarch_restir = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_raymarch_restir"),
        layout: Some(&restir_pl),
        module: &raymarch_module,
        entry_point: Some("raymarch_restir"),
        compilation_options: Default::default(),
        cache: None,
    });

    let composite_src =
        std::fs::read_to_string("assets/shaders/voxel_rt_composite.wgsl").expect("read voxel_rt_composite.wgsl");
    let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_rt_composite"),
        source: wgpu::ShaderSource::Wgsl(composite_src.into()),
    });
    let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_composite_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("voxel_rt_composite_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // --- DLSS-RR (Stage 4c) pipelines + layouts ---
    #[cfg(feature = "dlss")]
    let (raymarch_dlss, raymarch_dlss_restir, dlss_view_layout, dlss_resolve_layout) =
        init_dlss_pipelines(device, &scene_layout, &reservoir_layout, &raymarch_module, &composite_module);

    commands.insert_resource(VoxelRtPipelines {
        scene_layout,
        view_layout,
        reservoir_layout,
        raymarch,
        raymarch_restir,
        composite_module,
        composite_layout,
        composite_sampler,
        #[cfg(feature = "dlss")]
        raymarch_dlss,
        #[cfg(feature = "dlss")]
        raymarch_dlss_restir,
        #[cfg(feature = "dlss")]
        dlss_view_layout,
        #[cfg(feature = "dlss")]
        dlss_resolve_layout,
    });
    commands.init_resource::<VoxelRtResources>();
}

/// Build the DLSS-RR (`--features dlss`) compute pipeline + bind-group layouts. The `group(1)` "dlss view"
/// layout mirrors `raymarch_dlss`'s bindings: 0 = camera uniform, 1 = colour storage tex (rgba16f),
/// 2 = lighting uniform, 5/6 = diffuse/specular albedo storage (rgba8), 7 = normal+roughness storage
/// (rgba16f), 8 = depth storage (r32f), 9 = motion storage (rg16f), 10 = prev/cur view-proj uniform.
/// The resolve layout feeds the fullscreen resolve pass: 1 = sampler, 2/3/4 = colour/depth/motion sampled.
#[cfg(feature = "dlss")]
fn init_dlss_pipelines(
    device: &wgpu::Device,
    scene_layout: &wgpu::BindGroupLayout,
    reservoir_layout: &wgpu::BindGroupLayout,
    raymarch_module: &wgpu::ShaderModule,
    composite_module: &wgpu::ShaderModule,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_tex = |binding: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    let dlss_view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_dlss_view_layout"),
        entries: &[
            uniform(0),                                                  // camera
            storage_tex(1, OUTPUT_FORMAT),                              // out_tex (colour, rgba16f)
            uniform(2),                                                  // lighting
            storage_tex(5, wgpu::TextureFormat::Rgba8Unorm),           // diffuse_albedo
            storage_tex(6, wgpu::TextureFormat::Rgba8Unorm),           // specular_albedo
            storage_tex(7, wgpu::TextureFormat::Rgba16Float),          // normal_roughness
            storage_tex(8, wgpu::TextureFormat::R32Float),             // depth
            storage_tex(9, wgpu::TextureFormat::Rgba16Float),          // motion (.xy used; rg16f storage isn't universal)
            uniform(10),                                                // dlss_cam (prev/cur view-proj)
        ],
    });
    let dlss_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_raymarch_dlss_pl"),
        bind_group_layouts: &[Some(scene_layout), Some(&dlss_view_layout)],
        immediate_size: 0,
    });
    let raymarch_dlss = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_raymarch_dlss"),
        layout: Some(&dlss_pl),
        module: raymarch_module,
        entry_point: Some("raymarch_dlss"),
        compilation_options: Default::default(),
        cache: None,
    });
    // The ReSTIR variant: same DLSS guide layout + the group(2) reservoir buffers.
    let dlss_restir_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_raymarch_dlss_restir_pl"),
        bind_group_layouts: &[Some(scene_layout), Some(&dlss_view_layout), Some(reservoir_layout)],
        immediate_size: 0,
    });
    let raymarch_dlss_restir = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_raymarch_dlss_restir"),
        layout: Some(&dlss_restir_pl),
        module: raymarch_module,
        entry_point: Some("raymarch_dlss_restir"),
        compilation_options: Default::default(),
        cache: None,
    });

    let sampled = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let dlss_resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_dlss_resolve_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            sampled(2), // colour
            sampled(3), // depth
            sampled(4), // motion
        ],
    });
    let _ = composite_module; // resolve render pipeline is built lazily (format-keyed) in the pass
    (raymarch_dlss, raymarch_dlss_restir, dlss_view_layout, dlss_resolve_layout)
}

/// True iff two column-major 4×4 matrices are equal within a tight tolerance — the camera-move test for
/// temporal accumulation. Any element differing by more than `eps` (sub-pixel jitter excluded) counts as a
/// move and resets the accumulator so a moving camera never ghosts. `eps` is loose enough to ignore FP
/// re-derivation noise in a perfectly static view (so a still camera actually accumulates).
fn matrices_close(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> bool {
    const EPS: f32 = 1e-6;
    for c in 0..4 {
        for r in 0..4 {
            if (a[c][r] - b[c][r]).abs() > EPS {
                return false;
            }
        }
    }
    true
}

/// A read-only storage-buffer bind-group-layout entry at `binding`, visible to compute.
fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A read-write storage-buffer bind-group-layout entry at `binding`, visible to compute (the ReSTIR reservoirs).
fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A uniform-buffer bind-group-layout entry at `binding`, visible to compute.
fn uniform_buf(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Bytes per WGSL `Reservoir` (3×vec4 = 48). One reservoir per pixel in each ping-pong buffer.
const RESERVOIR_SIZE: u64 = 48;

/// Mirror of the WGSL `RestirParams` (group 2 binding 2): reset flag + frame index + viewport.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RestirParamsData {
    reset: u32,
    frame_index: u32,
    viewport_x: u32,
    viewport_y: u32,
}

/// [`Render`]/[`RenderSystems::PrepareResources`]: upload the streamed patch buffers + build the AABB BLAS
/// / brick TLAS for the CURRENT patch generation, then swap in a fresh `group(0)` scene bind group. Rebuilds
/// whenever the extracted patch's `generation` differs from the one already built (and ONLY then), so a
/// static camera does no GPU rebuild. Keep-old-until-revealed: the new BLAS/TLAS/buffers are built into
/// locals and only assigned at the end — the previous scene bind group (and its TLAS) stays bound for any
/// in-flight pass until this function completes the swap. Skips when the toggle is off or the patch is empty.
fn prepare_voxel_rt(
    toggle: Res<VoxelRtToggle>,
    patch_res: Option<Res<VoxelRtPatch>>,
    pipelines: Option<Res<VoxelRtPipelines>>,
    mut resources: ResMut<VoxelRtResources>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let (Some(patch_res), Some(pipelines)) = (patch_res, pipelines) else {
        return;
    };
    // Rebuild only on a NEW non-empty generation. An empty patch (no resident bricks yet) leaves any
    // previously-built scene untouched (keep-old), and a static camera (same generation) does nothing.
    if !toggle.enabled || patch_res.patch.is_empty() {
        return;
    }
    if resources.built_generation == Some(patch_res.generation) {
        return; // already built this generation — keep the current scene
    }
    let device = render_device.wgpu_device();
    let patch = &patch_res.patch;
    let n = patch.brick_count() as u32;

    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_aabbs"),
        contents: bytemuck::cast_slice(&patch.aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_metas"),
        contents: bytemuck::cast_slice(&patch.metas),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let voxel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_voxels"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: n,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("voxel_rt_blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size_desc.clone()] },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("voxel_rt_tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    tlas[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xff,
    ));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel_rt_build_accel"),
    });
    encoder.build_acceleration_structures(
        core::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &size_desc,
                stride: core::mem::size_of::<GpuBrickAabb>() as wgpu::BufferAddress,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }),
        core::iter::once(&tlas),
    );
    render_queue.submit(core::iter::once(encoder.finish()));

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_scene_bg"),
        layout: &pipelines.scene_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::AccelerationStructure(&tlas) },
            wgpu::BindGroupEntry { binding: 1, resource: meta_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: voxel_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: palette_buf.as_entire_binding() },
        ],
    });

    // Swap in the new scene atomically (the old `_keep` + bind group drop only now that the new ones are
    // fully built — keep-old-until-revealed).
    resources.scene_bind_group = Some(scene_bind_group);
    resources.brick_count = n;
    resources.built_generation = Some(patch_res.generation);
    resources._keep = Some(SceneKeepAlive {
        _blas: blas,
        _tlas: tlas,
        _buffers: [aabb_buf, meta_buf, voxel_buf, palette_buf],
    });
    debug!("voxel-RT: built accel structures for patch gen {} — {n} bricks", patch_res.generation);
}

/// [`Core3d`]/[`Core3dSystems::MainPass`]: when the toggle is on and the scene is built, dispatch the
/// raymarch compute pass into a per-view output texture, then composite it over the [`ViewTarget`]. When
/// off, returns immediately so the Stage-1 cubes render unchanged.
fn voxel_rt_pass(
    #[cfg(not(feature = "dlss"))] view: ViewQuery<(&ExtractedView, &ViewTarget)>,
    #[cfg(feature = "dlss")] view: ViewQuery<(
        &ExtractedView,
        &ViewTarget,
        Option<&Dlss<DlssRayReconstructionFeature>>,
    )>,
    toggle: Res<VoxelRtToggle>,
    lighting: Res<VoxelRtLighting>,
    pipelines: Option<Res<VoxelRtPipelines>>,
    mut resources: ResMut<VoxelRtResources>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !toggle.enabled {
        return;
    }
    let Some(pipelines) = pipelines else { return };
    if resources.scene_bind_group.is_none() {
        return; // scene not built yet (e.g. toggled on this very frame)
    }
    // Under `--features dlss`: when DLSS-RR is active on this view, the dedicated `voxel_rt_dlss_pass`
    // (between MainPass and EarlyPostProcess) produces the guides + colour instead — skip the temporal-accum
    // composite so we don't double-write. When DLSS-RR is NOT on the camera (unsupported machine), fall
    // through to the normal temporal-accum composite (the non-RTX fallback).
    #[cfg(feature = "dlss")]
    let (extracted_view, target) = {
        let (ev, tgt, dlss) = view.into_inner();
        if dlss.is_some() {
            return;
        }
        (ev, tgt)
    };
    #[cfg(not(feature = "dlss"))]
    let (extracted_view, target) = view.into_inner();
    let size = target.main_texture().size();
    let viewport = UVec2::new(size.width, size.height);
    if viewport.x == 0 || viewport.y == 0 {
        return;
    }
    let target_format = target.main_texture_format();
    let device = render_device.wgpu_device();

    // (Re)allocate the output + temporal-history textures if the view size changed. The output gains
    // COPY_SRC (it is copied into history each frame); the history gains COPY_DST + TEXTURE_BINDING (the
    // raymarch samples it as the previous accumulated frame). A resize reallocates both and resets the
    // accumulator below (the history content is stale at a new resolution).
    let need_alloc = resources.output.as_ref().map(|(_, _, s)| *s != viewport).unwrap_or(true);
    if need_alloc {
        let make = |label: &str, extra: wgpu::TextureUsages| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: viewport.x, height: viewport.y, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: OUTPUT_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | extra,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (otex, oview) = make("voxel_rt_output", wgpu::TextureUsages::COPY_SRC);
        let (htex, hview) = make("voxel_rt_history", wgpu::TextureUsages::COPY_DST);
        resources.output = Some((otex, oview, viewport));
        resources.history = Some((htex, hview));
        // ReSTIR per-pixel reservoirs (cur/prev ping-pong). Uninitialised — the `reset` flag (set below
        // because prev_view_proj is now None) makes the shader ignore the stale `prev` this frame.
        let reservoir_bytes = (viewport.x as u64) * (viewport.y as u64) * RESERVOIR_SIZE;
        let mk_res = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: reservoir_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        resources.reservoirs =
            Some((mk_res("voxel_rt_reservoir_a"), mk_res("voxel_rt_reservoir_b"), viewport));
        // A fresh-size history is uninitialised — force a reset (full new frame) this frame.
        resources.accum_samples = 0;
        resources.prev_view_proj = None;
    }

    // Build the composite render pipeline lazily for the live view-target format (cached).
    let rebuild_composite = resources.composite.as_ref().map(|(f, _)| *f != target_format).unwrap_or(true);
    if rebuild_composite {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel_rt_composite_pl"),
            bind_group_layouts: &[Some(&pipelines.composite_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel_rt_composite"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &pipelines.composite_module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &pipelines.composite_module,
                entry_point: Some("fs_composite"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        resources.composite = Some((target_format, pipeline));
    }

    // Advance the per-frame counter (before borrowing `output` immutably below) so the GI bounce-direction
    // hash decorrelates each frame.
    resources.frame_index = resources.frame_index.wrapping_add(1);
    let frame_index = resources.frame_index;

    // Camera uniform: world_from_clip + cam_pos + t_max + viewport.
    let world_from_view = extracted_view.world_from_view.to_matrix();
    let clip_from_view = extracted_view.clip_from_view;
    let world_from_clip = world_from_view * clip_from_view.inverse();
    let cam_pos = extracted_view.world_from_view.translation();

    // --- Temporal accumulation control: reset on a camera move or a geometry re-pack, else run a 1/n mean. ---
    // The view-projection (clip_from_world) fully captures both camera POSITION and PROJECTION; any change
    // means the previous history no longer aligns pixel-for-pixel, so we must reset (show the fresh frame) to
    // avoid ghosting. A scene re-pack (new geometry) likewise invalidates the history. Otherwise the camera is
    // still: grow the sample count and blend at 1/n so the image converges to the clean average over n frames.
    let view_proj = (clip_from_view * world_from_view.inverse()).to_cols_array_2d();
    let cur_generation = resources.built_generation;
    let moved = resources.prev_view_proj.map(|p| !matrices_close(&p, &view_proj)).unwrap_or(true);
    let geometry_changed = resources.accum_generation != cur_generation;
    if moved || geometry_changed {
        resources.accum_samples = 1; // fresh frame is sample #1 (weight 1.0 → no history)
    } else {
        // Cap the running mean so the accumulator keeps adapting to slow changes (e.g. an editor light tweak)
        // instead of freezing forever — past ~256 samples the variance is already negligible.
        resources.accum_samples = (resources.accum_samples + 1).min(256);
    }
    resources.prev_view_proj = Some(view_proj);
    resources.accum_generation = cur_generation;
    let accum_weight = 1.0 / resources.accum_samples as f32;

    let out_view = &resources.output.as_ref().expect("just allocated").1;

    let cam_uniform = CameraUniformData {
        world_from_clip: world_from_clip.to_cols_array_2d(),
        cam_pos: cam_pos.into(),
        t_max: 1.0e4,
        viewport: [viewport.x, viewport.y],
        accum_weight,
        _pad: 0,
    };
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_camera"),
        contents: bytemuck::bytes_of(&cam_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // The lighting+GI uniform (SSOT knobs), uploaded fresh each frame so editor tweaks take effect live.
    // Stamp the render-world frame counter into `frame_index` so the GI bounce-direction hash advances each
    // frame (temporal variation; the seed for a future history-buffer temporal accumulator).
    let mut light_data = lighting.data;
    light_data.frame_index = frame_index;
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_lighting"),
        contents: bytemuck::bytes_of(&light_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let hist_view = &resources.history.as_ref().expect("allocated with output").1;
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_view_bg"),
        layout: &pipelines.view_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_view) },
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(hist_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&pipelines.composite_sampler) },
        ],
    });
    let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_composite_bg"),
        layout: &pipelines.composite_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(out_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pipelines.composite_sampler) },
        ],
    });

    // ReSTIR group(2): the restir params + the ping-ponged reservoir buffers (write cur, read prev). `reset`
    // (camera move / geometry re-pack / fresh allocation) makes the shader ignore the stale previous frame.
    let restir_params = RestirParamsData {
        reset: u32::from(moved || geometry_changed),
        frame_index,
        viewport_x: viewport.x,
        viewport_y: viewport.y,
    };
    let restir_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_restir_params"),
        contents: bytemuck::bytes_of(&restir_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let (res_a, res_b, _) = resources.reservoirs.as_ref().expect("allocated with output");
    let (cur, prev) = if frame_index & 1 == 0 { (res_a, res_b) } else { (res_b, res_a) };
    let reservoir_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_reservoir_bg"),
        layout: &pipelines.reservoir_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cur.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: prev.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: restir_buf.as_entire_binding() },
        ],
    });

    let scene_bg = resources.scene_bind_group.as_ref().expect("checked above");
    let raymarch = &pipelines.raymarch_restir;
    let composite = &resources.composite.as_ref().expect("just built").1;
    let main_view = target.main_texture_view();
    // Texture handles for the post-pass output→history copy (the accumulator feedback).
    let out_tex = &resources.output.as_ref().expect("just allocated").0;
    let hist_tex = &resources.history.as_ref().expect("allocated with output").0;
    let copy_size = wgpu::Extent3d { width: viewport.x, height: viewport.y, depth_or_array_layers: 1 };

    // Use the RenderContext's raw wgpu command encoder for both passes (compute + composite).
    let encoder = ctx.command_encoder();
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel_rt_raymarch"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(raymarch);
        cpass.set_bind_group(0, scene_bg, &[]);
        cpass.set_bind_group(1, &view_bg, &[]);
        cpass.set_bind_group(2, &reservoir_bg, &[]);
        cpass.dispatch_workgroups(viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
    }
    // Feed this frame's accumulated output back into history for the next frame's blend (the running mean).
    encoder.copy_texture_to_texture(
        out_tex.as_image_copy(),
        hist_tex.as_image_copy(),
        copy_size,
    );
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel_rt_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: main_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(composite);
        rpass.set_bind_group(0, &composite_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}

// --- DLSS Ray Reconstruction render-world systems (Stage 4c, `--features dlss`) ---------------------

/// [`Render`]/[`RenderSystems::PrepareResources`]: for every view that has a [`Dlss`]`<RR>` component but no
/// [`ViewDlssRayReconstructionTextures`] yet (or whose textures are the wrong render-resolution), allocate the
/// 3 DLSS-RR GUIDE textures (diffuse/specular albedo, normal+roughness) at the FULL view-target size and
/// insert the component. (DLSS reads only the top-left `MainPassResolutionOverride` subrect via
/// `partial_texture_size`, so full-size textures are correct.) Mirrors Solari's `prepare.rs`. The `raymarch_dlss`
/// compute storage-writes these directly; bevy_anti_alias's DLSS-RR node then consumes the component.
#[cfg(feature = "dlss")]
#[allow(clippy::type_complexity)]
fn prepare_voxel_rt_dlss_textures(
    views: Query<
        (
            bevy::ecs::entity::Entity,
            &bevy::render::camera::ExtractedCamera,
            Option<&ViewDlssRayReconstructionTextures>,
        ),
        bevy::ecs::query::With<Dlss<DlssRayReconstructionFeature>>,
    >,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    use bevy::render::render_resource::{
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    };
    use bevy::render::texture::CachedTexture;
    for (entity, camera, existing) in &views {
        // The guide textures are sized to the FULL render-target (DLSS reads a subrect). Use the camera's
        // physical viewport size (the upscaled/output size).
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.map(|_| size).is_some() {
            // Already created. The textures are full-size; a window resize re-runs `prepare_dlss` which
            // re-creates the DlssRenderContext, but our guide textures are keyed off the full size which only
            // changes on a real resize — re-create only if absent. (A resize drops the component via the
            // bevy_anti_alias cleanup path on the camera; simplest correct behaviour is to leave existing.)
            continue;
        }
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        let make = |label: &str, format: TextureFormat| {
            let tex = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&TextureViewDescriptor::default());
            CachedTexture { texture: tex, default_view: view }
        };
        commands.entity(entity).insert(ViewDlssRayReconstructionTextures {
            diffuse_albedo: make("voxel_rt_dlss_diffuse_albedo", TextureFormat::Rgba8Unorm),
            specular_albedo: make("voxel_rt_dlss_specular_albedo", TextureFormat::Rgba8Unorm),
            normal_roughness: make("voxel_rt_dlss_normal_roughness", TextureFormat::Rgba16Float),
            specular_motion_vectors: make("voxel_rt_dlss_specular_motion", TextureFormat::Rg16Float),
        });
    }
}

/// DLSS camera uniform (WGSL `DlssCamera`, group 1 binding 10): previous- and current-frame clip_from_world
/// (JITTERED projection, to match the depth/motion the DLSS node expects). 128 bytes.
#[cfg(feature = "dlss")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DlssCameraData {
    prev_clip_from_world: [[f32; 4]; 4],
    clip_from_world: [[f32; 4]; 4],
}

/// [`Core3d`] (the `VoxelRtDlssSet`, between `MainPass` and `EarlyPostProcess`): the DLSS-RR raymarch. Runs
/// the `raymarch_dlss` compute (full lit colour + the 5 guide storage textures, at the DLSS render
/// resolution into the top-left of full-size textures), then a fullscreen RESOLVE render pass that lands the
/// colour into the HDR view target and the depth + motion into the RENDER_ATTACHMENT-only prepass textures.
/// bevy_anti_alias's DLSS-RR node (in `EarlyPostProcess`) then denoises+upscales. Skips views without
/// DLSS-RR (the non-dlss composite handles them).
#[cfg(feature = "dlss")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn voxel_rt_dlss_pass(
    view: ViewQuery<(
        &ExtractedView,
        &ViewTarget,
        &ViewDlssRayReconstructionTextures,
        &ViewPrepassTextures,
        &bevy::render::camera::TemporalJitter,
        Option<&bevy::camera::MainPassResolutionOverride>,
    )>,
    toggle: Res<VoxelRtToggle>,
    lighting: Res<VoxelRtLighting>,
    pipelines: Option<Res<VoxelRtPipelines>>,
    mut resources: ResMut<VoxelRtResources>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    if !toggle.enabled {
        return;
    }
    let Some(pipelines) = pipelines else { return };
    if resources.scene_bind_group.is_none() {
        return;
    }
    let (extracted_view, target, rr_textures, prepass, temporal_jitter, resolution_override) =
        view.into_inner();

    // The DLSS-RR node needs depth + motion prepass textures; if they aren't present this frame, bail (the
    // node would also skip). Both have RENDER_ATTACHMENT usage (we write them via the resolve render pass).
    let (Some(depth_attach), Some(motion_attach)) = (&prepass.depth, &prepass.motion_vectors) else {
        return;
    };

    // Full output size (the view target / prepass / guide textures are all allocated at this size).
    let full = target.main_texture().size();
    let full = UVec2::new(full.width, full.height);
    if full.x == 0 || full.y == 0 {
        return;
    }
    // DLSS render resolution = the MainPassResolutionOverride subrect (or full if absent on the first frame
    // before bevy_anti_alias's prepare_dlss has set it). We render into the top-left `render_res` subrect.
    let render_res = resolution_override.map(|r| r.0).unwrap_or(full);
    let device = render_device.wgpu_device();
    let motion_format = motion_attach.texture.texture.format();

    // (Re)allocate the dlss intermediate colour/depth/motion storage textures at FULL size (so the resolve
    // pass can copy any subrect 1:1 into the full-size view target / prepass textures).
    let need_alloc = resources.dlss_size != Some(full);
    if need_alloc {
        let make = |label: &str, format: wgpu::TextureFormat| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: full.x, height: full.y, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        resources.dlss_color = Some(make("voxel_rt_dlss_color", OUTPUT_FORMAT));
        resources.dlss_depth = Some(make("voxel_rt_dlss_depth", wgpu::TextureFormat::R32Float));
        // Rgba16Float (not Rg16Float) for the intermediate motion: `rg16float` storage isn't universally
        // supported; .xy carry the motion, the resolve pass writes the real Rg16Float prepass attachment.
        resources.dlss_motion = Some(make("voxel_rt_dlss_motion", wgpu::TextureFormat::Rgba16Float));
        resources.dlss_size = Some(full);
        resources.dlss_prev_clip_from_world = None;
        // ReSTIR reservoirs at FULL size (≥ the render-res dispatch); reset forces a fresh frame.
        let reservoir_bytes = (full.x as u64) * (full.y as u64) * RESERVOIR_SIZE;
        let mk_res = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: reservoir_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        resources.dlss_reservoirs =
            Some((mk_res("voxel_rt_dlss_reservoir_a"), mk_res("voxel_rt_dlss_reservoir_b"), full));
        resources.dlss_restir_prev = None;
    }

    // Build the resolve render pipeline lazily, keyed by (view-target format, motion format).
    let target_format = target.main_texture_format();
    let rebuild = resources
        .dlss_resolve
        .as_ref()
        .map(|(f, _)| *f != target_format)
        .unwrap_or(true);
    if rebuild {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel_rt_dlss_resolve_pl"),
            bind_group_layouts: &[Some(&pipelines.dlss_resolve_layout)],
            immediate_size: 0,
        });
        let depth_format = depth_attach.texture.texture.format();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel_rt_dlss_resolve"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &pipelines.composite_module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                // Reverse-Z: the prepass uses GreaterEqual, but we OVERWRITE rather than test (no prior
                // geometry — voxel-rt has no meshes). Always pass so our frag_depth always lands.
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &pipelines.composite_module,
                entry_point: Some("fs_resolve_dlss"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: motion_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        resources.dlss_resolve = Some((target_format, pipeline));
    }

    resources.frame_index = resources.frame_index.wrapping_add(1);
    let frame_index = resources.frame_index;

    // Camera basis for primary rays — using the JITTERED projection (TemporalJitter perturbs clip space; DLSS
    // expects the jittered camera + the jitter_offset to resolve). Mirror `prepare_view_uniforms`: jitter the
    // projection over the RENDER-resolution viewport.
    let world_from_view = extracted_view.world_from_view.to_matrix();
    let mut clip_from_view = extracted_view.clip_from_view;
    temporal_jitter.jitter_projection(&mut clip_from_view, render_res.as_vec2());
    let world_from_clip = world_from_view * clip_from_view.inverse();
    let cam_pos = extracted_view.world_from_view.translation();
    let clip_from_world = clip_from_view * world_from_view.inverse();
    let clip_from_world_arr = clip_from_world.to_cols_array_2d();
    let prev = resources.dlss_prev_clip_from_world.unwrap_or(clip_from_world_arr);

    let cam_uniform = CameraUniformData {
        world_from_clip: world_from_clip.to_cols_array_2d(),
        cam_pos: cam_pos.into(),
        t_max: 1.0e4,
        viewport: [render_res.x, render_res.y],
        accum_weight: 1.0, // unused by raymarch_dlss (DLSS denoises), kept for layout parity
        _pad: 0,
    };
    let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_camera"),
        contents: bytemuck::bytes_of(&cam_uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut light_data = lighting.data;
    light_data.frame_index = frame_index;
    let light_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_lighting"),
        contents: bytemuck::bytes_of(&light_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let dlss_cam = DlssCameraData {
        prev_clip_from_world: prev,
        clip_from_world: clip_from_world_arr,
    };
    let dlss_cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_cam"),
        contents: bytemuck::bytes_of(&dlss_cam),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    resources.dlss_prev_clip_from_world = Some(clip_from_world_arr);

    // ReSTIR reset: a camera move, a render-resolution change, or a geometry re-pack invalidates the
    // same-pixel reservoirs (no reprojection yet — that's R2). DLSS-RR still reprojects the resolved colour.
    // CRUCIAL: use the UN-JITTERED view-projection for the move test. `clip_from_world_arr` is built from the
    // TemporalJitter-perturbed projection, which changes EVERY frame — comparing it would fire `reset` every
    // frame and the reservoirs would never accumulate (the GI would boil exactly as before). The sub-pixel
    // jitter must not count as camera motion. `extracted_view.clip_from_view` is the original un-jittered proj.
    let view_proj_unjittered =
        (extracted_view.clip_from_view * world_from_view.inverse()).to_cols_array_2d();
    let built_gen = resources.built_generation;
    let reset_restir = match resources.dlss_restir_prev {
        None => true,
        Some((r, vp, g)) => {
            r != render_res || !matrices_close(&vp, &view_proj_unjittered) || g != built_gen
        }
    };
    resources.dlss_restir_prev = Some((render_res, view_proj_unjittered, built_gen));

    let color_view = &resources.dlss_color.as_ref().expect("just allocated").1;
    let depth_view = &resources.dlss_depth.as_ref().expect("just allocated").1;
    let motion_view = &resources.dlss_motion.as_ref().expect("just allocated").1;

    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_dlss_view_bg"),
        layout: &pipelines.dlss_view_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(color_view) },
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&rr_textures.diffuse_albedo.default_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&rr_textures.specular_albedo.default_view) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&rr_textures.normal_roughness.default_view) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(motion_view) },
            wgpu::BindGroupEntry { binding: 10, resource: dlss_cam_buf.as_entire_binding() },
        ],
    });
    let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_dlss_resolve_bg"),
        layout: &pipelines.dlss_resolve_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&pipelines.composite_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(color_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(motion_view) },
        ],
    });

    // ReSTIR group(2): params + ping-ponged reservoirs (write cur, read prev).
    let restir_params = RestirParamsData {
        reset: u32::from(reset_restir),
        frame_index,
        viewport_x: render_res.x,
        viewport_y: render_res.y,
    };
    let restir_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_restir_params"),
        contents: bytemuck::bytes_of(&restir_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let (res_a, res_b, _) = resources.dlss_reservoirs.as_ref().expect("allocated above");
    let (cur, prev) = if frame_index & 1 == 0 { (res_a, res_b) } else { (res_b, res_a) };
    let reservoir_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_dlss_reservoir_bg"),
        layout: &pipelines.reservoir_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: cur.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: prev.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: restir_buf.as_entire_binding() },
        ],
    });

    let scene_bg = resources.scene_bind_group.as_ref().expect("checked above");
    let raymarch = &pipelines.raymarch_dlss_restir;
    let resolve = &resources.dlss_resolve.as_ref().expect("just built").1;
    let main_view = target.main_texture_view();
    let depth_target = &depth_attach.texture.default_view;
    let motion_target = &motion_attach.texture.default_view;

    let encoder = ctx.command_encoder();
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel_rt_raymarch_dlss"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(raymarch);
        cpass.set_bind_group(0, scene_bg, &[]);
        cpass.set_bind_group(1, &view_bg, &[]);
        cpass.set_bind_group(2, &reservoir_bg, &[]);
        cpass.dispatch_workgroups(render_res.x.div_ceil(8), render_res.y.div_ceil(8), 1);
    }
    {
        // Resolve into the view target (colour) + the prepass motion (colour 1) + prepass depth (frag_depth).
        // The viewport is clamped to the render resolution so we only write the DLSS-read subrect.
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel_rt_dlss_resolve"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: main_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: motion_target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_target,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_viewport(0.0, 0.0, render_res.x as f32, render_res.y as f32, 0.0, 1.0);
        rpass.set_pipeline(resolve);
        rpass.set_bind_group(0, &resolve_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
