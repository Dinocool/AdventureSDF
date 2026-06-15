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
use rustc_hash::FxHashSet;
use wgpu::util::DeviceExt;

#[cfg(feature = "dlss")]
use bevy::anti_alias::dlss::{
    Dlss, DlssRayReconstructionFeature, DlssRayReconstructionSupported,
    ViewDlssRayReconstructionTextures,
};
#[cfg(feature = "dlss")]
use bevy::core_pipeline::prepass::ViewPrepassTextures;

use super::brickmap::BrickMap;
use super::cornell::{build_cornell, build_cornell_with_edits};
use super::gallery::{GALLERY_SCENES, load_gallery};
use super::edits::{VoxelEdits, VoxelHit, pick_voxel};
use super::gpu::{GpuAliasEntry, GpuBrickAabb, GpuBrickPatch, GpuVoxelLight, pack_brickmap, pack_resident_set};
use super::palette::{BlockId, BlockRegistry, CornellBlock};
use super::source::{BrickSource, StaticVoxSource, WorldgenSource};
use super::streaming::{BrickKey, ResidencyManager, StreamingConfig, camera_brick_coord, region_half_extent_m};
use super::vox::load_vox;
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
    /// Cache of the baked Sponza `.vox` scene `(BrickMap, BlockRegistry)`, loaded LAZILY the first time the
    /// Sponza scene is selected and kept thereafter (loading + parsing the `.vox` is not free, so we never
    /// reload it per frame — the static scene is packed once from this cache). `None` until the first Sponza
    /// switch; on a load FAILURE it stays `None` and the residency falls back to Cornell (never panics).
    sponza: Option<(BrickMap, BlockRegistry)>,
    /// The Sponza scene's [`StaticVoxSource`] — its per-LOD MIP PYRAMID (a downsample of the WHOLE building,
    /// ~hundreds of ms) built ONCE on the Sponza switch and reused every frame. Built here, NOT per-frame in
    /// the drain: calling `StaticVoxSource::new` in the per-frame sourcing block rebuilt the entire pyramid
    /// every streaming frame — the Sponza load-lag root cause. `None` off Sponza (freed on a switch away).
    sponza_source: Option<StaticVoxSource>,
    /// Cache of the merged GALLERY scene `(BrickMap, BlockRegistry)` — the [`super::gallery::GALLERY_SCENES`]
    /// row loaded + merged ONCE the first time the Gallery is selected, then kept (loading + merging several
    /// `.vox` is not free, so we never re-merge per frame — mirrors [`sponza`](Self::sponza)). `None` until the
    /// first Gallery switch; absent assets are skipped during the merge (never a load FAILURE, so this is
    /// `Some` with whatever scenes existed — possibly an empty map if none were baked).
    gallery: Option<(BrickMap, BlockRegistry)>,
    /// The Gallery's [`StaticVoxSource`] — its MIP PYRAMID over the MERGED multi-scene map, built ONCE on the
    /// Gallery switch and reused every frame (NOT per-frame in the drain — the same build-once rule as
    /// [`sponza_source`](Self::sponza_source), so the merged row never re-downsamples per streaming frame).
    /// `None` off Gallery (freed on a switch away).
    gallery_source: Option<StaticVoxSource>,
    /// Which scene the last packed patch was built for. `None` until the first pack; on a scene switch this
    /// differs from the live [`VoxelScene`], triggering a one-shot re-pack of the new scene.
    packed_scene: Option<VoxelScene>,
    /// The [`VoxelEdits`] generation the last Cornell pack reflected. When the live delta's generation differs
    /// (the user placed/removed a voxel) the static Cornell box is re-baked with the new overlay and re-packed
    /// — making the edit visible next frame. `None` until the first Cornell pack.
    packed_edit_gen: Option<u64>,
    /// Worldgen re-pack AMORTIZATION (perf): whether the resident set has changed since the last pack, and how
    /// many frames since the last pack. `pack_resident_set` + the BLAS rebuild are O(resident); running them on
    /// EVERY dirty drain while streaming a region was a dominant cost. We instead pack on a settle
    /// (`pending() == 0`) OR every [`WORLDGEN_REPACK_INTERVAL`] frames during a long stream — so terrain still
    /// reveals progressively (keep-old-until-revealed) but the per-frame pack frequency is bounded.
    worldgen_dirty_pending: bool,
    worldgen_frames_since_pack: u32,
    /// The INCREMENTAL re-packer for the streamed (worldgen / Sponza / Gallery) arm: the O(changed) successor
    /// to the per-move full [`pack_resident_set`]. It owns a per-brick slot allocator, a fixed-block voxel
    /// arena, and shadow copies of each slot's bytes, so a re-pack re-`pack_one`s ONLY the entered/dropped
    /// bricks plus their resident 26-neighbourhood (the halo dependency), then assembles the extracted patch by
    /// memcpy of the cached bytes. The result is byte-identical to a from-scratch pack (proven by
    /// `snapshot_patch_matches_full_pack`), so the render path and shader are unchanged and pixel-identical.
    /// `None` until the first streamed pack; a fresh packer is set on every scene switch (so a switch streams in
    /// cleanly), mirroring `manager`.
    packer: Option<super::incremental::ResidentPacker>,
}

impl VoxelRtStreaming {
    /// Read-only access to the live residency manager (resident / pruned / pending counts, LOD histogram) for
    /// the editor perf/stats panel.
    #[inline]
    pub fn manager(&self) -> &ResidencyManager {
        &self.manager
    }

    /// The active streaming config (clip_half, caps) — for the perf/stats panel's view-radius / LOD0-reach.
    #[inline]
    pub fn cfg(&self) -> &StreamingConfig {
        &self.cfg
    }
}

/// Max frames a worldgen stream batch drains before forcing a re-pack (so a big cold fill reveals in chunks
/// rather than one final pop, while still bounding the O(resident) pack to ~once per this many frames).
const WORLDGEN_REPACK_INTERVAL: u32 = 6;

/// Asset path of the baked Sponza `.vox` — the default static GI-measurement scene. Voxelized offline once
/// (see `examples/voxelize_scene.rs`); the runtime only reads this baked file. Relative to the crate root /
/// working directory (matching how `biomes.ron` is read), so headless tests run from the crate root resolve
/// it too. The single SSOT for the Sponza asset location (the editor scene-selector path table points here).
pub const SPONZA_VOX_PATH: &str = "assets/models/sponza.vox";

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
            .init_resource::<VoxelRtSky>()
            .init_resource::<RestirSettings>()
            .init_resource::<WorldCacheSettings>()
            .init_resource::<VoxelEdits>()
            .init_resource::<VoxelEditBrush>()
            .add_plugins(ExtractResourcePlugin::<VoxelRtToggle>::default())
            .add_plugins(ExtractResourcePlugin::<VoxelRtPatch>::default())
            .add_plugins(ExtractResourcePlugin::<VoxelRtLighting>::default())
            .add_plugins(ExtractResourcePlugin::<VoxelRtSky>::default())
            .add_plugins(ExtractResourcePlugin::<RestirSettings>::default())
            .add_plugins(ExtractResourcePlugin::<WorldCacheSettings>::default())
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
        "voxel-RT clipmap streaming ready (clip_half {} bricks ⇒ view radius ≈ {:.0} m over {} nested LOD shells, ≤{} bricks/frame, cap {} resident)",
        cfg.clip_half_bricks,
        region_half_extent_m(&cfg),
        crate::voxel::brickmap::MAX_LOD + 1,
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
        sponza: None,
        sponza_source: None,
        gallery: None,
        gallery_source: None,
        packed_scene: None,
        packed_edit_gen: None,
        worldgen_dirty_pending: false,
        worldgen_frames_since_pack: 0,
        packer: None,
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
    mut sky: ResMut<VoxelRtSky>,
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
                // Reset the sky to its default so a switch BACK from worldgen (which set the bright worldgen
                // sky) restores Cornell's look exactly — Cornell is closed, but keep it identical regardless.
                sky.data = SkyUniformData::default();
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

    // --- Streamed scenes (Sponza + Worldgen): the SAME camera-following clipmap residency. ---
    // Both run the identical pipeline — `update` (reconcile the desired clipmap) → `drain_work_from(source)`
    // (bounded parallel sourcing + the shared edit overlay) → `take_dirty` → `pack_resident_set` — and differ
    // ONLY in their BrickSource: worldgen samples the procedural surface; Sponza reads the baked `.vox`
    // BrickMap (LOD0 extract / coarse downsample, all-air outside its bounds so the clipmap naturally bounds
    // the building). There is no bespoke static path anymore — Sponza supports LOD/clipmaps + editing exactly
    // like worldgen.
    //
    // On a switch INTO a streamed scene, reset the residency so the new set streams in cleanly and apply that
    // scene's lighting/sky presets (knobs-as-uniforms — the editor can still override afterward; we set them
    // only on the SWITCH so a later edit doesn't clobber a user's tweaks). For Sponza we lazily load + cache
    // the `.vox` here; a load failure falls back to a static Cornell pack so the engine never panics.
    if streaming.packed_scene != Some(*scene) {
        // On the Sponza switch, ensure the `.vox` is loaded + cached (once). A failure leaves the cache empty.
        if matches!(*scene, VoxelScene::Sponza) && streaming.sponza.is_none() {
            match load_vox(SPONZA_VOX_PATH) {
                Ok(loaded) => streaming.sponza = Some(loaded),
                Err(e) => error!(
                    "voxel-RT: could not load {SPONZA_VOX_PATH}: {e} — falling back to the Cornell box \
                     (bake Sponza via `cargo run --example voxelize_scene`)"
                ),
            }
        }
        // On the Gallery switch, load + MERGE the data-driven scene row ONCE + cache it (mirrors the Sponza
        // `.vox` cache). `load_gallery` never FAILS — absent assets are skipped with a warn — so this is always
        // `Some`; an empty merged map (no scene baked) is treated as "nothing to stream" below, exactly like a
        // missing Sponza asset (Cornell fallback), so the engine still renders + never panics.
        if matches!(*scene, VoxelScene::Gallery) && streaming.gallery.is_none() {
            streaming.gallery = Some(load_gallery(GALLERY_SCENES));
        }
        // A static `.vox`-backed scene whose merged/loaded map is MISSING or EMPTY (the asset(s) aren't baked):
        // pack a static Cornell box this frame so the engine still renders + never panics, latch
        // packed_scene = *scene (so we don't re-pack every frame), and bail out of the streaming path for this
        // scene until the asset exists / the scene changes. Sponza: `sponza == None` (load failed). Gallery:
        // the merge produced an empty map (no rows loaded).
        let static_map_missing = match *scene {
            VoxelScene::Sponza => streaming.sponza.is_none(),
            VoxelScene::Gallery => streaming.gallery.as_ref().is_none_or(|(map, _)| map.is_empty()),
            _ => false,
        };
        if static_map_missing {
            let map = build_cornell_with_edits(&streaming.cornell_registry, &edits);
            let patch = pack_brickmap(&map, &streaming.cornell_registry);
            patch_res.patch = patch;
            patch_res.generation = patch_res.generation.wrapping_add(1);
            lighting.data = LightingUniformData::cornell();
            sky.data = SkyUniformData::default();
            streaming.packed_scene = Some(*scene);
            streaming.packed_edit_gen = Some(edits.generation());
            return;
        }

        // Fresh residency for the new streamed scene (worldgen surface, the loaded Sponza map, or the merged
        // Gallery map).
        streaming.manager = ResidencyManager::new();
        // Fresh incremental packer for the new scene (sized to the resident cap) — a switch streams in cleanly,
        // and the slot allocator / arena / shadows start empty (mirrors the manager reset).
        streaming.packer = Some(super::incremental::ResidentPacker::new(streaming.cfg.max_resident_bricks as u32));
        streaming.last_cam_brick = None;
        streaming.worldgen_dirty_pending = false;
        streaming.worldgen_frames_since_pack = 0;
        streaming.packed_edit_gen = Some(edits.generation());
        streaming.packed_scene = Some(*scene);
        // Build the static scene's mip PYRAMID ONCE here, on the switch — NOT per-frame in the drain below
        // (that rebuilt the whole-building/whole-row downsample every streaming frame: the load-lag root
        // cause). It is owned, so it lives in the resource across frames; worldgen / a switch away frees it.
        // Each static scene gets its own source field (mutually exclusive — only the live scene's is Some).
        streaming.sponza_source = if matches!(*scene, VoxelScene::Sponza) {
            streaming.sponza.as_ref().map(|(map, _)| StaticVoxSource::new(map))
        } else {
            None
        };
        streaming.gallery_source = if matches!(*scene, VoxelScene::Gallery) {
            streaming.gallery.as_ref().map(|(map, _)| StaticVoxSource::new(map))
        } else {
            None
        };
        match *scene {
            VoxelScene::Sponza => {
                lighting.data = LightingUniformData::sponza();
                sky.data = SkyUniformData::sponza();
                info!("voxel-RT: switched to SPONZA scene — streaming the baked .vox through the clipmap");
            }
            VoxelScene::Gallery => {
                // The gallery is a row of baked scenes (Sponza et al.) under the same open-sky GI preset Sponza
                // uses — the row is a GI/LOD COMPARISON, so all scenes share one lighting/sky so differences read
                // as scene-geometry, not lighting. Knobs-as-uniforms; the editor overrides afterward.
                lighting.data = LightingUniformData::sponza();
                sky.data = SkyUniformData::sponza();
                info!("voxel-RT: switched to GALLERY scene — streaming the MERGED multi-.vox row through the clipmap");
            }
            _ => {
                lighting.data = LightingUniformData::worldgen();
                sky.data = SkyUniformData::worldgen();
                info!("voxel-RT: switched to WORLDGEN scene — applied worldgen sun + directional sky presets");
            }
        }
    }

    // LATCHED missing-static-asset guard (never-panic invariant): on the SWITCH frame the block above packs the
    // Cornell fallback + latches `packed_scene = *scene` + returns — but on EVERY subsequent frame that switch
    // block is skipped (`packed_scene == *scene`), so without this guard execution would fall through to the
    // static streaming arm below and hit `.expect(...)` on a `None` source → panic. When a `.vox`-backed scene
    // is selected but its map never loaded / merged empty (the asset(s) are absent), there is nothing to
    // stream: the Cornell fallback is already packed and stays valid, so just bail every frame until the asset
    // exists / the scene changes. Worldgen is unaffected (it has no static-map dependency).
    let static_map_missing = match *scene {
        VoxelScene::Sponza => streaming.sponza.is_none(),
        VoxelScene::Gallery => streaming.gallery.as_ref().is_none_or(|(map, _)| map.is_empty()),
        _ => false,
    };
    if static_map_missing {
        return;
    }

    let Ok(cam_tf) = cam.single() else {
        return; // camera not spawned yet — try next frame
    };
    let cam_world: [f32; 3] = cam_tf.translation().into();
    // The LOD0 brick the camera sits in — the FINEST clipmap boundary, so it crosses whenever ANY level's
    // shell could shift (a coarse boundary is `2^L×` farther apart, so a LOD0 crossing strictly implies it).
    let cam_brick = camera_brick_coord(cam_world);

    // An EDIT (build/destroy) re-queues exactly the affected resident bricks so the change re-sources +
    // re-packs LOCALLY (it ADAPTS — the resident set, GI reservoirs, and world cache all stay; never a full
    // clear, see [[feedback-gi-adapt-not-reset]]). Detected by the delta generation changing since the last
    // pack. Works for EVERY streamed scene through the shared `apply_edit_overlay` in `drain_work_from`.
    let edits_changed = streaming.packed_edit_gen != Some(edits.generation());
    if edits_changed {
        let dirty = affected_resident_keys(&edits);
        // Mark the affected bricks REWRITTEN in the incremental packer too, so the edit re-packs exactly those
        // bricks + their 26-neighbourhood next re-pack (the edit/dig path stays incremental — it ADAPTS, never
        // a full clear). The manager re-sources them; the packer re-`pack_one`s them.
        if let Some(packer) = streaming.packer.as_mut() {
            packer.mark_rewritten(dirty.iter().copied());
        }
        streaming.manager.requeue_keys(dirty);
        streaming.packed_edit_gen = Some(edits.generation());
    }

    // Bounded SOURCING of queued bricks from this scene's BrickSource + the shared edit overlay. Split-borrow
    // the streaming fields so the static `.vox` map (`sponza`) can back the source while the manager drains.
    // The packing PALETTE/registry is scene-specific: worldgen bricks index the worldgen registry; Sponza
    // bricks index the `.vox`'s OWN palette (loaded alongside its map), so both the drain (face-exposure /
    // light gather) and the pack below use that registry — never the worldgen one — or the colours would be
    // wrong.
    let scene_now = *scene;
    let VoxelRtStreaming {
        manager,
        cfg,
        layer,
        lib,
        registry,
        seed,
        sponza,
        sponza_source,
        gallery,
        gallery_source,
        worldgen_dirty_pending,
        worldgen_frames_since_pack,
        last_cam_brick,
        packer,
        ..
    } = &mut *streaming;

    // The scene's BrickSource — built ONCE here so it backs BOTH `update`'s classify FILTER (the
    // SURFACE-FOLLOWING RESIDENCY prune) and the drain's sourcing. Worldgen wraps the procedural surface (a
    // height-based classify that prunes the deep underground + high sky); the static scenes reuse the CACHED
    // pyramid source (default-Surface classify — the wholly-outside reject already bounds them). The registry
    // whose palette this scene's bricks index travels alongside (drain + pack must agree on it).
    let worldgen_source = WorldgenSource::new(layer, lib, *seed);
    let (source, active_registry): (&dyn BrickSource, &BlockRegistry) = match scene_now {
        VoxelScene::Sponza => {
            // Map + palette + the prebuilt source are ready by here (the source's pyramid was built ONCE on the
            // switch; the missing-asset case returned above). Reuse the CACHED source — never rebuild the
            // pyramid per frame (that was the load-lag root cause).
            let (_, vox_registry) = sponza.as_ref().expect("sponza map loaded before streaming");
            let src = sponza_source.as_ref().expect("sponza source built on the switch");
            (src, vox_registry)
        }
        VoxelScene::Gallery => {
            // The MERGED multi-scene map + its concatenated registry + the prebuilt source are ready by here
            // (the source's pyramid over the whole row was built ONCE on the switch; the empty-merge case
            // returned above). Reuse the CACHED source — never re-merge / re-downsample per frame.
            let (_, vox_registry) = gallery.as_ref().expect("gallery map merged before streaming");
            let src = gallery_source.as_ref().expect("gallery source built on the switch");
            (src, vox_registry)
        }
        _ => (&worldgen_source, registry),
    };

    // Reconcile only when the camera crosses into a new LOD0 brick (a shell could shift), an edit re-queued
    // bricks, OR there is still pending work to drain. This avoids recomputing the clipmap every idle frame.
    // The per-move enqueue/drop is O(shell) — only the LOD0 face-slab shifts on a small move; coarse shells
    // are unchanged. `update` takes the source so its classify FILTER can prune non-surface bricks at enqueue.
    let cam_changed = *last_cam_brick != Some(cam_brick);
    if cam_changed {
        let dropped = manager.update(cam_world, cfg, source);
        *last_cam_brick = Some(cam_brick);
        if dropped > 0 {
            debug!("voxel streaming: dropped {dropped} bricks left behind by camera move");
        }
    } else if manager.pending() == 0 {
        return; // nothing to do this frame
    }

    manager.drain_work_from(cfg, source, active_registry, &edits);

    // AMORTIZE the O(resident) re-pack (pack_resident_set ~60 ms + the full BLAS rebuild): accumulate "resident
    // set changed" and pack only on a SETTLE (queue drained) OR every WORLDGEN_REPACK_INTERVAL frames during a
    // long stream — NOT on every dirty drain (which made each streaming frame pay the full O(resident) pack +
    // rebuild). Terrain still reveals progressively (keep-old-until-revealed); the per-frame cost while
    // streaming drops to just the bounded sourcing drain. `take_dirty` is consumed every frame so the dirty
    // flag is never lost; we OR it into the accumulator.
    if manager.take_dirty() {
        *worldgen_dirty_pending = true;
    }
    *worldgen_frames_since_pack = worldgen_frames_since_pack.saturating_add(1);
    let settled = manager.pending() == 0;
    if *worldgen_dirty_pending && (settled || *worldgen_frames_since_pack >= WORLDGEN_REPACK_INTERVAL) {
        let entries = manager.resident_entries();
        // INCREMENTAL re-pack: `update` re-`pack_one`s ONLY the entered/dropped bricks + their resident
        // 26-neighbourhood (the O(changed) cost), then `snapshot_patch` assembles the extracted patch by MEMCPY
        // of the cached per-brick bytes — byte-identical to a from-scratch `pack_resident_set` (proven by
        // `snapshot_patch_matches_full_pack`), so the render path + shader are UNCHANGED + pixel-identical. The
        // packer is `Some` for every streamed scene (set on the switch); the `unwrap_or_else` is a defensive
        // first-tick guard that falls back to the full pack.
        let patch = match packer.as_mut() {
            Some(p) => {
                p.update(&entries);
                p.snapshot_patch(active_registry)
            }
            None => pack_resident_set(&entries, active_registry),
        };
        let (n, v) = (patch.brick_count(), patch.voxels.len());
        patch_res.patch = patch;
        patch_res.generation = patch_res.generation.wrapping_add(1);
        *worldgen_dirty_pending = false;
        *worldgen_frames_since_pack = 0;
        debug!(
            "voxel-RT: re-packed resident set gen {} — {n} bricks, {v} cells, {} pending (settled={settled})",
            patch_res.generation,
            manager.pending()
        );
    }
}

/// The resident clipmap keys an edit-delta change INVALIDATES: for each overridden world voxel, the LOD0
/// brick that OWNS it plus its LOD0 halo-neighbour bricks (so a boundary edit refreshes the neighbour's stale
/// halo). Bricks not yet resident are still re-queued (a place into empty space appears). The SSOT mapping
/// "edit → bricks to re-source", shared by every streamed scene (it consults only the [`VoxelEdits`] delta).
///
/// LOD0 ONLY, by design: the edit overlay ([`apply_edit_overlay`], applied in
/// [`ResidencyManager::drain_work_from`]) is keyed on the LOD0 world-voxel grid and is SKIPPED for any
/// `lod > 0` brick — a `0.2 m` LOD0 edit does not align with a coarse brick's wider cells, so a coarse brick
/// cannot carry it. Re-queuing coarse bricks here was therefore DEAD WORK: they would be re-sourced (paying
/// the coarse extract cost) and come back byte-identical, never reflecting the edit. So we re-queue ONLY the
/// LOD0 owner + halo neighbours. CONSEQUENCE (stated honestly): once the camera retreats far enough that the
/// edited region falls into a COARSE clipmap shell, the edit is no longer visible — the coarse brick shows the
/// un-edited baked/sourced geometry. This is acceptable: edits happen near the camera (where the region is
/// LOD0); a coarse-aware edit mip (folding the LOD0 override down the pyramid) is a future item.
fn affected_resident_keys(edits: &VoxelEdits) -> FxHashSet<BrickKey> {
    use super::edits::dirty_bricks_for_edit;
    let mut keys: FxHashSet<BrickKey> = FxHashSet::default();
    for (wv, _block) in edits.iter() {
        // LOD0 owner + halo neighbours (the bricks whose stored voxels / halo read this edited voxel). The
        // edit grid IS the LOD0 grid, so these are the only bricks the overlay can change.
        for bc in dirty_bricks_for_edit(wv) {
            keys.insert(BrickKey { coord: bc, lod: 0 });
        }
    }
    keys
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
        Self { enabled: true, mode: bevy::anti_alias::dlss::DlssPerfQualityMode::Quality }
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
                // Never reset RR on a terrain edit (or a camera move) — RR reprojects via motion and the
                // ReSTIR reservoirs adapt locally, so the GI smoothly follows edits instead of full-clearing.
                d.reset = false;
            }
            (true, None) => {
                commands.entity(cam).insert(Dlss::<DlssRayReconstructionFeature> {
                    perf_quality_mode: settings.mode,
                    reset: true, // clean start only when RR is first attached
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

/// Number of entries in the world-space radiance-cache hash table (Phase 2.1). MUST be a power of two in the
/// range 2^10..=2^20 (the prefix-sum compaction is a two-level scan over 1024-wide blocks, so 2^20 is the
/// natural ceiling — one block-scan workgroup covers up to 1024 blocks). The live render path uses this full
/// size; the headless test shrinks it via [`voxel_raytrace_shader_src`] so the cache buffers + compaction
/// stay small + fast. Structural (resolution-independent) — the cache is WORLD-space, allocated ONCE.
pub const WORLD_CACHE_SIZE: u32 = 1 << 20;

/// **SSOT loader for `voxel_raytrace.wgsl`.** The world-cache section is parameterised by the hash-table size
/// via the `#{WORLD_CACHE_SIZE}` token (so the headless test can use a tiny table); every shader-load site —
/// the live pipelines here AND every GPU test — MUST go through this so the token is substituted before naga
/// sees it (raw `read_to_string` would feed naga an un-substituted `#{...}` and fail to parse). `size` MUST be
/// a power of two in `[2^10, 2^20]`.
pub fn voxel_raytrace_shader_src(size: u32) -> String {
    debug_assert!(size.is_power_of_two() && (1024..=WORLD_CACHE_SIZE).contains(&size));
    let src = std::fs::read_to_string("assets/shaders/voxel_raytrace.wgsl")
        .expect("read voxel_raytrace.wgsl");
    src.replace("#{WORLD_CACHE_SIZE}", &size.to_string())
}

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
    /// The `raymarch` compute pipeline (legacy `gather_gi` GI). Dispatched when `RestirSettings.restir` is
    /// off — the `gi_mode` A/B toggle (legacy vs ReSTIR in one build).
    raymarch: wgpu::ComputePipeline,
    /// Two-pass ReSTIR (non-DLSS). Pass 1 (`restir_p1`) = initial RIS + temporal → `reservoirs_b` + surface;
    /// pass 2 (`restir_p2`) = same-frame spatial from `reservoirs_b` → `reservoirs_a` + shade → out_tex. Both
    /// share `restir_pl`; dispatched back-to-back in one compute pass (the intra-pass storage barrier orders
    /// pass-1-writes-b before pass-2-reads-b). The live GI path.
    restir_p1: wgpu::ComputePipeline,
    restir_p2: wgpu::ComputePipeline,
    /// `group(3)` (Phase 2.1 world-cache): the cache uniform (`wc`) + the 11 persistent cache storage buffers.
    /// A DEDICATED bind group (not group 2) so `restir_p1` is never forced to bind all of them — in 2.2 the
    /// reservoir path will add this group ALONGSIDE group 2, not merge into it.
    world_cache_layout: wgpu::BindGroupLayout,
    /// The cache passes' minimal `group(1)` layout ({2: light, 11: sky}) + the `group(2)` indirect-dispatch
    /// layout ({0: dispatch buffer}). Stored so the per-frame cache dispatch can build the matching bind
    /// groups (compaction layout is positional: scene(0), view(1), dispatch(2), cache(3); update/blend omit 2).
    world_cache_view_layout: wgpu::BindGroupLayout,
    world_cache_dispatch_layout: wgpu::BindGroupLayout,
    /// The six world-cache compute pipelines, dispatched IN THIS ORDER each frame BEFORE `restir_p1`:
    /// decay → compact_single_block → compact_blocks → compact_write_active → update (indirect) →
    /// blend (indirect). All share one pipeline layout `[scene(0), view(1), <empty>(2), cache(3)]`.
    wc_decay: wgpu::ComputePipeline,
    wc_compact_single_block: wgpu::ComputePipeline,
    wc_compact_blocks: wgpu::ComputePipeline,
    wc_compact_write_active: wgpu::ComputePipeline,
    wc_update: wgpu::ComputePipeline,
    wc_blend: wgpu::ComputePipeline,
    /// The composite shader module + its bind-group layout + sampler. The composite render pipeline is
    /// built lazily (and cached) once the live view-target format is known.
    composite_module: wgpu::ShaderModule,
    composite_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    /// DLSS-RR (Stage 4c): the `raymarch_dlss` compute pipeline (writes the full lit colour + the 5 guide
    /// storage textures) + its `group(1)` view layout, and the resolve render pass's bind-group layout
    /// (samples the colour/depth/motion storage textures → view target + prepass depth/motion). The resolve
    /// render pipeline itself is built lazily (format-keyed) in the pass.
    /// Legacy DLSS guide-writing pass (`gather_gi` GI). Dispatched when `RestirSettings.restir` is off (A/B).
    #[cfg(feature = "dlss")]
    raymarch_dlss: wgpu::ComputePipeline,
    /// Two-pass ReSTIR (DLSS). `restir_dlss_p1` = initial RIS + reprojected temporal → `reservoirs_b` +
    /// surface (no guides); `restir_dlss_p2` = same-frame spatial → `reservoirs_a` + shade → out_tex + the 5
    /// DLSS-RR guides. Both share the DLSS restir pipeline layout; dispatched back-to-back in one pass.
    #[cfg(feature = "dlss")]
    restir_dlss_p1: wgpu::ComputePipeline,
    #[cfg(feature = "dlss")]
    restir_dlss_p2: wgpu::ComputePipeline,
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
    /// ReSTIR per-pixel reservoir storage buffers (a, b) + the size they were allocated for. With the two-pass
    /// split these are FIXED-ROLE (NOT ping-ponged): `a` (binding 0) = the FINAL/history pool (pass 1's
    /// temporal tap reads last frame's final; pass 2 writes this frame's final); `b` (binding 1) = the
    /// intermediate POST-TEMPORAL pool (pass 1 writes; pass 2's same-frame spatial reads). Reallocated on view
    /// resize; contents discarded via the `reset` flag (camera move / resize). Used by both ReSTIR paths.
    reservoirs: Option<(wgpu::Buffer, wgpu::Buffer, UVec2)>,
    /// Per-pixel RECEIVER surface (world pos + normal) buffers (cur/prev). These DO ping-pong by frame parity:
    /// pass 1 writes `cur` (this frame) + reads `prev` (last frame) for the temporal Jacobian + dissimilarity
    /// reject; pass 2 reads `cur` (same-frame) for the spatial neighbour. Non-DLSS path.
    surfaces: Option<(wgpu::Buffer, wgpu::Buffer, UVec2)>,
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
    /// changed) resets the HISTORY-TEXTURE accumulation (not the reservoirs). `None` until the first frame.
    prev_view_proj: Option<[[f32; 4]; 4]>,
    /// The scene patch generation the accumulator is valid for; a re-pack (geometry changed) resets it.
    accum_generation: Option<u64>,
    /// Previous-frame UN-jittered `clip_from_world` for the non-DLSS ReSTIR temporal reprojection (fed to the
    /// shader as `camera.prev_clip_from_world`; mirrors `dlss_prev_clip_from_world`). The non-DLSS path is not
    /// jittered, so the current frame's `clip_from_world` IS its un-jittered clip. `None` on the first frame
    /// (then `prev == cur`, so the reprojection returns the current pixel).
    prev_clip_from_world: Option<[[f32; 4]; 4]>,
    /// `(viewport, built_generation)` at the last non-DLSS frame — drives the ReSTIR `reset` flag. Reset fires
    /// ONLY on the first frame or a viewport (resolution) change; camera motion is handled by motion-vector
    /// reprojection and an edit ADAPTS locally (never full-clears the reservoirs). `None` until the first frame.
    restir_prev: Option<(UVec2, Option<u64>)>,

    // --- Phase 2.1 world-space radiance cache (PERSISTENT; allocated ONCE, never realloc'd on resize) ---
    /// The 11 persistent cache buffers + the bind group over them. The cache is WORLD-space /
    /// resolution-independent, so this is built ONCE on the first frame (zero-initialised → every cell starts
    /// empty) and is NEVER reallocated on a view resize and NEVER cleared on a terrain edit (stale cells decay
    /// + re-fill locally; [[feedback-gi-adapt-not-reset]]). `None` until the first frame allocates it.
    world_cache: Option<WorldCacheBuffers>,
    /// `false` until the first cache dispatch has run, so the `reset` flag (blend overwrites instead of
    /// accumulating) fires exactly ONCE — on the first frame after allocation — and never again (no clear on
    /// edit / camera move).
    world_cache_initialized: bool,

    /// Phase 2.5 NEE: the emissive-voxel LIGHT LIST + its power-weighted alias table as GPU storage buffers,
    /// plus the live light COUNT and the patch generation they were built for. Rebuilt (in `prepare_voxel_rt`)
    /// whenever the patch generation changes — the lights are derived from the resident set, so they follow the
    /// streamed/edited geometry. `count == 0` ⇒ no emitters ⇒ NEE is skipped (the buffers are 1-long dummies so
    /// the bind group is always valid; the shader never indexes them). `None` until the first non-empty patch.
    world_cache_lights: Option<WorldCacheLights>,

    /// Phase 2.4 GPU per-pass timing (editor only): the persistent timestamp QuerySet + resolve/read-back
    /// buffers. `None` until the first timed frame allocates it — OR permanently `None` if the device lacks the
    /// TIMESTAMP features (then per-pass timing is reported unavailable once and every pass is dispatched
    /// untimed, zero behaviour change). `gpu_timer_checked` gates the one-time allocation attempt + the
    /// unavailable log so we don't retry/spam every frame.
    #[cfg(feature = "editor")]
    gpu_timer: Option<WorldCacheGpuTimer>,
    #[cfg(feature = "editor")]
    gpu_timer_checked: bool,

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
    /// DLSS-path per-pixel surface buffers (cur/prev), sized to the full render res (like dlss_reservoirs).
    #[cfg(feature = "dlss")]
    dlss_surfaces: Option<(wgpu::Buffer, wgpu::Buffer, UVec2)>,
    /// (render_res, clip_from_world, built_generation) at the last DLSS frame — drives the ReSTIR `reset`
    /// (a camera move, a resolution change, or a geometry re-pack invalidates the same-pixel reservoirs).
    #[cfg(feature = "dlss")]
    dlss_restir_prev: Option<(UVec2, [[f32; 4]; 4], Option<u64>)>,
}

/// Holders that must outlive the scene bind group / TLAS for the program's lifetime.
struct SceneKeepAlive {
    _blas: wgpu::Blas,
    _tlas: wgpu::Tlas,
    _buffers: [wgpu::Buffer; 5],
}

/// The PERSISTENT world-space radiance-cache GPU state (Phase 2.1): the 11 storage buffers + the `group(3)`
/// bind group over them (+ a re-uploaded `wc` uniform each frame). Allocated ONCE (zero-initialised so every
/// cell starts empty), resolution-independent, never realloc'd on resize, never globally cleared on an edit.
/// The buffers are retained here so they outlive the bind group for the program's lifetime.
struct WorldCacheBuffers {
    checksums: wgpu::Buffer,
    life: wgpu::Buffer,
    radiance: wgpu::Buffer,
    geometry: wgpu::Buffer,
    luminance_deltas: wgpu::Buffer,
    new_radiance: wgpu::Buffer,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    active_cell_indices: wgpu::Buffer,
    active_cells_count: wgpu::Buffer,
    /// INDIRECT|STORAGE — the update + blend passes dispatch indirect over this.
    active_cells_dispatch: wgpu::Buffer,
}

/// Phase 2.5 NEE: the emissive-voxel LIGHT LIST GPU buffers (light array + power-weighted alias table) + the
/// live light count + the patch generation they were built for. Unlike the persistent world-cache buffers
/// (allocated once, world-space), these are REBUILT whenever the patch generation changes — the lights are
/// derived from the resident set, so they track the streamed/edited geometry. When the scene has NO emitters
/// the buffers are 1-long dummies (count 0) so the cache bind group is always valid; the shader skips NEE.
struct WorldCacheLights {
    lights: wgpu::Buffer,
    alias: wgpu::Buffer,
    /// Number of lights (== `patch.lights.len()`); stamped into `WorldCacheUniform.light_count`. 0 ⇒ NEE off.
    count: u32,
}

impl WorldCacheLights {
    /// Build the light-list buffers from a packed patch's `lights` + `alias`. An EMPTY list (no emitters) is
    /// uploaded as a single zeroed dummy element each so the storage bindings are never zero-length (wgpu
    /// rejects a 0-byte storage binding) — `count == 0` makes the shader skip NEE so the dummies are never read.
    fn new(device: &wgpu::Device, patch: &GpuBrickPatch) -> Self {
        let count = patch.lights.len() as u32;
        // Non-empty contents (≥1 element) so the storage buffers are valid even with no emitters.
        let dummy_light = GpuVoxelLight { pos: [0.0; 3], area: 0.0, radiance: [0.0; 3], inv_pdf: 0.0 };
        let dummy_alias = GpuAliasEntry { prob: 1.0, alias: 0 };
        let lights_bytes: Vec<u8> = if patch.lights.is_empty() {
            bytemuck::bytes_of(&dummy_light).to_vec()
        } else {
            bytemuck::cast_slice(&patch.lights).to_vec()
        };
        let alias_bytes: Vec<u8> = if patch.alias.is_empty() {
            bytemuck::bytes_of(&dummy_alias).to_vec()
        } else {
            bytemuck::cast_slice(&patch.alias).to_vec()
        };
        let lights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel_rt_nee_lights"),
            contents: &lights_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let alias = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel_rt_nee_alias"),
            contents: &alias_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self { lights, alias, count }
    }
}

impl WorldCacheBuffers {
    /// Allocate the persistent cache buffers, ZERO-INITIALISED (`mapped_at_creation` zero-fill via wgpu's
    /// default-zeroed mapping is not guaranteed, so we create them un-mapped — wgpu zeroes new buffers — and
    /// rely on that; `checksums == 0` ⇒ every cell empty, `life == 0` ⇒ inactive). `size` = `WORLD_CACHE_SIZE`.
    fn new(device: &wgpu::Device, size: u32) -> Self {
        let n = size as u64;
        let mk = |label: &str, bytes: u64, indirect: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE;
            if indirect {
                usage |= wgpu::BufferUsages::INDIRECT;
            }
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                // wgpu guarantees a freshly-created buffer is zero-initialised on first use, so every cell
                // starts empty (checksum 0, life 0) with no explicit clear.
                mapped_at_creation: false,
            })
        };
        Self {
            checksums: mk("voxel_rt_wc_checksums", n * 4, false),
            life: mk("voxel_rt_wc_life", n * 4, false),
            radiance: mk("voxel_rt_wc_radiance", n * 16, false),
            geometry: mk("voxel_rt_wc_geometry", n * 32, false),
            luminance_deltas: mk("voxel_rt_wc_luminance_deltas", n * 4, false),
            new_radiance: mk("voxel_rt_wc_new_radiance", n * 16, false),
            a: mk("voxel_rt_wc_a", n * 4, false),
            b: mk("voxel_rt_wc_b", 1024 * 4, false),
            active_cell_indices: mk("voxel_rt_wc_active_cell_indices", n * 4, false),
            active_cells_count: mk("voxel_rt_wc_active_cells_count", 4, false),
            active_cells_dispatch: mk("voxel_rt_wc_active_cells_dispatch", 12, true),
        }
    }

    /// Build the `group(3)` cache bind group: binding 0 = the per-frame `wc` uniform, bindings 1..=10 = the
    /// 10 persistent storage buffers, bindings 15/16 = the Phase-2.5 NEE light list + alias table (the
    /// indirect-dispatch buffer is in its own group 2 — see `dispatch_bg`). The light buffers are rebuilt per
    /// patch generation (`WorldCacheLights`); they're always ≥1-long (dummy when no emitters), so the binding
    /// is valid even with NEE effectively off.
    fn bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        wc_uniform: &wgpu::Buffer,
        lights: &WorldCacheLights,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel_rt_world_cache_bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wc_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.checksums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.life.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.radiance.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.geometry.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.luminance_deltas.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.new_radiance.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.active_cell_indices.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.active_cells_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: lights.lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: lights.alias.as_entire_binding() },
            ],
        })
    }

    /// Build the `group(2)` indirect-dispatch bind group ({0: the dispatch-args buffer}), used ONLY by the
    /// decay/compaction passes — unbound before the update/blend indirect dispatches consume it.
    fn dispatch_bg(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel_rt_world_cache_dispatch_bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.active_cells_dispatch.as_entire_binding(),
            }],
        })
    }
}

// --- Phase 2.4 GPU per-pass timing (issue #123, editor builds only) ----------------------------------

/// The world-cache + ReSTIR passes this timer measures, in dispatch order. Each consumes TWO timestamp slots
/// (begin, end). The compaction is three tiny back-to-back passes; we bracket them as ONE "compact" segment
/// (their individual times are sub-µs noise). Index here == the segment's position in the QuerySet.
#[cfg(feature = "editor")]
const WC_TIMING_LABELS: [&str; 6] = [
    "world_cache: decay",
    "world_cache: compact",
    "world_cache: update",
    "world_cache: blend",
    "restir: p1",
    "restir: p2",
];

/// One begin+end timestamp pair per measured segment.
#[cfg(feature = "editor")]
const WC_TIMING_SLOTS: u32 = (WC_TIMING_LABELS.len() as u32) * 2;

/// **GPU per-pass timing for the world-cache + ReSTIR passes** (editor builds; issue #123). A persistent
/// timestamp `QuerySet` (one begin/end pair per segment) + a resolve buffer (`resolve_query_set` target,
/// `QUERY_RESOLVE | COPY_SRC`) + a `MAP_READ` read-back buffer. The flow per frame, all on the pass's existing
/// command encoder (so it adds NO extra submit): (a) read back the PREVIOUS frame's resolved timestamps (the
/// buffer was submitted + the device polled by Bevy on the prior frame, so the map is ready — 1-frame latency,
/// which is fine for a diagnostic); (b) convert ticks → ms via `queue.get_timestamp_period()` and push each
/// segment into the global `instrument` GPU sink (the editor profiler drains it into the Performance panel's
/// GPU stack + a debug log); (c) record THIS frame's begin/end timestamps via `cpass.write_timestamp` between
/// the pipeline switches (needs `TIMESTAMP_QUERY_INSIDE_PASSES`), `resolve_query_set` into the resolve buffer,
/// and copy it into the read-back buffer to be mapped next frame.
///
/// Built only when the device exposes the timestamp features; otherwise the whole timer is `None` and the
/// passes dispatch untimed (zero behaviour change). NEVER touches the cache data — it only reads the clock.
#[cfg(feature = "editor")]
struct WorldCacheGpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    /// `true` once a frame's timestamps have been written (so the FIRST read-back, before any write, is skipped
    /// rather than reading an unmapped/garbage buffer).
    have_pending: bool,
}

#[cfg(feature = "editor")]
impl WorldCacheGpuTimer {
    /// Allocate the timer iff the device supports timestamp queries INSIDE compute passes. Returns `None`
    /// (timing unavailable, reported once by the caller) otherwise — the passes then run untimed.
    fn new(device: &wgpu::Device) -> Option<Self> {
        let feats = device.features();
        if !feats.contains(wgpu::Features::TIMESTAMP_QUERY)
            || !feats.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("voxel_rt_wc_timing_qs"),
            ty: wgpu::QueryType::Timestamp,
            count: WC_TIMING_SLOTS,
        });
        let bytes = u64::from(WC_TIMING_SLOTS) * 8; // u64 ticks per slot
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_rt_wc_timing_resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel_rt_wc_timing_readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self { query_set, resolve_buf, readback_buf, have_pending: false })
    }

    /// Read back the PREVIOUS frame's resolved timestamps (if any), convert to ms via `period` (ns/tick), and
    /// push each segment into the `instrument` GPU sink. Must run BEFORE re-recording this frame's queries (it
    /// leaves the read-back buffer UNMAPPED so the encoder can copy into it again). 1-frame latency.
    ///
    /// The buffer was submitted + presented last frame, so its GPU writes are done; a NON-BLOCKING `Poll` fires
    /// the map callback immediately without stalling the render thread. If the map somehow isn't ready yet (a
    /// stalled frame), we just unmap + skip this read — next frame picks it up. NEVER waits on the GPU.
    fn read_previous(&mut self, device: &wgpu::Device, period: f32) {
        if !self.have_pending {
            return;
        }
        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        // Non-blocking poll: completes the map for work that already finished (last frame) without waiting.
        let _ = device.poll(wgpu::PollType::Poll);
        if let Ok(view) = slice.get_mapped_range() {
            let ticks: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&view).to_vec();
            drop(view);
            for (i, label) in WC_TIMING_LABELS.iter().enumerate() {
                let begin = ticks.get(i * 2).copied().unwrap_or(0);
                let end = ticks.get(i * 2 + 1).copied().unwrap_or(0);
                // ticks increase within a frame; guard against a zeroed/garbage pair (skip if end < begin).
                if end >= begin {
                    let ms = (end - begin) as f32 * period * 1e-6;
                    crate::instrument::record_gpu(label, ms);
                }
            }
        }
        // Always unmap so the encoder can copy into the read-back buffer again this frame.
        self.readback_buf.unmap();
    }

    /// Write `segment`'s BEGIN timestamp on the open compute pass (call right before `set_pipeline`+dispatch).
    fn begin(&self, cpass: &mut wgpu::ComputePass, segment: usize) {
        cpass.write_timestamp(&self.query_set, segment as u32 * 2);
    }

    /// Write `segment`'s END timestamp on the open compute pass (call right after the dispatch).
    fn end(&self, cpass: &mut wgpu::ComputePass, segment: usize) {
        cpass.write_timestamp(&self.query_set, segment as u32 * 2 + 1);
    }

    /// After the compute pass closes: resolve the QuerySet into the resolve buffer and copy it into the
    /// MAP_READ read-back buffer (mapped + read NEXT frame). Marks a pending read-back.
    fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..WC_TIMING_SLOTS, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.readback_buf,
            0,
            u64::from(WC_TIMING_SLOTS) * 8,
        );
        self.have_pending = true;
    }
}

/// Camera uniform mirroring the WGSL `CameraUniform` (group 1, binding 0): `world_from_clip` (64) +
/// `cam_pos` (12) + `t_max` (4) + `viewport` (8) + `accum_weight` (4) + pad (4) + `prev_clip_from_world` (64)
/// = 160 bytes.
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
    /// Previous-frame UN-jittered `clip_from_world`, for the non-DLSS ReSTIR temporal reprojection
    /// (`reproject_pixel`). The non-DLSS path is not jittered, so the current frame's `clip_from_world` IS its
    /// un-jittered clip; we store it each frame and feed last frame's here. On the first frame `prev == cur`
    /// (so `reproject_pixel` returns the current pixel). The DLSS path fills this for layout parity but ignores
    /// it (it reprojects via `dlss_cam.motion_prev`).
    prev_clip_from_world: [[f32; 4]; 4],
}

/// **SSOT for the direct-lighting knobs** (the WGSL `LightingUniform`, group 1 binding 2). All values are
/// runtime UNIFORMS (knobs-as-uniforms mandate) — the GUI/editor can drive any of them; nothing here is a
/// shader const. 80 bytes (std140-safe: each `Vec3` is followed by a scalar to fill its 16-byte slot; the
/// GI knobs form a packed 16-byte row; the final row is `emissive_strength, frame_index, debug_view, _pad`
/// — exactly 16 bytes, no trailing pad). Mirrored field-for-field by both the WGSL shader
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
    /// Was `gi_firefly_clamp` (a biased per-bounce-sample radiance cap), discarded in Phase 2.2 as best
    /// practice — fireflies are now handled correctly by ReSTIR resampling + the world-cache temporal
    /// averaging + DLSS-RR, with no biased clamp anywhere. Kept as a pad so the struct stays EXACTLY 80 bytes
    /// (same offsets, no UBO re-layout).
    pub _pad: f32,
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
            // Open-world default: a bounce reaches far enough to hit distant terrain (and otherwise returns the
            // procedural sky), so GI is plausible outside a closed box. `cornell()` keeps its own tuned value,
            // and the GPU tests pin their own `gi_bounce_dist`, so closed-scene tests are unaffected.
            gi_bounce_dist: 64.0,
            emissive_strength: 4.0,
            frame_index: 0,
            debug_view: 0,
            _pad: 0.0, // was gi_firefly_clamp (discarded in 2.2 — best practice; no biased clamp anywhere)
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
            _pad: 0.0, // firefly clamping discarded in 2.2 (best practice) — ReSTIR + cache + DLSS-RR handle fireflies
        }
    }

    /// Lighting tuned for the LARGE streamed WORLDGEN terrain (the Phase-2.6 GI showcase + stress scene).
    /// Unlike the closed Cornell box, this is an OPEN world: a strong, crisp directional SUN (hard sun
    /// shadows shaping the mountains + deep valleys) plus the directional sky (set via [`SkyUniformData::
    /// worldgen`]) that the GI bounce reads when a ray escapes the resident clipmap — so open slopes are
    /// SKY-LIT and shadowed valleys fill from multi-bounce + the emissive lava/crystal. A modest neutral
    /// ambient keeps deep shadow off pure black before GI converges; a longer `gi_bounce_dist` so a bounce
    /// can reach the far walls of a wide valley. All runtime uniforms (knobs-as-uniforms) — an editor panel
    /// still overrides any of them.
    pub fn worldgen() -> Self {
        // A high afternoon sun coming down from +X/+Z (so peaks cast long valley shadows). Direction the
        // light TRAVELS (away from the sun); Lambert uses `-sun_direction`.
        let sun = Vec3::new(-0.45, -0.78, -0.42).normalize();
        Self {
            sun_direction: sun.into(),
            sun_intensity: 3.2,
            sun_color: [1.0, 0.95, 0.85],
            shadow_bias: 0.06,
            ambient_color: [0.06, 0.08, 0.11],
            ao_radius: 1.5,
            ao_samples: 4,
            gi_rays: 8,
            gi_intensity: 1.0,
            gi_bounce_dist: 96.0,
            emissive_strength: 4.0,
            frame_index: 0,
            debug_view: 0,
            _pad: 0.0,
        }
    }

    /// Lighting tuned for the baked SPONZA atrium — the GI-MEASUREMENT scene. Sponza is a half-open building
    /// (a colonnaded courtyard with an open roof + open ends), so the classic key light is a strong, warm
    /// directional SUN raking STEEPLY DOWN through the open roof to strike the floor + lower colonnade. That
    /// directly-lit floor and the coloured drapes then bounce a MEASURABLE indirect signal up into the shaded
    /// arcades (single-bounce floor→arch fill + multi-bounce colour bleed off the red/green hangings) — the
    /// whole reason this scene exists. The values are picked for a strong, clearly-attributable GI signal:
    /// a bright sun (direct lighting that drives a strong first bounce), plenty of GI rays for low-variance
    /// colour bleed, a long-enough `gi_bounce_dist` to cross the ~30 m nave so a bounce off one wall reaches
    /// the far colonnade, and only a faint neutral ambient (so the GI — not a flat fill — is what lights the
    /// shadowed arcades). The sky ([`SkyUniformData::sponza`]) supplies the open-roof ambient + sky-GI a
    /// bounce escaping upward returns. All runtime uniforms (knobs-as-uniforms) — the editor overrides any.
    pub fn sponza() -> Self {
        // NOTE the baked Khronos Sponza is a mostly-ROOFED interior — the sun reaches almost none of the floor,
        // so the dominant light is SKY-GI entering through the open nave + the colonnade arches (see
        // `SkyUniformData::sponza`, with a boosted `gi_sky_intensity`). The sun is a strong KEY on the nave +
        // exposed upper geometry it DOES reach. Direction the light travels (away from the sun); Lambert uses
        // `-sun_direction`; steep down the nave. The editor sliders re-aim/re-weight it for your measurement.
        let sun = Vec3::new(0.32, -0.90, -0.28).normalize();
        Self {
            sun_direction: sun.into(),
            sun_intensity: 2.4,
            sun_color: [1.0, 0.94, 0.82], // warm midday sun
            shadow_bias: 0.05,
            // A modest neutral-cool fill so the deep, sky-occluded interior still reads (the asset is roofed),
            // kept low enough that the sky-GI bounce — not a flat fill — is what shapes the arcades.
            ambient_color: [0.09, 0.10, 0.13],
            ao_radius: 1.2,
            ao_samples: 4,
            gi_rays: 16, // a strong, low-variance colour-bleed signal for the GI measurement
            gi_intensity: 1.0,
            gi_bounce_dist: 48.0, // long enough to cross the ~30 m nave (wall→far-colonnade bounce)
            emissive_strength: 4.0, // Sponza has no baked emitters, but keep the knob at the open-world default
            frame_index: 0,
            debug_view: 0,
            _pad: 0.0,
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

/// **SSOT for the procedural-sky / environment knobs** (the WGSL `Sky`, group 1 binding 11). A SEPARATE UBO
/// from [`LightingUniformData`] (which is full at 80 bytes). All runtime UNIFORMS (knobs-as-uniforms) — the
/// editor drives any of them; nothing here is a shader const. 64 bytes (std140-safe: each `[f32;3]` vec3 is
/// followed by a scalar to fill its 16-byte slot — NOT `[scalar;N]` padding, which `encase`/`bytemuck` would
/// misalign). Mirrored field-for-field by the WGSL `Sky` struct, so the sky layout has exactly one SSOT.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniformData {
    /// Linear-RGB sky colour at the horizon (`dir.y == 0`).
    pub horizon_color: [f32; 3],
    /// Scalar multiplier on ALL sky radiance (the gradient + the sun disk).
    pub intensity: f32,
    /// Linear-RGB sky colour straight up (`dir.y == +1`).
    pub zenith_color: [f32; 3],
    /// How strongly a diffuse bounce that ESCAPES to sky lights the GI (multiplies `sky_radiance` for bounces).
    pub gi_sky_intensity: f32,
    /// Linear-RGB lower-hemisphere fill colour straight down (`dir.y == -1`).
    pub ground_color: [f32; 3],
    /// Angular HALF-size of the soft sun disk, in radians (`0` disables the disk). Tinted by `sun_tint`.
    pub sun_size: f32,
    /// Linear-RGB tint on the sun disk (multiplied by `light.sun_color × sun_intensity`).
    pub sun_tint: [f32; 3],
    /// Fills the last std140 slot, so the struct is exactly 64 bytes.
    pub _pad: f32,
}

impl Default for SkyUniformData {
    /// Reproduces the CURRENT look: the same horizon/zenith gradient the inline primary-miss sky used
    /// (`horizon (0.55,0.65,0.78)`, `zenith (0.12,0.22,0.45)`), a modest unit intensity, full `gi_sky_intensity`
    /// (a bounce into the sky returns the sky it sees), a ground fill near the horizon, and a small warm sun disk.
    fn default() -> Self {
        Self {
            horizon_color: [0.55, 0.65, 0.78],
            intensity: 1.0,
            zenith_color: [0.12, 0.22, 0.45],
            gi_sky_intensity: 1.0,
            // Lower hemisphere: a dim earth-toned fill so a downward bounce isn't pure black (it was the flat
            // ambient before). Kept dark so it doesn't wash out GI.
            ground_color: [0.18, 0.17, 0.16],
            sun_size: 0.04, // ~2.3° half-angle — a soft sun disk
            sun_tint: [1.0, 0.95, 0.85],
            _pad: 0.0,
        }
    }
}

impl SkyUniformData {
    /// Sky tuned for the open WORLDGEN terrain (Phase 2.6): a BRIGHT directional daytime sky so open slopes
    /// are strongly sky-lit and a bounce escaping the resident clipmap returns plenty of fill (the open-world
    /// counterpart to Cornell's closed box). A deep-blue zenith → pale-blue horizon gradient, full
    /// `gi_sky_intensity` (the sky is the dominant ambient source outdoors), an earthy lower-hemisphere fill,
    /// and a crisp warm sun disk aligned with [`LightingUniformData::worldgen`]'s sun. Knobs-as-uniforms.
    pub fn worldgen() -> Self {
        Self {
            horizon_color: [0.70, 0.80, 0.95],
            intensity: 2.2,
            zenith_color: [0.18, 0.34, 0.62],
            gi_sky_intensity: 1.0,
            ground_color: [0.20, 0.18, 0.15],
            sun_size: 0.035,
            sun_tint: [1.0, 0.93, 0.80],
            _pad: 0.0,
        }
    }

    /// Sky tuned for the baked SPONZA interior (the GI-measurement scene). The baked Khronos Sponza is mostly
    /// ROOFED, so the sky is the DOMINANT light: it enters through the open nave + the colonnade arches and is
    /// the main fill the shaded arcades see (hence the boosted `gi_sky_intensity`), plus the radiance a GI
    /// bounce returns when it escapes upward through an opening. It is
    /// a deep-blue zenith → pale-blue horizon gradient (a clear midday sky), full `gi_sky_intensity` (the sky
    /// is the dominant FILL on surfaces the sun doesn't reach directly — measurable sky-GI in the arcades), a
    /// warm-grey lower-hemisphere fill, and a crisp warm sun disk aligned with [`LightingUniformData::sponza`]'s
    /// sun. Knobs-as-uniforms — the editor overrides any of them.
    pub fn sponza() -> Self {
        Self {
            horizon_color: [0.72, 0.81, 0.93],
            intensity: 2.6,
            zenith_color: [0.20, 0.36, 0.64],
            gi_sky_intensity: 2.5, // sky is the DOMINANT interior light (enters via the open nave + the arches)
            ground_color: [0.16, 0.15, 0.13],
            sun_size: 0.03,
            sun_tint: [1.0, 0.92, 0.78],
            _pad: 0.0,
        }
    }
}

/// Runtime sky resource: the SSOT [`SkyUniformData`] knobs, extracted to the render world each frame and
/// uploaded to the WGSL `Sky` (group 1 binding 11). The Render/GI editor panel mutates this; defaults
/// preserve the current look so existing GPU tests stay green. Knobs-as-uniforms.
#[derive(Resource, Clone, Copy, Debug, Default, ExtractResource)]
pub struct VoxelRtSky {
    pub data: SkyUniformData,
}

/// **SSOT for the world-space radiance-cache knobs** (Phase 2.1; the WGSL `WorldCacheUniform`, group 3
/// binding 0). All runtime UNIFORMS (knobs-as-uniforms mandate) — editor sliders land in 2.4 (Render/GI panel's
/// "World Cache" section); nothing here is a WGSL const. Mirrors Solari's `WORLD_CACHE_*` tunables. 64 bytes
/// (std140/std430-safe: 13 live scalars + 3 pad = four 16-byte rows). `frame_index`, `reset`, and `view_x/y/z`
/// are stamped by the render pass, not user knobs.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldCacheUniformData {
    /// Size of a cache cell at the lowest LOD, in metres (Solari 0.15).
    pub cell_base_size: f32,
    /// How fast the cell LOD grows with camera distance (Solari 15.0).
    pub lod_scale: f32,
    /// Max length of an update-pass GI bounce ray, in metres (Solari 50.0).
    pub gi_ray_distance: f32,
    /// Frames a cell survives un-queried before the decay pass clears it (Solari 10).
    pub cell_lifetime: u32,
    /// Temporal-blend sample-count cap — higher is smoother but laggier (Solari 32.0).
    pub max_temporal_samples: f32,
    /// Per-frame counter (decorrelates the update-pass RNG). Stamped by the render pass.
    pub frame_index: u32,
    /// 1 = first-allocation clear (blend overwrites instead of accumulating). Stamped by the render pass.
    pub reset: u32,
    /// Phase 2.2 A/B gate (knobs-as-uniforms): `1` = the ReSTIR initial reservoir reads the cache
    /// (`reservoir_from_bounce_cached`, the live default); `0` = the FRESH `reservoir_from_bounce` path (no
    /// cache query → no cell marked alive → the cache stays idle, exactly like Phase 2.1).
    pub use_world_cache: u32,
    /// Phase 2.3 A/B gate (knobs-as-uniforms): `1` = the cache UPDATE pass feeds the cache forward at each
    /// bounce hit (`new_radiance += albedo·query_world_cache`, multi-bounce, the live default); `0` =
    /// single-bounce (the Phase 2.1 behaviour — direct+emissive / sky only). Reversible at runtime; an edit
    /// never clears the cache either way (adapt-not-reset — stale cells decay + refill).
    pub gi_multibounce: u32,
    /// Camera world position (X), stamped by the render pass — feeds the update pass's multi-bounce cache
    /// query LOD. Three scalars (not a `Vec3`) keep the std140 layout a clean four-16-byte-row 64 bytes with
    /// no vec3 alignment padding (`[scalar;N]`/encase-pad foot-guns avoided per the GPU-uniform-verify note).
    pub view_x: f32,
    /// Camera world position (Y), stamped by the render pass.
    pub view_y: f32,
    /// Camera world position (Z), stamped by the render pass.
    pub view_z: f32,
    /// Phase 2.4 SOFT per-frame active-cell cap (knobs-as-uniforms). `0` (the default) = UNLIMITED — every
    /// active cell is updated + blended each frame, exactly the pre-2.4 behaviour (byte-identical GPU tests).
    /// When `> 0`, `compact_write_active` clamps the indirect update/blend dispatch to `ceil(N/64)` workgroups
    /// AND the update/blend entries early-out for `active_index >= N`, so at most `N` active cells are processed
    /// this frame; the rest keep their last radiance + life and are picked up next frame as they stay alive.
    /// NEVER corrupts the cache (skipped cells are untouched, not cleared). Trade: lower steady GPU cost vs a
    /// slower cache fill (more frames to converge). OFF by default.
    pub max_active_cells_per_frame: u32,
    /// Phase 2.5 NEE: number of emissive-voxel lights in the bound light list (`0` ⇒ NEE skipped — no emitters).
    /// Stamped by the render pass from the packed patch's `lights.len()`; never a user knob.
    pub light_count: u32,
    /// Phase 2.5 NEE A/B gate (knobs-as-uniforms): `1` = the update pass adds direct emissive-voxel light
    /// sampling (NEE) with MIS (the live default — the principled firefly/variance fix); `0` = bounce-only (the
    /// pre-2.5 behaviour: emitters found only by the random cosine bounce). Reversible at runtime; the editor
    /// slider toggles it. Adapt-not-reset (toggling never clears the cache — stale cells decay + refill).
    pub nee_enabled: u32,
    /// Phase 2.5 NEE: shadow-ray light samples per cell update (clamped to `≥1` in-shader). Higher cuts the
    /// direct-light variance further at a linear shadow-ray cost; `1` is the Solari-class default (the temporal
    /// blend averages successive frames). A runtime uniform (knobs-as-uniforms).
    pub nee_samples: u32,
}

impl Default for WorldCacheUniformData {
    /// Solari's defaults (`world_cache_query.wgsl`). `frame_index`/`reset`/`view_*` are runtime-stamped, default 0.
    fn default() -> Self {
        Self {
            cell_base_size: 0.15,
            lod_scale: 15.0,
            gi_ray_distance: 50.0,
            cell_lifetime: 10,
            max_temporal_samples: 32.0,
            frame_index: 0,
            reset: 0,
            use_world_cache: 1, // 2.2 default: the initial reservoir reads the cache (A/B gate, editor-toggled)
            gi_multibounce: 1,  // 2.3 default: the update pass feeds the cache forward (multi-bounce, A/B-gated)
            view_x: 0.0,
            view_y: 0.0,
            view_z: 0.0,
            max_active_cells_per_frame: 0, // 2.4 default: UNLIMITED (no behaviour change; cap is opt-in)
            light_count: 0,                // 2.5: stamped per frame from the packed light list (0 ⇒ NEE skipped)
            nee_enabled: 1,                // 2.5 default: NEE ON (the principled firefly/variance fix; A/B-gated)
            nee_samples: 1,                // 2.5 default: 1 shadow ray/cell/frame (the temporal blend averages)
        }
    }
}

/// **SSOT for the editor-tunable world-cache knobs** (knobs-as-uniforms). Drives [`WorldCacheUniformData`]
/// each frame; the Render/GI editor sliders (Stage 2.4) write it. Extracted to the render world. In 2.1 the
/// cache RUNS off these knobs but is not yet read by the live image. `Default` = [`WorldCacheUniformData`]'s
/// Solari-tuned defaults.
#[derive(Resource, Clone, Copy, Debug, Default, ExtractResource)]
pub struct WorldCacheSettings {
    pub data: WorldCacheUniformData,
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
            storage_ro(1),  // metas
            storage_ro(2),  // voxel_indices (R2b bit-packed index stream)
            storage_ro(3),  // palette (block id → colour)
            storage_ro(12), // brick_palettes (R2b per-brick palettes)
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
            // binding 11: the procedural-sky uniform (`Sky`), updated per frame. Shared by the primary-miss
            // sky, the GI bounce-miss sky, and the ReSTIR bounce-miss sky sample (one sky SSOT).
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let raymarch_src = voxel_raytrace_shader_src(WORLD_CACHE_SIZE);
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

    // ReSTIR group(2): reservoir storage buffers (cur/prev) + restir params uniform + per-pixel receiver
    // surface buffers (cur/prev) for neighbour-reuse Jacobian + dissimilarity rejection.
    let reservoir_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_reservoir_layout"),
        entries: &[storage_rw(0), storage_rw(1), uniform_buf(2), storage_rw(3), storage_rw(4)],
    });
    // group(3) world-cache layout (Phase 2.1). Created BEFORE `restir_pl` because Phase 2.2 binds the cache
    // into `restir_p1`/`restir_dlss_p1` so the initial reservoir can `query_world_cache` (lazy-insert → the
    // live query is what POPULATES the cache). Binding 0 = the `wc` uniform, bindings 1..=10 = the 10
    // persistent cache storage buffers. The indirect-dispatch buffer lives in its OWN group(2) (see below).
    let world_cache_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_world_cache_layout"),
        entries: &[
            uniform_buf(0),
            storage_rw(1),
            storage_rw(2),
            storage_rw(3),
            storage_rw(4),
            storage_rw(5),
            storage_rw(6),
            storage_rw(7),
            storage_rw(8),
            storage_rw(9),
            storage_rw(10),
            // Phase 2.5 NEE: the emissive-voxel light list (15) + power-weighted alias table (16), read-only.
            // In `world_cache_layout` (not a new group) so the update pass reads them AND the restir passes —
            // which share this layout for the cache query — keep a valid (unused) binding.
            storage_ro(15),
            storage_ro(16),
        ],
    });
    // The two-pass ReSTIR pipeline layout. group(3) = the world cache: `restir_p1` queries it (read_write — the
    // query lazy-inserts), and `restir_p2` shares the layout (it never touches the cache; binding an unused
    // group is legal). The cache `group(3)` bind group set by the world-cache passes (which run earlier in the
    // same compute pass) stays bound through both restir passes, so no extra `set_bind_group(3, ...)` is needed.
    let restir_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_raymarch_restir_pl"),
        bind_group_layouts: &[
            Some(&scene_layout),
            Some(&view_layout),
            Some(&reservoir_layout),
            Some(&world_cache_layout),
        ],
        immediate_size: 0,
    });
    let restir_p1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_restir_p1"),
        layout: Some(&restir_pl),
        module: &raymarch_module,
        entry_point: Some("restir_p1"),
        compilation_options: Default::default(),
        cache: None,
    });
    let restir_p2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_restir_p2"),
        layout: Some(&restir_pl),
        module: &raymarch_module,
        entry_point: Some("restir_p2"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- Phase 2.1 world-cache: the 6 compute pipelines (the group(3) `world_cache_layout` is created above,
    // shared with `restir_pl` so Phase 2.2's initial reservoir can query the cache). ---
    // A MINIMAL group(1) layout for the cache passes holding ONLY the two uniforms the UPDATE pass reads from
    // group 1 — `light` (binding 2) + `sky` (binding 11) — used by `direct_lighting` / `sky_radiance`. The
    // cache passes never write `out_tex` / sample `history`, so we omit those (a smaller, dedicated layout
    // avoids threading the full view bind group — camera/output/history/sampler — into the cache dispatch).
    let world_cache_view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_world_cache_view_layout"),
        entries: &[uniform_buf(2), uniform_buf(11)],
    });
    // group(2) — the indirect-dispatch-args buffer, in its OWN bind group. wgpu forbids a buffer being both
    // bound read-write storage AND used as an indirect-dispatch source in one compute-pass usage scope, so the
    // decay/compaction passes (which WRITE it) bind this group, while the update/blend passes (which CONSUME
    // it as the indirect arg) use a layout that OMITS group 2 — and we unbind it at dispatch. Mirrors Solari's
    // `bind_group_world_cache_active_cells_dispatch` + `set_bind_group(2, None)`.
    let world_cache_dispatch_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel_rt_world_cache_dispatch_layout"),
        entries: &[storage_rw(0)],
    });
    // Pipeline layout A — decay + the 3 compaction passes: scene(0), view(1), dispatch(2), cache(3). (Only
    // `compact_write_active` actually writes group 2, but sharing one layout across the 4 whole-table passes
    // keeps the wiring uniform; naga prunes the unused dispatch global from the others.)
    let world_cache_compact_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_world_cache_compact_pl"),
        bind_group_layouts: &[
            Some(&scene_layout),
            Some(&world_cache_view_layout),
            Some(&world_cache_dispatch_layout),
            Some(&world_cache_layout),
        ],
        immediate_size: 0,
    });
    // Pipeline layout B — update + blend: scene(0), view(1), <no group 2>, cache(3). Omitting group 2 lets the
    // indirect dispatch consume the (now-unbound) dispatch buffer legally. The `trace` (group 0) + `light`/
    // `sky` (group 1) are needed by the update pass; the cache (group 3) by both.
    let world_cache_update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_world_cache_update_pl"),
        bind_group_layouts: &[
            Some(&scene_layout),
            Some(&world_cache_view_layout),
            None, // group 2 deliberately absent — the dispatch buffer is unbound when used as indirect args
            Some(&world_cache_layout),
        ],
        immediate_size: 0,
    });
    let mk_wc = |label: &'static str, entry: &'static str, layout: &wgpu::PipelineLayout| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &raymarch_module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let wc_decay = mk_wc("voxel_rt_wc_decay", "world_cache_decay", &world_cache_compact_pl);
    let wc_compact_single_block = mk_wc(
        "voxel_rt_wc_compact_single_block",
        "world_cache_compact_single_block",
        &world_cache_compact_pl,
    );
    let wc_compact_blocks =
        mk_wc("voxel_rt_wc_compact_blocks", "world_cache_compact_blocks", &world_cache_compact_pl);
    let wc_compact_write_active = mk_wc(
        "voxel_rt_wc_compact_write_active",
        "world_cache_compact_write_active",
        &world_cache_compact_pl,
    );
    let wc_update = mk_wc("voxel_rt_wc_update", "world_cache_update", &world_cache_update_pl);
    let wc_blend = mk_wc("voxel_rt_wc_blend", "world_cache_blend", &world_cache_update_pl);

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
    let (raymarch_dlss, restir_dlss_p1, restir_dlss_p2, dlss_view_layout, dlss_resolve_layout) =
        init_dlss_pipelines(
            device,
            &scene_layout,
            &reservoir_layout,
            &world_cache_layout,
            &raymarch_module,
            &composite_module,
        );

    commands.insert_resource(VoxelRtPipelines {
        scene_layout,
        view_layout,
        reservoir_layout,
        raymarch,
        restir_p1,
        restir_p2,
        world_cache_layout,
        world_cache_view_layout,
        world_cache_dispatch_layout,
        wc_decay,
        wc_compact_single_block,
        wc_compact_blocks,
        wc_compact_write_active,
        wc_update,
        wc_blend,
        composite_module,
        composite_layout,
        composite_sampler,
        #[cfg(feature = "dlss")]
        raymarch_dlss,
        #[cfg(feature = "dlss")]
        restir_dlss_p1,
        #[cfg(feature = "dlss")]
        restir_dlss_p2,
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
    world_cache_layout: &wgpu::BindGroupLayout,
    raymarch_module: &wgpu::ShaderModule,
    composite_module: &wgpu::ShaderModule,
) -> (
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::ComputePipeline,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
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
            uniform(11),                                                // sky (procedural-sky uniform, one SSOT)
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
    // The two-pass ReSTIR variant: same DLSS guide layout + the group(2) reservoir buffers + group(3) world
    // cache, two entries. group(3) lets `restir_dlss_p1`'s initial reservoir `query_world_cache` (lazy-insert
    // → the live query populates the cache); `restir_dlss_p2` shares the layout but never touches the cache.
    let dlss_restir_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxel_rt_raymarch_dlss_restir_pl"),
        bind_group_layouts: &[
            Some(scene_layout),
            Some(&dlss_view_layout),
            Some(reservoir_layout),
            Some(world_cache_layout),
        ],
        immediate_size: 0,
    });
    let restir_dlss_p1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_restir_dlss_p1"),
        layout: Some(&dlss_restir_pl),
        module: raymarch_module,
        entry_point: Some("restir_dlss_p1"),
        compilation_options: Default::default(),
        cache: None,
    });
    let restir_dlss_p2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxel_rt_restir_dlss_p2"),
        layout: Some(&dlss_restir_pl),
        module: raymarch_module,
        entry_point: Some("restir_dlss_p2"),
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
    (raymarch_dlss, restir_dlss_p1, restir_dlss_p2, dlss_view_layout, dlss_resolve_layout)
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

/// Bytes per WGSL `PixelSurface` (2×vec4 = 32): world pos + valid flag, world normal + pad.
const SURFACE_SIZE: u64 = 32;

/// Mirror of the WGSL `RestirParams` (group 2 binding 2): reset + frame + viewport + the editor ReSTIR knobs.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RestirParamsData {
    reset: u32,
    frame_index: u32,
    viewport_x: u32,
    viewport_y: u32,
    spatial_samples: u32,
    confidence_weight_cap: f32,
    spatial_radius: f32,
    _pad: u32,
}

/// **SSOT for the editor-tunable ReSTIR knobs** (knobs-as-uniforms). Drives `RestirParamsData` each frame; the
/// Render/GI panel writes it. `gi_mode` selects the live GI path: `false` = legacy `gather_gi`, `true` = ReSTIR
/// (the A/B toggle). Extracted to the render world.
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct RestirSettings {
    /// `true` = ReSTIR GI (default), `false` = legacy `gather_gi` (for A/B comparison).
    pub restir: bool,
    /// Spatial reuse SEARCH budget: disk taps tried per pixel to find ONE valid neighbour to merge (0 =
    /// temporal-only). NOT an accumulation count — one neighbour is merged per frame (variance-stable).
    pub spatial_samples: u32,
    /// Spatial-neighbour disk radius in pixels.
    pub spatial_radius: f32,
    /// Temporal/spatial history confidence cap (frames). Higher = smoother + more lag.
    pub confidence_cap: f32,
}

impl Default for RestirSettings {
    fn default() -> Self {
        Self { restir: true, spatial_samples: 4, spatial_radius: 16.0, confidence_cap: 8.0 }
    }
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
        label: Some("voxel_rt_voxel_indices"),
        contents: bytemuck::cast_slice(&patch.voxels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let palette_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_palette"),
        contents: bytemuck::cast_slice(&patch.palette),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Storage plan R2b — the per-brick palettes the bit-packed index stream indirects through.
    let brick_palettes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_brick_palettes"),
        contents: bytemuck::cast_slice(&patch.brick_palettes),
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
            wgpu::BindGroupEntry { binding: 12, resource: brick_palettes_buf.as_entire_binding() },
        ],
    });

    // Phase 2.5 NEE: rebuild the emissive-voxel light list + alias table for this generation (the lights are
    // derived from the resident set, so they follow streamed/edited geometry). Built alongside the scene swap
    // so the light list and the BLAS/voxels always describe the SAME generation (never a frame out of sync).
    let lights = WorldCacheLights::new(device, patch);
    debug!(
        "voxel-RT NEE: built light list for gen {} — {} emissive-voxel lights",
        patch_res.generation, lights.count
    );
    resources.world_cache_lights = Some(lights);

    // Swap in the new scene atomically (the old `_keep` + bind group drop only now that the new ones are
    // fully built — keep-old-until-revealed).
    resources.scene_bind_group = Some(scene_bind_group);
    resources.brick_count = n;
    resources.built_generation = Some(patch_res.generation);
    resources._keep = Some(SceneKeepAlive {
        _blas: blas,
        _tlas: tlas,
        _buffers: [aabb_buf, meta_buf, voxel_buf, palette_buf, brick_palettes_buf],
    });
    debug!("voxel-RT: built accel structures for patch gen {} — {n} bricks", patch_res.generation);
}

/// The objects the per-frame world-cache dispatch needs, built before the compute pass opens (so they can be
/// `set_bind_group`'d into it): the three bind groups (scene group 0 is the caller's; here groups 1/2/3) and a
/// handle to the indirect-dispatch buffer (cloned — wgpu `Buffer` is an `Arc`, so this is cheap and keeps the
/// borrow off `resources` while the compute pass runs).
struct WorldCachePrepared {
    view_bg: wgpu::BindGroup,
    dispatch_bg: wgpu::BindGroup,
    cache_bg: wgpu::BindGroup,
    dispatch_buf: wgpu::Buffer,
}

/// Allocate the persistent world-cache buffers on first use, upload the per-frame `wc` uniform (stamping
/// `frame_index` + the one-shot `reset`), and build the bind groups for the six cache passes. Returns `None`
/// only if there's nothing to do (never errors). `light_buf`/`sky_buf` are the caller's already-uploaded
/// lighting/sky uniforms (the update pass reads `light`/`sky` via group 1). The cache is WORLD-space: the
/// buffers are allocated ONCE and `reset` fires exactly once (first frame after allocation) — never on resize
/// or edit ([[feedback-gi-adapt-not-reset]]).
#[allow(clippy::too_many_arguments)]
fn prepare_world_cache(
    device: &wgpu::Device,
    pipelines: &VoxelRtPipelines,
    resources: &mut VoxelRtResources,
    settings: &WorldCacheSettings,
    frame_index: u32,
    cam_pos: [f32; 3],
    light_buf: &wgpu::Buffer,
    sky_buf: &wgpu::Buffer,
) -> WorldCachePrepared {
    if resources.world_cache.is_none() {
        resources.world_cache = Some(WorldCacheBuffers::new(device, WORLD_CACHE_SIZE));
        resources.world_cache_initialized = false; // first dispatch this frame uses reset=1
    }
    // First cache frame after allocation: blend overwrites instead of accumulating (the buffers are
    // zero-initialised, but reset=1 makes the very first blend exact). Cleared exactly ONCE.
    let reset = !resources.world_cache_initialized;
    resources.world_cache_initialized = true;

    let mut wc = settings.data;
    wc.frame_index = frame_index;
    wc.reset = u32::from(reset);
    // Stamp the camera position so the multi-bounce update-pass cache query (`wc_view_position`) uses the same
    // distance-adaptive cell LOD as the live `reservoir_from_bounce_cached` consumer (which reads `camera`).
    [wc.view_x, wc.view_y, wc.view_z] = cam_pos;
    // Phase 2.5 NEE: stamp the live light count from the packed list. 0 (no emitters, or the list not yet built)
    // ⇒ the shader skips NEE cleanly. The `nee_enabled`/`nee_samples` knobs ride in from `settings.data`.
    wc.light_count = resources.world_cache_lights.as_ref().map(|l| l.count).unwrap_or(0);
    let wc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_world_cache_uniform"),
        contents: bytemuck::bytes_of(&wc),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // The NEE light buffers (bound at cache bindings 15/16). On the very first frames (cache allocated before
    // the first patch is packed) the list may not exist yet — bind a zeroed dummy so the bind group is valid;
    // `light_count == 0` then keeps the shader off the dummy. The real list swaps in on the next packed gen.
    if resources.world_cache_lights.is_none() {
        resources.world_cache_lights = Some(WorldCacheLights::new(device, &GpuBrickPatch::default()));
    }
    let cache = resources.world_cache.as_ref().expect("just allocated");
    let lights = resources.world_cache_lights.as_ref().expect("just ensured");
    let view_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_world_cache_view_bg"),
        layout: &pipelines.world_cache_view_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
        ],
    });
    let dispatch_bg = cache.dispatch_bg(device, &pipelines.world_cache_dispatch_layout);
    let cache_bg = cache.bind_group(device, &pipelines.world_cache_layout, &wc_buf, lights);
    let dispatch_buf = cache.active_cells_dispatch.clone();
    WorldCachePrepared { view_bg, dispatch_bg, cache_bg, dispatch_buf }
}

/// Dispatch the six world-cache passes IN ORDER on an open compute pass: decay → compact_single_block →
/// compact_blocks → compact_write_active → update (indirect) → blend (indirect). The caller has already set
/// the scene bind group at index 0 and (for the live raymarch/restir) may rebind groups afterward. Consecutive
/// dispatches in one compute pass get WebGPU's implicit storage barrier, so each pass sees the prior's writes.
///
/// Editor builds pass an optional [`WorldCacheGpuTimer`] (issue #123): we bracket each measured segment
/// (decay=0, compact=1 over the three compaction passes, update=2, blend=3) with begin/end timestamps. The
/// timing is purely additive — the dispatches are byte-identical with or without it.
fn dispatch_world_cache_passes(
    cpass: &mut wgpu::ComputePass,
    pipelines: &VoxelRtPipelines,
    prepared: &WorldCachePrepared,
    #[cfg(feature = "editor")] timer: Option<&WorldCacheGpuTimer>,
) {
    cpass.set_bind_group(1, &prepared.view_bg, &[]);
    cpass.set_bind_group(2, &prepared.dispatch_bg, &[]); // group 2 = the indirect-dispatch buffer (written here)
    cpass.set_bind_group(3, &prepared.cache_bg, &[]);
    // Whole-table passes: one thread per cell (workgroup_size 1024).
    let table_groups = WORLD_CACHE_SIZE / 1024;
    #[cfg(feature = "editor")]
    if let Some(t) = timer {
        t.begin(cpass, 0);
    }
    cpass.set_pipeline(&pipelines.wc_decay);
    cpass.dispatch_workgroups(table_groups, 1, 1);
    #[cfg(feature = "editor")]
    if let Some(t) = timer {
        t.end(cpass, 0); // decay
        t.begin(cpass, 1); // compact (×3)
    }
    cpass.set_pipeline(&pipelines.wc_compact_single_block);
    cpass.dispatch_workgroups(table_groups, 1, 1);
    cpass.set_pipeline(&pipelines.wc_compact_blocks);
    cpass.dispatch_workgroups(1, 1, 1);
    cpass.set_pipeline(&pipelines.wc_compact_write_active);
    cpass.dispatch_workgroups(table_groups, 1, 1);
    #[cfg(feature = "editor")]
    if let Some(t) = timer {
        t.end(cpass, 1); // compact
    }
    // UNBIND group 2 before the indirect dispatches: wgpu forbids the dispatch buffer being bound read-write
    // storage AND used as the indirect-args source in one usage scope. The update/blend pipeline layout omits
    // group 2, so this clears it (Solari's `set_bind_group(2, None)` pattern).
    cpass.set_bind_group(2, None, &[]);
    // Active-cell passes: indirect over the compacted count (ceil(active / 64) workgroups).
    #[cfg(feature = "editor")]
    if let Some(t) = timer {
        t.begin(cpass, 2); // update
    }
    cpass.set_pipeline(&pipelines.wc_update);
    cpass.dispatch_workgroups_indirect(&prepared.dispatch_buf, 0);
    #[cfg(feature = "editor")]
    if let Some(t) = timer {
        t.end(cpass, 2); // update
        t.begin(cpass, 3); // blend
    }
    cpass.set_pipeline(&pipelines.wc_blend);
    cpass.dispatch_workgroups_indirect(&prepared.dispatch_buf, 0);
    #[cfg(feature = "editor")]
    if let Some(t) = timer {
        t.end(cpass, 3); // blend
    }
}

/// [`Core3d`]/[`Core3dSystems::MainPass`]: when the toggle is on and the scene is built, dispatch the
/// raymarch compute pass into a per-view output texture, then composite it over the [`ViewTarget`]. When
/// off, returns immediately so the Stage-1 cubes render unchanged.
#[allow(clippy::too_many_arguments)]
fn voxel_rt_pass(
    #[cfg(not(feature = "dlss"))] view: ViewQuery<(&ExtractedView, &ViewTarget)>,
    #[cfg(feature = "dlss")] view: ViewQuery<(
        &ExtractedView,
        &ViewTarget,
        Option<&Dlss<DlssRayReconstructionFeature>>,
    )>,
    toggle: Res<VoxelRtToggle>,
    lighting: Res<VoxelRtLighting>,
    sky: Res<VoxelRtSky>,
    restir_settings: Res<RestirSettings>,
    world_cache_settings: Res<WorldCacheSettings>,
    pipelines: Option<Res<VoxelRtPipelines>>,
    mut resources: ResMut<VoxelRtResources>,
    render_device: Res<RenderDevice>,
    // Phase 2.4 GPU per-pass timing (editor builds): the queue converts timestamp ticks → ms via
    // `get_timestamp_period()`. Unused by the GI math; cfg-gated so the non-editor signature is untouched.
    #[cfg(feature = "editor")] render_queue: Res<RenderQueue>,
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
        // ReSTIR per-pixel reservoirs (fixed-role a/b) + receiver-surface buffers (cur/prev ping-pong).
        // Uninitialised — the `reset` flag (set below because prev_view_proj is now None) makes the shader
        // ignore stale history (pass 1's temporal tap) on the first frame.
        let px = (viewport.x as u64) * (viewport.y as u64);
        let mk_buf = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        resources.reservoirs = Some((
            mk_buf("voxel_rt_reservoir_a", px * RESERVOIR_SIZE),
            mk_buf("voxel_rt_reservoir_b", px * RESERVOIR_SIZE),
            viewport,
        ));
        resources.surfaces = Some((
            mk_buf("voxel_rt_surface_a", px * SURFACE_SIZE),
            mk_buf("voxel_rt_surface_b", px * SURFACE_SIZE),
            viewport,
        ));
        // A fresh-size history is uninitialised — force a reset (full new frame) this frame.
        resources.accum_samples = 0;
        resources.prev_view_proj = None;
        // Reprojection has no stale prev across a resize; self-tap on the next frame (the `restir_prev`
        // viewport change also forces the reservoir `reset`, so the stale-size reservoirs are discarded).
        resources.prev_clip_from_world = None;
        resources.restir_prev = None;
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

    // The current frame's UN-jittered `clip_from_world`. The non-DLSS path applies NO TemporalJitter, so this
    // IS the un-jittered clip (identical to the `view_proj` move-test matrix below). It feeds both the next
    // frame's reservoir reprojection (stored at the end of this block) and the history-texture move test.
    let view_proj = (clip_from_view * world_from_view.inverse()).to_cols_array_2d();

    // --- HISTORY-TEXTURE accumulation control: reset on a camera move or a geometry re-pack, else run a 1/n
    // mean. --- The view-projection (clip_from_world) fully captures both camera POSITION and PROJECTION; any
    // change means the previous history no longer aligns pixel-for-pixel, so we must reset (show the fresh
    // frame) to avoid ghosting. A scene re-pack (new geometry) likewise invalidates the history. Otherwise the
    // camera is still: grow the sample count and blend at 1/n so the image converges to the clean average over n
    // frames. NOTE: this controls ONLY the on-top history TEXTURE blend (out_tex/history_tex `accum_weight`),
    // NOT the ReSTIR reservoirs — those now reproject through camera motion (see the reset trigger below).
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

    // Previous-frame un-jittered clip for the ReSTIR temporal REPROJECTION. On the first frame there is no
    // prev → use the current clip so `reproject_pixel` returns the current pixel (a no-op self-tap). Mirrors the
    // DLSS path's `dlss_prev_clip_from_world.unwrap_or(view_proj)`. We store THIS frame's clip below for next.
    let prev_clip_from_world = resources.prev_clip_from_world.unwrap_or(view_proj);
    resources.prev_clip_from_world = Some(view_proj);

    let out_view = &resources.output.as_ref().expect("just allocated").1;

    let cam_uniform = CameraUniformData {
        world_from_clip: world_from_clip.to_cols_array_2d(),
        cam_pos: cam_pos.into(),
        t_max: 1.0e4,
        viewport: [viewport.x, viewport.y],
        accum_weight,
        _pad: 0,
        prev_clip_from_world,
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
    // The procedural-sky uniform (SSOT knobs), uploaded fresh each frame so editor tweaks take effect live.
    let sky_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_sky"),
        contents: bytemuck::bytes_of(&sky.data),
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
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
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

    // ReSTIR group(2): the restir params + the fixed-role reservoir buffers (a = final/history, b =
    // intermediate). The reservoir `reset` fires ONLY on the first frame or a viewport (resolution) change —
    // NEVER on a camera move and NEVER on a geometry edit. Camera motion is now handled by motion-vector
    // reprojection (`reproject_pixel(p, camera.prev_clip_from_world, ...)` in `restir_p1`; disocclusions on
    // fast motion are caught by the `surfaces_dissimilar` reject in `restir_p1_core`), and a geometry edit
    // ADAPTS locally (fresh candidates re-trace the new geometry, the visibility trace drops now-occluded
    // samples, dissimilarity rejects moved surfaces) — never a full clear. This mirrors the DLSS path's
    // `reset_restir` keying. (The on-top history TEXTURE accumulator above still resets on a move/`geometry_changed`
    // — that just shows the fresh frame, it is NOT a reservoir clear.)
    let reset_restir = match resources.restir_prev {
        None => true,
        Some((vp, _g)) => vp != viewport,
    };
    resources.restir_prev = Some((viewport, cur_generation));
    let restir_params = RestirParamsData {
        reset: u32::from(reset_restir),
        frame_index,
        viewport_x: viewport.x,
        viewport_y: viewport.y,
        spatial_samples: restir_settings.spatial_samples,
        confidence_weight_cap: restir_settings.confidence_cap,
        spatial_radius: restir_settings.spatial_radius,
        _pad: 0,
    };
    let restir_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_restir_params"),
        contents: bytemuck::bytes_of(&restir_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let (res_a, res_b, _) = resources.reservoirs.as_ref().expect("allocated with output");
    let (surf_a, surf_b, _) = resources.surfaces.as_ref().expect("allocated with output");
    let even = frame_index & 1 == 0;
    // Reservoirs are FIXED-ROLE (binding 0 = `reservoirs_a` = history/final, binding 1 = `reservoirs_b` =
    // intermediate) — NOT ping-ponged. Pass 1 reads `a` (last frame's final, at the reprojected/permuted tap)
    // + writes `b`; pass 2 reads `b` (this frame, same-frame spatial) + writes `a` (this frame's final). Read
    // and write of `a` within one frame are ordered by the intra-pass storage barrier between the p1/p2
    // dispatches. Surfaces still ping-pong (pass 1 writes `cur` + reads `prev` for the temporal validity test).
    let (surf_cur, surf_prev) = if even { (surf_a, surf_b) } else { (surf_b, surf_a) };
    let reservoir_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_reservoir_bg"),
        layout: &pipelines.reservoir_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: res_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: res_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: restir_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: surf_cur.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: surf_prev.as_entire_binding() },
        ],
    });

    // Phase 2.1 world-space radiance cache: allocate the persistent buffers (once) + build the cache bind
    // groups. Dispatched BEFORE the live raymarch/restir, but it does NOT feed the live image this stage — the
    // cache just runs + accumulates (zero visual change; 2.2 wires it into the reservoir). Mutably borrows
    // `resources` here, before the immutable scene/output borrows below.
    let wc_prepared = prepare_world_cache(
        device,
        &pipelines,
        &mut resources,
        &world_cache_settings,
        frame_index,
        cam_pos.into(),
        &light_buf,
        &sky_buf,
    );

    // Phase 2.4 GPU per-pass timing (editor only). Take the persistent timer OUT of `resources` so the
    // immutable scene/output borrows below can coexist with the timer's `&mut` use; it goes back at the end of
    // the pass. Allocate it lazily (and read back LAST frame's timestamps now, before re-recording).
    #[cfg(feature = "editor")]
    let mut gpu_timer = {
        if !resources.gpu_timer_checked {
            resources.gpu_timer_checked = true;
            resources.gpu_timer = WorldCacheGpuTimer::new(device);
            if resources.gpu_timer.is_none() {
                bevy::log::warn!(
                    "voxel-rt: GPU per-pass timing unavailable — device lacks TIMESTAMP_QUERY / \
                     TIMESTAMP_QUERY_INSIDE_PASSES; world-cache passes run untimed"
                );
            }
        }
        let mut t = resources.gpu_timer.take();
        if let Some(timer) = t.as_mut() {
            timer.read_previous(device, render_queue.get_timestamp_period());
        }
        t
    };

    let scene_bg = resources.scene_bind_group.as_ref().expect("checked above");
    // `gi_mode` A/B: ReSTIR GI (group-2 reservoirs, two passes) vs the legacy `gather_gi` raymarch (no group 2).
    let use_restir = restir_settings.restir;
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
        cpass.set_bind_group(0, scene_bg, &[]);
        // World-cache passes FIRST (decay → compact ×3 → update → blend), sharing scene group 0. They set
        // groups 1/2/3 themselves; the live raymarch/restir below rebinds groups 1/2 to the view + reservoirs.
        dispatch_world_cache_passes(
            &mut cpass,
            &pipelines,
            &wc_prepared,
            #[cfg(feature = "editor")]
            gpu_timer.as_ref(),
        );
        cpass.set_bind_group(1, &view_bg, &[]);
        let groups = (viewport.x.div_ceil(8), viewport.y.div_ceil(8), 1);
        if use_restir {
            // Two-pass ReSTIR: pass 1 (initial + temporal → reservoirs_b) then pass 2 (same-frame spatial →
            // reservoirs_a + shade → out_tex), back-to-back. The intra-pass storage barrier orders p1's writes
            // to reservoirs_b before p2 reads them (WebGPU guarantees inter-dispatch storage visibility).
            cpass.set_bind_group(2, &reservoir_bg, &[]);
            // group(3) = the world cache (Phase 2.2): `restir_p1`'s initial reservoir queries it (lazy-insert →
            // the query is what POPULATES the cache). Re-set explicitly even though the cache passes left it
            // bound — rebinding group 2 above can invalidate inheritance of higher-indexed groups.
            cpass.set_bind_group(3, &wc_prepared.cache_bg, &[]);
            #[cfg(feature = "editor")]
            if let Some(t) = gpu_timer.as_ref() {
                t.begin(&mut cpass, 4);
            }
            cpass.set_pipeline(&pipelines.restir_p1);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
            #[cfg(feature = "editor")]
            if let Some(t) = gpu_timer.as_ref() {
                t.end(&mut cpass, 4); // restir p1
                t.begin(&mut cpass, 5); // restir p2
            }
            cpass.set_pipeline(&pipelines.restir_p2);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
            #[cfg(feature = "editor")]
            if let Some(t) = gpu_timer.as_ref() {
                t.end(&mut cpass, 5); // restir p2
            }
        } else {
            cpass.set_pipeline(&pipelines.raymarch);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
    }
    // Phase 2.4: resolve this frame's world-cache/ReSTIR timestamps into the read-back buffer (mapped + read
    // next frame). Additive — the GI passes above are byte-identical with or without the timer.
    #[cfg(feature = "editor")]
    if let Some(t) = gpu_timer.as_mut() {
        t.resolve(encoder);
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
    // Return the persistent timer to `resources` (the scene/output borrows above have ended).
    #[cfg(feature = "editor")]
    {
        resources.gpu_timer = gpu_timer.take();
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

/// DLSS camera uniform (WGSL `DlssCamera`, group 1 binding 10). 192 bytes.
/// `depth_clip_from_world` is the JITTERED projection (matches Bevy's jittered reverse-Z depth prepass — used
/// only for the depth write). `motion_prev`/`motion_cur` are the UN-JITTERED previous/current clip_from_world:
/// the motion vector must be geometry motion only, because the DLSS node is given the sub-pixel jitter offset
/// separately and resolves it itself. (Differencing jittered matrices double-counts the jitter → camera shake.)
#[cfg(feature = "dlss")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DlssCameraData {
    depth_clip_from_world: [[f32; 4]; 4],
    motion_prev: [[f32; 4]; 4],
    motion_cur: [[f32; 4]; 4],
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
    sky: Res<VoxelRtSky>,
    restir_settings: Res<RestirSettings>,
    world_cache_settings: Res<WorldCacheSettings>,
    pipelines: Option<Res<VoxelRtPipelines>>,
    mut resources: ResMut<VoxelRtResources>,
    render_device: Res<RenderDevice>,
    // Phase 2.4 GPU per-pass timing (editor builds): ticks → ms via `get_timestamp_period()`.
    #[cfg(feature = "editor")] render_queue: Res<RenderQueue>,
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
        // ReSTIR reservoirs + receiver-surface buffers at FULL size (≥ the render-res dispatch); reset forces
        // a fresh frame.
        let px = (full.x as u64) * (full.y as u64);
        let mk_buf = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        resources.dlss_reservoirs = Some((
            mk_buf("voxel_rt_dlss_reservoir_a", px * RESERVOIR_SIZE),
            mk_buf("voxel_rt_dlss_reservoir_b", px * RESERVOIR_SIZE),
            full,
        ));
        resources.dlss_surfaces = Some((
            mk_buf("voxel_rt_dlss_surface_a", px * SURFACE_SIZE),
            mk_buf("voxel_rt_dlss_surface_b", px * SURFACE_SIZE),
            full,
        ));
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
    let clip_from_world_arr = clip_from_world.to_cols_array_2d(); // JITTERED — depth write only
    // UN-jittered current clip_from_world: motion vectors must exclude the jitter (DLSS resolves it itself),
    // and this is also the stable matrix the ReSTIR reset move-test compares.
    let view_proj_unjittered =
        (extracted_view.clip_from_view * world_from_view.inverse()).to_cols_array_2d();
    let motion_prev = resources.dlss_prev_clip_from_world.unwrap_or(view_proj_unjittered);

    let cam_uniform = CameraUniformData {
        world_from_clip: world_from_clip.to_cols_array_2d(),
        cam_pos: cam_pos.into(),
        t_max: 1.0e4,
        viewport: [render_res.x, render_res.y],
        accum_weight: 1.0, // unused by raymarch_dlss (DLSS denoises), kept for layout parity
        _pad: 0,
        // Unused by the DLSS path (it reprojects via `dlss_cam.motion_prev`); filled for layout parity.
        prev_clip_from_world: motion_prev,
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
    let sky_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_sky"),
        contents: bytemuck::bytes_of(&sky.data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let dlss_cam = DlssCameraData {
        depth_clip_from_world: clip_from_world_arr,
        motion_prev,
        motion_cur: view_proj_unjittered,
    };
    let dlss_cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_cam"),
        contents: bytemuck::bytes_of(&dlss_cam),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    resources.dlss_prev_clip_from_world = Some(view_proj_unjittered);

    // ReSTIR reset: ONLY a render-resolution change or the first frame fully clears the reservoirs. Camera
    // motion is handled by motion-vector reprojection, and — deliberately — a GEOMETRY EDIT does NOT reset:
    // the world-space reservoirs adapt locally (fresh candidates re-trace the new geometry, the visibility
    // trace drops now-occluded samples, dissimilarity rejects moved surfaces), so editing terrain makes the
    // GI smoothly follow the change over a few frames instead of full-screen clearing.
    let built_gen = resources.built_generation;
    let reset_restir = match resources.dlss_restir_prev {
        None => true,
        Some((r, _vp, _g)) => r != render_res,
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
            wgpu::BindGroupEntry { binding: 11, resource: sky_buf.as_entire_binding() },
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

    // ReSTIR group(2): params + fixed-role reservoirs (a = final/history, b = intermediate).
    // Scale the spatial-reuse radius by the upscale factor so it covers a constant WORLD/output area at
    // upscaling DLSS modes (the knob is in output pixels; the dispatch is at render_res). At DLAA this is 1.0.
    let upscale = full.x as f32 / render_res.x.max(1) as f32;
    let restir_params = RestirParamsData {
        reset: u32::from(reset_restir),
        frame_index,
        viewport_x: render_res.x,
        viewport_y: render_res.y,
        spatial_samples: restir_settings.spatial_samples,
        confidence_weight_cap: restir_settings.confidence_cap,
        spatial_radius: restir_settings.spatial_radius * upscale,
        _pad: 0,
    };
    let restir_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("voxel_rt_dlss_restir_params"),
        contents: bytemuck::bytes_of(&restir_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Phase 2.1 world cache: allocate (once) + build the cache bind groups, BEFORE the immutable
    // reservoir/scene borrows below. Same PERSISTENT world-space cache as the non-DLSS path (one shared
    // `VoxelRtResources.world_cache`), so it keeps accumulating regardless of which present path runs.
    let wc_prepared = prepare_world_cache(
        device,
        &pipelines,
        &mut resources,
        &world_cache_settings,
        frame_index,
        cam_pos.into(),
        &light_buf,
        &sky_buf,
    );

    // Phase 2.4 GPU per-pass timing (editor only) — same pattern as the non-DLSS pass: take the persistent
    // timer out so it can be `&mut`-used alongside the immutable scene/reservoir borrows below, read back last
    // frame's timestamps, and return it at the end. The shared `world_cache`/timer means either present path
    // produces the per-pass numbers.
    #[cfg(feature = "editor")]
    let mut gpu_timer = {
        if !resources.gpu_timer_checked {
            resources.gpu_timer_checked = true;
            resources.gpu_timer = WorldCacheGpuTimer::new(device);
            if resources.gpu_timer.is_none() {
                bevy::log::warn!(
                    "voxel-rt: GPU per-pass timing unavailable — device lacks TIMESTAMP_QUERY / \
                     TIMESTAMP_QUERY_INSIDE_PASSES; world-cache passes run untimed"
                );
            }
        }
        let mut t = resources.gpu_timer.take();
        if let Some(timer) = t.as_mut() {
            timer.read_previous(device, render_queue.get_timestamp_period());
        }
        t
    };

    let (res_a, res_b, _) = resources.dlss_reservoirs.as_ref().expect("allocated above");
    let (surf_a, surf_b, _) = resources.dlss_surfaces.as_ref().expect("allocated above");
    let even = frame_index & 1 == 0;
    // FIXED-ROLE reservoirs (a = history/final, b = intermediate); surfaces still ping-pong. See the non-DLSS
    // pass for the full ordering note — both passes run in one compute dispatch sequence below.
    let (surf_cur, surf_prev) = if even { (surf_a, surf_b) } else { (surf_b, surf_a) };
    let reservoir_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel_rt_dlss_reservoir_bg"),
        layout: &pipelines.reservoir_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: res_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: res_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: restir_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: surf_cur.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: surf_prev.as_entire_binding() },
        ],
    });

    let scene_bg = resources.scene_bind_group.as_ref().expect("checked above");
    // `gi_mode` A/B: two-pass ReSTIR GI (group-2 reservoirs) vs the legacy `gather_gi` DLSS raymarch (no group 2).
    let use_restir = restir_settings.restir;
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
        cpass.set_bind_group(0, scene_bg, &[]);
        // World-cache passes FIRST (shared scene group 0); they set + leave groups 1/2/3, which the DLSS
        // raymarch/restir below rebinds (group 1 = dlss view_bg, group 2 = reservoirs). The cache does NOT
        // feed the live image this stage.
        dispatch_world_cache_passes(
            &mut cpass,
            &pipelines,
            &wc_prepared,
            #[cfg(feature = "editor")]
            gpu_timer.as_ref(),
        );
        cpass.set_bind_group(1, &view_bg, &[]);
        let groups = (render_res.x.div_ceil(8), render_res.y.div_ceil(8), 1);
        if use_restir {
            // Two-pass ReSTIR: pass 1 (initial + reprojected temporal → reservoirs_b + surface) then pass 2
            // (same-frame spatial → reservoirs_a + shade → out_tex + DLSS guides), back-to-back. The intra-pass
            // storage barrier orders p1's reservoirs_b writes before p2 reads them.
            cpass.set_bind_group(2, &reservoir_bg, &[]);
            // group(3) = the world cache (Phase 2.2): `restir_dlss_p1`'s initial reservoir queries it
            // (lazy-insert → populates the cache). Re-set explicitly (rebinding group 2 can drop higher groups).
            cpass.set_bind_group(3, &wc_prepared.cache_bg, &[]);
            #[cfg(feature = "editor")]
            if let Some(t) = gpu_timer.as_ref() {
                t.begin(&mut cpass, 4);
            }
            cpass.set_pipeline(&pipelines.restir_dlss_p1);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
            #[cfg(feature = "editor")]
            if let Some(t) = gpu_timer.as_ref() {
                t.end(&mut cpass, 4); // restir p1
                t.begin(&mut cpass, 5); // restir p2
            }
            cpass.set_pipeline(&pipelines.restir_dlss_p2);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
            #[cfg(feature = "editor")]
            if let Some(t) = gpu_timer.as_ref() {
                t.end(&mut cpass, 5); // restir p2
            }
        } else {
            cpass.set_pipeline(&pipelines.raymarch_dlss);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
    }
    // Phase 2.4: resolve this frame's timestamps (mapped + read next frame). Additive — no GI-math change.
    #[cfg(feature = "editor")]
    if let Some(t) = gpu_timer.as_mut() {
        t.resolve(encoder);
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
    // Return the persistent timer to `resources` (the scene/reservoir borrows above have ended).
    #[cfg(feature = "editor")]
    {
        resources.gpu_timer = gpu_timer.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_render::worldgen::coord::LayerId;

    /// Build a [`VoxelRtStreaming`] in the LATCHED, asset-MISSING Sponza state: `sponza == None` (the `.vox`
    /// never loaded) and `packed_scene == Some(Sponza)` (the switch frame already packed the Cornell fallback
    /// and latched). This is the exact post-switch state that, before the latched guard was re-added, fell
    /// through to `sponza.expect(None)` on frame 2 and panicked. Mirrors `init_voxel_rt_streaming`'s
    /// construction with empty worldgen context (Cornell/Sponza never touch it).
    fn latched_missing_sponza_streaming() -> VoxelRtStreaming {
        let layer = HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default());
        let lib = BiomeLibrary::default();
        let registry = BlockRegistry::from_biome_library(&lib);
        VoxelRtStreaming {
            cfg: StreamingConfig::default(),
            manager: ResidencyManager::new(),
            layer,
            lib,
            registry,
            seed: WORLDGEN_SLICE_SEED,
            last_cam_brick: None,
            cornell_registry: BlockRegistry::cornell(),
            sponza: None, // the missing-asset case: never loaded
            sponza_source: None,
            gallery: None,
            gallery_source: None,
            packed_scene: Some(VoxelScene::Sponza), // already latched on the switch frame
            packed_edit_gen: Some(0),
            worldgen_dirty_pending: false,
            worldgen_frames_since_pack: 0,
            packer: None,
        }
    }

    /// REGRESSION (the documented never-panic path): when Sponza is selected but its `.vox` is absent, the
    /// streaming system must NOT panic on the 2nd+ frame. We drive the REAL `stream_voxel_rt_residency` system
    /// for several ticks in the latched asset-missing state (the switch frame already packed the Cornell
    /// fallback) and assert it runs to completion every tick. Before the latched guard was re-added, frame 2
    /// skipped the switch block (`packed_scene == *scene`) and fell through to `sponza.expect(...)` with
    /// `sponza == None` → panic. No GPU / no render app — this is the main-world system only.
    #[test]
    fn missing_sponza_does_not_panic_across_ticks() {
        let mut app = App::new();
        // Only the resources the main-world system reads (no MinimalPlugins/Input/GPU needed).
        app.insert_resource(VoxelScene::Sponza)
            .init_resource::<VoxelEdits>()
            .init_resource::<VoxelRtPatch>()
            .init_resource::<VoxelRtLighting>()
            .init_resource::<VoxelRtSky>()
            .insert_resource(latched_missing_sponza_streaming())
            .add_systems(Update, stream_voxel_rt_residency);
        // A real camera so `cam.single()` SUCCEEDS and execution reaches the Sponza streaming arm — this is what
        // makes the test actually EXERCISE the panic path: without the latched missing-Sponza guard, frame 2 would
        // fall through to `sponza.expect(None)` and panic. With the guard it returns before the camera query, so
        // the camera's presence is precisely what makes this test FAIL if the guard is ever removed (rather than
        // being silently short-circuited by an empty-query early-return).
        app.world_mut().spawn((SdfCamera, GlobalTransform::default()));

        // Drive 3 ticks: frame "2", "3", "4" of the asset-missing latched state. None may panic; the patch
        // generation must stay put (no streaming work happens — the fallback is already packed + bound).
        let gen_before = app.world().resource::<VoxelRtPatch>().generation;
        for _ in 0..3 {
            app.update();
        }
        let gen_after = app.world().resource::<VoxelRtPatch>().generation;
        assert_eq!(
            gen_before, gen_after,
            "the latched missing-Sponza guard must bail without re-packing (and without panicking)"
        );
        // And the residency never started streaming (sponza stayed None ⇒ no source ⇒ no resident bricks).
        let streaming = app.world().resource::<VoxelRtStreaming>();
        assert!(streaming.sponza.is_none(), "the missing asset stays unloaded");
        assert_eq!(streaming.manager.resident_count(), 0, "no bricks stream while the asset is missing");
    }
}
