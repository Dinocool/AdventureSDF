//! Procedural world generator — a LayerProcGen-style layer stack adapted to this engine's GPU SDF
//! clipmap renderer. See `docs/WORLD_GEN_PLAN.md` for the full design.
//!
//! Structure (built bottom-up; each module is independently unit-tested):
//! - [`noise`] — the deterministic, cross-platform noise basis. The foundation of every
//!   *authoritative* (gameplay-relevant) layer: shared-seed multiplayer requires bit-identical
//!   generation across GPU vendors / CPUs / OSes (WORLD_GEN_PLAN §2.8), so authoritative layers run
//!   on the CPU using integer-hash entropy + IEEE basic-op interpolation (no GPU floats, no
//!   transcendentals, no FMA contraction). The `worldgen_parity` test harness pins its outputs.
//!
//! Subsequent modules (coordinates, artifacts, the `Layer` trait, the `LayerManager`, the height
//! layer, and the GPU upload seam) land in later increments of the Phase-1 vertical slice.

pub mod artifact;
pub mod biome;
pub mod coord;
pub mod graph;
pub mod layer;
pub mod layers;
pub mod manager;
pub mod noise;
pub mod plan;
pub mod spline;
pub mod store;
pub mod upload;

use std::sync::Arc;

use bevy::math::DVec2;
use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;

use crate::node::Node3D;
use crate::scene_manager::{AppScene, SceneEntity};
use crate::sdf_render::atlas::SdfAtlas;
use crate::sdf_render::edits::{
    CsgKind, MaterialFields, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive,
};
use crate::sdf_render::{SdfCamera, SdfOrbitCamera, SdfVolume};

use layers::erosion::ErosionParams;
use layers::height::{HEIGHT_CHUNK_CELLS, HeightParams};
use manager::LayerManager;
use upload::{
    HeightRingCpu, build_height_clipmap, build_height_ring, set_cpu_height_clipmap,
    set_cpu_height_ring, set_cpu_terrain_offset,
};

/// Master switch for the procedural worldgen vertical slice. Default ON so the terrain shows when
/// the editor scene loads; flip off to fall back to a plain authored scene with no streamed terrain.
#[derive(Resource, Clone, Copy)]
pub struct WorldGenEnabled(pub bool);

impl Default for WorldGenEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// World-anchored fixed seed for the slice. A real game would source this from the save/session;
/// the slice pins it so the streamed terrain is reproducible across runs.
pub const WORLDGEN_SLICE_SEED: u64 = 0xA15E_C0DE_2026;

/// Tier-0 generation radius (world metres) the manager keeps resident around the focus — the finest
/// clipmap tier's rolling ring window. Each coarser tier `t` uses `WORLDGEN_SLICE_RADIUS·2^t` (handled
/// in [`LayerManager::new_clipmap`]). The render volume no longer needs to fit inside this radius: the
/// volume is world-anchored and effectively infinite (`WORLDGEN_TERRAIN_HALF_XZ` below), and the
/// residency COVERAGE GATE (`mesh_bake::mesh_resident_chunks` + `upload::clipmap_covers_aabb`) is what
/// now guarantees no chunk meshes ground the clipmap hasn't loaded — so the old "radius > volume
/// half-extent margin" rationale is gone (the gate, not a margin, is the guarantee). Per-tier invariant
/// retained: `2·radius = 960 < RING·chunk = 8·128 = 1024` (and it scales with the tier, since both the
/// radius and the chunk size double per tier), so no two resident chunks alias one ring slot
/// (`slice_radius_respects_ring_invariant`).
pub const WORLDGEN_SLICE_RADIUS: f64 = HEIGHT_CHUNK_CELLS as f64 * 3.75;

/// World reach (±metres from the focus) tier `t` of the height clipmap covers:
/// `(HEIGHT_RING_CHUNKS/2)·HEIGHT_CHUNK_CELLS·2^t = 512·2^t` with the defaults. The coarsest tier
/// (`T-1`) must reach the mesh-bake clipmap's coarsest-LOD outer reach so terrain extends to the full
/// LOD-`(lod_count-1)` extent.
pub fn height_tier_reach(tier: u32) -> f64 {
    (upload::HEIGHT_RING_CHUNKS as f64 / 2.0) * HEIGHT_CHUNK_CELLS as f64 * (1u64 << tier) as f64
}

/// SSOT for the height-clipmap TIER COUNT `T`, derived from the mesh-bake clipmap so terrain auto-extends
/// to the full LOD-`(lod_count-1)` reach (and auto-grows if the default `lod_count` changes). Choose the
/// smallest `T` whose COARSEST tier's covered radius (`512·2^(T-1)`) ≥ `reach` (the coarsest-LOD outer
/// half-extent from `mesh_bake::coarsest_lod_outer_reach`). `T ≥ 1` always.
pub fn height_clipmap_tiers(reach: f64) -> u32 {
    let mut t = 1u32;
    while height_tier_reach(t - 1) < reach {
        t += 1;
    }
    t
}

/// World half-extent of the single global `Terrain` volume. Now a LARGE, effectively-infinite extent so
/// the ONE world-anchored volume spans the whole explorable area. World-anchored ⇒ its `inv_model` never
/// changes ⇒ its content hash is stable ⇒ terrain stages PER-CHUNK by construction (no whole-band
/// re-mesh on a camera roll — the old camera-following volume changed `inv_model` every chunk crossing,
/// re-hashing every chunk). What actually meshes is restricted by two camera-driven mechanisms, not by
/// this extent: the mesh-bake CLIPMAP (only chunks near the camera) and the residency COVERAGE GATE
/// (`mesh_bake::mesh_resident_chunks` — a terrain chunk is resident only when its full XZ footprint is
/// covered by loaded height, so an oversized far-LOD chunk can't sample outside the ±radius ring and
/// render a corrupt slab). Phase B Step 2 extends loaded coverage outward (tiered rings).
pub const WORLDGEN_TERRAIN_HALF_XZ: f32 = 131072.0;

/// Safety margin applied to the derived vertical band (covers the ridge fold + central-difference
/// gradient slop + a little headroom). The dist-band cull only bakes the thin surface shell regardless,
/// so a slightly generous AABB costs nothing but keeps the surface comfortably inside the band.
pub const WORLDGEN_TERRAIN_BAND_MARGIN: f32 = 1.2;

/// DERIVE the vertical AABB half-band the global terrain volume occupies from the LIVE height + erosion
/// params, so it tracks the editor sliders (taller mountains ⇒ taller band). The carved surface lives in
/// `sea_level ± swing`, where `swing = amplitude_sum·(1 + ridge) + erosion_strength`, all × the margin.
/// `(1 + ridge)` covers the ridge fold's extra reach (the fold can push toward `amplitude_sum`); erosion
/// only carves DOWN but we pad symmetrically for simplicity. Pure function of the two param resources.
pub fn terrain_band_half(height: &HeightParams, erosion: &ErosionParams) -> f32 {
    let amp_sum = height.amplitude_sum() as f32;
    let ridge_factor = 1.0 + height.ridge.max(0.0);
    let erosion_swing = if erosion.enabled { erosion.strength.max(0.0) } else { 0.0 };
    (amp_sum * ridge_factor + erosion_swing) * WORLDGEN_TERRAIN_BAND_MARGIN
}

/// The terrain volume's vertical band `[min_y, max_y]` derived from the live params (see
/// [`terrain_band_half`]): `sea_level ± terrain_band_half`.
pub fn terrain_band(height: &HeightParams, erosion: &ErosionParams) -> (f32, f32) {
    let half = terrain_band_half(height, erosion);
    (height.sea_level - half, height.sea_level + half)
}

/// The active biome terrain node-graph (the surface the bake samples). A `Resource` so the editor /
/// asset hot-reload can swap it live; the `LayerManager` republishes it into every tier on change
/// (`roll_worldgen` → `set_graph`). Defaults to the "mountains placed in plains" preset (the peaks+plains
/// look); the editor + the shipped `assets/worldgen/*.graph.ron` can replace it.
#[derive(Resource, Clone)]
pub struct WorldGraph(pub Arc<graph::Graph>);

impl Default for WorldGraph {
    fn default() -> Self {
        Self(Arc::new(graph::preset::mountains_plains_graph(graph::preset::MOUNTAINS_PLAINS_AMPLITUDE)))
    }
}

/// Vertical AABB band `[min_y, max_y]` for a terrain GRAPH — derived from the graph's conservative
/// output bound (the tallest peak it can produce) × the safety margin, symmetric about 0 (the graph's
/// node offsets already encode base elevation). Keeps towering graph peaks inside the volume AABB.
pub fn terrain_band_graph(g: &graph::Graph) -> (f32, f32) {
    let half = (g.value_bound() as f32) * WORLDGEN_TERRAIN_BAND_MARGIN;
    (-half, half)
}

/// The built GPU height ring, handed from the main world to the render world's bake extract. Bumps
/// `generation` on every rebuild so the render world re-uploads only on a change (it caches the last
/// uploaded gen). `ring = None` until the first chunk streams in (the bake then binds a dummy).
#[derive(Resource, Default)]
pub struct WorldGenGpuRing {
    pub ring: Option<HeightRingCpu>,
    pub generation: u32,
}

/// Marks the single global worldgen terrain volume (so `roll_worldgen` spawns exactly one).
#[derive(Component)]
struct WorldGenTerrainVolume;

/// Marks the worldgen-provided sun, so the slice is lit even when no scene file (which would
/// otherwise own the `DirectionalLight`) is loaded.
#[derive(Component)]
struct WorldGenSun;

/// The procedural-worldgen plugin: owns the [`LayerManager`], rolls generation around the camera,
/// rebuilds the GPU height ring on a delta, and spawns the one world-spanning [`SdfPrimitive::Terrain`]
/// volume the bake samples. Registered from `SdfScenePlugin::build`.
pub struct WorldGenPlugin;

impl Plugin for WorldGenPlugin {
    fn build(&self, app: &mut App) {
        // Derive the height-clipmap tier count `T` from the mesh-bake clipmap (SSOT): `T` tiers must
        // cover the coarsest-LOD outer reach so terrain extends to the full LOD-`(lod_count-1)` extent.
        let grid = crate::sdf_render::SdfGridConfig::default();
        let mesh_cfg = crate::sdf_render::mesh_bake::MeshBakeConfig::default();
        let reach = crate::sdf_render::mesh_bake::coarsest_lod_outer_reach(&grid, &mesh_cfg) as f64;
        let tiers = height_clipmap_tiers(reach);
        info!(
            "worldgen height clipmap: {tiers} tiers (tier 0 = ±{} m, coarsest tier {} = ±{} m) covering \
             mesh-bake coarsest-LOD reach ±{reach:.0} m",
            height_tier_reach(0),
            tiers - 1,
            height_tier_reach(tiers - 1),
        );

        app.init_resource::<WorldGenEnabled>()
            .init_resource::<HeightParams>()
            .init_resource::<ErosionParams>()
            .init_resource::<WorldGenGpuRing>()
            .init_resource::<WorldGraph>()
            // Biome terrain graphs follow the same resource pipeline as materials (load/hot-reload/save).
            .init_asset::<graph::GraphAsset>()
            .register_asset_loader(graph::GraphAssetLoader)
            .register_type::<graph::GraphAsset>()
            // Stage-1 terrain-materials: the biome/strata/material library (CPU/data only this stage —
            // INERT visually; Stage 2/3 bake + shader consume it). Same load/hot-reload pipeline.
            .init_resource::<biome::BiomeLibrary>()
            .init_asset::<biome::BiomeLibraryAsset>()
            .register_asset_loader(biome::BiomeLibraryAssetLoader)
            .register_type::<biome::BiomeLibraryAsset>()
            .register_type::<biome::TerrainSurfaceMaterial>()
            .register_type::<biome::BiomeDef>()
            .register_type::<biome::StrataLayer>()
            .register_type::<biome::TerrainMatId>()
            .register_type::<biome::BiomeId>()
            .register_type::<biome::BiomeLibrary>()
            .insert_resource(LayerManager::new_clipmap(
                WORLDGEN_SLICE_SEED,
                HeightParams::default(),
                ErosionParams::default(),
                tiers,
            ))
            .register_type::<HeightParams>()
            .register_type::<ErosionParams>()
            // Load the active biome graph from its .ron at startup + hot-reload it into `WorldGraph`
            // (the authored graph "plugs into" the live world; editor Save → .ron → re-mesh).
            .add_systems(Startup, (load_active_graph, load_biome_library))
            .add_systems(Update, (apply_active_graph, apply_biome_library))
            .add_systems(
                OnEnter(AppScene::SdfEditor),
                spawn_terrain_volume.run_if(|e: Res<WorldGenEnabled>| e.0),
            )
            // BEFORE the bake scheduler so a terrain rebuild re-bakes the affected chunks the same
            // frame (mirrors `update_height_field`'s ordering).
            .add_systems(
                Update,
                (
                    // BEFORE the bake scheduler so a terrain rebuild re-bakes the affected chunks the
                    // same frame (mirrors `update_height_field`'s ordering).
                    roll_worldgen.before(crate::sdf_render::bake_scheduler::schedule_bakes),
                    reframe_worldgen_camera,
                )
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|e: Res<WorldGenEnabled>| e.0),
            );

        // Editor-only: the "World Gen" dock panel — Height + Erosion sliders, live re-gen.
        #[cfg(feature = "editor")]
        crate::editor::panels::register_panel(
            app,
            "sdf/worldgen",
            "World Gen",
            crate::editor::panels::DockSide::Right,
            12,
            worldgen_panel,
        );
    }
}

/// Spawn the single world-spanning `Terrain` volume when worldgen is enabled. A low [`SdfOrder`] (0)
/// so authored edits (higher orders) compose OVER the terrain; a plain Union with a default inline
/// material. Mirrors the gallery's volume-spawn shape (`SdfPrimitive` + `SdfOp` + `SdfOrder` +
/// `SdfMaterialSource` + `SdfVolume`).
///
/// Spawned at `Transform::IDENTITY` and STATIC/WORLD-ANCHORED — it NEVER moves (the old camera-follow
/// is gone). World-anchored ⇒ a stable `inv_model` ⇒ a stable content hash ⇒ per-chunk staging by
/// construction (a camera roll no longer re-hashes the whole terrain band). The `half_xz` is a large,
/// effectively-infinite extent (`WORLDGEN_TERRAIN_HALF_XZ`) spanning the whole explorable area; the
/// mesh-bake clipmap (camera-driven) and the residency COVERAGE GATE restrict what actually meshes to
/// the rolling loaded region. Because the volume sits at the origin, local space IS world space, so the
/// CPU terrain offset is ZERO (published once below); `local.xz == world.xz` for the ring sampler.
fn spawn_terrain_volume(
    mut commands: Commands,
    existing: Query<(), With<WorldGenTerrainVolume>>,
    world_graph: Res<WorldGraph>,
) {
    if !existing.is_empty() {
        return; // already spawned (re-entering the editor scene)
    }

    // DERIVE the vertical band from the active terrain GRAPH's conservative peak bound so the volume's
    // AABB covers the tallest peaks it can produce. The narrow-band cull still bakes only the thin shell.
    let (min_y, max_y) = terrain_band_graph(&world_graph.0);

    // World-anchored volume ⇒ local space == world space ⇒ the CPU Terrain eval's world-XZ offset is
    // ZERO. Publish it once (the static defaults to ZERO already; this makes the invariant explicit and
    // resets it if a prior session left a stale follow offset).
    set_cpu_terrain_offset(Vec2::ZERO);

    // The slice's own sun (tagged `SceneEntity` so the SDF lit pass's sun query picks it up — see
    // `render::prepare_sdf_camera`). Without this the terrain renders BLACK when no scene file
    // supplies a `DirectionalLight`. ~10000 lux matches the renderer's exposure (SDF_EXPOSURE_EV100).
    commands.spawn((
        Name::new("Worldgen Sun"),
        DirectionalLight { illuminance: 10000.0, shadows_enabled: true, ..default() },
        // Terrain spans km, so the default (small-scene) cascade range would shadow only a tiny bubble.
        // Four cascades out to 2 km cover the near+mid terrain the camera actually reads; the baked terrain
        // meshes render through Bevy PBR so directional shadows + cascades work natively.
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            maximum_distance: 2000.0,
            ..default()
        }
        .build(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.9, 0.6, 0.0)),
        Node3D,
        SceneEntity,
        WorldGenSun,
    ));

    commands.spawn((
        Name::new("Worldgen Terrain"),
        Transform::IDENTITY,
        SdfPrimitive::Terrain {
            half_xz: Vec2::splat(WORLDGEN_TERRAIN_HALF_XZ),
            min_height: min_y,
            max_height: max_y,
        },
        SdfOp { kind: CsgKind::Union, smoothing: 0.0 },
        SdfOrder(0),
        // Explicit inline material (asset: None ⇒ defined entirely by overrides). A bare
        // `SdfMaterialSource::default()` leaves every override `None`, which the inline path resolves
        // to a ZERO (black) albedo — hence the all-black terrain. Give it a visible mossy-green
        // dielectric so the surface shades correctly out of the box.
        SdfMaterialSource {
            asset: None,
            overrides: MaterialFields {
                base_color: Some([0.16, 0.27, 0.10, 1.0]), // linear RGBA, earthy green
                metallic: Some(0.0),
                roughness: Some(0.95),
                ..default()
            },
        },
        SdfVolume,
        WorldGenTerrainVolume,
    ));
}

/// Roll worldgen each frame: stream the rolling generation window around the camera and rebuild the
/// GPU + CPU height ring on a store delta. The render `Terrain` volume is now WORLD-ANCHORED and never
/// moves (the camera-follow machinery is gone), so this system no longer touches the volume transform.
///
/// Streaming: feed the camera XZ to the [`LayerManager`]; when it reports a store delta (chunks
/// streamed/dropped) OR the [`HeightParams`] changed (editor tweak ⇒ full regen), rebuild + publish the
/// GPU + CPU height ring (the eval needs the fresh data). THE mirror of `update_height_field`.
///
/// Re-mesh pulse: the mesh-bake re-meshes streaming terrain PER-CHUNK by construction now — the
/// world-anchored volume keeps a stable content hash, and the residency COVERAGE GATE
/// (`mesh_bake::mesh_resident_chunks`) enters newly-loaded chunks / reaps evicted ones automatically as
/// the ring rolls. So a plain streaming `delta` does NOT pulse `MeshBakeRebuild`. ONLY an explicit
/// editor terrain-PARAM edit (`params_changed`, a full regen — rare) pulses `mesh_rebuild`/`rebake_all`
/// to re-mesh the whole loaded region against the new params.
#[expect(clippy::too_many_arguments, reason = "Bevy system: one param per resource/query it touches")]
fn roll_worldgen(
    mut manager: ResMut<LayerManager>,
    params: Res<HeightParams>,
    erosion: Res<ErosionParams>,
    world_graph: Res<WorldGraph>,
    // The mesh-bake clipmap config (the LOD slider) — the height-clipmap window tracks its `lod_count`.
    grid_cfg: Res<crate::sdf_render::SdfGridConfig>,
    mesh_cfg: Res<crate::sdf_render::mesh_bake::MeshBakeConfig>,
    mut gpu_ring: ResMut<WorldGenGpuRing>,
    mut atlas: ResMut<SdfAtlas>,
    // The MESH renderer's full-rebake nudge. Pulsed ONLY on an explicit terrain-param edit (full regen);
    // streaming deltas re-mesh per-chunk via the residency coverage gate, no global pulse needed.
    mut mesh_rebuild: ResMut<crate::sdf_render::mesh_bake::MeshBakeRebuild>,
    // Read-only camera transform: the generation focus is the camera EYE.
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
    mut last_params: Local<Option<HeightParams>>,
    mut last_erosion: Local<Option<ErosionParams>>,
    mut last_graph: Local<Option<Arc<graph::Graph>>>,
) {
    let _span = crate::instrument::span("worldgen roll");
    let cam_pos = camera.single().map(|t| t.translation).unwrap_or(Vec3::ZERO);

    // Apply the active biome terrain graph to EVERY tier, republishing on change (editor edit / asset
    // hot-reload / first run). `set_graph` rebuilds all tiers + evicts residency → a full regen, so it
    // runs BEFORE the tier-count/param steps (so appended/rebuilt tiers carry the graph). Cheap when
    // unchanged (a single `Arc` pointer compare).
    let graph_changed = last_graph.as_ref().is_none_or(|g| !Arc::ptr_eq(g, &world_graph.0));
    if graph_changed {
        manager.set_graph(Some(world_graph.0.clone()));
        *last_graph = Some(world_graph.0.clone());
    }

    // The LayerManager's generation window ALWAYS follows the camera (the LayerProcGen GenerationSource):
    // each layer maintains its rolling region around the focus, evicting what leaves it and generating
    // what enters (WORLD_GEN_PLAN §2.7). The render VOLUME no longer follows — it's world-anchored — but
    // the generation focus must still track the viewer so the right region streams in/out.
    let focus = DVec2::new(cam_pos.x as f64, cam_pos.z as f64);

    // DYNAMIC WINDOW: the height-clipmap tier count tracks the live mesh-bake `lod_count` (the LOD slider),
    // so the loaded sample-area window always covers the configured LOD reach. Recompute the needed tiers
    // from the mesh-bake clipmap each frame; `set_tier_count` only changes the stack on an actual change —
    // GROW appends coarse tiers (keeps loaded terrain, no flicker), SHRINK drops the coarsest tiers.
    let reach = crate::sdf_render::mesh_bake::coarsest_lod_outer_reach(&grid_cfg, &mesh_cfg) as f64;
    let tiers_changed = manager.set_tier_count(height_clipmap_tiers(reach), *params, *erosion);

    // Editor param tweak (height OR erosion) → rebuild ALL tiers from the new params + evict residency so
    // `update` regenerates. Tracked so an unchanged frame doesn't needlessly clear. A param edit is an
    // EXPLICIT full regen — the one case that pulses the mesh-bake full-rebake. (A pure tier change keeps
    // the existing params, so it needs no rebuild — `set_params` is only for an actual param edit.)
    let params_changed = *last_params != Some(*params) || *last_erosion != Some(*erosion);
    if params_changed {
        manager.set_params(*params, *erosion);
    }
    if params_changed || tiers_changed {
        *last_params = Some(*params);
        *last_erosion = Some(*erosion);
    }

    // Stream the rolling window (and regenerate if params just changed). `update` returns true iff
    // the store has a pending delta (chunks generated or dropped) — exactly when the ring changed.
    let delta = manager.update(focus);
    if !delta {
        return; // nothing streamed and no param edit → ring unchanged, nothing to publish.
    }

    // Rebuild the tiered CLIPMAP from the resident store and publish it as the CPU snapshot the Terrain
    // eval + coverage gate read (fine-near/coarse-far terrain to the full mesh-bake reach). One ring per
    // tier; tier `t`'s chunk edge is `HEIGHT_CHUNK_CELLS·2^t`.
    let tier_cells: Vec<u32> =
        (0..manager.tier_count()).map(|t| HEIGHT_CHUNK_CELLS << t).collect();
    let clipmap = build_height_clipmap(manager.height_store(), &tier_cells);
    // Publish the tier-0 hi-fi DETAIL-NORMAL sampler FIRST (before the clipmap), so any bake that grabs the
    // new clipmap also sees the matching hi-fi terrain source (raw `sample_world` slope) — the two stay in
    // lockstep (same params/erosion/graph/seed the clipmap was built from). A derived render attribute; the
    // height itself is unchanged (no `HEIGHT_GEN_VERSION` bump).
    upload::set_cpu_terrain_hifi(Some(Arc::new(manager.make_terrain_hifi())));
    set_cpu_height_clipmap(Some(Arc::new(clipmap)));

    // ALSO keep the single tier-0 ring published for the gated GPU bake (which binds ONE ring) + the
    // per-ring parity tests. The GPU path is NOT extended to multiple tiers here (out of scope) — it
    // simply renders tier 0 where present; the on-screen MESH path uses the full clipmap above.
    let ring = build_height_ring(manager.height_store());
    set_cpu_height_ring(Some(Arc::new(ring.clone())));
    gpu_ring.ring = Some(ring);
    gpu_ring.generation = gpu_ring.generation.wrapping_add(1);

    // The ring now reflects the full resident store, so clear the store delta. Otherwise `has_delta()`
    // stays true forever and we'd rebuild the ring EVERY frame (the window never "settles"). Draining
    // resets the delta so `update` only reports it again when chunks actually stream in or evict.
    manager.height_store_mut().drain_dirty();
    manager.height_store_mut().drain_dropped();

    // Re-mesh pulse ONLY on an explicit param edit (a full regen). A plain streaming delta needs NO
    // pulse: the world-anchored volume's content hash is stable, so the residency coverage gate meshes
    // newly-loaded chunks per-chunk and reaps evicted ones on its own (no whole-band re-mesh). On a
    // param edit, every loaded chunk's surface changed, so force a full rebake (atlas lever = gated-off
    // cloud foundation; `mesh_rebuild` = the real on-screen path).
    if params_changed || graph_changed {
        atlas.rebake_all = true;
        mesh_rebuild.0 = true;
    }
}

/// Handle to the active biome terrain graph asset — the production graph the world loads from
/// `assets/worldgen/*.graph.ron` (the editor's Save target). Loading/hot-reloading it republishes into
/// [`WorldGraph`], which `roll_worldgen` re-meshes. This is how the authored graph "plugs into" the world.
#[derive(Resource)]
pub struct ActiveGraphHandle(pub Handle<graph::GraphAsset>);

/// The on-disk graph the world loads by default (relative to `assets/`) — the multi-biome world graph.
const ACTIVE_GRAPH_ASSET: &str = "worldgen/world.graph.ron";

/// Kick off loading the active graph asset at startup (async; the preset default drives the world until
/// it lands, then [`apply_active_graph`] swaps it in).
fn load_active_graph(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(ActiveGraphHandle(assets.load(ACTIVE_GRAPH_ASSET)));
}

/// On load / hot-reload of the active graph asset, republish it into [`WorldGraph`] (→ `roll_worldgen`
/// re-meshes). So editing + saving the `.graph.ron` (in the node editor or on disk) updates the live
/// world; this is the proper resource-pipeline plug-in (same as material hot-reload).
fn apply_active_graph(
    mut events: MessageReader<AssetEvent<graph::GraphAsset>>,
    assets: Res<Assets<graph::GraphAsset>>,
    active: Option<Res<ActiveGraphHandle>>,
    mut world_graph: ResMut<WorldGraph>,
) {
    let Some(active) = active else { return };
    for ev in events.read() {
        let changed = matches!(ev, AssetEvent::Added { id } | AssetEvent::Modified { id } if *id == active.0.id());
        if changed && let Some(asset) = assets.get(&active.0) {
            world_graph.0 = std::sync::Arc::new(asset.graph.clone());
            info!("worldgen: active biome graph loaded ({} nodes)", asset.graph.nodes.len());
        }
    }
}

/// Handle to the active biome library asset (`assets/worldgen/biomes.ron`). Loaded/hot-reloaded and
/// compiled into the [`biome::BiomeLibrary`] resource — Stage-1 terrain-materials data (INERT visually
/// this stage; Stage 2/3 consume the library).
#[derive(Resource)]
pub struct ActiveBiomeLibraryHandle(pub Handle<biome::BiomeLibraryAsset>);

/// The on-disk biome library the world loads by default (relative to `assets/`).
const ACTIVE_BIOMES_ASSET: &str = "worldgen/biomes.ron";

/// Kick off loading the biome library asset at startup (async; compiled into [`biome::BiomeLibrary`]
/// once it lands by [`apply_biome_library`]).
fn load_biome_library(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(ActiveBiomeLibraryHandle(assets.load(ACTIVE_BIOMES_ASSET)));
}

/// On load / hot-reload of the biome library asset, compile + validate it into the
/// [`biome::BiomeLibrary`] resource. Mirrors [`apply_active_graph`]; a malformed/invalid library logs an
/// error and leaves the previous (or empty default) library in place rather than panicking at runtime.
fn apply_biome_library(
    mut events: MessageReader<AssetEvent<biome::BiomeLibraryAsset>>,
    assets: Res<Assets<biome::BiomeLibraryAsset>>,
    active: Option<Res<ActiveBiomeLibraryHandle>>,
    mut library: ResMut<biome::BiomeLibrary>,
) {
    let Some(active) = active else { return };
    for ev in events.read() {
        let changed = matches!(ev, AssetEvent::Added { id } | AssetEvent::Modified { id } if *id == active.0.id());
        if changed && let Some(asset) = assets.get(&active.0) {
            match biome::BiomeLibrary::compile(asset) {
                Ok(lib) => {
                    info!(
                        "worldgen: biome library loaded ({} biomes, {} materials)",
                        lib.biomes.len(),
                        lib.materials.len()
                    );
                    *library = lib;
                }
                Err(e) => error!("worldgen: biome library invalid, keeping previous: {e}"),
            }
        }
    }
}

/// Reframe the orbit camera above the terrain, ONCE, after the camera entity exists.
///
/// Done in `Update` (not `OnEnter`) because the Startup-spawned camera isn't queryable at the first
/// `OnEnter`. Writes the camera `Transform` DIRECTLY: `orbit_camera` only syncs orbit→transform while
/// the pointer is over the viewport (`ViewportInputAllowed`), so otherwise the camera keeps its
/// buried distance-8 startup transform until the user first interacts. We also set the orbit resource
/// so it stays consistent once the user does grab the view.
fn reframe_worldgen_camera(
    mut done: Local<bool>,
    mut orbit: ResMut<SdfOrbitCamera>,
    mut cam: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    if *done {
        return;
    }
    let Ok(mut tf) = cam.single_mut() else {
        return; // camera not spawned yet — retry next frame
    };
    orbit.target = Vec3::ZERO;
    orbit.distance = 320.0;
    orbit.yaw = 0.7;
    orbit.pitch = 0.55;
    *tf = orbit.view_transform();
    *done = true;
}

/// The "World Gen" editor dock panel: live sliders for the height + erosion params. An edit mutates the
/// reflected resource → `roll_worldgen`'s `params_changed` gate rebuilds all tiers + re-meshes the loaded
/// terrain that frame (see [`roll_worldgen`]). Two groups: Height and Erosion.
#[cfg(feature = "editor")]
fn worldgen_panel(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    use bevy_egui::egui::{DragValue, Slider};

    ui.label("Procedural terrain — edits re-mesh the loaded region live.");
    ui.separator();

    // ---- Height group ----
    ui.collapsing("Height", |ui| {
        let p = *world.resource::<HeightParams>();

        let mut octaves = p.octaves;
        if ui.add(Slider::new(&mut octaves, 1..=8).text("Octaves")).changed() {
            world.resource_mut::<HeightParams>().octaves = octaves;
        }
        // base_freq is tiny; expose it as a wavelength (metres) for an intuitive slider.
        let mut wavelength = if p.base_freq > 0.0 { 1.0 / p.base_freq } else { 1536.0 };
        if ui
            .add(Slider::new(&mut wavelength, 256.0..=4096.0).text("Mountain wavelength (m)"))
            .changed()
        {
            world.resource_mut::<HeightParams>().base_freq = 1.0 / wavelength.max(1.0);
        }
        let mut lacunarity = p.lacunarity;
        if ui.add(Slider::new(&mut lacunarity, 1.5..=3.0).text("Lacunarity")).changed() {
            world.resource_mut::<HeightParams>().lacunarity = lacunarity;
        }
        let mut gain = p.gain;
        if ui.add(Slider::new(&mut gain, 0.2..=0.7).text("Gain")).changed() {
            world.resource_mut::<HeightParams>().gain = gain;
        }
        let mut amplitude = p.amplitude;
        if ui.add(Slider::new(&mut amplitude, 10.0..=800.0).text("Amplitude (m)")).changed() {
            world.resource_mut::<HeightParams>().amplitude = amplitude;
        }
        let mut ridge = p.ridge;
        if ui
            .add(Slider::new(&mut ridge, 0.0..=1.0).text("Ridge (0=fBm, 1=sharp peaks)"))
            .changed()
        {
            world.resource_mut::<HeightParams>().ridge = ridge;
        }
        let mut sea_level = p.sea_level;
        if ui.add(Slider::new(&mut sea_level, -200.0..=200.0).text("Sea level (m)")).changed() {
            world.resource_mut::<HeightParams>().sea_level = sea_level;
        }
        let mut band_limit = p.band_limit;
        if ui
            .add(Slider::new(&mut band_limit, 0.0..=8.0).text("Crest band-limit (node radii)"))
            .changed()
        {
            world.resource_mut::<HeightParams>().band_limit = band_limit;
        }
    });

    // ---- Erosion group ----
    ui.collapsing("Erosion", |ui| {
        let e = *world.resource::<ErosionParams>();

        let mut enabled = e.enabled;
        if ui.checkbox(&mut enabled, "Enabled").changed() {
            world.resource_mut::<ErosionParams>().enabled = enabled;
        }
        let mut strength = e.strength;
        if ui.add(Slider::new(&mut strength, 0.0..=200.0).text("Strength / carve depth (m)")).changed() {
            world.resource_mut::<ErosionParams>().strength = strength;
        }
        let mut octaves = e.octaves;
        if ui.add(Slider::new(&mut octaves, 1..=8).text("Octaves")).changed() {
            world.resource_mut::<ErosionParams>().octaves = octaves;
        }
        let mut base_cell = e.base_cell_size;
        if ui
            .add(Slider::new(&mut base_cell, 64.0..=2048.0).text("Base cell size (m)"))
            .changed()
        {
            world.resource_mut::<ErosionParams>().base_cell_size = base_cell;
        }
        let mut lacunarity = e.lacunarity;
        if ui.add(Slider::new(&mut lacunarity, 1.5..=3.0).text("Lacunarity")).changed() {
            world.resource_mut::<ErosionParams>().lacunarity = lacunarity;
        }
        let mut gain = e.gain;
        if ui.add(Slider::new(&mut gain, 0.2..=0.7).text("Gain")).changed() {
            world.resource_mut::<ErosionParams>().gain = gain;
        }
        let mut gully = e.gully_weight;
        if ui.add(Slider::new(&mut gully, 0.0..=2.0).text("Gully weight")).changed() {
            world.resource_mut::<ErosionParams>().gully_weight = gully;
        }
        let mut fade = e.peak_valley_fade;
        if ui.add(Slider::new(&mut fade, 0.0..=1.0).text("Peak/valley fade")).changed() {
            world.resource_mut::<ErosionParams>().peak_valley_fade = fade;
        }
        let mut salt = e.seed_salt;
        if ui.add(DragValue::new(&mut salt).prefix("Seed salt: ")).changed() {
            world.resource_mut::<ErosionParams>().seed_salt = salt;
        }
    });
}

#[cfg(test)]
mod plugin_tests {
    use super::*;

    /// The slice invariants hold: `2·radius < RING·chunk_size` (no ring-slot aliasing), and the
    /// terrain band brackets the default height amplitude. Both the radius and the chunk size double per
    /// tier, so the tier-0 check implies it for every tier.
    #[test]
    fn slice_radius_respects_ring_invariant() {
        let ring_span = upload::HEIGHT_RING_CHUNKS as f64 * HEIGHT_CHUNK_CELLS as f64;
        assert!(
            2.0 * WORLDGEN_SLICE_RADIUS < ring_span,
            "2·radius ({}) must be < RING·chunk ({ring_span})",
            2.0 * WORLDGEN_SLICE_RADIUS
        );
    }

    /// The derived tier count `T` covers the mesh-bake coarsest-LOD reach for the DEFAULT configs — so
    /// "terrain to the full LOD-8 reach" is guaranteed, and `T` auto-extends if the default `lod_count`
    /// changes. Asserts `512·2^(T-1) ≥ reach` and that `T-1` is the SMALLEST such tier (no over-build).
    #[test]
    fn clipmap_tiers_cover_mesh_bake_reach() {
        let grid = crate::sdf_render::SdfGridConfig::default();
        let mesh_cfg = crate::sdf_render::mesh_bake::MeshBakeConfig::default();
        let reach = crate::sdf_render::mesh_bake::coarsest_lod_outer_reach(&grid, &mesh_cfg) as f64;
        let t = height_clipmap_tiers(reach);
        assert!(t >= 1);
        assert!(
            height_tier_reach(t - 1) >= reach,
            "coarsest tier {} covers ±{} m but reach is ±{reach} m",
            t - 1,
            height_tier_reach(t - 1),
        );
        // Minimal: one fewer tier would NOT cover (unless T==1, i.e. tier 0 already covers).
        if t > 1 {
            assert!(
                height_tier_reach(t - 2) < reach,
                "T is not minimal: tier {} already covers ±{} m ≥ reach ±{reach} m",
                t - 2,
                height_tier_reach(t - 2),
            );
        }
    }

    /// The DERIVED terrain band brackets the full default surface swing (fBm amplitude sum × the ridge
    /// fold + erosion strength, with margin) — so the volume AABB always contains the carved surface.
    #[test]
    fn terrain_band_brackets_default_surface_swing() {
        let h = HeightParams::default();
        let e = ErosionParams::default();
        let (min_y, max_y) = terrain_band(&h, &e);
        // Bare amplitude-sum swing (no ridge/erosion/margin) must be comfortably inside the band.
        let amp_sum = h.amplitude_sum() as f32;
        assert!(max_y > amp_sum && min_y < -amp_sum, "band [{min_y},{max_y}] must bracket ±{amp_sum}");
        // And the band grows with erosion strength.
        let stronger = ErosionParams { strength: e.strength + 100.0, ..e };
        assert!(terrain_band_half(&h, &stronger) > terrain_band_half(&h, &e));
    }

    /// The world-anchored terrain volume's half-extent is large enough to span the whole explorable
    /// area (effectively infinite vs the ±radius loaded ring): the coverage gate, not this extent,
    /// bounds what meshes — so the volume can sit static at the origin without re-hashing on a roll.
    #[test]
    fn terrain_volume_is_effectively_infinite() {
        assert!(WORLDGEN_TERRAIN_HALF_XZ > WORLDGEN_SLICE_RADIUS as f32 * 100.0);
    }
}
