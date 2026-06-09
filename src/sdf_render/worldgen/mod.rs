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
pub mod coord;
pub mod layer;
pub mod layers;
pub mod manager;
pub mod noise;
pub mod plan;
pub mod store;
pub mod upload;

use std::sync::Arc;

use bevy::math::DVec2;
use bevy::prelude::*;

use crate::node::Node3D;
use crate::scene_manager::{AppScene, SceneEntity};
use crate::sdf_render::atlas::SdfAtlas;
use crate::sdf_render::edits::{
    CsgKind, MaterialFields, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive,
};
use crate::sdf_render::{SdfCamera, SdfOrbitCamera, SdfVolume};

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

/// Vertical AABB band the global terrain volume occupies. Tightened to bound the height layer's full
/// fBm swing (default ≈ Σ octave amplitudes ≈ 70 m) with margin — NOT the old ±256 m. The bake's
/// dist-band cull only bakes the thin surface shell regardless, but a tight AABB keeps the BVH/brick
/// classification focused on where the surface actually is, cutting wasted far/empty bricks.
pub const WORLDGEN_TERRAIN_MIN_Y: f32 = -96.0;
pub const WORLDGEN_TERRAIN_MAX_Y: f32 = 96.0;

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
            .init_resource::<WorldGenGpuRing>()
            .insert_resource(LayerManager::new_clipmap(
                WORLDGEN_SLICE_SEED,
                HeightParams::default(),
                tiers,
            ))
            .register_type::<HeightParams>()
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
fn spawn_terrain_volume(mut commands: Commands, existing: Query<(), With<WorldGenTerrainVolume>>) {
    if !existing.is_empty() {
        return; // already spawned (re-entering the editor scene)
    }

    // World-anchored volume ⇒ local space == world space ⇒ the CPU Terrain eval's world-XZ offset is
    // ZERO. Publish it once (the static defaults to ZERO already; this makes the invariant explicit and
    // resets it if a prior session left a stale follow offset).
    set_cpu_terrain_offset(Vec2::ZERO);

    // The slice's own sun (tagged `SceneEntity` so the SDF lit pass's sun query picks it up — see
    // `render::prepare_sdf_camera`). Without this the terrain renders BLACK when no scene file
    // supplies a `DirectionalLight`. ~10000 lux matches the renderer's exposure (SDF_EXPOSURE_EV100).
    commands.spawn((
        Name::new("Worldgen Sun"),
        DirectionalLight { illuminance: 10000.0, shadows_enabled: false, ..default() },
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
            min_height: WORLDGEN_TERRAIN_MIN_Y,
            max_height: WORLDGEN_TERRAIN_MAX_Y,
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
) {
    let _span = crate::instrument::span("worldgen roll");
    let cam_pos = camera.single().map(|t| t.translation).unwrap_or(Vec3::ZERO);

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
    let tiers_changed = manager.set_tier_count(height_clipmap_tiers(reach), *params);

    // Editor param tweak → rebuild ALL tiers from the new params + evict residency so `update` regenerates.
    // Tracked so an unchanged frame doesn't needlessly clear. A param edit is an EXPLICIT full regen — the
    // one case that pulses the mesh-bake full-rebake. (A pure tier change keeps the existing fBm params, so
    // it needs no rebuild — `set_height_params` is only for an actual param edit.)
    let params_changed = *last_params != Some(*params);
    if params_changed {
        manager.set_height_params(*params);
    }
    if params_changed || tiers_changed {
        *last_params = Some(*params);
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
    if params_changed {
        atlas.rebake_all = true;
        mesh_rebuild.0 = true;
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

    #[test]
    fn terrain_band_brackets_default_amplitude() {
        let amp = HeightParams::default().amplitude;
        assert!(WORLDGEN_TERRAIN_MAX_Y > amp && WORLDGEN_TERRAIN_MIN_Y < -amp);
    }

    /// The world-anchored terrain volume's half-extent is large enough to span the whole explorable
    /// area (effectively infinite vs the ±radius loaded ring): the coverage gate, not this extent,
    /// bounds what meshes — so the volume can sit static at the origin without re-hashing on a roll.
    #[test]
    fn terrain_volume_is_effectively_infinite() {
        assert!(WORLDGEN_TERRAIN_HALF_XZ > WORLDGEN_SLICE_RADIUS as f32 * 100.0);
    }
}
