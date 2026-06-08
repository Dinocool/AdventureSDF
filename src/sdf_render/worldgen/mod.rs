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
use upload::{HeightRingCpu, build_height_ring, set_cpu_height_ring, set_cpu_terrain_offset};

/// Master switch for the procedural worldgen vertical slice. Default ON so the terrain shows when
/// the editor scene loads; flip off to fall back to a plain authored scene with no streamed terrain.
#[derive(Resource, Clone, Copy)]
pub struct WorldGenEnabled(pub bool);

impl Default for WorldGenEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// Whether the terrain STREAMS with the camera (volume + generation focus follow the camera eye) or
/// stays a FIXED region anchored at the world origin. Default OFF — a stable, reproducible island at
/// the origin (handy for authoring/testing). Toggle ON for free exploration (terrain follows you).
#[derive(Resource, Clone, Copy, Default)]
pub struct WorldGenFollowCamera(pub bool);

/// World-anchored fixed seed for the slice. A real game would source this from the save/session;
/// the slice pins it so the streamed terrain is reproducible across runs.
pub const WORLDGEN_SLICE_SEED: u64 = 0xA15E_C0DE_2026;

/// Generation radius (world metres) the manager keeps resident around the focus. Deliberately LARGER
/// than the terrain volume's half-extent below (480 vs 384, a 96 m margin), so EVERY brick inside the
/// volume samples real generated height. A brick straddling the generated/ungenerated boundary would
/// sample a mix of real height and the miss fallback → a torn/"corrupted" surface at the far extents;
/// the margin keeps the whole volume clear of that boundary. Invariant: `2·radius = 960 < RING·chunk =
/// 8·128 = 1024`, so no two resident chunks alias one ring slot (`slice_radius_respects_ring_invariant`).
pub const WORLDGEN_SLICE_RADIUS: f64 = HEIGHT_CHUNK_CELLS as f64 * 3.75;

/// World half-extent of the single global `Terrain` volume. Strictly LESS than the generation radius
/// (see above) so the whole volume is backed by generated height with a coarse-brick margin.
pub const WORLDGEN_TERRAIN_HALF_XZ: f32 = HEIGHT_CHUNK_CELLS as f32 * 3.0;

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
        app.init_resource::<WorldGenEnabled>()
            .init_resource::<WorldGenFollowCamera>()
            .init_resource::<HeightParams>()
            .init_resource::<WorldGenGpuRing>()
            .insert_resource(LayerManager::new_slice(
                WORLDGEN_SLICE_SEED,
                HeightParams::default(),
                WORLDGEN_SLICE_RADIUS,
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
/// `SdfMaterialSource` + `SdfVolume`). Spawned at IDENTITY; `roll_worldgen` then snaps its translation
/// to follow the camera each chunk crossing (the Terrain sampler is world-anchored, so the moved
/// footprint samples the correct world height via the CPU offset / GPU world-XZ — see `roll_worldgen`).
fn spawn_terrain_volume(mut commands: Commands, existing: Query<(), With<WorldGenTerrainVolume>>) {
    if !existing.is_empty() {
        return; // already spawned (re-entering the editor scene)
    }

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

/// Snap a world-XZ position to the `HEIGHT_CHUNK_CELLS` grid (the streamed-chunk lattice). The
/// terrain volume's translation only moves on chunk crossings, so it stays put within a chunk (no
/// per-frame transform jitter → no constant re-bakes), yet always re-centres near the camera.
fn snap_to_chunk_grid(xz: Vec2) -> Vec2 {
    let cell = HEIGHT_CHUNK_CELLS as f32;
    (xz / cell).round() * cell
}

/// Roll worldgen WITH the camera each frame: stream the rolling window around the camera eye, move
/// the `Terrain` volume to follow the camera (snapped to the chunk grid), and rebake when either the
/// terrain streamed/regenerated OR the volume moved.
///
/// Streaming: feed the camera XZ to the [`LayerManager`]; when it reports a store delta (chunks
/// streamed/dropped) OR the [`HeightParams`] changed (editor tweak ⇒ full regen), rebuild + publish
/// the GPU + CPU height ring and force a rebake (THE mirror of `update_height_field`).
///
/// Following: the `Terrain` volume used to sit at IDENTITY (a fixed ±384 m island at the origin),
/// so its extents never generated as the camera explored past them. Now its `Transform.translation`
/// is set to the camera XZ snapped to the chunk grid (y = 0); the world-anchored ring/CPU-offset make
/// the moved footprint sample the correct world height. The generation radius
/// (`WORLDGEN_SLICE_RADIUS` = 480) stays LARGER than the volume half-extent
/// (`WORLDGEN_TERRAIN_HALF_XZ` = 384), so the WHOLE footprint is backed by generated height (no torn
/// boundary bricks). On a move we publish the new CPU offset and force a rebake (the moved bricks must
/// re-evaluate the ring at their new world XZ).
#[expect(clippy::too_many_arguments, reason = "Bevy system: one param per resource/query it touches")]
fn roll_worldgen(
    mut manager: ResMut<LayerManager>,
    params: Res<HeightParams>,
    mut gpu_ring: ResMut<WorldGenGpuRing>,
    mut atlas: ResMut<SdfAtlas>,
    // The MESH renderer's full-rebake nudge. A ring rebuild (param edit / streaming delta) doesn't
    // change the Terrain volume's content hash (its params are fixed; the ring is a process-global the
    // hash can't see), so the mesh-bake needs an explicit pulse to re-mesh the new surface. (The GPU
    // atlas `rebake_all` below is the gated-off cloud foundation — this is the real on-screen path.)
    mut mesh_rebuild: ResMut<crate::sdf_render::mesh_bake::MeshBakeRebuild>,
    // Read-only camera transform: the generation focus + volume follow target is the camera EYE.
    // (`reframe_worldgen_camera` queries `&mut Transform` With<SdfCamera> in a DIFFERENT system; a
    // second read-only query here is fine — Bevy serializes the conflicting access.)
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
    mut volume: Query<&mut Transform, (With<WorldGenTerrainVolume>, Without<SdfCamera>)>,
    follow: Res<WorldGenFollowCamera>,
    mut last_params: Local<Option<HeightParams>>,
) {
    let _span = crate::instrument::span("worldgen roll");
    let cam_pos = camera.single().map(|t| t.translation).unwrap_or(Vec3::ZERO);

    // Follow mode (toggle, default OFF): the generation window + the Terrain volume track the camera
    // eye XZ, so terrain streams around wherever the camera is. OFF: both stay fixed at the world
    // origin — a stable bounded island (the volume sits at origin, focus = origin).
    let following = follow.0;
    let focus = if following {
        DVec2::new(cam_pos.x as f64, cam_pos.z as f64)
    } else {
        DVec2::ZERO
    };
    let target_xz = if following {
        // Snap to the chunk grid so the volume only moves on chunk crossings (no per-frame jitter / rebake).
        snap_to_chunk_grid(Vec2::new(cam_pos.x, cam_pos.z))
    } else {
        Vec2::ZERO
    };

    // Move the Terrain volume to `target_xz`. On a move (incl. toggling follow off → snap back to
    // origin), publish the CPU world-XZ offset (so the CPU Terrain eval samples the ring at world XZ)
    // and force a rebake of the moved footprint.
    let mut volume_moved = false;
    if let Ok(mut tf) = volume.single_mut() {
        let want = Vec3::new(target_xz.x, 0.0, target_xz.y);
        if tf.translation != want {
            tf.translation = want;
            set_cpu_terrain_offset(target_xz);
            volume_moved = true;
        }
    }

    // Editor param tweak → evict residency so `update` regenerates from the new params. Track the
    // last-applied params so an unchanged frame doesn't needlessly clear (mirrors the fingerprint
    // gate in `update_height_field`).
    let params_changed = *last_params != Some(*params);
    if params_changed {
        manager.set_height_params(*params);
        *last_params = Some(*params);
    }

    // Stream the rolling window (and regenerate if params just changed). `update` returns true iff
    // the store has a pending delta (chunks generated or dropped) — exactly when the ring changed.
    let delta = manager.update(focus);
    if !delta {
        // No terrain change. But if the volume moved this frame (camera crossed a chunk boundary
        // without streaming new chunks — e.g. inside the resident window), the moved footprint must
        // STILL rebake to re-sample the ring at its new world XZ.
        if volume_moved {
            atlas.rebake_all = true;
            // The moved volume's content hash changes (translation ⇒ new `inv_model`), so the mesh
            // bake re-meshes the affected chunks on its own; pulse anyway so the behaviour matches the
            // ring-rebuild path below regardless of where the surface change came from.
            mesh_rebuild.0 = true;
        }
        return;
    }

    // Rebuild the ring from the resident store, publish to the GPU + CPU consumers. Build once and
    // share via clone (the CPU snapshot + the GPU payload) rather than running the fBm twice.
    let ring = build_height_ring(manager.height_store());
    set_cpu_height_ring(Some(Arc::new(ring.clone())));
    gpu_ring.ring = Some(ring);
    gpu_ring.generation = gpu_ring.generation.wrapping_add(1);

    // The ring now reflects the full resident store, so clear the store delta. Otherwise `has_delta()`
    // stays true forever and we'd rebuild the ring + force a full rebake EVERY frame (the window never
    // "settles"). Draining resets the delta so `update` only reports it again when chunks actually
    // stream in or evict (camera move / param change).
    manager.height_store_mut().drain_dirty();
    manager.height_store_mut().drain_dropped();

    // Force a rebake so the regenerated surface is folded into the field. The GPU atlas lever
    // (gated-off cloud foundation) AND the MESH renderer's full-rebake pulse: the Terrain content hash
    // is blind to the ring swap (its params are unchanged), so without this nudge the mesh-bake would
    // keep the stale surface. This is what actually re-meshes the regenerated terrain on screen.
    atlas.rebake_all = true;
    mesh_rebuild.0 = true;
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
    /// terrain band brackets the default height amplitude.
    #[test]
    fn slice_radius_respects_ring_invariant() {
        let ring_span = upload::HEIGHT_RING_CHUNKS as f64 * HEIGHT_CHUNK_CELLS as f64;
        assert!(
            2.0 * WORLDGEN_SLICE_RADIUS < ring_span,
            "2·radius ({}) must be < RING·chunk ({ring_span})",
            2.0 * WORLDGEN_SLICE_RADIUS
        );
    }

    #[test]
    fn terrain_band_brackets_default_amplitude() {
        let amp = HeightParams::default().amplitude;
        assert!(WORLDGEN_TERRAIN_MAX_Y > amp && WORLDGEN_TERRAIN_MIN_Y < -amp);
    }

    /// The follow-snap quantizes the camera XZ to the `HEIGHT_CHUNK_CELLS` grid: the volume only
    /// translates on chunk crossings (so it doesn't re-bake every frame), and a camera within a
    /// chunk maps to the same snapped translation.
    #[test]
    fn follow_snap_quantizes_to_chunk_grid() {
        let cell = HEIGHT_CHUNK_CELLS as f32;
        // Near origin → snaps to 0.
        assert_eq!(snap_to_chunk_grid(Vec2::new(10.0, -20.0)), Vec2::ZERO);
        // Just past half a chunk → snaps to one chunk.
        let half = cell * 0.5 + 1.0;
        assert_eq!(snap_to_chunk_grid(Vec2::new(half, half)), Vec2::splat(cell));
        // Exactly on a chunk multiple stays put; small wander within the same chunk maps to the
        // SAME snapped value (no jitter → no per-frame re-bake).
        let on_grid = Vec2::new(3.0 * cell, -2.0 * cell);
        assert_eq!(snap_to_chunk_grid(on_grid), on_grid);
        assert_eq!(
            snap_to_chunk_grid(on_grid + Vec2::new(cell * 0.2, -cell * 0.3)),
            on_grid
        );
        // Negative coords snap symmetrically (round-half away maps -0.6·cell → -cell).
        assert_eq!(snap_to_chunk_grid(Vec2::splat(-0.6 * cell)), Vec2::splat(-cell));
    }
}
