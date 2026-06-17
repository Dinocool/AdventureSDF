//! Voxel ray-tracing engine — **Stage 1**: voxelize the procedural worldgen into 0.05 m cubes and render
//! a small patch around the origin as flat-lit coloured cubes. The first visual proof of the
//! voxelization + palette + terrain-height chain (the SDF/mesh-bake renderers were pruned in the
//! voxel-RT rebuild; this is the seed of the replacement).
//!
//! PIPELINE (all CPU, Stage 1):
//! 1. [`palette`] — a [`palette::BlockRegistry`] built from the worldgen [`BiomeLibrary`] palette
//!    (`TerrainMatId → BlockId → colour`, one SSOT).
//! 2. [`brickmap`] — a sparse [`brickmap::BrickMap`] of `8³` bricks (0.05 m voxels), empty bricks absent.
//! 3. [`voxelize`] — [`voxelize::voxelize_brick`] samples the REAL [`HeightLayer::sample_world`] surface
//!    + climate/strata materials per voxel.
//! 4. [`VoxelPlugin`] — on startup, voxelize a bounded patch around the origin and spawn ONE shared cube
//!    mesh per EXPOSED-surface voxel (buried voxels are skipped to keep entity count low), coloured by a
//!    cached per-`BlockId` [`StandardMaterial`]. Adds its own sun + ambient so the patch is lit.
//!
//! WORLDGEN ACCESS (Stage 1): the streaming `LayerManager` makes synchronous sampling awkward, so we
//! construct a [`HeightLayer`] DIRECTLY from the worldgen param resources ([`HeightParams`] /
//! [`ErosionParams`] / [`WorldGraph`] / [`WorldBiomeShapes`]) and call `sample_world` — a deterministic
//! direct sample that matches what the streamed terrain would produce. The [`BiomeLibrary`] is parsed
//! synchronously from `assets/worldgen/biomes.ron` (robust — no async-load ordering for the first proof).

pub mod brickmap;
pub mod cornell;
pub mod edits;
pub mod gallery;
pub mod gpu;
/// Stage G1 — the GPU brick voxelizer (host side of `worldgen_voxelize.wgsl`): assemble the compute
/// shader + flatten the worldgen library/registry/brick into its uniform. Correctness-only (not yet live).
pub mod gpu_voxelize;
pub mod incremental;
pub mod palette;
/// Stage 6 — voxel physics (the player walks the cubes). Feature-gated on `physics` (pulls `rapier3d`).
#[cfg(feature = "physics")]
pub mod physics;
pub mod raytrace;
/// Phase G "G-c.0" — the GPU-resident sparse brick OCCUPANCY structure (the face-cull input for the
/// GPU-driven readback-free streaming front end). Built + uploaded from the static `.vxo`/merged source;
/// wired to NO pipeline yet (no behaviour change). See `docs/PHASE_G_GC_PLAN.md` §2.2.
pub mod residency_gpu;
/// Phase G "G-c.4" — the LIVE readback-free GPU residency FRONT END (the production home of the GPU-driven
/// pipeline proven in `tests/voxel_gpu_residency_converge.rs`). Drives the residency DECISION + pack + AABB-fill
/// on the GPU into the live scene pool, gated by the non-blocking change_count mirror. See `PHASE_G_GC_PLAN.md` §1/§3/§4.
pub mod residency_front_end;
/// Phase G "G-c.4-paging" — the STREAMED `.vxo` region PREFETCHER + the demand-paged GPU occupancy / core store
/// that drive the GPU residency front end over a region-paged `.vxo` (Bistro), constant-RAM + readback-free. See
/// `PHASE_G_GC_PLAN.md` §8.
pub mod residency_pager;
pub mod source;
pub mod streaming;
pub mod vox;
pub mod voxelize;
pub mod vxo;

use bevy::math::IVec3;
use bevy::prelude::*;

use crate::sdf_render::{SdfCamera, SdfCameraMode, SdfOrbitCamera};
use crate::sdf_render::worldgen::biome::{BiomeLibrary, BiomeLibraryAsset};
use crate::sdf_render::worldgen::coord::LayerId;
use crate::sdf_render::worldgen::layers::erosion::ErosionParams;
use crate::sdf_render::worldgen::layers::height::{HeightLayer, HeightParams};
use crate::sdf_render::worldgen::{WORLDGEN_SLICE_SEED, WorldBiomeShapes, WorldGraph};

use brickmap::{BRICK_EDGE, BrickMap, VOXEL_SIZE};
use palette::BlockRegistry;
use voxelize::voxelize_brick;

/// Half-extent (world metres) of the Stage-1 voxel patch in X and Z around the origin — a `32 m × 32 m`
/// patch (`±16 m`). Bounded so the cube-entity count stays small (exposed-surface voxels only).
pub const PATCH_HALF_XZ: f32 = 16.0;

/// How far BELOW the local surface the patch voxelizes (metres). A few metres of depth so the dug strata
/// (surface → sub-surface → stone) are present beneath the visible skin; only the exposed shell spawns.
pub const PATCH_DEPTH_BELOW: f32 = 4.0;

/// How far ABOVE the local surface the patch voxelizes (metres) — a small headroom so the topmost surface
/// voxels are captured (and any overhang would be, though Stage-1 terrain is a heightfield).
pub const PATCH_HEIGHT_ABOVE: f32 = 1.0;

/// Which voxel scene the engine renders. The DEFAULT is [`VoxelScene::Sponza`] — a baked classic
/// GI-measurement scene (the Crytek Sponza atrium voxelized once offline into `assets/models/sponza.vox`) that
/// now STREAMS through the SAME camera-following clipmap residency as worldgen (via a
/// [`source::StaticVoxSource`] over the loaded `.vox` [`BrickMap`]) — NOT the old pack-once static path. Only
/// [`VoxelScene::Cornell`] is still fully resident: a static Cornell box, the canonical GI correctness anchor
/// (colour bleed, an emissive area light, soft shadows). [`VoxelScene::Worldgen`] — the LARGE, streamed,
/// GI-rich procedural terrain (Phase 2.6: sky GI on slopes, multi-bounce fill in deep valleys, emissive
/// lava/crystal colour bleed, the world-cache at scale), now PERF-OPTIMIZED (cold fill ~1.7 s, ~2.5 ms/frame
/// streaming drain). All three stay reachable via the **`V`** toggle (and the editor scene selector). The
/// single SSOT knob the streaming + camera-framing systems read to decide which path runs.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VoxelScene {
    /// Static Cornell box — fully resident, no streaming. The GI correctness anchor (reachable via **`V`**).
    Cornell,
    /// Infinite streaming worldgen terrain (the original Stage-3 path) — the large GI-rich scene, now
    /// perf-optimized (cold fill ~1.7 s, ~2.5 ms/frame streaming drain).
    Worldgen,
    /// Baked `.vox` scene (Sponza) — loaded from `assets/models/sponza.vox` once and then STREAMED through the
    /// SAME clipmap residency worldgen uses (a [`source::StaticVoxSource`] over the loaded [`BrickMap`]: LOD0
    /// extract / coarse mip-pyramid downsample, all-air outside its bounds so the clipmap naturally bounds the
    /// building). NOT pack-once-static anymore — Sponza supports LOD/clipmaps + editing exactly like worldgen.
    /// The default boot scene: a classic GI-measurement atrium (strong single + multi-bounce colour bleed off
    /// the floor + coloured drapes under a raking sun).
    #[default]
    Sponza,
    /// Several baked `.vox` scenes placed SIDE BY SIDE in one world for a GI / LOD comparison row (the
    /// pre-instancing MERGE — see [`gallery`]). On the switch in, the DATA-DRIVEN [`gallery::GALLERY_SCENES`]
    /// list is loaded + merged ONCE into a single [`BrickMap`] + [`BlockRegistry`] (each scene shifted into a
    /// non-overlapping region, palettes concatenated) and cached; that merged map then STREAMS through the
    /// IDENTICAL [`source::StaticVoxSource`] + clipmap residency Sponza uses (source built ONCE on the switch,
    /// never per frame). Starts as just Sponza but is trivially extensible — add a `GalleryEntry` row per baked
    /// classic (Sibenik / San Miguel / …); absent assets are skipped with a `warn!` (never a panic).
    Gallery,
}

impl VoxelScene {
    /// True iff this is the static Cornell scene (no streaming, one-shot residency + framing).
    #[inline]
    pub fn is_cornell(self) -> bool {
        matches!(self, VoxelScene::Cornell)
    }

    /// True iff this is the PROCEDURAL worldgen scene. Note this is NOT "the only streamed scene": Sponza now
    /// streams through the same clipmap residency too (via a [`source::StaticVoxSource`]). The distinction this
    /// predicate draws is the SOURCE — worldgen samples the procedural surface; Sponza reads a baked `.vox`
    /// map; Cornell is the only remaining fully-resident, packed-once box.
    #[inline]
    pub fn is_worldgen(self) -> bool {
        matches!(self, VoxelScene::Worldgen)
    }

    /// The next scene in the **`V`**-key cycle: Sponza → Cornell → Gallery → Sponza. The SSOT for the cycle order,
    /// shared by the keyboard toggle and (for parity) any other caller that wants "advance the scene". **Worldgen
    /// is SHELVED (2026-06) and excluded from the cycle** — the variant still exists (reachable programmatically /
    /// for when worldgen un-shelves) but `V` skips it; a current Worldgen scene advances to Gallery.
    #[inline]
    pub fn next(self) -> Self {
        match self {
            VoxelScene::Sponza => VoxelScene::Cornell,
            VoxelScene::Cornell => VoxelScene::Gallery,
            VoxelScene::Worldgen => VoxelScene::Gallery,
            VoxelScene::Gallery => VoxelScene::Sponza,
        }
    }
}

/// Stage-1 voxel engine plugin: builds the block registry + a small voxelized patch around the origin and
/// drives the camera reframe onto it. The patch is rendered by the HW-RT path ([`raytrace::VoxelRtPlugin`]),
/// not cube entities. Registered from `main.rs`.
pub struct VoxelPlugin;

impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VoxelScene>()
            .init_resource::<SceneReframed>()
            .add_systems(Update, switch_voxel_scene_input);
        // The Stage-1 cube ENTITIES are no longer spawned: `StandardMaterial`'s bindless PBR shader is broken
        // on the wgpu-trunk fork (`the wgpu_binding_array enable extension is not enabled`), so the cubes
        // never rendered anyway. The HW-RT path ([`raytrace::VoxelRtPlugin`], default ON) is now the renderer.
        // The voxelization SSOT (brickmap / voxelize / streaming) is untouched — only cube spawning is gone.
        //
        // Reframe the editor orbit camera onto the patch in Update with a one-shot latch (the Startup-spawned
        // camera isn't queryable on the first frame, and worldgen frames the camera for km-scale terrain at
        // distance 320, which would make the 0.05 m voxels invisible).
        app.add_systems(Update, reframe_camera_on_patch);

        // Stage 6 — voxel physics (the player walks the cubes). First-person: `P` drops you into the
        // Cornell box and the SdfCamera becomes your eyes. Gated on `physics` (rapier3d) + the SdfEditor
        // scene (where the SdfCamera is active). The toggle/rebuild/move run in order each frame; the
        // orbit/fps camera input is already gated off while `SdfCameraMode::player` is on (SdfScenePlugin).
        #[cfg(feature = "physics")]
        {
            use crate::scene_manager::AppScene;
            app.init_resource::<physics::VoxelColliders>()
                .init_resource::<physics::VoxelWalk>()
                .add_systems(
                    Update,
                    (
                        physics::toggle_walk_mode,
                        physics::rebuild_walk_colliders,
                        physics::walk_player,
                    )
                        .chain()
                        .run_if(in_state(AppScene::SdfEditor)),
                );
        }
    }
}

/// Build a [`HeightLayer`] from the live worldgen param resources — the DIRECT-construction path (the
/// streaming `LayerManager` is awkward to sample synchronously; a direct deterministic sample is fine for
/// Stage 1 and matches the streamed surface bit-for-bit, since both evaluate the same `sample_world`). The
/// default [`WorldGraph`] / [`WorldBiomeShapes`] reproduce exactly the rendered terrain's surface.
fn build_height_layer(
    height: &HeightParams,
    erosion: &ErosionParams,
    graph: &WorldGraph,
    biome_shapes: &WorldBiomeShapes,
) -> HeightLayer {
    HeightLayer::new(LayerId(0), *height, *erosion)
        .with_graph(Some(graph.0.clone()))
        .with_biome_shapes(biome_shapes.0.clone())
}

/// Load the [`BiomeLibrary`] synchronously from `assets/worldgen/biomes.ron`, or fall back to the empty
/// default if the file is missing/invalid (logged). Stage-1 robustness: a missing/invalid library yields
/// an all-air patch rather than a panic. (We parse it directly instead of waiting on the async asset
/// loader so the Startup voxelization has the data immediately.)
fn load_biome_library_sync() -> BiomeLibrary {
    let path = std::path::Path::new("assets/worldgen/biomes.ron");
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            warn!("voxel: could not read {}: {e} — using empty biome library", path.display());
            return BiomeLibrary::default();
        }
    };
    let asset = match ron::de::from_bytes::<BiomeLibraryAsset>(&bytes) {
        Ok(a) => a,
        Err(e) => {
            warn!("voxel: biomes.ron parse error: {e} — using empty biome library");
            return BiomeLibrary::default();
        }
    };
    match BiomeLibrary::compile(&asset) {
        Ok(lib) => lib,
        Err(e) => {
            warn!("voxel: biome library invalid: {e} — using empty biome library");
            BiomeLibrary::default()
        }
    }
}

/// The integer voxel Y range `[min, max]` the patch covers, derived from the worldgen surface height range
/// over the XZ patch (sampled on a coarse grid) padded by [`PATCH_DEPTH_BELOW`] / [`PATCH_HEIGHT_ABOVE`].
/// Returns world-VOXEL coordinates (metres / VOXEL_SIZE). Pure function of the layer + seed + patch consts.
fn patch_y_voxel_range(layer: &HeightLayer, seed: u64) -> (i32, i32) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    // Coarse 9×9 probe of the patch surface to bound the height band (cheap; the surface is band-limited).
    let probes = 8;
    for j in 0..=probes {
        for i in 0..=probes {
            let wx = (-PATCH_HALF_XZ as f64) + (i as f64 / probes as f64) * (2.0 * PATCH_HALF_XZ as f64);
            let wz = (-PATCH_HALF_XZ as f64) + (j as f64 / probes as f64) * (2.0 * PATCH_HALF_XZ as f64);
            let h = layer.sample_world(wx, wz, seed).height as f64;
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    let min_y = ((lo - PATCH_DEPTH_BELOW as f64) / VOXEL_SIZE as f64).floor() as i32;
    let max_y = ((hi + PATCH_HEIGHT_ABOVE as f64) / VOXEL_SIZE as f64).ceil() as i32;
    (min_y, max_y)
}

/// Voxelize the bounded origin patch into a [`BrickMap`] (only the bricks covering the patch XZ × the
/// derived surface Y band are voxelized; empty bricks are absent). Returns the map + the patch's voxel
/// XZ/Y bounds (for the exposed-voxel scan, which must clip to the patch so out-of-patch neighbours read
/// as air, exposing the patch's outer faces too).
fn voxelize_patch(
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    registry: &BlockRegistry,
    seed: u64,
) -> BrickMap {
    let mut map = BrickMap::new();
    // Voxel XZ bounds of the patch.
    let half_v = (PATCH_HALF_XZ / VOXEL_SIZE).ceil() as i32;
    let (min_vy, max_vy) = patch_y_voxel_range(layer, seed);

    // Brick coordinate bounds covering the patch (Euclidean floor of the voxel bounds by BRICK_EDGE).
    let bmin = IVec3::new(-half_v, min_vy, -half_v);
    let bmax = IVec3::new(half_v, max_vy, half_v);
    let bc_min = IVec3::new(bmin.x.div_euclid(BRICK_EDGE), bmin.y.div_euclid(BRICK_EDGE), bmin.z.div_euclid(BRICK_EDGE));
    let bc_max = IVec3::new(bmax.x.div_euclid(BRICK_EDGE), bmax.y.div_euclid(BRICK_EDGE), bmax.z.div_euclid(BRICK_EDGE));

    for bz in bc_min.z..=bc_max.z {
        for by in bc_min.y..=bc_max.y {
            for bx in bc_min.x..=bc_max.x {
                let coord = IVec3::new(bx, by, bz);
                let brick = voxelize_brick(coord, 0, layer, lib, registry, seed); // static patch = all LOD0
                map.insert(coord, brick); // empty bricks are dropped by insert
            }
        }
    }
    map
}

/// Public wrappers re-exporting the Stage-1 voxelization helpers so the Stage-2 HW-RT path
/// ([`raytrace`]) can voxelize the SAME patch from the same SSOT (no divergence between the cube view and
/// the ray-traced view). Thin pass-throughs to the private functions above.
pub fn build_height_layer_pub(
    height: &HeightParams,
    erosion: &ErosionParams,
    graph: &WorldGraph,
    biome_shapes: &WorldBiomeShapes,
) -> HeightLayer {
    build_height_layer(height, erosion, graph, biome_shapes)
}

/// See [`build_height_layer_pub`] — public wrapper over the Stage-1 biome-library loader.
pub fn load_biome_library_pub() -> BiomeLibrary {
    load_biome_library_sync()
}

/// See [`build_height_layer_pub`] — public wrapper over the Stage-1 patch voxelizer.
pub fn voxelize_patch_pub(
    layer: &HeightLayer,
    lib: &BiomeLibrary,
    registry: &BlockRegistry,
    seed: u64,
) -> BrickMap {
    voxelize_patch(layer, lib, registry, seed)
}

/// Runtime input: press **V** to cycle the voxel scene Sponza → Cornell → Worldgen → Sponza. Switching
/// resets the one-shot camera-reframe latch (via [`SceneReframed`]) so the camera re-frames onto the
/// newly-selected scene next frame. The editor scene selector mutates the SAME [`VoxelScene`] resource (and
/// must reset the latch the same way) — this is one of two entry points to the single SSOT.
fn switch_voxel_scene_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut scene: ResMut<VoxelScene>,
    mut reframed: ResMut<SceneReframed>,
) {
    if keys.just_pressed(KeyCode::KeyV) {
        *scene = scene.next();
        reframed.0 = false; // re-frame the camera onto the new scene
        info!("voxel scene: {:?}", *scene);
    }
}

/// Latch for the one-shot camera reframe: `true` once the camera has been framed onto the CURRENT scene.
/// A resource (not a system `Local`) so [`switch_voxel_scene_input`] can reset it on a scene switch, forcing
/// [`reframe_camera_on_patch`] to re-frame onto the newly-selected scene.
#[derive(Resource, Default)]
pub struct SceneReframed(pub bool);

/// One-shot: frame the editor orbit camera onto the current voxel scene. For [`VoxelScene::Cornell`] it
/// frames the OPEN front of the static box (camera outside `-Z`, looking `+Z` so the box fills the view).
/// For [`VoxelScene::Worldgen`] it frames the origin-surface patch (the original behaviour — runs after
/// worldgen's own km-scale reframe, which is far too far for 0.05 m voxels). Latches via [`SceneReframed`] so
/// the user can move the camera freely afterward; the latch resets on a scene switch.
#[allow(clippy::too_many_arguments)]
fn reframe_camera_on_patch(
    mut reframed: ResMut<SceneReframed>,
    scene: Res<VoxelScene>,
    mut commands: Commands,
    height: Res<HeightParams>,
    erosion: Res<ErosionParams>,
    graph: Res<WorldGraph>,
    biome_shapes: Res<WorldBiomeShapes>,
    mut orbit: ResMut<SdfOrbitCamera>,
    mut mode: ResMut<SdfCameraMode>,
    mut cam: Query<(Entity, &mut Transform), With<SdfCamera>>,
) {
    if reframed.0 {
        return;
    }
    let Ok((cam_entity, mut tf)) = cam.single_mut() else {
        return; // camera not spawned yet — retry next frame
    };
    match *scene {
        VoxelScene::Cornell => {
            // Cornell is a small static box → the ORBIT camera (orbit a point) is the right interaction.
            mode.fps = false;
            // Frame the OPEN front (−Z) of the box, looking +Z into it, centred so the box fills the view.
            let [cx, cy, cz] = cornell::interior_center_world();
            orbit.target = Vec3::new(cx, cy, cz);
            // Stand back along −Z (yaw = −π/2 puts the eye on −Z relative to the target) at a distance that
            // fits the ~9.6 m box comfortably, with a slight downward pitch so the floor + boxes read.
            orbit.distance = cornell::interior_extent_world() * 1.4;
            orbit.yaw = -std::f32::consts::FRAC_PI_2;
            orbit.pitch = 0.12;
            *tf = orbit.view_transform();
        }
        VoxelScene::Worldgen => {
            // Worldgen is a streamed, camera-FOLLOWING world → use the FREE-FLY (FPS) camera, NOT orbit.
            // Rotating the orbit camera moves the EYE on a ~40 m sphere, so the streaming region (centred on
            // the eye brick) re-streamed on every rotate ("the whole world regenerates"), and the resident
            // bubble chased the orbiting eye so the terrain sat at its far edge ("barely see anywhere"). In
            // free-fly, rotation (right-mouse) is LOOK-only — the eye stays put, so no re-stream — and the eye
            // sits IN the terrain (the resident region centres on you). WASD/Space/Ctrl fly; wheel = speed.
            mode.fps = true;
            mode.yaw = 0.7;
            mode.pitch = -0.18; // look out + slightly down over the terrain
            let layer = build_height_layer(&height, &erosion, &graph, &biome_shapes);
            let surface_y = layer.sample_world(0.0, 0.0, WORLDGEN_SLICE_SEED).height;
            let eye = Vec3::new(0.0, surface_y + 10.0, 0.0); // stand ~10 m above the origin ground
            let forward = Vec3::new(
                mode.yaw.cos() * mode.pitch.cos(),
                mode.pitch.sin(),
                mode.yaw.sin() * mode.pitch.cos(),
            )
            .normalize_or_zero();
            *tf = Transform::from_translation(eye).looking_at(eye + forward, Vec3::Y);
            orbit.target = eye + forward * orbit.distance; // sensible if the user toggles back to orbit
        }
        VoxelScene::Sponza => {
            // Sponza is a fixed, bounded building streamed through the SAME clipmap as worldgen (the `.vox`
            // loader anchors it floor-at-y=0, centred on X/Z). The baked Khronos Sponza is LARGE — it spans
            // ~122 m along its long (X) axis, ~74 m in Z, and ~51 m tall — so the clipmap (≈8192 m view radius,
            // 160 · 0.4 · 2^7) covers it fully and the bricks stream in from the StaticVoxSource. The right
            // interaction is the
            // FREE-FLY (FPS) camera — you stand INSIDE the nave looking down its long axis at the colonnade,
            // drapes, and lit floor (an orbit would put the eye outside the building). Seed the eye near the
            // −X short end at a vantage height, looking along +X down the full 122 m hall with a slight upward
            // tilt so the upper colonnade + the sky-lit roof line read. WASD/Space/Ctrl fly; right-mouse looks.
            mode.fps = true;
            mode.yaw = 0.0; // look toward +X (down the long axis of the building)
            mode.pitch = 0.06; // a touch upward so the tall colonnade + sky-lit roof line read
            // Stand near the −X end (the floor spans roughly ±61 m in X), ~9 m up so the eye clears the floor
            // clutter and takes in the receding 122 m hall, a hair off the X/Z centreline so the colonnade on
            // both sides frames the nave. Well inside the clipmap's inner LOD0 cube (centred on the eye).
            let eye = Vec3::new(-52.0, 9.0, 1.0);
            let forward = Vec3::new(
                mode.yaw.cos() * mode.pitch.cos(),
                mode.pitch.sin(),
                mode.yaw.sin() * mode.pitch.cos(),
            )
            .normalize_or_zero();
            *tf = Transform::from_translation(eye).looking_at(eye + forward, Vec3::Y);
            orbit.target = eye + forward * orbit.distance; // sensible if the user toggles to orbit
        }
        VoxelScene::Gallery => {
            // The Gallery is a ROW of scenes laid out side by side along +X (the merge auto-spaces them past the
            // first scene, which is anchored at the world origin like standalone Sponza). The right interaction
            // is the FREE-FLY (FPS) camera so the user can fly DOWN the row comparing GI/LOD on each scene. Seed
            // the eye in front of the first scene (near the −X / −Z corner), raised to a vantage height, looking
            // along +X (and a touch toward +Z) so the row recedes across the view — exactly Sponza's framing but
            // aimed to take in the aisle of scenes rather than one nave. WASD/Space/Ctrl fly; right-mouse looks.
            mode.fps = true;
            mode.yaw = 0.18; // look down the +X row, angled slightly toward the scenes
            mode.pitch = 0.04; // a touch upward so the tall scenes' upper geometry reads
            let eye = Vec3::new(-52.0, 12.0, -40.0);
            let forward = Vec3::new(
                mode.yaw.cos() * mode.pitch.cos(),
                mode.pitch.sin(),
                mode.yaw.sin() * mode.pitch.cos(),
            )
            .normalize_or_zero();
            *tf = Transform::from_translation(eye).looking_at(eye + forward, Vec3::Y);
            orbit.target = eye + forward * orbit.distance; // sensible if the user toggles to orbit
        }
    }
    // `AmbientLight` is a per-camera component in 0.19 — give the viewport camera a soft ambient so the
    // cube faces turned away from the sun aren't fully black.
    commands.entity(cam_entity).insert(AmbientLight { color: Color::WHITE, brightness: 600.0, ..default() });
    reframed.0 = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layer() -> HeightLayer {
        HeightLayer::new(LayerId(0), HeightParams::default(), ErosionParams::default())
    }

    /// The derived Y voxel band brackets the origin-column surface height (the patch contains the surface
    /// it's meant to render) and is non-degenerate (min < max).
    #[test]
    fn patch_y_range_brackets_surface() {
        let layer = test_layer();
        let (min_vy, max_vy) = patch_y_voxel_range(&layer, WORLDGEN_SLICE_SEED);
        assert!(min_vy < max_vy, "patch Y band must be non-empty");
        let h = layer.sample_world(0.0, 0.0, WORLDGEN_SLICE_SEED).height as f64;
        let surf_vy = (h / VOXEL_SIZE as f64).floor() as i32;
        assert!(min_vy <= surf_vy && surf_vy <= max_vy, "patch Y band must contain the origin surface");
    }

    /// Voxelizing the patch yields a non-empty, bounded brick set (the patch actually contains terrain),
    /// and the count is small enough to render fast (sanity bound on the proof patch).
    #[test]
    fn patch_voxelizes_to_bounded_bricks() {
        let layer = test_layer();
        // A tiny library so the test is self-contained (one solid material, every biome uses it).
        use crate::sdf_render::worldgen::biome::{BiomeDef, BiomeId, StrataLayer, TerrainMatId, TerrainSurfaceMaterial};
        let materials = vec![TerrainSurfaceMaterial {
            name: "stone".into(),
            base_color: [0.5, 0.5, 0.5, 1.0],
            roughness: 0.9,
            blend: 0.0,
            texture: None,
            tiling: 4.0,
            ..Default::default()
        }];
        let biomes = BiomeId::ALL
            .iter()
            .map(|_| BiomeDef {
                name: "b".into(),
                surface: TerrainMatId(0),
                surface_rules: vec![],
                strata: vec![StrataLayer { material: TerrainMatId(0), thickness: 1000.0 }],
                bedrock: TerrainMatId(0),
            })
            .collect();
        let lib = BiomeLibrary { materials, biomes };
        let registry = BlockRegistry::from_biome_library(&lib);
        let map = voxelize_patch(&layer, &lib, &registry, WORLDGEN_SLICE_SEED);
        assert!(!map.is_empty(), "the origin patch must contain terrain bricks");
        // Bounded: the patch is ~32 m × band × 32 m of 0.4 m bricks — far fewer than a runaway count. The flip
        // to 0.05 m quartered the brick span, so the same fixed-WORLD-size patch holds ~16× more XZ brick
        // columns (a surface sheet plus a shallow depth band) — the sanity ceiling scales with it.
        assert!(map.len() < 200_000, "brick count {} should stay bounded for the small patch", map.len());
    }
}
