//! # SDF clipmap renderer
//!
//! Renders an editable signed-distance-field world by raymarching a sparse brick atlas,
//! with camera-centred LOD shells so it can reach vast distances. The data flow, in order,
//! and where each stage lives:
//!
//! 1. **Edits → analytic CSG field** (`edits`). Each [`SdfVolume`] is a primitive + CSG op
//!    (`fold_csg`). This field is *resolution-independent*: callable at any point and any
//!    scale. Everything downstream samples it.
//! 2. **Per-LOD bake** (`atlas`). For each resident brick, every voxel stores the analytic
//!    CSG distance (`edits::fold_csg`) sampled at the voxel centre — a true trilinear SDF.
//!    A coarse brick samples the analytic field at its own (larger) voxel scale, so far
//!    geometry bakes correctly without any LOD-0 data, and the surface sits at the same
//!    place at every LOD (no inter-LOD seam). Trade-off: a feature thinner than a voxel can
//!    be missed at coarse LOD (its zero-crossing falls between samples) — accepted as the
//!    cost of a clean, un-inflated field.
//! 3. **Sparse storage + GPU lookup** (`chunk`, `render`, `bindings.wgsl`). Bricks group
//!    into 4³=64-brick **chunks** addressed by an *absolute* world-lattice key (independent
//!    of the camera, so CPU and GPU agree by construction). Resident chunks live in a per-LOD
//!    **toroidal directory** — a dense `R³` array per LOD where chunk `c` sits at the fixed slot
//!    `c mod R`, so the GPU resolves it by a direct index + key-tag compare (no sort, no binary
//!    search) and the CPU inserts/evicts in O(1). Each slot carries a 64-bit occupancy mask +
//!    popcount index into a packed (sparse) tile-run buffer. Brick texels live in a 2D-tiled
//!    atlas texture.
//! 4. **Async incremental bake** (`bake_scheduler`). The camera-centred chunk ring recenters
//!    as the camera moves; entered chunks bake on a task pool, exited chunks evict IMMEDIATELY
//!    (the march falls back to a coarser resident LOD during the brief handoff) — never blocking
//!    the main thread.
//! 5. **Unified raymarch** (`sdf_raymarch.wgsl`, helpers in `brick`). One loop:
//!    resolve the finest resident LOD at `p`; skip empty space by brick-DDA; otherwise
//!    sphere-trace the trilinear field and accept the hit once the surface is within the
//!    pixel cone (screen-space termination — the vast-distance speed win). There is **no GPU
//!    BVH** in the march; the field + brick-geometry DDA drive all skipping. The `bvh` module
//!    is CPU-only, used solely as the bake's edit-culling acceleration structure.
//!
//! Editor-only pieces (`debug`, `gizmo`, `picking`, overlays) sit alongside but are not on
//! the render hot path.

pub mod atlas;
pub mod bake_scheduler;
pub mod bc7;
pub mod bvh;
pub mod chunk;
#[cfg(feature = "editor")]
pub mod debug;
pub mod editor_camera;
pub mod edits;
pub mod gallery;
pub mod gizmo;
pub mod height;
pub mod node_gizmos;
pub mod overlays;
pub mod picking;
pub mod render;
pub mod scatter;
pub mod stress;
pub mod textures;
pub mod tower_field;

use bevy::prelude::*;

use crate::scene_manager::AppScene;

// The editor viewport cameras (orbit + free-fly) live in `editor_camera`, and the gizmo overlays in
// `overlays`. Their public types are re-exported here so cross-module consumers keep the stable
// `sdf_render::` path.
pub use editor_camera::{
    CameraInput, OrbitFocus, SdfCameraMode, SdfOrbitCamera, sync_orbit_camera_transform,
};
pub use overlays::{LodRingsVisible, SdfGridGizmos, SdfNodeGizmos, SdfOverlayGizmos};

// --- Components ---

// Edit primitives, CSG ops, ordering, and material live in `edits`. Re-exported
// here so the rest of the module (and external callers) keep a stable
// `sdf_render::` path.
pub use edits::{CsgKind, MaterialFields, SdfMaterial, SdfMaterialSource, SdfOp, SdfOrder, SdfPrimitive};

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
#[require(crate::node::Node3D)]
pub struct SdfVolume;

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SdfCamera;

/// Whether the debug toolkit is currently drawing SDF bounds wireframes. Owned by
/// the core module so the gizmo-draw system can live behind the feature gate
/// without the resource type vanishing from the core build.
#[derive(Resource, Default)]
pub struct WireframeBoundsVisible(pub bool);

/// Diagnostic: world-space center + size of recently-baked bricks, each tagged with the time it
/// was baked so the editor can FADE the wire box out over a few seconds. Lets you SEE which
/// bricks an edit move dirties (e.g. confirm a far small object doesn't touch the heightmap) AND
/// how rapidly — a continuous drag leaves a bright cloud, an idle frame fades to nothing.
/// Entries accumulate across frames (NOT cleared each frame); the draw system drops expired ones.
/// `enabled` gates collection so it costs nothing when off. Owned by the core module so the
/// scheduler can fill it without the editor feature; the draw system is editor-gated.
#[derive(Resource, Default)]
pub struct BakedBrickDebug {
    pub enabled: bool,
    /// (center, edge_size, baked_at_secs) per recently-baked brick.
    pub bricks: Vec<(Vec3, f32, f32)>,
}

/// How long (seconds) a baked-brick marker stays visible, fading to transparent over its life.
pub const BAKED_BRICK_FADE_SECS: f32 = 2.0;

/// Last CPU ray-step capture from the debug ray inspector. Empty until a capture
/// is requested.
#[derive(Resource, Default)]
pub struct RayStepCapture {
    pub steps: Vec<picking::RayStep>,
}

/// Toggle for the SDF fullscreen raymarch pass. F1 flips this.
#[derive(Resource)]
pub struct SdfRenderEnabled(pub bool);

impl Default for SdfRenderEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// Whether viewport input (orbit/pick/gizmo-drag) is allowed this frame. The
/// editor sets this from the pointer-in-viewport test so clicks on dock panels
/// don't fall through to the 3D scene. Defaults to `true` so the non-editor build
/// (full-window viewport, no panels) keeps working unchanged.
#[derive(Resource)]
pub struct ViewportInputAllowed(pub bool);

impl Default for ViewportInputAllowed {
    fn default() -> Self {
        Self(true)
    }
}

/// Live raymarch tuning, fed to the shader each frame via the camera uniform's
/// `debug_params`. Always present (defaults match the historical shader constants)
/// so the render path never depends on the debug toolkit feature.
#[derive(Resource, Reflect)]
#[reflect(Resource)]
pub struct SdfRaymarchParams {
    pub max_steps: u32,
    pub max_dist: f32,
    pub sdf_eps: f32,
    /// Multiplier on the per-pixel cone half-width used for screen-space march
    /// termination. The march stops when the conservative field drops below
    /// `pixel_cone · t` (surface within ~`cone_scale` pixels), so far geometry resolves
    /// at coarse LOD instead of marching down to LOD 0 — the vast-distance speed win.
    /// 1.0 = exactly one pixel; larger = coarser/cheaper, smaller = sharper/costlier.
    pub cone_scale: f32,
    /// Sphere-trace over-relaxation factor (Keinert 2014). The march steps `over_relax · d`
    /// with a safe fallback when consecutive unbounding spheres separate, converging on
    /// grazing rays in fewer steps. 1.0 = plain sphere tracing; (1,2) accelerates. Default
    /// 1.6: measured (tests/sdf_march_sim.rs) big step cut on grazing-MISS rays (the slow
    /// tangent-band crawl) with zero hit↔miss flips — the fallback undoes any overshoot on
    /// hits, and the cross-fade shell forces ω=1 where the blended field is non-eikonal.
    /// (1.8 cut more in the sim but showed visual artifacts on the real scene, so backed off
    /// to 1.6 for margin below the ω<2 overlapping-sphere safety ceiling.)
    pub over_relax: f32,
    /// LOD cross-fade band width, as a fraction of each clipmap ring's half-extent. In the
    /// outer `lod_blend_band` shell of a ring the marched field is `mix`-faded from the
    /// serving LOD toward its coarser neighbour, so the surface morphs smoothly across the
    /// ring boundary instead of snapping (removes the visible LOD pop/seam). 0 = disabled
    /// (hard LOD seams, the original behaviour). Tunable live via the editor raymarch panel.
    pub lod_blend_band: f32,
}

impl Default for SdfRaymarchParams {
    fn default() -> Self {
        Self {
            // Raised for vast-distance marching: cone termination keeps the step count
            // bounded even though the reach is far larger than the old 100-unit cap.
            max_steps: 192,
            max_dist: 5000.0,
            sdf_eps: 0.001,
            cone_scale: 1.0,
            over_relax: 1.6,
            lod_blend_band: 0.2,
        }
    }
}

// --- Selection ---

/// The currently-selected SDF volume. Click-picking sets `entity`; the transform
/// manipulator (transform-gizmo-bevy) is attached to this entity via `GizmoTarget`.
#[derive(Resource, Default)]
pub struct SdfSelection {
    pub entity: Option<Entity>,
}

// --- Grid Config ---

/// Number of LOD levels the clipmap generates by default. Level 0 is the base
/// resolution; each coarser level doubles `voxel_size` (and so covers 2× the linear
/// extent / 8× the volume) of the one below it.
pub const DEFAULT_LOD_COUNT: u32 = 8;
/// Bricks per axis in each LOD ring window centred on the camera. The ring at level
/// `L` covers `ring_bricks · cell_stride · voxel_size · 2^L` world units per axis, so
/// the same count reaches twice as far each coarser level (the clipmap nesting). Must be
/// a multiple of [`chunk::CHUNK_BRICKS`] (= 4; the ring is enumerated in whole chunks).
/// 128 = 4·32: each LOD window spans twice as many bricks per axis as before, so every level
/// reaches 2x further at the SAME voxel resolution — distant geometry is served a finer LOD
/// (eases the far-LOD shrink), at the cost of a larger resident shell (the sparse cull still
/// keeps only non-empty bricks). Must be a multiple of `CHUNK_BRICKS` (= 4).
pub const DEFAULT_RING_BRICKS: u32 = 128;
/// Default ring-recenter hysteresis, in whole chunks (see
/// [`SdfGridConfig::recenter_snap_chunks`]). With `CHUNK_BRICKS = 4` and a 128-brick ring
/// (32 chunks/axis), snapping to 2 chunks means the window recenters every ~5.6 m at LOD
/// 0 instead of every brick crossing, while still keeping the camera 14+ chunks from any
/// window edge.
pub const DEFAULT_RECENTER_SNAP_CHUNKS: i32 = 2;

#[derive(Resource, Clone)]
pub struct SdfGridConfig {
    pub grid_size: u32,
    pub brick_size: u32,
    /// Base (level-0) voxel size in world units. Level `L` uses `voxel_size · 2^L`.
    pub voxel_size: f32,
    /// How many LOD levels the clipmap bakes (level `0..lod_count`).
    pub lod_count: u32,
    /// Bricks per axis in each LOD ring window centred on the camera.
    pub ring_bricks: u32,
    /// Hysteresis: the ring window only recenters when the camera crosses this many
    /// whole chunks, so the per-LOD origin snaps to a coarse `recenter_snap_chunks`
    /// lattice instead of moving every brick crossing (~0.7 m at LOD 0). `1` = recenter
    /// on every chunk crossing (no hysteresis). Must stay well below
    /// `ring_bricks / CHUNK_BRICKS` so the camera never leaves its own window.
    pub recenter_snap_chunks: i32,
}

impl Default for SdfGridConfig {
    fn default() -> Self {
        Self {
            grid_size: 1024,
            brick_size: 8,
            voxel_size: 0.1,
            lod_count: DEFAULT_LOD_COUNT,
            ring_bricks: DEFAULT_RING_BRICKS,
            recenter_snap_chunks: DEFAULT_RECENTER_SNAP_CHUNKS,
        }
    }
}

impl SdfGridConfig {
    /// Spatial stride between bricks, in voxels. A brick stores `brick_size`
    /// samples (8) but covers `brick_size - 1` cells (7); adjacent bricks share
    /// (duplicate) the boundary sample plane so trilinear interpolation never
    /// needs to read a neighbouring brick. This "apron" is what the paper's
    /// Sparse Brick Set uses to avoid cracks at brick seams.
    pub fn cell_stride(&self) -> i32 {
        (self.brick_size - 1) as i32
    }

    /// Ring chunks per axis: `R = ring_bricks / CHUNK_BRICKS`. The edge of each per-LOD toroidal
    /// directory window and the SINGLE source for this derivation (CPU mirror of `ring_chunks() /
    /// CHUNK_BRICKS` in `bindings.wgsl`). `LiveChunkTables`/`ChunkTables` cache it and `dir_index`
    /// resolves against it, so every site MUST agree — route through here, never recompute ad hoc.
    pub fn ring_chunks_per_axis(&self) -> i32 {
        self.ring_bricks as i32 / chunk::CHUNK_BRICKS
    }

    /// Half the ring window in chunks (`R / 2`) — the camera-centred window's reach from its origin.
    pub fn ring_half_chunks(&self) -> i32 {
        self.ring_chunks_per_axis() / 2
    }

    /// Total per-LOD toroidal directory length: `R³ × lod_count` fixed `ChunkLookup` slots.
    pub fn directory_len(&self) -> usize {
        let r = self.ring_chunks_per_axis() as usize;
        r * r * r * self.lod_count as usize
    }

    pub fn world_extent(&self) -> f32 {
        self.grid_size as f32 * self.voxel_size
    }
    pub fn world_origin(&self) -> Vec3 {
        Vec3::splat(-self.world_extent() * 0.5)
    }

    /// Voxel size (world units) at LOD level `lod`: `base · 2^lod`.
    pub fn voxel_size_at(&self, lod: u32) -> f32 {
        self.voxel_size * (1u32 << lod) as f32
    }

    /// World-space edge length of one brick at LOD `lod` (`cell_stride · voxel_size`).
    pub fn brick_world_size(&self, lod: u32) -> f32 {
        self.cell_stride() as f32 * self.voxel_size_at(lod)
    }

    /// Convert world position to brick origin (grid-relative voxel coords,
    /// snapped down to the brick stride). Single-resolution (level-0, centred grid);
    /// kept for the non-LOD bake/test paths. LOD bakes use [`Self::world_to_brick_lod`].
    pub fn world_to_brick(&self, world_pos: Vec3) -> IVec3 {
        let s = self.cell_stride();
        let relative = world_pos - self.world_origin();
        let vox_x = (relative.x / self.voxel_size) as i32;
        let vox_y = (relative.y / self.voxel_size) as i32;
        let vox_z = (relative.z / self.voxel_size) as i32;
        IVec3::new((vox_x / s) * s, (vox_y / s) * s, (vox_z / s) * s)
    }

    /// Brick origin (stride-aligned voxel coords at LOD `lod`) containing `world_pos`.
    /// Each LOD lattice is anchored at world 0 (not the centred grid origin), so coords
    /// are signed and a ring can sit anywhere around the camera. `div_euclid` floors
    /// toward negative infinity so the lattice is continuous across the origin.
    pub fn world_to_brick_lod(&self, world_pos: Vec3, lod: u32) -> IVec3 {
        let s = self.cell_stride();
        let vs = self.voxel_size_at(lod);
        let vox = IVec3::new(
            (world_pos.x / vs).floor() as i32,
            (world_pos.y / vs).floor() as i32,
            (world_pos.z / vs).floor() as i32,
        );
        IVec3::new(
            vox.x.div_euclid(s) * s,
            vox.y.div_euclid(s) * s,
            vox.z.div_euclid(s) * s,
        )
    }

    /// World-space minimum corner of the brick at LOD `lod` with origin coord `coord`.
    pub fn brick_min_world(&self, coord: IVec3, lod: u32) -> Vec3 {
        let vs = self.voxel_size_at(lod);
        Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32) * vs
    }

    /// The ring window's corner brick coord at LOD `lod` for a camera at `camera_pos`:
    /// the camera's brick minus half the ring on each axis, so the ring is centred on
    /// the camera. Coords are multiples of `cell_stride`.
    pub fn ring_origin(&self, camera_pos: Vec3, lod: u32) -> IVec3 {
        let s = self.cell_stride();
        let center = self.world_to_brick_lod(camera_pos, lod);
        let half = (self.ring_bricks / 2) as i32 * s;
        center - IVec3::splat(half)
    }

    // Chunk addressing (absolute keys, sparse occupancy) lives in `super::chunk`.
}

// --- Plugin ---

pub struct SdfScenePlugin;

impl Plugin for SdfScenePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SdfGridConfig>()
            .init_resource::<SdfSelection>()
            .init_resource::<OrbitFocus>()
            .init_resource::<SdfOrbitCamera>()
            .init_resource::<SdfCameraMode>()
            .init_resource::<edits::MaterialRegistry>()
            .init_resource::<atlas::SdfAtlas>()
            .init_resource::<bake_scheduler::PrevEditAabbs>()
            .init_resource::<bake_scheduler::BakeScheduler>()
            .init_resource::<bake_scheduler::PendingGpuBakes>()
            .init_resource::<bake_scheduler::BakeTaskState>()
            .init_resource::<LodRingsVisible>()
            .init_resource::<bvh::Bvh>()
            .init_resource::<SdfRenderEnabled>()
            .init_resource::<SdfRaymarchParams>()
            .init_resource::<WireframeBoundsVisible>()
            .init_resource::<BakedBrickDebug>()
            .init_resource::<RayStepCapture>()
            .init_resource::<ViewportInputAllowed>()
            .init_resource::<gizmo::GizmoState>()
            .register_type::<SdfVolume>()
            .register_type::<SdfCamera>()
            .register_type::<SdfPrimitive>()
            .register_type::<SdfOp>()
            .register_type::<SdfOrder>()
            .register_type::<SdfMaterial>()
            .register_type::<edits::SdfMaterialSource>()
            .register_type::<edits::MaterialFields>()
            .register_type::<CsgKind>()
            .register_type::<SdfRaymarchParams>()
            .register_type::<stress::TowerSpawner>()
            // Spawn the scene. Material ids come from the demand-driven asset table
            // (loaded MaterialAssets get stable registry ids); the compile step in
            // `assets::compile` fills the registry once assets resolve, and the GPU
            // table re-uploads via change detection.
            // The viewport camera persists across scene-state transitions (editor infra),
            // spawned once at startup and activated only while in the SDF editor.
            .add_systems(Startup, editor_camera::spawn_editor_camera)
            .add_systems(Update, editor_camera::sync_editor_camera_active)
            .add_systems(
                OnEnter(AppScene::SdfEditor),
                (setup_sdf_scene, load_default_gallery).chain(),
            )
            // Camera control: skipped when the pointer is over a dock panel (editor
            // sets ViewportInputAllowed). Non-editor build leaves it true.
            .add_systems(
                Update,
                (
                    editor_camera::orbit_camera.run_if(|m: Res<SdfCameraMode>| !m.fps),
                    editor_camera::fps_camera.run_if(|m: Res<SdfCameraMode>| m.fps),
                )
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
            )
            // Focus easing runs even while the pointer is over a dock panel, so a
            // Hierarchy double-click animates the camera without re-entering the
            // viewport. NOT gated on ViewportInputAllowed (unlike orbit_camera).
            .add_systems(
                Update,
                editor_camera::ease_orbit_focus
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|m: Res<SdfCameraMode>| !m.fps),
            )
            // Gizmo interaction THEN click-selection, both in `Last`, chained so the
            // gizmo claims a handle click before `sdf_picking` would reselect the
            // volume underneath (`sdf_picking` bails when `GizmoState.claimed_click`).
            .add_systems(
                Last,
                (gizmo::gizmo_update, sdf_picking, focus_on_double_click)
                    .chain()
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
            )
            // Ungated: end any active gizmo drag on mouse release even when the pointer
            // is over a dock panel, so a stale drag never carries into the next click.
            .add_systems(
                Last,
                gizmo::clear_gizmo_drag_on_release.run_if(in_state(AppScene::SdfEditor)),
            )
            // Bake/upload/render-toggle always run in the editor scene — property
            // edits in the inspector (and gizmo drags) must still re-bake. The GPU bake is
            // the only path: `schedule_bakes` does topology (edit detection + camera
            // recenter) and emits GPU compute jobs.
            // Rebuild the bake-time height cache when materials change, BEFORE the baker, so a
            // displacement edit triggers a rebake the same frame.
            // Expand any loaded TowerSpawner node into its tower-field volumes (the stress scene).
            // Before the baker so the spawned volumes bake the same frame they appear.
            // Clear last frame's incremental chunk-table delta record at the START of the frame —
            // AFTER the render world extracted it (extract runs at the end of the prior frame) and
            // BEFORE `schedule_bakes` accumulates this frame's. See `clear_chunk_table_dirty`.
            .add_systems(
                First,
                clear_chunk_table_dirty.run_if(in_state(AppScene::SdfEditor)),
            )
            .add_systems(
                Update,
                stress::expand_tower_spawners
                    .run_if(in_state(AppScene::SdfEditor))
                    .before(bake_scheduler::schedule_bakes),
            )
            .add_systems(
                Update,
                update_height_field
                    .run_if(in_state(AppScene::SdfEditor))
                    .before(bake_scheduler::schedule_bakes),
            )
            .add_systems(
                Update,
                bake_scheduler::schedule_bakes.run_if(in_state(AppScene::SdfEditor)),
            )
            .add_systems(
                Update,
                toggle_sdf_render.run_if(in_state(AppScene::SdfEditor)),
            );

        // Overlay gizmos (ground grid + bounds) need GizmoPlugin (Assets<GizmoAsset>).
        // Present in the real app (DefaultPlugins) but not in MinimalPlugins test
        // harnesses, so register the group + drawing only when present.
        if app.world().is_resource_added::<Assets<GizmoAsset>>()
            || app.world().get_resource::<Assets<GizmoAsset>>().is_some()
        {
            // `GizmoRenderPlugin` (the filled-overlay renderer consumed by `draw_gizmo`) is added
            // explicitly in `main.rs`; here we only wire the gizmo groups, which need `GizmoPlugin`
            // (`Assets<GizmoAsset>`) — present under DefaultPlugins, absent in MinimalPlugins tests.
            app.init_gizmo_group::<SdfOverlayGizmos>()
                .init_gizmo_group::<SdfGridGizmos>()
                .init_gizmo_group::<SdfNodeGizmos>()
                .add_systems(OnEnter(AppScene::SdfEditor), overlays::configure_overlay_gizmos)
                .add_systems(
                    Update,
                    (overlays::draw_ground_grid, gizmo::draw_gizmo)
                        .run_if(in_state(AppScene::SdfEditor)),
                )
                // LOD ring overlay: only while the toggle is on (LodRingsVisible, F8),
                // so it doesn't clutter the normal view.
                .add_systems(
                    Update,
                    overlays::draw_lod_rings
                        .run_if(in_state(AppScene::SdfEditor))
                        .run_if(|v: Res<LodRingsVisible>| v.0),
                );

            // Per-node-type gizmos (light glyphs, point-light ring + radius drag, axes)
            // own their draw/pick/interaction in `node_gizmos`.
            node_gizmos::register(app);
        }

        #[cfg(feature = "editor")]
        app.add_plugins(debug::SdfDebugPlugin);
    }
}

// --- Scene Setup ---

fn setup_sdf_scene(mut asset_table: ResMut<crate::assets::MaterialAssetTable>) {
    asset_table.ensure_fallback();

    // Materials are no longer hardcoded here: each volume in the loaded scene carries an
    // `SdfMaterialSource` (a file path and/or inline overrides), and `resolve_materials`
    // loads + derives the GPU registry dynamically from whatever the scene contains.
    //
    // The viewport camera is EDITOR infrastructure (see `spawn_editor_camera`), not scene
    // content — it persists across scene loads/switches and is never serialized. The gallery
    // geometry + light come from `assets/scenes/gallery.scene` via `load_default_gallery`.
    //
    // Initial bake happens on the first `schedule_bakes` tick (atlas starts dirty), once the
    // loaded edit entities exist and the BVH can be built from them.
}

/// Path to the editor's default scene (the stress tower-field — heavy SDF load, used while
/// profiling/optimizing the raymarch). The PBR gallery lives at `assets/scenes/gallery.scene`
/// and can be loaded manually.
pub const DEFAULT_SCENE_PATH: &str = "assets/scenes/stress.scene";

/// Load the default scene into the world on editor enter. Exclusive (scene load
/// needs `&mut World` + the type registry). Runs after `setup_sdf_scene` so the materials
/// it registers exist before the volumes that reference them appear — though the load only
/// needs the registry, since `registry_id`s are baked into the file.
fn load_default_gallery(world: &mut World) {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let path = std::path::Path::new(DEFAULT_SCENE_PATH);
    match crate::soul_scene::load_scene(world, path, &registry.read()) {
        Ok(roots) => info!("loaded default scene ({} roots)", roots.len()),
        Err(e) => error!("failed to load default scene: {e}"),
    }
    // Restore the editor camera saved with the scene (if any), so launching frames the
    // gallery the way it was last saved.
    if let Some(cam) = world.resource::<crate::soul_scene::LoadedEditorCamera>().0 {
        {
            let mut orbit = world.resource_mut::<SdfOrbitCamera>();
            orbit.target = Vec3::from_array(cam.target);
            orbit.distance = cam.distance;
            orbit.yaw = cam.yaw;
            orbit.pitch = cam.pitch;
        }
        sync_orbit_camera_transform(world);
    }
}

// --- Picking ---

/// A volume entity paired with its resolved edit + world AABB, sorted by `SdfOrder`
/// (ties by entity index) so CSG evaluation order is deterministic. Each edit's
/// material id is its `SdfMaterial.registry_id` — a global id into the material
/// registry, independent of spawn/sort order.
pub struct GatheredEdit {
    pub entity: Entity,
    pub edit: edits::ResolvedEdit,
    pub aabb: bevy::math::bounding::Aabb3d,
}

/// Query data for reading an SDF volume edit's full definition. Aliased so the same
/// (6-field) query reads identically across the bake, picking, and debug systems
/// without tripping the type-complexity lint.
pub type VolumeQueryData = (
    Entity,
    // World transform, so a volume parented under another node inherits its parent's
    // motion (Bevy propagates `Transform` → `GlobalTransform`). Baking/picking operate
    // in world space, so this is the value they need.
    &'static GlobalTransform,
    &'static SdfPrimitive,
    &'static SdfOp,
    &'static SdfOrder,
    &'static SdfMaterial,
);

/// Collect all SDF volume edits from the world, sorted by `SdfOrder` (ties broken by
/// entity index for determinism). The material id comes from each edit's
/// `SdfMaterial` registry reference.
pub fn gather_sorted_edits(volumes: &Query<VolumeQueryData, With<SdfVolume>>) -> Vec<GatheredEdit> {
    let mut rows: Vec<(
        SdfOrder,
        Entity,
        Transform,
        SdfPrimitive,
        SdfOp,
        SdfMaterial,
    )> = volumes
        .iter()
        .map(|(e, t, p, op, order, m)| (*order, e, t.compute_transform(), p.clone(), *op, *m))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.index().cmp(&b.1.index())));

    rows.into_iter()
        .map(|(_, entity, transform, prim, op, material)| {
            let aabb = edits::edit_world_aabb(&prim, &transform, op.smoothing);
            GatheredEdit {
                entity,
                edit: edits::ResolvedEdit::new(prim, transform, op, material.registry_id as u16),
                aabb,
            }
        })
        .collect()
}

/// Left-click selects the SDF volume under the cursor (CPU raymarch pick). Runs
/// after `gizmo_update` in `Last`; if the gizmo claimed the click (a handle was
/// grabbed), it bails so grabbing a handle doesn't reselect the volume underneath.
/// Query filter for non-SDF spatial nodes pickable via their gizmo bounds (lights/empties).
type GizmoNodeFilter = (Without<SdfVolume>, Without<SdfCamera>);

#[allow(clippy::too_many_arguments)]
fn sdf_picking(
    mouse: Res<ButtonInput<MouseButton>>,
    mut selection: ResMut<SdfSelection>,
    gizmo_state: Res<gizmo::GizmoState>,
    cameras: Query<(&Camera, &GlobalTransform, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    // Non-SDF spatial nodes (lights, empties) have no raymarchable geometry, so they're
    // picked by ray-testing the oriented bounding box of their drawn editor gizmo.
    gizmo_nodes: Query<(Entity, &GlobalTransform, &crate::node::EditorGizmo), GizmoNodeFilter>,
    // Point lights are also pickable by clicking their drawn range sphere (a large target).
    point_lights: Query<&PointLight>,
    bvh: Res<bvh::Bvh>,
) {
    if !mouse.just_pressed(MouseButton::Left) || gizmo_state.claimed_click {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(mouse_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, _cam_global, cam_transform)) = cameras.single() else {
        return;
    };
    let Some(ray) = picking::mouse_to_ray(camera, cam_transform, window, mouse_pos) else {
        return;
    };

    // 1. Raymarch the SDF volumes (the geometric pick), keeping the hit depth `t` so a
    //    node gizmo in front of the surface can win the click.
    let gathered = gather_sorted_edits(&volumes);
    let sdf_hit = picking::pick_entity(&bvh, &ray, &gathered);

    // 2. Ray-test each node gizmo's oriented bounding box (matching the drawn glyph),
    //    keeping the nearest entry distance — directly comparable to the SDF hit's `t`.
    let mut best_node: Option<(f32, Entity)> = None; // (ray_depth, entity)
    let consider = |t: f32, e: Entity, best: &mut Option<(f32, Entity)>| {
        if best.is_none_or(|(bt, _)| t < bt) {
            *best = Some((t, e));
        }
    };
    for (entity, xf, gizmo) in &gizmo_nodes {
        let (center, half) = node_gizmos::pick_bounds(gizmo);
        let obb = picking::Obb::from_local(center, half, xf);
        if let Some(t) = obb.ray_hit(&ray) {
            consider(t, entity, &mut best_node);
        }
        // A point light is also pickable by clicking its drawn range sphere (its two great
        // circles), a much larger target than the central bulb. Tolerance scales with
        // distance so the line stays ~8px thick on screen.
        if let Ok(light) = point_lights.get(entity) {
            let origin = xf.translation();
            let tol = (8.0 * (origin - cam_transform.translation).length()
                / camera.clip_from_view().y_axis.y)
                / window.height();
            for normal in node_gizmos::draw::SPHERE_CIRCLE_NORMALS {
                if let Some(t) = picking::ray_circle(&ray, origin, normal, light.range, tol) {
                    consider(t, entity, &mut best_node);
                }
            }
        }
    }

    // 3. Depth arbitration: a node in front of the SDF surface (or when the ray missed
    //    the SDF entirely) wins; otherwise the SDF hit wins. A click on truly empty space
    //    deselects (matching the prior raymarch-miss behaviour).
    selection.entity = match (sdf_hit, best_node) {
        (Some((sdf_e, sdf_t)), Some((node_t, node_e))) => {
            if node_t <= sdf_t {
                Some(node_e)
            } else {
                Some(sdf_e)
            }
        }
        (Some((sdf_e, _)), None) => Some(sdf_e),
        (None, Some((_, node_e))) => Some(node_e),
        (None, None) => None,
    };
}

/// CPU-pick the nearest SDF volume under a window-space cursor position, for callers
/// outside the `sdf_picking` system (e.g. the dock's material drag-drop handler, which runs
/// with `&mut World`). Returns the hit `SdfVolume` entity, or `None` on a miss. Reuses the
/// same ray + raymarch as `sdf_picking`; only SDF volumes are considered (gizmo nodes don't
/// accept a material).
pub fn pick_sdf_volume(world: &mut World, cursor: Vec2) -> Option<Entity> {
    let (camera, cam_transform) = {
        let mut q = world.query_filtered::<(&Camera, &Transform), With<SdfCamera>>();
        let (c, t) = q.single(world).ok()?;
        (c.clone(), *t)
    };
    let window = {
        let mut q = world.query::<&Window>();
        q.single(world).ok()?.clone()
    };
    let ray = picking::mouse_to_ray(&camera, &cam_transform, &window, cursor)?;

    let gathered = {
        let mut q = world.query_filtered::<VolumeQueryData, With<SdfVolume>>();
        gather_sorted_edits(&q.query(world))
    };
    let bvh = world.resource::<bvh::Bvh>();
    picking::pick_entity(bvh, &ray, &gathered).map(|(e, _t)| e)
}

/// Double-click (within 300ms) on the selected volume eases the orbit camera onto
/// it. Runs right after `sdf_picking` so `SdfSelection.entity` is already current;
/// kept separate so picking stays a single-responsibility pick. Orbit-mode only —
/// the FPS camera flies freely and ignores the orbit target.
fn focus_on_double_click(
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mode: Res<SdfCameraMode>,
    selection: Res<SdfSelection>,
    mut focus: ResMut<OrbitFocus>,
    mut gizmo_state: ResMut<gizmo::GizmoState>,
    volumes: Query<&Transform, With<SdfVolume>>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let now = time.elapsed_secs();
    let double_click = now - focus.last_click < 0.3;
    focus.last_click = now;
    if !double_click {
        return;
    }

    // The first click of a double-click selects the object, which makes the transform
    // gizmo appear centered on it — so the SECOND click lands on the view-plane translate
    // handle and `gizmo_update` (earlier in this chain) just started a drag. Cancel it so
    // a double-click focuses without dragging the object.
    gizmo_state.drag = None;
    gizmo_state.claimed_click = false;

    if !mode.fps
        && let Some(entity) = selection.entity
        && let Ok(transform) = volumes.get(entity)
    {
        focus.target = Some(transform.translation);
    }
}

/// Clear the incremental chunk-table delta record (dirty rows/slots/sentinel) accumulated last
/// frame. Runs in `First`, AFTER the render world extracted the delta (extract runs at the end of
/// the previous frame) and BEFORE `schedule_bakes` accumulates this frame's changes — so each
/// frame's `dirty_*` sets carry exactly that frame's topology mutations. `schedule_bakes` only
/// APPENDS to these sets (never reads them), so a start-of-frame clear can't drop pending work.
fn clear_chunk_table_dirty(mut atlas: ResMut<atlas::SdfAtlas>) {
    atlas.live_chunks.clear_dirty();
}

/// Rebuild the bake-time height cache when the material registry's displacement columns
/// (`tex_layers[3]`, `parallax_scale`) change, snapshot it into the scheduler for async tasks,
/// and force a rebake so the new relief is folded into the field. A no-op when nothing
/// displacement-relevant changed (fingerprint match) — colour-only edits don't rebake.
fn update_height_field(
    registry: Res<edits::MaterialRegistry>,
    library: Res<crate::assets::MaterialTextureLibrary>,
    mut sched: ResMut<bake_scheduler::BakeScheduler>,
    mut atlas: ResMut<atlas::SdfAtlas>,
    mut last_fingerprint: Local<u64>,
) {
    if let Some(rebuilt) = height::build(&registry, &library, *last_fingerprint) {
        *last_fingerprint = rebuilt.fingerprint;
        // The scheduler owns the canonical Arc snapshot (async bake tasks clone it; sync_bake
        // reads it via `height_field`). A registry change that alters displacement forces a
        // full rebake so the relief is folded into the field.
        sched.set_height(std::sync::Arc::new(rebuilt));
        atlas.rebake_all = true;
    }
}

fn toggle_sdf_render(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut enabled: ResMut<SdfRenderEnabled>,
    mut lod_rings: ResMut<LodRingsVisible>,
) {
    if keyboard.just_pressed(KeyCode::F1) {
        enabled.0 = !enabled.0;
        info!("SDF render pass: {}", if enabled.0 { "ON" } else { "OFF" });
    }
    if keyboard.just_pressed(KeyCode::F8) {
        lod_rings.0 = !lod_rings.0;
        info!("LOD ring overlay: {}", if lod_rings.0 { "ON" } else { "OFF" });
    }
}
