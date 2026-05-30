//! # SDF clipmap renderer
//!
//! Renders an editable signed-distance-field world by raymarching a sparse brick atlas,
//! with camera-centred LOD shells so it can reach vast distances. The data flow, in order,
//! and where each stage lives:
//!
//! 1. **Edits → analytic CSG field** (`edits`). Each [`SdfVolume`] is a primitive + CSG op
//!    (`fold_csg`). This field is *resolution-independent*: callable at any point and any
//!    scale. Everything downstream samples it.
//! 2. **Conservative per-LOD bake** (`atlas`). For each resident brick, every voxel stores
//!    the **minimum** analytic distance over a small sub-grid spanning the voxel cell
//!    (`atlas::SUBSAMPLES`). The min is a *conservative lower bound*: thin features survive
//!    at coarse LOD (no "grain") and a sphere-trace step of the stored value can never
//!    overshoot a surface. A coarse brick samples the analytic field at its own scale, so
//!    far geometry bakes correctly without any LOD-0 data.
//! 3. **Sparse storage + GPU lookup** (`chunk`, `render`, `bindings.wgsl`). Bricks group
//!    into 4³=64-brick **chunks** addressed by an *absolute* world-lattice key (independent
//!    of the camera, so CPU and GPU agree by construction). Resident chunks form a sorted
//!    table (binary-searched on the GPU) with a 64-bit occupancy mask + popcount index into
//!    a packed tile-run buffer. Brick texels live in a 2D-tiled atlas texture.
//! 4. **Async incremental bake** (`bake_scheduler`). The camera-centred chunk ring recenters
//!    as the camera moves; entered chunks bake on a task pool, exited chunks evict — never
//!    blocking the main thread.
//! 5. **Unified raymarch** (`sdf_raymarch.wgsl`, helpers in `brick`/`cubic`). One loop:
//!    resolve the finest resident LOD at `p`; skip empty space by brick-DDA; at LOD 0 near
//!    the surface solve the exact analytic **cubic** for a crisp silhouette; otherwise
//!    sphere-trace the conservative field and accept the hit once the surface is within the
//!    pixel cone (screen-space termination — the vast-distance speed win). There is **no GPU
//!    BVH** in the march; the conservative field drives all skipping. The `bvh` module is
//!    CPU-only, used solely as the bake's edit-culling acceleration structure.
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
pub mod edits;
pub mod gizmo;
pub mod picking;
pub mod render;
pub mod textures;

use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use crate::scene_manager::{AppScene, SceneEntity};

/// Gizmo config group for editor overlays (transform handles, bounds). Uses
/// `depth_bias = -1.0` so overlays always draw on top of the SDF surface — the
/// editor convention. Drawn via immediate-mode gizmos, not the SDF shader.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct SdfOverlayGizmos;

/// Gizmo config group for the editor ground grid. Separate from the overlay group
/// so it keeps default depth (the SDF surface and geometry occlude grid lines
/// behind them) rather than always drawing on top.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct SdfGridGizmos;

// --- Components ---

// Edit primitives, CSG ops, ordering, and material live in `edits`. Re-exported
// here so the rest of the module (and external callers) keep a stable
// `sdf_render::` path.
pub use edits::{CsgKind, SdfMaterial, SdfOp, SdfOrder, SdfPrimitive};

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SdfVolume;

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SdfCamera;

/// Whether the debug toolkit is currently drawing SDF bounds wireframes. Owned by
/// the core module so the gizmo-draw system can live behind the feature gate
/// without the resource type vanishing from the core build.
#[derive(Resource, Default)]
pub struct WireframeBoundsVisible(pub bool);

/// Whether the per-LOD clipmap ring wire boxes are drawn (toggled with F8). Off by
/// default so the overlay stays clean; see `draw_lod_rings`.
#[derive(Resource, Default)]
pub struct LodRingsVisible(pub bool);

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
    /// Near-surface distance band (world units) within which a LOD-0 sample switches to
    /// the exact analytic cubic for a crisp silhouette. Outside it (or at any coarser
    /// LOD) the march sphere-traces the conservative field.
    pub cubic_band: f32,
}

impl Default for SdfRaymarchParams {
    fn default() -> Self {
        Self {
            // Raised for vast-distance marching: cone termination keeps the step count
            // bounded even though the reach is far larger than the old 100-unit cap.
            max_steps: 192,
            max_dist: 2000.0,
            sdf_eps: 0.001,
            cone_scale: 1.0,
            cubic_band: 0.5,
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

/// Double-click-to-focus state for the orbit camera. `sdf_picking` records each
/// left-click time to detect double-clicks; a double-click on a volume sets
/// `target`, which `orbit_camera` eases `SdfOrbitCamera.target` toward.
#[derive(Resource, Default)]
pub struct OrbitFocus {
    /// World point the orbit target is easing toward; cleared once reached.
    pub target: Option<Vec3>,
    /// Elapsed-seconds timestamp of the previous left-click (double-click detection).
    last_click: f32,
}

// --- Orbit Camera ---

#[derive(Resource)]
pub struct SdfOrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for SdfOrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 8.0,
            yaw: 0.0,
            pitch: 0.4,
        }
    }
}

/// SDF editor camera mode. Default is the orbit camera; the viewport toolbar toggles
/// `fps` to switch to a free-fly (WASD + mouse-look) camera, useful for flying out
/// across the km-scale clipmap terrain instead of orbiting a point.
#[derive(Resource)]
pub struct SdfCameraMode {
    /// True = free-fly (FPS) camera; false = orbit camera.
    pub fps: bool,
    /// Free-fly yaw/pitch (radians). Seeded from the orbit camera on each toggle so the
    /// view doesn't jump.
    pub yaw: f32,
    pub pitch: f32,
    /// Movement speed in world units/second (adjustable with the mouse wheel in FPS).
    pub speed: f32,
}

impl Default for SdfCameraMode {
    fn default() -> Self {
        Self {
            fps: false,
            yaw: 0.0,
            pitch: 0.0,
            speed: 15.0,
        }
    }
}

// --- Grid Config ---

/// Number of LOD levels the clipmap generates by default. Level 0 is the base
/// resolution; each coarser level doubles `voxel_size` (and so covers 2× the linear
/// extent / 8× the volume) of the one below it.
pub const DEFAULT_LOD_COUNT: u32 = 8;
/// Bricks per axis in each LOD ring window centred on the camera. The ring at level
/// `L` covers `ring_bricks · cell_stride · voxel_size · 2^L` world units per axis, so
/// the same count reaches twice as far each coarser level (the clipmap nesting). Must be
/// a multiple of [`chunk::CHUNK_BRICKS`] (the ring is enumerated in whole chunks).
pub const DEFAULT_RING_BRICKS: u32 = 12;

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
}

impl Default for SdfGridConfig {
    fn default() -> Self {
        Self {
            grid_size: 1024,
            brick_size: 8,
            voxel_size: 0.1,
            lod_count: DEFAULT_LOD_COUNT,
            ring_bricks: DEFAULT_RING_BRICKS,
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
    pub fn bricks_per_axis(&self) -> u32 {
        self.grid_size / (self.brick_size - 1)
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

    /// Compute linear brick ID from a brick origin coordinate (single-resolution,
    /// level-0). Kept for the non-LOD path.
    pub fn brick_id(&self, coord: IVec3) -> u32 {
        let bpa = self.bricks_per_axis();
        let s = self.cell_stride();
        let bx = (coord.x / s) as u32;
        let by = (coord.y / s) as u32;
        let bz = (coord.z / s) as u32;
        bz * bpa * bpa + by * bpa + bx
    }
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
            .init_resource::<LodRingsVisible>()
            .init_resource::<bvh::Bvh>()
            .init_resource::<SdfRenderEnabled>()
            .init_resource::<SdfRaymarchParams>()
            .init_resource::<WireframeBoundsVisible>()
            .init_resource::<RayStepCapture>()
            .init_resource::<ViewportInputAllowed>()
            .init_resource::<gizmo::GizmoState>()
            .register_type::<SdfVolume>()
            .register_type::<SdfCamera>()
            .register_type::<SdfPrimitive>()
            .register_type::<SdfOp>()
            .register_type::<SdfOrder>()
            .register_type::<SdfMaterial>()
            .register_type::<CsgKind>()
            .register_type::<SdfRaymarchParams>()
            // Spawn the scene. Material ids come from the demand-driven asset table
            // (loaded MaterialAssets get stable registry ids); the compile step in
            // `assets::compile` fills the registry once assets resolve, and the GPU
            // table re-uploads via change detection.
            .add_systems(OnEnter(AppScene::SdfEditor), setup_sdf_scene)
            // Camera control: skipped when the pointer is over a dock panel (editor
            // sets ViewportInputAllowed). Non-editor build leaves it true.
            .add_systems(
                Update,
                (
                    orbit_camera.run_if(|m: Res<SdfCameraMode>| !m.fps),
                    fps_camera.run_if(|m: Res<SdfCameraMode>| m.fps),
                )
                    .run_if(in_state(AppScene::SdfEditor))
                    .run_if(|allowed: Res<ViewportInputAllowed>| allowed.0),
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
            // Bake/upload/render-toggle always run in the editor scene — property
            // edits in the inspector (and gizmo drags) must still re-bake.
            .add_systems(
                Update,
                (
                    bake_scheduler::schedule_bakes,
                    bake_scheduler::apply_bakes,
                    upload_sdf_buffers,
                    toggle_sdf_render,
                )
                    .chain()
                    .run_if(in_state(AppScene::SdfEditor)),
            );

        // Overlay gizmos (ground grid + bounds) need GizmoPlugin (Assets<GizmoAsset>).
        // Present in the real app (DefaultPlugins) but not in MinimalPlugins test
        // harnesses, so register the group + drawing only when present.
        if app.world().is_resource_added::<Assets<GizmoAsset>>()
            || app.world().get_resource::<Assets<GizmoAsset>>().is_some()
        {
            // The filled-overlay gizmo renderer (reusable; consumed by `draw_gizmo`).
            if !app.is_plugin_added::<crate::gizmo_render::GizmoRenderPlugin>() {
                app.add_plugins(crate::gizmo_render::GizmoRenderPlugin);
            }
            app.init_gizmo_group::<SdfOverlayGizmos>()
                .init_gizmo_group::<SdfGridGizmos>()
                .add_systems(OnEnter(AppScene::SdfEditor), configure_overlay_gizmos)
                .add_systems(
                    Update,
                    (draw_ground_grid, gizmo::draw_gizmo).run_if(in_state(AppScene::SdfEditor)),
                )
                // LOD ring overlay: only while the toggle is on (LodRingsVisible, F8),
                // so it doesn't clutter the normal view.
                .add_systems(
                    Update,
                    draw_lod_rings
                        .run_if(in_state(AppScene::SdfEditor))
                        .run_if(|v: Res<LodRingsVisible>| v.0),
                );
        }

        #[cfg(feature = "editor")]
        app.add_plugins(debug::SdfDebugPlugin);
    }
}

// --- Scene Setup ---

fn setup_sdf_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut material_assets: ResMut<Assets<crate::assets::MaterialAsset>>,
    mut asset_table: ResMut<crate::assets::MaterialAssetTable>,
) {
    use crate::assets::{MaterialAsset, TexRef};

    asset_table.ensure_fallback();

    // Demo materials. Each references a texture variant by path (slug/dir). If a
    // matching `assets/materials/<name>.material.ron` exists on disk we load it (the
    // authored source of truth); otherwise we synthesize one in-memory so the demo
    // scene still renders. Either way it gets a stable registry id from the table,
    // and `assets::compile` fills the GPU registry once the asset resolves.
    let mut mat = |name: &str, slug: &str, dir: &str| -> u32 {
        let path = format!("materials/{name}.material.ron");
        let handle = if std::path::Path::new(&format!("assets/{path}")).exists() {
            asset_server.load::<MaterialAsset>(path)
        } else {
            material_assets.add(MaterialAsset {
                base_color: [1.0, 1.0, 1.0, 1.0],
                blend_softness: 0.0,
                maps: std::array::from_fn(|_| {
                    Some(TexRef {
                        slug: slug.to_string(),
                        dir: dir.to_string(),
                    })
                }),
            })
        };
        asset_table.register(handle)
    };

    let mat_sand = mat("sand", "sand", "1");
    let mat_cobble = mat("cobble", "cobble_stone", "1");
    let mat_cobble2 = mat("cobble2", "cobble_stone", "3");
    let mat_ground = mat("ground", "ground", "1");
    let mat_ground2 = mat("ground2", "ground", "4");

    // Camera
    let orbit = SdfOrbitCamera::default();
    let pos = orbit.target
        + Vec3::new(
            orbit.distance * orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.distance * orbit.pitch.sin(),
            orbit.distance * orbit.yaw.sin() * orbit.pitch.cos(),
        );
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(pos).looking_at(orbit.target, Vec3::Y),
        Msaa::Off,
        SdfCamera,
        // Target for the filled gizmo overlay (gizmo_render).
        crate::gizmo_render::GizmoCamera,
        DepthPrepass,
        SceneEntity,
    ));

    // Demo gallery: a wide, flat sand "ground plane" cube with a spread of distinct
    // primitives resting on its top surface. All plain unions (no subtracts). The
    // plane is centred so its top face sits at y = 0; each object's centre is then
    // placed at y = its half-height so it rests exactly on the surface.
    // (order, transform, primitive, material)
    const PLANE_HALF_Y: f32 = 0.15; // thin slab → reads like a plane
    let demo: [(u32, Transform, SdfPrimitive, u32); 7] = [
        // Ground plane: wide + thin, top face at y = 0 (centre at y = -half_y).
        (
            0,
            Transform::from_xyz(0.0, -PLANE_HALF_Y, 0.0),
            SdfPrimitive::Box {
                half_extents: Vec3::new(4.0, PLANE_HALF_Y, 3.0),
            },
            mat_sand,
        ),
        // Box resting on the plane (half-height 0.4 → centre at y = 0.4).
        (
            1,
            Transform::from_xyz(-2.4, 0.4, 0.4),
            SdfPrimitive::Box {
                half_extents: Vec3::splat(0.4),
            },
            mat_cobble,
        ),
        (
            2,
            Transform::from_xyz(-1.1, 0.55, -0.3),
            SdfPrimitive::Sphere { radius: 0.55 },
            mat_cobble2,
        ),
        // Torus lies flat: its half-thickness above centre is `minor` (0.18).
        (
            3,
            Transform::from_xyz(0.2, 0.18, 0.5),
            SdfPrimitive::Torus {
                major: 0.5,
                minor: 0.18,
            },
            mat_ground,
        ),
        // Capsule standing up: half-height + radius above centre.
        (
            4,
            Transform::from_xyz(1.3, 0.68, -0.4),
            SdfPrimitive::Capsule {
                half_height: 0.4,
                radius: 0.28,
            },
            mat_ground2,
        ),
        // Cylinder standing up: half-height above centre.
        (
            5,
            Transform::from_xyz(2.4, 0.5, 0.3),
            SdfPrimitive::Cylinder {
                radius: 0.4,
                half_height: 0.5,
            },
            mat_cobble,
        ),
        (
            6,
            Transform::from_xyz(0.6, 0.45, -1.1),
            SdfPrimitive::Sphere { radius: 0.45 },
            mat_ground,
        ),
    ];

    for (order, transform, prim, registry_id) in demo {
        commands.spawn((
            transform,
            prim,
            SdfOp {
                kind: CsgKind::Union,
                smoothing: 0.0,
            },
            SdfOrder(order),
            SdfMaterial { registry_id },
            SdfVolume,
            SceneEntity,
        ));
    }

    // Directional light so 3D geometry (and debug wireframes) are visible.
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5)),
        SceneEntity,
    ));

    // Initial bake happens on the first `schedule_bakes` tick (atlas starts
    // dirty), once the edit entities exist and the BVH can be built from them.
}

// --- Orbit Camera ---

/// The raw per-frame input the editor cameras share: mouse buttons, keyboard, frame
/// time, and the mouse-motion / scroll message readers. Bundled so both camera
/// systems (and any future view tool) take one param instead of repeating the same
/// five reads.
#[derive(SystemParam)]
pub struct CameraInput<'w, 's> {
    pub mouse: Res<'w, ButtonInput<MouseButton>>,
    pub keyboard: Res<'w, ButtonInput<KeyCode>>,
    pub time: Res<'w, Time>,
    pub motion: MessageReader<'w, 's, MouseMotion>,
    pub scroll: MessageReader<'w, 's, MouseWheel>,
}

/// Godot-style editor camera: middle-mouse orbits, Shift+middle pans, wheel zooms.
/// The camera transform is recomputed every frame so zoom/pan take effect
/// immediately (the previous version only rebuilt it while orbiting, so scroll
/// appeared to do nothing until you dragged).
fn orbit_camera(
    mut orbit: ResMut<SdfOrbitCamera>,
    mut focus: ResMut<OrbitFocus>,
    mut input: CameraInput,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    // Wheel zoom (dolly toward/away from the target).
    for ev in input.scroll.read() {
        orbit.distance = (orbit.distance - ev.y * 0.5).clamp(0.5, 50.0);
    }

    // Smoothly ease the orbit target toward a double-click focus point. Exponential
    // smoothing (frame-rate independent); cleared once we're within a hair of it.
    if let Some(dest) = focus.target {
        let t = 1.0 - (-12.0 * input.time.delta_secs()).exp();
        orbit.target = orbit.target.lerp(dest, t);
        if orbit.target.distance(dest) < 0.01 {
            orbit.target = dest;
            focus.target = None;
        }
    }

    let orbiting = input.mouse.pressed(MouseButton::Middle);
    let panning = orbiting
        && (input.keyboard.pressed(KeyCode::ShiftLeft)
            || input.keyboard.pressed(KeyCode::ShiftRight));

    if orbiting {
        // Basis vectors of the current view for screen-space panning.
        let dir = Vec3::new(
            orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.pitch.sin(),
            orbit.yaw.sin() * orbit.pitch.cos(),
        );
        let right = dir.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(dir).normalize_or_zero();

        for ev in input.motion.read() {
            if panning {
                // Shift+MMB: pan the target across the view plane (scaled by distance
                // so the world tracks the cursor at any zoom).
                let pan = orbit.distance * 0.0015;
                orbit.target += -right * ev.delta.x * pan + up * ev.delta.y * pan;
                // Manual pan overrides any in-progress double-click focus ease.
                focus.target = None;
            } else {
                // MMB: orbit yaw/pitch.
                orbit.yaw -= ev.delta.x * 0.005;
                orbit.pitch = (orbit.pitch + ev.delta.y * 0.005).clamp(-1.4, 1.4);
            }
        }
    } else {
        input.motion.clear();
    }

    // Always recompute so zoom/pan/orbit all apply immediately.
    let pos = orbit.target
        + Vec3::new(
            orbit.distance * orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.distance * orbit.pitch.sin(),
            orbit.distance * orbit.yaw.sin() * orbit.pitch.cos(),
        );

    for mut transform in &mut camera_query {
        *transform = Transform::from_translation(pos).looking_at(orbit.target, Vec3::Y);
    }
}

/// Free-fly (FPS) camera for the SDF editor: hold right mouse to look, WASD to move,
/// Space/Ctrl for up/down, wheel adjusts speed. Lets you fly out across the km-scale
/// clipmap terrain. Active only when `SdfCameraMode.fps` is set (the viewport toolbar
/// toggle); the orbit camera is disabled in that mode so they don't fight.
fn fps_camera(
    mut mode: ResMut<SdfCameraMode>,
    mut orbit: ResMut<SdfOrbitCamera>,
    mut input: CameraInput,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    // Wheel adjusts fly speed (exponential feel), clamped to a sane range.
    for ev in input.scroll.read() {
        mode.speed = (mode.speed * (1.0 + ev.y * 0.1)).clamp(1.0, 500.0);
    }

    // Mouse-look only while holding right mouse (so panel clicks don't spin the view).
    let looking = input.mouse.pressed(MouseButton::Right);
    if looking {
        for ev in input.motion.read() {
            mode.yaw += ev.delta.x * 0.003;
            mode.pitch = (mode.pitch - ev.delta.y * 0.003)
                .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);
        }
    } else {
        input.motion.clear();
    }

    let forward = Vec3::new(
        mode.yaw.cos() * mode.pitch.cos(),
        mode.pitch.sin(),
        mode.yaw.sin() * mode.pitch.cos(),
    )
    .normalize_or_zero();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let up = Vec3::Y;

    let mut dir = Vec3::ZERO;
    if input.keyboard.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if input.keyboard.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if input.keyboard.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if input.keyboard.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if input.keyboard.pressed(KeyCode::Space) {
        dir += up;
    }
    if input.keyboard.pressed(KeyCode::ControlLeft) || input.keyboard.pressed(KeyCode::ControlRight)
    {
        dir -= up;
    }

    let Some(mut transform) = camera_query.iter_mut().next() else {
        return;
    };

    let mut pos = transform.translation;
    if dir != Vec3::ZERO {
        let mut speed = mode.speed;
        if input.keyboard.pressed(KeyCode::ShiftLeft) || input.keyboard.pressed(KeyCode::ShiftRight)
        {
            speed *= 3.0; // sprint
        }
        pos += dir.normalize() * speed * input.time.delta_secs();
    }
    *transform = Transform::from_translation(pos).looking_at(pos + forward, Vec3::Y);

    // Keep the orbit camera's target tracking in front of us, so toggling back to orbit
    // resumes smoothly around what we're looking at rather than snapping to the origin.
    orbit.target = pos + forward * orbit.distance;
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
    &'static Transform,
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
        .map(|(e, t, p, op, order, m)| (*order, e, *t, p.clone(), *op, *m))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.index().cmp(&b.1.index())));

    rows.into_iter()
        .map(|(_, entity, transform, prim, op, material)| {
            let aabb = edits::edit_world_aabb(&prim, &transform, op.smoothing);
            GatheredEdit {
                entity,
                edit: edits::ResolvedEdit {
                    prim,
                    transform,
                    op,
                    material_id: material.registry_id as u16,
                },
                aabb,
            }
        })
        .collect()
}

/// Left-click selects the SDF volume under the cursor (CPU raymarch pick). Runs
/// after `gizmo_update` in `Last`; if the gizmo claimed the click (a handle was
/// grabbed), it bails so grabbing a handle doesn't reselect the volume underneath.
fn sdf_picking(
    mouse: Res<ButtonInput<MouseButton>>,
    mut selection: ResMut<SdfSelection>,
    gizmo_state: Res<gizmo::GizmoState>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
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
    let Ok((camera, cam_transform)) = cameras.single() else {
        return;
    };
    let Some(ray) = picking::mouse_to_ray(camera, cam_transform, window, mouse_pos) else {
        return;
    };

    let gathered = gather_sorted_edits(&volumes);
    selection.entity = picking::pick_entity(&bvh, &ray, &gathered);
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
    volumes: Query<&Transform, With<SdfVolume>>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let now = time.elapsed_secs();
    let double_click = now - focus.last_click < 0.3;
    focus.last_click = now;
    if double_click
        && !mode.fps
        && let Some(entity) = selection.entity
        && let Ok(transform) = volumes.get(entity)
    {
        focus.target = Some(transform.translation);
    }
}

/// Push the overlay gizmo group in front of everything (always-on-top handles).
fn configure_overlay_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<SdfOverlayGizmos>();
    config.depth_bias = -1.0;
    config.line.width = 3.0;

    // Grid uses default depth (occluded by geometry) and thin lines.
    let (grid, _) = store.config_mut::<SdfGridGizmos>();
    grid.depth_bias = 0.0;
    grid.line.width = 1.0;
}

/// Draw a Godot-style infinite ground grid on the XZ plane: faint minor lines
/// every unit, brighter major lines every `MAJOR` units, and colored X (red) /
/// Z (blue) axis lines through the origin. Centred on the camera target snapped to
/// the grid so it reads as infinite as the view pans.
fn draw_ground_grid(mut gizmos: Gizmos<SdfGridGizmos>, orbit: Res<SdfOrbitCamera>) {
    const HALF: i32 = 50; // lines each side of centre
    const STEP: f32 = 1.0; // grid spacing in world units (Godot-style 1m cells)
    let step = STEP;

    let minor = Color::srgba(0.35, 0.35, 0.38, 0.5);
    let major = Color::srgba(0.55, 0.55, 0.60, 0.8);
    let x_axis = Color::srgb(0.86, 0.24, 0.24);
    let z_axis = Color::srgb(0.26, 0.49, 0.92);

    // Snap the grid centre to the target so lines stay put as the camera orbits.
    let cx = (orbit.target.x / step).round() as i32;
    let cz = (orbit.target.z / step).round() as i32;
    let extent = HALF as f32 * step;

    for i in -HALF..=HALF {
        let gx = cx + i;
        let gz = cz + i;
        let wx = gx as f32 * step;
        let wz = gz as f32 * step;

        // Line parallel to Z at x = wx. At gx == 0 this lies on the Z axis (blue).
        let color = line_color(gx, z_axis, major, minor);
        gizmos.line(
            Vec3::new(wx, 0.0, cz as f32 * step - extent),
            Vec3::new(wx, 0.0, cz as f32 * step + extent),
            color,
        );
        // Line parallel to X at z = wz. At gz == 0 this lies on the X axis (red).
        let color = line_color(gz, x_axis, major, minor);
        gizmos.line(
            Vec3::new(cx as f32 * step - extent, 0.0, wz),
            Vec3::new(cx as f32 * step + extent, 0.0, wz),
            color,
        );
    }
}

/// Pick a grid line's colour: the axis colour at index 0 (the origin line), else a
/// major or minor tone depending on divisibility by `MAJOR`.
fn line_color(index: i32, axis: Color, major: Color, minor: Color) -> Color {
    const MAJOR: i32 = 10;
    if index == 0 {
        axis
    } else if index % MAJOR == 0 {
        major
    } else {
        minor
    }
}

/// Draw each LOD clipmap ring's world-AABB as a wire box, colour-matched to the
/// `SDF_DEBUG_LOD` shader ramp (green = fine/near, red = coarse/far). Makes the nested
/// ring extents and their camera-centred recentering directly visible. Uses the same
/// `ring_origin` math the bake centres each ring on, so the boxes track the resident set.
fn draw_lod_rings(
    mut gizmos: Gizmos<SdfOverlayGizmos>,
    config: Res<SdfGridConfig>,
    camera: Query<&Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    let Some(cam) = camera.iter().next() else {
        return;
    };
    let cam_pos = cam.translation;

    for lod in 0..config.lod_count {
        let origin = config.ring_origin(cam_pos, lod);
        let min = config.brick_min_world(origin, lod);
        // The ring spans `ring_bricks` bricks per axis at this LOD's voxel size.
        let extent = Vec3::splat(config.brick_world_size(lod) * config.ring_bricks as f32);
        let center = min + extent * 0.5;

        // Discrete colours matching the SDF_DEBUG_LOD shader: 0 white, 1 green,
        // 2 blue, 3 red, 4+ yellow.
        let color = match lod {
            0 => Color::srgb(1.0, 1.0, 1.0),
            1 => Color::srgb(0.0, 1.0, 0.0),
            2 => Color::srgb(0.0, 0.4, 1.0),
            3 => Color::srgb(1.0, 0.0, 0.0),
            _ => Color::srgb(1.0, 1.0, 0.0),
        };
        gizmos.primitive_3d(
            &Cuboid::new(extent.x, extent.y, extent.z),
            Isometry3d::from_translation(center),
            color,
        );
    }
}

// --- Upload to GPU (placeholder — render.rs handles actual upload) ---

fn upload_sdf_buffers(_atlas: Res<atlas::SdfAtlas>) {
    // Render world will pick up atlas changes via extract
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
