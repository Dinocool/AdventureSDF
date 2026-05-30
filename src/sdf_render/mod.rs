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
}

impl Default for SdfRaymarchParams {
    fn default() -> Self {
        Self {
            max_steps: 128,
            max_dist: 100.0,
            sdf_eps: 0.001,
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
                (gizmo::gizmo_update, sdf_picking)
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

    // --- Clipmap LOD test scene: km-scale heightmap terrain + scattered pillars ---
    //
    // A large noise heightmap is the base terrain (Union, order 0). Cube pillars are
    // sparsely scattered across it (deterministic from a hash), each topped by a sphere
    // of a DIFFERENT material than the pillar, so the LOD rings and material handling
    // are both visible as the camera moves out across the terrain.
    let terrain_mats = [mat_sand, mat_ground, mat_ground2];
    let pillar_mats = [mat_cobble, mat_cobble2];
    let sphere_mats = [mat_ground2, mat_sand, mat_cobble];

    let union = || SdfOp {
        kind: CsgKind::Union,
        smoothing: 0.0,
    };

    let mut order = 0u32;
    let mut spawn_edit =
        |commands: &mut Commands, transform: Transform, prim: SdfPrimitive, registry_id: u32| {
            commands.spawn((
                transform,
                prim,
                union(),
                SdfOrder(order),
                SdfMaterial { registry_id },
                SdfVolume,
                SceneEntity,
            ));
            order += 1;
        };

    // Base terrain: a wide heightmap. half_xz spans hundreds of metres so the clipmap
    // rings have far terrain to coarsen. The field is a vertical-distance approximation
    // (valid when densely sampled, which the fine LODs near the camera guarantee).
    const TERRAIN_HALF_XZ: f32 = 400.0;
    spawn_edit(
        &mut commands,
        Transform::from_xyz(0.0, 0.0, 0.0),
        SdfPrimitive::Heightmap {
            half_xz: Vec2::splat(TERRAIN_HALF_XZ),
            max_height: 40.0,
            freq: 0.02,
            amp: 18.0,
            seed: 1337,
        },
        terrain_mats[0],
    );

    // Sparse pillars on a jittered grid. Deterministic pseudo-random from a small
    // integer hash so the scene is stable across runs. Each pillar is a tall thin box;
    // a sphere of a different material caps it.
    let hash = |x: i32, z: i32, salt: u32| -> f32 {
        let mut h = (x as u32).wrapping_mul(73856093)
            ^ (z as u32).wrapping_mul(19349663)
            ^ salt.wrapping_mul(83492791);
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1e995);
        h ^= h >> 15;
        (h & 0xffff) as f32 / 65535.0 // [0,1)
    };
    let terrain_h = |x: f32, z: f32| -> f32 {
        // Mirror the Heightmap primitive's vertical field closely enough to seat the
        // pillars on the surface (value-noise * amp). Exact placement isn't critical —
        // pillars sink slightly into / rise above terrain, both fine for the demo.
        let n = (x * 0.02).sin() * (z * 0.02).cos();
        n * 18.0
    };

    const GRID: i32 = 6; // pillars on a -GRID..=GRID grid (jittered), pruned by density
    const SPACING: f32 = 22.0;
    for gz in -GRID..=GRID {
        for gx in -GRID..=GRID {
            // ~45% of cells get a pillar — sparse scatter.
            if hash(gx, gz, 7) > 0.45 {
                continue;
            }
            let jitter_x = (hash(gx, gz, 11) - 0.5) * SPACING * 0.6;
            let jitter_z = (hash(gx, gz, 13) - 0.5) * SPACING * 0.6;
            let x = gx as f32 * SPACING + jitter_x;
            let z = gz as f32 * SPACING + jitter_z;
            let base_y = terrain_h(x, z);

            let pillar_h = 4.0 + hash(gx, gz, 17) * 6.0; // 4..10 m tall
            let pillar_half = Vec3::new(1.2, pillar_h * 0.5, 1.2);
            let pillar_cy = base_y + pillar_half.y;
            let pi = ((gx + gz).rem_euclid(pillar_mats.len() as i32)) as usize;
            spawn_edit(
                &mut commands,
                Transform::from_xyz(x, pillar_cy, z),
                SdfPrimitive::Box {
                    half_extents: pillar_half,
                },
                pillar_mats[pi],
            );

            // Sphere cap, different material, resting on the pillar top.
            let sphere_r = 1.8;
            let si = ((gx + gz + 1).rem_euclid(sphere_mats.len() as i32)) as usize;
            spawn_edit(
                &mut commands,
                Transform::from_xyz(x, base_y + pillar_h + sphere_r * 0.5, z),
                SdfPrimitive::Sphere { radius: sphere_r },
                sphere_mats[si],
            );
        }
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

    // Initial bake happens on the first `bake_dirty_bricks` tick (atlas starts
    // dirty), once the edit entities exist and the BVH can be built from them.
}

// --- Orbit Camera ---

/// Godot-style editor camera: middle-mouse orbits, Shift+middle pans, wheel zooms.
/// The camera transform is recomputed every frame so zoom/pan take effect
/// immediately (the previous version only rebuilt it while orbiting, so scroll
/// appeared to do nothing until you dragged).
fn orbit_camera(
    mut orbit: ResMut<SdfOrbitCamera>,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
) {
    // Wheel zoom (dolly toward/away from the target).
    for ev in scroll.read() {
        orbit.distance = (orbit.distance - ev.y * 0.5).clamp(0.5, 50.0);
    }

    let orbiting = mouse.pressed(MouseButton::Middle);
    let panning =
        orbiting && (keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight));

    if orbiting {
        // Basis vectors of the current view for screen-space panning.
        let dir = Vec3::new(
            orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.pitch.sin(),
            orbit.yaw.sin() * orbit.pitch.cos(),
        );
        let right = dir.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(dir).normalize_or_zero();

        for ev in motion.read() {
            if panning {
                // Shift+MMB: pan the target across the view plane (scaled by distance
                // so the world tracks the cursor at any zoom).
                let pan = orbit.distance * 0.0015;
                orbit.target += -right * ev.delta.x * pan + up * ev.delta.y * pan;
            } else {
                // MMB: orbit yaw/pitch.
                orbit.yaw -= ev.delta.x * 0.005;
                orbit.pitch = (orbit.pitch + ev.delta.y * 0.005).clamp(-1.4, 1.4);
            }
        }
    } else {
        motion.clear();
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
#[expect(clippy::too_many_arguments)]
fn fps_camera(
    mut mode: ResMut<SdfCameraMode>,
    mut orbit: ResMut<SdfOrbitCamera>,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
) {
    // Wheel adjusts fly speed (exponential feel), clamped to a sane range.
    for ev in scroll.read() {
        mode.speed = (mode.speed * (1.0 + ev.y * 0.1)).clamp(1.0, 500.0);
    }

    // Mouse-look only while holding right mouse (so panel clicks don't spin the view).
    let looking = mouse.pressed(MouseButton::Right);
    if looking {
        for ev in motion.read() {
            mode.yaw -= ev.delta.x * 0.003;
            mode.pitch = (mode.pitch - ev.delta.y * 0.003)
                .clamp(-std::f32::consts::FRAC_PI_2 + 0.01, std::f32::consts::FRAC_PI_2 - 0.01);
        }
    } else {
        motion.clear();
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
    if keyboard.pressed(KeyCode::KeyW) {
        dir += forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        dir -= forward;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        dir += right;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        dir -= right;
    }
    if keyboard.pressed(KeyCode::Space) {
        dir += up;
    }
    if keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight) {
        dir -= up;
    }

    let Some(mut transform) = camera_query.iter_mut().next() else {
        return;
    };

    let mut pos = transform.translation;
    if dir != Vec3::ZERO {
        let mut speed = mode.speed;
        if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
            speed *= 3.0; // sprint
        }
        pos += dir.normalize() * speed * time.delta_secs();
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
/// `ring_origin` math the bake froze, so the boxes track exactly what got baked.
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
