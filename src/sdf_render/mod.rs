pub mod atlas;
pub mod bc7;
pub mod bvh;
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

// --- Grid Config ---

#[derive(Resource, Clone)]
pub struct SdfGridConfig {
    pub grid_size: u32,
    pub brick_size: u32,
    pub voxel_size: f32,
}

impl Default for SdfGridConfig {
    fn default() -> Self {
        Self {
            grid_size: 1024,
            brick_size: 8,
            voxel_size: 0.1,
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

    /// Convert world position to brick origin (grid-relative voxel coords,
    /// snapped down to the brick stride).
    pub fn world_to_brick(&self, world_pos: Vec3) -> IVec3 {
        let s = self.cell_stride();
        let relative = world_pos - self.world_origin();
        let vox_x = (relative.x / self.voxel_size) as i32;
        let vox_y = (relative.y / self.voxel_size) as i32;
        let vox_z = (relative.z / self.voxel_size) as i32;
        IVec3::new((vox_x / s) * s, (vox_y / s) * s, (vox_z / s) * s)
    }

    /// Convert world position to voxel index within its brick (0..=stride).
    pub fn world_to_voxel(&self, world_pos: Vec3) -> IVec3 {
        let s = self.cell_stride();
        let relative = world_pos - self.world_origin();
        let vox_x = (relative.x / self.voxel_size) as i32;
        let vox_y = (relative.y / self.voxel_size) as i32;
        let vox_z = (relative.z / self.voxel_size) as i32;
        IVec3::new(vox_x % s, vox_y % s, vox_z % s)
    }

    /// Compute linear brick ID from a brick origin coordinate.
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
            .init_resource::<edits::MaterialRegistry>()
            .init_resource::<atlas::SdfAtlas>()
            .init_resource::<PrevEditAabbs>()
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
                orbit_camera
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
                (bake_dirty_bricks, upload_sdf_buffers, toggle_sdf_render)
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

// --- Atlas Baking ---

/// Last frame's per-edit world AABB, keyed by entity. Lets `bake_dirty_bricks`
/// dirty an edit's *former* footprint (not just where it moved to) so vacated
/// bricks get rebuilt/removed. Also serves as the previous entity set for
/// add/remove detection.
#[derive(Resource, Default)]
struct PrevEditAabbs {
    map: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d>,
}

/// Any component that affects an edit's baked result. A change to one of these
/// triggers a targeted rebake of the bricks the edit touches.
type ChangedEdit = Or<(
    Changed<Transform>,
    Changed<SdfOp>,
    Changed<SdfPrimitive>,
    Changed<SdfMaterial>,
)>;

fn bake_dirty_bricks(
    mut atlas: ResMut<atlas::SdfAtlas>,
    mut bvh: ResMut<bvh::Bvh>,
    mut prev_aabbs: ResMut<PrevEditAabbs>,
    config: Res<SdfGridConfig>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    changed: Query<Entity, (With<SdfVolume>, ChangedEdit)>,
) {
    let gathered = gather_sorted_edits(&volumes);
    let resolved: Vec<edits::ResolvedEdit> = gathered.iter().map(|g| g.edit.clone()).collect();
    let aabbs: Vec<bevy::math::bounding::Aabb3d> = gathered.iter().map(|g| g.aabb).collect();
    let current: std::collections::HashMap<Entity, bevy::math::bounding::Aabb3d> =
        gathered.iter().map(|g| (g.entity, g.aabb)).collect();

    // An edit added or removed changes the whole BVH → full rebuild. (Equal count
    // with a swapped entity is caught by the membership check.)
    let set_changed = current.len() != prev_aabbs.map.len()
        || current.keys().any(|e| !prev_aabbs.map.contains_key(e));

    if atlas.rebake_all || set_changed {
        *bvh = bvh::Bvh::build(&aabbs);
        atlas.full_bake(&resolved, &aabbs, &bvh, &config);
        prev_aabbs.map = current;
        return;
    }

    // Existing edits only: union each changed edit's old+new footprint into the
    // dirty set. Nothing changed → idle, no bake, no BVH rebuild.
    if changed.is_empty() {
        return;
    }

    let mut dirty = std::mem::take(&mut atlas.dirty_bricks);
    for entity in &changed {
        if let Some(old) = prev_aabbs.map.get(&entity) {
            dirty.extend(atlas::bricks_in_aabb(&config, old));
        }
        if let Some(new) = current.get(&entity) {
            dirty.extend(atlas::bricks_in_aabb(&config, new));
        }
    }

    // An edit moved → its BVH leaf AABB moved; rebuild so the incremental bake culls
    // against current positions.
    *bvh = bvh::Bvh::build(&aabbs);
    atlas.bake_incremental(&dirty, &resolved, &bvh, &config);
    prev_aabbs.map = current;
}

// --- Upload to GPU (placeholder — render.rs handles actual upload) ---

fn upload_sdf_buffers(_atlas: Res<atlas::SdfAtlas>) {
    // Render world will pick up atlas changes via extract
}

fn toggle_sdf_render(keyboard: Res<ButtonInput<KeyCode>>, mut enabled: ResMut<SdfRenderEnabled>) {
    if keyboard.just_pressed(KeyCode::F1) {
        enabled.0 = !enabled.0;
        info!("SDF render pass: {}", if enabled.0 { "ON" } else { "OFF" });
    }
}
