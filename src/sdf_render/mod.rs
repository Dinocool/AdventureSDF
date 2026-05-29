pub mod atlas;
pub mod bc7;
pub mod bvh;
#[cfg(feature = "debug_toolkit")]
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

// --- Components ---

// Edit primitives, CSG ops, ordering, and material live in `edits`. Re-exported
// here so the rest of the module (and external callers) keep a stable
// `sdf_render::` path.
pub use edits::{CsgKind, SdfMaterial, SdfOp, SdfOrder, SdfPrimitive};

#[derive(Component)]
pub struct SdfVolume;

#[derive(Component)]
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

// --- Gizmo ---

#[derive(Resource, Default)]
pub struct SdfSelection {
    pub entity: Option<Entity>,
    pub dragging: Option<DragState>,
    /// Handle currently hovered or being dragged, for highlight. `None` => none.
    pub active_handle: Option<gizmo::HandleId>,
}

pub struct DragState {
    pub handle: gizmo::HandleId,
    /// Axis the mouse parameter is projected along during the drag.
    pub axis: Vec3,
    pub start_mouse_proj: f32,
    pub start_transform: Transform,
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
            .init_resource::<textures::TextureLibrary>()
            .init_resource::<atlas::SdfAtlas>()
            .init_resource::<PrevEditAabbs>()
            .init_resource::<bvh::Bvh>()
            .init_resource::<SdfRenderEnabled>()
            .init_resource::<SdfRaymarchParams>()
            .init_resource::<WireframeBoundsVisible>()
            .init_resource::<RayStepCapture>()
            .register_type::<SdfPrimitive>()
            .register_type::<SdfOp>()
            .register_type::<SdfOrder>()
            .register_type::<SdfMaterial>()
            .register_type::<CsgKind>()
            .register_type::<SdfRaymarchParams>()
            // Build the material registry from the texture-library manifests, then
            // spawn the scene — chained so the registry is populated before the
            // spawns resolve their material ids. (The initial-state `OnEnter` fires
            // during startup state-transition, *before* the `Startup` schedule, so
            // a plain `Startup` system would run too late.)
            .add_systems(
                OnEnter(AppScene::SdfEditor),
                (textures::build_texture_library, setup_sdf_scene).chain(),
            )
            .add_systems(
                Update,
                (
                    orbit_camera,
                    sdf_picking,
                    gizmo_interaction,
                    gizmo_hover,
                    bake_dirty_bricks,
                    upload_sdf_buffers,
                    toggle_sdf_render,
                )
                    .chain()
                    .run_if(in_state(AppScene::SdfEditor)),
            );

        // Overlay gizmos need GizmoPlugin (Assets<GizmoAsset>). Present in the real
        // app (DefaultPlugins) but not in MinimalPlugins test harnesses, so register
        // the group + drawing only when that infrastructure exists.
        if app.world().is_resource_added::<Assets<GizmoAsset>>()
            || app.world().get_resource::<Assets<GizmoAsset>>().is_some()
        {
            app.init_gizmo_group::<SdfOverlayGizmos>()
                .add_systems(OnEnter(AppScene::SdfEditor), configure_overlay_gizmos)
                .add_systems(
                    Update,
                    draw_gizmo_handles.run_if(in_state(AppScene::SdfEditor)),
                );
        }

        #[cfg(feature = "debug_toolkit")]
        app.add_plugins(debug::SdfDebugPlugin);
    }
}

// --- Scene Setup ---

fn setup_sdf_scene(mut commands: Commands, library: Res<textures::TextureLibrary>) {
    // Demo materials reference library variants by slug (registry id = 1 + layer,
    // built in `build_texture_library`). Falls back to id 0 if the library is
    // missing, so the scene still renders.
    let mat_of = |slug: &str| -> u32 {
        library
            .variants
            .iter()
            .position(|v| v.slug == slug)
            .map(|layer| 1 + layer as u32)
            .unwrap_or(0)
    };
    let mat_cobble = mat_of("cobble_stone");
    let mat_sand = mat_of("sand");
    let mat_ground = mat_of("ground");
    // Distinct variants within a slug so neighbours look different (registry id =
    // 1 + layer; consecutive ids are consecutive variants). `nth` clamps the offset
    // so it stays a valid registry id even if a slug has few variants.
    let n_mats = library.variants.len() as u32 + 1; // +1 for the fallback at id 0
    let nth = |base: u32, off: u32| -> u32 {
        if base == 0 {
            0
        } else {
            (base + off).min(n_mats.saturating_sub(1))
        }
    };
    let mat_cobble2 = nth(mat_cobble, 2);
    let mat_ground2 = nth(mat_ground, 3);

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

fn orbit_camera(
    mut orbit: ResMut<SdfOrbitCamera>,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    mut scroll: MessageReader<MouseWheel>,
) {
    for ev in scroll.read() {
        orbit.distance = (orbit.distance - ev.y * 0.5).clamp(1.0, 50.0);
    }

    if !mouse.pressed(MouseButton::Right) {
        motion.clear();
        return;
    }

    for ev in motion.read() {
        orbit.yaw -= ev.delta.x * 0.005;
        orbit.pitch = (orbit.pitch + ev.delta.y * 0.005).clamp(-1.4, 1.4);
    }

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

fn sdf_picking(
    mouse: Res<ButtonInput<MouseButton>>,
    mut selection: ResMut<SdfSelection>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    volumes: Query<VolumeQueryData, With<SdfVolume>>,
    bvh: Res<bvh::Bvh>,
) {
    if !mouse.just_pressed(MouseButton::Left) || selection.dragging.is_some() {
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

    // Don't change selection if clicking on a gizmo handle.
    if let Some(entity) = selection.entity
        && let Ok((_, transform, _, _, _, _)) = volumes.get(entity)
        && picking::raymarch_gizmo(&ray, transform.translation).is_some()
    {
        return;
    }

    let gathered = gather_sorted_edits(&volumes);
    selection.entity = picking::pick_entity(&bvh, &ray, &gathered);
}

// --- Gizmo Interaction ---

fn gizmo_interaction(
    mouse: Res<ButtonInput<MouseButton>>,
    mut selection: ResMut<SdfSelection>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    mut volumes: Query<&mut Transform, (With<SdfVolume>, Without<SdfCamera>)>,
) {
    let selected_entity = match selection.entity {
        Some(e) => e,
        None => return,
    };

    if !mouse.pressed(MouseButton::Left) {
        selection.dragging = None;
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

    // Continue ongoing drag — the action is determined by which handle was grabbed.
    if let Some(ref drag) = selection.dragging {
        let current = project_onto_axis(&ray, drag.start_transform.translation, drag.axis);
        let delta = current - drag.start_mouse_proj;

        if let Ok(mut transform) = volumes.get_mut(selected_entity) {
            apply_handle_drag(drag, delta, &mut transform);
            // Mutating Transform triggers `Changed<Transform>`, which
            // `bake_dirty_bricks` picks up to rebake just the affected bricks.
        }
        return;
    }

    // Start a new drag on gizmo-handle click. Picks across ALL handles (no mode).
    if mouse.just_pressed(MouseButton::Left) {
        let Ok(transform) = volumes.get_mut(selected_entity) else {
            return;
        };
        let origin = transform.translation;
        let start_transform = *transform;

        if let Some(pick) = picking::raymarch_gizmo(&ray, origin) {
            let axis = gizmo::Handle {
                id: pick.handle,
                origin,
            }
            .drag_axis();
            let start_proj = project_onto_axis(&ray, origin, axis);
            selection.dragging = Some(DragState {
                handle: pick.handle,
                axis,
                start_mouse_proj: start_proj,
                start_transform,
            });
        }
    }
}

/// Apply a drag delta to the transform according to the grabbed handle's action.
fn apply_handle_drag(drag: &DragState, delta: f32, transform: &mut Transform) {
    use gizmo::HandleId;
    let start = &drag.start_transform;
    match drag.handle {
        HandleId::Translate(_) => {
            transform.translation = start.translation + drag.axis * delta;
        }
        HandleId::Rotate(_) => {
            // Negated so dragging matches the visual ring direction (was inverted).
            transform.rotation = Quat::from_axis_angle(drag.axis, -delta) * start.rotation;
        }
        HandleId::ScaleAxis(a) => {
            // Scale only the grabbed axis.
            let mut scale = start.scale;
            let factor = (1.0 + delta).abs().max(0.01);
            scale[a as usize] = start.scale[a as usize] * factor;
            transform.scale = scale;
        }
        HandleId::ScalePlane(a, b) => {
            // Scale the two in-plane axes together.
            let factor = (1.0 + delta).abs().max(0.01);
            let mut scale = start.scale;
            scale[a as usize] = start.scale[a as usize] * factor;
            scale[b as usize] = start.scale[b as usize] * factor;
            transform.scale = scale;
        }
    }
}

/// Push the overlay gizmo group in front of everything (always-on-top handles).
fn configure_overlay_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<SdfOverlayGizmos>();
    config.depth_bias = -1.0;
    config.line.width = 3.0;
}

/// Color for a handle: axis-tinted (X=red, Y=green, Z=blue; planar = blend of its
/// two axes), brightened toward yellow-white when it's the hovered/active handle
/// — the Blender/Maya highlight convention.
fn handle_color(id: gizmo::HandleId, active: Option<gizmo::HandleId>) -> Color {
    use gizmo::HandleId;
    let axis_rgb = |a: u8| match a {
        0 => Srgba::rgb(0.86, 0.24, 0.24),
        1 => Srgba::rgb(0.36, 0.78, 0.30),
        _ => Srgba::rgb(0.26, 0.49, 0.92),
    };
    let base = match id {
        HandleId::Translate(a) | HandleId::Rotate(a) | HandleId::ScaleAxis(a) => axis_rgb(a),
        HandleId::ScalePlane(a, b) => axis_rgb(a).mix(&axis_rgb(b), 0.5),
    };
    if active == Some(id) {
        base.mix(&Srgba::rgb(1.0, 0.95, 0.4), 0.7).into()
    } else {
        base.into()
    }
}

/// Draw the full universal manipulator: every translate/rotate/scale handle at
/// once. Geometry comes from gizmo::Handle — the same definition picking uses.
fn draw_gizmo_handles(
    mut gizmos: Gizmos<SdfOverlayGizmos>,
    selection: Res<SdfSelection>,
    volumes: Query<&Transform, With<SdfVolume>>,
) {
    let Some(entity) = selection.entity else {
        return;
    };
    let Ok(transform) = volumes.get(entity) else {
        return;
    };
    for handle in gizmo::Handle::all(transform.translation) {
        handle.draw(
            &mut gizmos,
            handle_color(handle.id, selection.active_handle),
        );
    }
}

/// Update `selection.active_handle` for highlighting: the dragged handle wins,
/// otherwise the handle currently under the cursor, otherwise none.
fn gizmo_hover(
    mut selection: ResMut<SdfSelection>,
    cameras: Query<(&Camera, &Transform), With<SdfCamera>>,
    windows: Query<&Window>,
    volumes: Query<&Transform, With<SdfVolume>>,
) {
    if let Some(ref drag) = selection.dragging {
        selection.active_handle = Some(drag.handle);
        return;
    }

    let Some(entity) = selection.entity else {
        selection.active_handle = None;
        return;
    };
    let (Ok(window), Ok((camera, cam_transform))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(mouse_pos) = window.cursor_position() else {
        selection.active_handle = None;
        return;
    };
    let Ok(transform) = volumes.get(entity) else {
        return;
    };
    let Some(ray) = picking::mouse_to_ray(camera, cam_transform, window, mouse_pos) else {
        return;
    };

    selection.active_handle =
        picking::raymarch_gizmo(&ray, transform.translation).map(|pick| pick.handle);
}

/// Project a ray onto an axis line and return the parameter along the axis.
fn project_onto_axis(ray: &picking::Ray, origin: Vec3, axis: Vec3) -> f32 {
    let w = ray.origin - origin;
    let a = ray.direction.dot(ray.direction);
    let b = ray.direction.dot(axis);
    let c = axis.dot(axis);
    let d = ray.direction.dot(w);
    let e = axis.dot(w);
    let denom = a * c - b * b;
    if denom.abs() < 1e-8 {
        return 0.0;
    }
    // Closest-point-on-line parameter along `axis` (line2). The classic result
    // is (a*e - b*d)/denom; the negated form drags the opposite way.
    (a * e - b * d) / denom
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
