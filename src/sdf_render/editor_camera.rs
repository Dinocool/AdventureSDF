//! The SDF editor's viewport cameras: a Godot-style orbit camera (middle-mouse orbit/pan, wheel
//! zoom) and a free-fly (FPS) camera for crossing the km-scale clipmap terrain, plus the shared
//! input bundle and the persistent editor-camera lifecycle. These are editor-interaction concerns;
//! the public types ([`SdfOrbitCamera`], [`SdfCameraMode`], [`OrbitFocus`], [`CameraInput`]) and
//! [`sync_orbit_camera_transform`] are re-exported from [`super`] so the `sdf_render::` path is stable.

use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::ecs::system::SystemParam;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;

use super::{SdfCamera, SdfVolume};

/// Double-click-to-focus state for the orbit camera. `sdf_picking` records each
/// left-click time to detect double-clicks; a double-click on a volume sets
/// `target`, which `orbit_camera` eases `SdfOrbitCamera.target` toward.
#[derive(Resource, Default)]
pub struct OrbitFocus {
    /// World point the orbit target is easing toward; cleared once reached.
    pub target: Option<Vec3>,
    /// Elapsed-seconds timestamp of the previous left-click (double-click detection).
    pub(super) last_click: f32,
}

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

impl SdfOrbitCamera {
    /// Eye (camera) position for the current orbit parameters.
    pub fn eye(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.yaw.cos() * self.pitch.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.sin() * self.pitch.cos(),
            )
    }

    /// View transform (eye placed on the orbit sphere, looking at the target). Single
    /// source for the orbit→transform mapping used by `orbit_camera`, focus easing, and
    /// the immediate re-sync after a scene swap.
    pub fn view_transform(&self) -> Transform {
        Transform::from_translation(self.eye()).looking_at(self.target, Vec3::Y)
    }
}

/// Apply the orbit resource to the SDF camera's transform right now. `orbit_camera` only
/// runs while the pointer is in the viewport, so after a scene swap restores a per-scene
/// camera we sync here — otherwise the view wouldn't update (it'd "jump" later, once the
/// cursor re-enters the viewport).
pub fn sync_orbit_camera_transform(world: &mut World) {
    let transform = world.resource::<SdfOrbitCamera>().view_transform();
    let mut query = world.query_filtered::<&mut Transform, With<SdfCamera>>();
    for mut t in query.iter_mut(world) {
        *t = transform;
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

/// Spawn the persistent SDF editor camera once at startup. It is activated only
/// while in the SDF editor so it doesn't fight the AdventureGame / WireframeTest cameras.
/// Guarded so a hot-reload / re-run can't double-spawn it.
pub(crate) fn spawn_editor_camera(mut commands: Commands, existing: Query<(), With<SdfCamera>>) {
    if !existing.is_empty() {
        return;
    }
    let orbit = SdfOrbitCamera::default();
    let pos = orbit.target
        + Vec3::new(
            orbit.distance * orbit.yaw.cos() * orbit.pitch.cos(),
            orbit.distance * orbit.pitch.sin(),
            orbit.distance * orbit.yaw.sin() * orbit.pitch.cos(),
        );
    commands.spawn((
        Camera3d::default(),
        // Inactive until in the SDF editor scene (see `sync_editor_camera_active`), so the
        // persistent editor camera doesn't fight other scenes' cameras.
        Camera {
            is_active: false,
            ..default()
        },
        // HDR so the view target is linear Rgba16Float and Bevy's Tonemapping pass converts
        // (linear→sRGB) for display. The SDF shader then writes LINEAR radiance, which lets the
        // SSR history buffer hold correct linear values for the reflection IBL term. In Bevy
        // 0.18 `hdr` is the `Hdr` marker component (was `Camera.hdr`).
        bevy::render::view::Hdr,
        Transform::from_translation(pos).looking_at(orbit.target, Vec3::Y),
        Msaa::Off,
        SdfCamera,
        // Target for the filled gizmo overlay (gizmo_render).
        crate::gizmo_render::GizmoCamera,
        DepthPrepass,
        crate::scene_manager::EditorEntity,
        crate::soul_scene::NonSerializable,
        crate::node::Node3D,
        Name::new("Editor Camera"),
    ));
    commands.insert_resource(orbit);
}

/// Activate the editor camera only while in the SDF editor scene. Other app scenes
/// (AdventureGame, WireframeTest) render their own cameras; deactivating ours keeps exactly
/// one active camera per window across state transitions.
pub(crate) fn sync_editor_camera_active(
    state: Res<State<crate::scene_manager::AppScene>>,
    mut cam: Query<&mut Camera, With<SdfCamera>>,
) {
    if let Ok(mut cam) = cam.single_mut() {
        let want = *state.get() == crate::scene_manager::AppScene::SdfEditor;
        if cam.is_active != want {
            cam.is_active = want;
        }
    }
}

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
pub(crate) fn orbit_camera(
    mut orbit: ResMut<SdfOrbitCamera>,
    mut focus: ResMut<OrbitFocus>,
    mut input: CameraInput,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    // Wheel zoom (dolly toward/away from the target). Hold Shift for 10x coarse zoom.
    let zoom_step = if input.keyboard.pressed(KeyCode::ShiftLeft)
        || input.keyboard.pressed(KeyCode::ShiftRight)
    {
        5.0
    } else {
        0.5
    };
    for ev in input.scroll.read() {
        orbit.distance = (orbit.distance - ev.y * zoom_step).clamp(0.5, 50.0);
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
    let view = orbit.view_transform();
    for mut transform in &mut camera_query {
        *transform = view;
    }
}

/// Ease the orbit target toward a double-click focus point and recompute the camera
/// transform. Separate from `orbit_camera` (and NOT gated on `ViewportInputAllowed`)
/// so a focus triggered from a dock panel — e.g. double-clicking a Hierarchy row —
/// animates immediately, instead of stalling until the pointer re-enters the
/// viewport. Orbit-mode only; the FPS camera ignores the orbit target.
pub(crate) fn ease_orbit_focus(
    mut orbit: ResMut<SdfOrbitCamera>,
    mut focus: ResMut<OrbitFocus>,
    time: Res<Time>,
    mut camera_query: Query<&mut Transform, (With<SdfCamera>, Without<SdfVolume>)>,
) {
    let Some(dest) = focus.target else {
        return;
    };

    // Exponential smoothing (frame-rate independent); snap + clear once we're close.
    let t = 1.0 - (-12.0 * time.delta_secs()).exp();
    orbit.target = orbit.target.lerp(dest, t);
    if orbit.target.distance(dest) < 0.01 {
        orbit.target = dest;
        focus.target = None;
    }

    let view = orbit.view_transform();
    for mut transform in &mut camera_query {
        *transform = view;
    }
}

/// Free-fly (FPS) camera for the SDF editor: hold right mouse to look, WASD to move,
/// Space/Ctrl for up/down, wheel adjusts speed. Lets you fly out across the km-scale
/// clipmap terrain. Active only when `SdfCameraMode.fps` is set (the viewport toolbar
/// toggle); the orbit camera is disabled in that mode so they don't fight.
pub(crate) fn fps_camera(
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
