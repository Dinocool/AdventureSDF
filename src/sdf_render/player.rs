//! WORLDGEN PLAYER mode — a 3rd-person controlled capsule that walks the baked terrain colliders (P1),
//! with the [`SdfCamera`] following it. Toggled via [`SdfCameraMode::player`] (the `P` key / the viewport
//! toolbar). Lets you drop in and test the terrain on foot. Active only in the `SdfEditor` scene; while on,
//! it overrides the orbit/free-fly camera (those systems are gated off in `SdfScenePlugin`).

use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::SdfCamera;
use super::editor_camera::SdfCameraMode;

const GRAVITY: f32 = 22.0; // m/s²
const JUMP_SPEED: f32 = 9.0; // m/s
const WALK_SPEED: f32 = 12.0; // m/s
const CAM_DISTANCE: f32 = 9.0; // 3rd-person camera distance (m)
const EYE_HEIGHT: f32 = 1.2; // look target above the player's origin (m)

/// The worldgen test player — a capsule on a Rapier kinematic character controller. At most one exists
/// (spawned when player mode turns on, despawned when off).
#[derive(Component)]
pub struct WorldgenPlayer {
    /// Vertical velocity (m/s), integrated manually for gravity + jump (the controller is position-based).
    vy: f32,
}

/// Toggle player mode with the `P` key (works in both build configs; the editor toolbar also toggles it).
pub(crate) fn toggle_player_mode(keyboard: Res<ButtonInput<KeyCode>>, mut mode: ResMut<SdfCameraMode>) {
    if keyboard.just_pressed(KeyCode::KeyP) {
        mode.player = !mode.player;
    }
}

/// Spawn the player when player mode turns ON (and none exists); despawn it when player mode turns OFF.
/// Spawns at the camera's XZ dropped onto the terrain surface (+ a margin so gravity settles it cleanly).
pub(crate) fn manage_player(
    mut commands: Commands,
    mode: Res<SdfCameraMode>,
    players: Query<Entity, With<WorldgenPlayer>>,
    cam: Query<&Transform, With<SdfCamera>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let exists = !players.is_empty();
    if mode.player && !exists {
        let cam_pos = cam.single().map(|t| t.translation).unwrap_or(Vec3::ZERO);
        // Drop onto the pristine surface height at the camera's XZ (falls onto the chunk collider).
        let ground = super::worldgen::upload::cpu_terrain_hifi()
            .map(|h| h.surface(cam_pos.x as f64, cam_pos.z as f64).0)
            .unwrap_or(cam_pos.y);
        let spawn = Vec3::new(cam_pos.x, ground + 3.0, cam_pos.z);
        commands.spawn((
            Name::new("Worldgen Player"),
            Transform::from_translation(spawn),
            Mesh3d(meshes.add(Capsule3d::new(0.5, 1.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.20, 0.50, 0.90),
                perceptual_roughness: 0.6,
                ..default()
            })),
            RigidBody::KinematicPositionBased,
            Collider::capsule_y(0.5, 0.5),
            KinematicCharacterController {
                slide: true,
                snap_to_ground: Some(CharacterLength::Absolute(0.1)),
                autostep: Some(CharacterAutostep {
                    max_height: CharacterLength::Absolute(0.4),
                    min_width: CharacterLength::Absolute(0.2),
                    include_dynamic_bodies: false,
                }),
                ..default()
            },
            WorldgenPlayer { vy: 0.0 },
            crate::scene_manager::EditorEntity,
            crate::soul_scene::NonSerializable,
            crate::node::Node3D,
        ));
    } else if !mode.player && exists {
        for e in &players {
            commands.entity(e).despawn();
        }
    }
}

/// WASD movement relative to the camera yaw + jump (Space) + gravity, driven through the kinematic character
/// controller (collide-and-slide on the chunk colliders).
pub(crate) fn move_worldgen_player(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mode: Res<SdfCameraMode>,
    mut q: Query<(
        &mut WorldgenPlayer,
        &mut KinematicCharacterController,
        Option<&KinematicCharacterControllerOutput>,
    )>,
) {
    let Ok((mut player, mut ctrl, out)) = q.single_mut() else {
        return;
    };
    let grounded = out.is_some_and(|o| o.grounded);
    if keyboard.just_pressed(KeyCode::Space) && grounded {
        player.vy = JUMP_SPEED;
    }
    if grounded && player.vy < 0.0 {
        player.vy = 0.0; // rest on the ground rather than accumulating downward velocity
    }
    player.vy -= GRAVITY * time.delta_secs();

    // Forward/right on the XZ plane from the camera yaw (matches the editor camera's yaw convention).
    let (s, c) = mode.yaw.sin_cos();
    let forward = Vec3::new(c, 0.0, s);
    let right = Vec3::new(-s, 0.0, c);
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
    if dir.length_squared() > 0.0 {
        dir = dir.normalize();
    }
    let dt = time.delta_secs();
    ctrl.translation = Some(dir * WALK_SPEED * dt + Vec3::Y * player.vy * dt);
}

/// 3rd-person follow: RMB-drag orbits the yaw/pitch; the `SdfCamera` sits behind + above the player looking
/// at it. Reuses [`SdfCameraMode`]'s `yaw`/`pitch` so toggling back to free-fly keeps the heading.
pub(crate) fn follow_player_camera(
    mut mode: ResMut<SdfCameraMode>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    players: Query<&Transform, (With<WorldgenPlayer>, Without<SdfCamera>)>,
    mut cam: Query<&mut Transform, With<SdfCamera>>,
) {
    let Ok(player) = players.single() else {
        return;
    };
    let Ok(mut cam_t) = cam.single_mut() else {
        return;
    };
    if mouse.pressed(MouseButton::Right) {
        for ev in motion.read() {
            mode.yaw -= ev.delta.x * 0.005;
            mode.pitch = (mode.pitch + ev.delta.y * 0.005).clamp(-1.3, 1.3);
        }
    } else {
        motion.clear();
    }
    // Camera "forward" from yaw/pitch (same basis as the editor cameras); sit opposite it, raised.
    let (sy, cy) = mode.yaw.sin_cos();
    let cp = mode.pitch.cos();
    let fwd = Vec3::new(cy * cp, mode.pitch.sin(), sy * cp);
    let head = player.translation + Vec3::Y * EYE_HEIGHT;
    cam_t.translation = head - fwd * CAM_DISTANCE;
    cam_t.look_at(head, Vec3::Y);
}
