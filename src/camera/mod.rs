use std::f32::consts::FRAC_PI_2;

use bevy::input::mouse::MouseMotion;
use bevy::math::{Dir3, Ray3d};
use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::player::CharacterController;
use crate::scene_manager::{AppScene, SceneEntity};
use crate::world::Terrain;

pub struct CameraPlugin;

#[derive(Resource)]
pub struct CameraMode {
    pub free_camera_active: bool,
    pub free_speed: f32,
    free_yaw: f32,
    free_pitch: f32,
}

impl Default for CameraMode {
    fn default() -> Self {
        Self {
            free_camera_active: false,
            free_speed: 10.0,
            free_yaw: 0.0,
            free_pitch: 0.0,
        }
    }
}

#[derive(Component)]
pub struct ThirdPersonCamera {
    pub distance: f32,
    pub height_offset: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub orbit_speed: f32,
}

impl Default for ThirdPersonCamera {
    fn default() -> Self {
        Self {
            distance: 10.0,
            height_offset: 1.5,
            yaw: 0.0,
            pitch: 0.3,
            orbit_speed: 0.003,
        }
    }
}

#[derive(Component)]
pub struct FreeCamera;

#[derive(Resource, Default)]
pub struct RightClickState {
    pub drag_accumulated: Vec2,
    pub dragging: bool,
}

#[derive(Message)]
pub struct RightClickEvent {
    pub screen_position: Vec2,
}

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CameraMode::default())
            .init_resource::<RightClickState>()
            .add_message::<RightClickEvent>()
            .add_systems(OnEnter(AppScene::AdventureGame), spawn_cameras)
            .add_systems(
                Update,
                (
                    track_right_click.run_if(third_person_active),
                    toggle_camera_mode,
                    orbit_camera.run_if(third_person_active),
                    free_camera_system.run_if(free_camera_active),
                )
                    .chain()
                    .run_if(in_state(AppScene::AdventureGame)),
            );
    }
}

fn third_person_active(mode: Res<CameraMode>) -> bool {
    !mode.free_camera_active
}

fn free_camera_active(mode: Res<CameraMode>) -> bool {
    mode.free_camera_active
}

fn spawn_cameras(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ThirdPersonCamera::default(),
        SceneEntity,
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        Transform::from_xyz(0.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        FreeCamera,
        SceneEntity,
    ));
}

fn track_right_click(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: MessageReader<MouseMotion>,
    mut state: ResMut<RightClickState>,
    mut messages: MessageWriter<RightClickEvent>,
    windows: Query<&Window>,
) {
    let delta: Vec2 = motion_events.read().map(|m| m.delta).sum();

    if mouse_buttons.pressed(MouseButton::Right) {
        state.drag_accumulated += delta;
        if state.drag_accumulated.length_squared() >= 25.0 {
            state.dragging = true;
        }
    }

    if mouse_buttons.just_released(MouseButton::Right) {
        if !state.dragging
            && let Ok(window) = windows.single()
            && let Some(pos) = window.cursor_position()
        {
            messages.write(RightClickEvent {
                screen_position: pos,
            });
        }
        state.drag_accumulated = Vec2::ZERO;
        state.dragging = false;
    }
}

pub fn toggle_camera_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<CameraMode>,
    mut tp_cam: Query<&mut Camera, (With<ThirdPersonCamera>, Without<FreeCamera>)>,
    mut free_cam: Query<&mut Camera, (With<FreeCamera>, Without<ThirdPersonCamera>)>,
    mut cursor: Query<&mut CursorOptions, With<Window>>,
) {
    if !keyboard.just_pressed(KeyCode::F10) {
        return;
    }

    mode.free_camera_active = !mode.free_camera_active;

    if let Ok(mut cam) = tp_cam.single_mut() {
        cam.is_active = !mode.free_camera_active;
    }
    if let Ok(mut cam) = free_cam.single_mut() {
        cam.is_active = mode.free_camera_active;
    }

    if let Ok(mut cursor) = cursor.single_mut() {
        if mode.free_camera_active {
            cursor.grab_mode = CursorGrabMode::Locked;
            cursor.visible = false;
        } else {
            cursor.grab_mode = CursorGrabMode::None;
            cursor.visible = true;
        }
    }
}

fn orbit_camera(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: MessageReader<MouseMotion>,
    mut ray_cast: MeshRayCast,
    terrain_query: Query<(), With<Terrain>>,
    mut player_query: Query<&mut Transform, With<CharacterController>>,
    mut camera_query: Query<(&mut ThirdPersonCamera, &mut Transform), Without<CharacterController>>,
    right_click: Res<RightClickState>,
) {
    let Ok(mut player_transform) = player_query.single_mut() else {
        return;
    };
    let Ok((mut cam, mut cam_transform)) = camera_query.single_mut() else {
        return;
    };

    let delta: Vec2 = motion_events.read().map(|m| m.delta).sum();

    let right_held = mouse_buttons.pressed(MouseButton::Right);

    if right_click.dragging {
        cam.yaw -= delta.x * cam.orbit_speed;
        cam.pitch -= delta.y * cam.orbit_speed;
        cam.pitch = cam.pitch.clamp(-0.5, 1.4);
    }

    if right_click.dragging && right_held {
        player_transform.rotation = Quat::from_rotation_y(cam.yaw);
    }

    let desired_offset = Vec3::new(
        cam.distance * cam.pitch.cos() * cam.yaw.sin(),
        cam.distance * cam.pitch.sin(),
        cam.distance * cam.pitch.cos() * cam.yaw.cos(),
    );

    let player_center = player_transform.translation + Vec3::new(0.0, cam.height_offset, 0.0);
    let desired_pos = player_center + desired_offset;

    let final_pos = if let Ok(direction) = Dir3::new(desired_offset) {
        let ray = Ray3d::new(player_center, direction);
        let terrain_filter = |entity: Entity| terrain_query.contains(entity);
        let settings = MeshRayCastSettings::default()
            .with_filter(&terrain_filter)
            .with_visibility(RayCastVisibility::Any);
        let hits = ray_cast.cast_ray(ray, &settings);

        if let Some((_, hit)) = hits.first() {
            if hit.distance < cam.distance {
                ray.get_point(hit.distance - 0.1)
            } else {
                desired_pos
            }
        } else {
            desired_pos
        }
    } else {
        desired_pos
    };

    cam_transform.translation = final_pos;
    cam_transform.look_at(player_center, Vec3::Y);
}

fn free_camera_system(
    mut motion_events: MessageReader<MouseMotion>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut mode: ResMut<CameraMode>,
    mut query: Query<&mut Transform, With<FreeCamera>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };

    let delta: Vec2 = motion_events.read().map(|m| m.delta).sum();
    if delta.length_squared() > 0.0 {
        mode.free_yaw -= delta.x * 0.002;
        mode.free_pitch -= delta.y * 0.002;
        mode.free_pitch = mode.free_pitch.clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
    }

    if keyboard.just_pressed(KeyCode::Equal) || keyboard.just_pressed(KeyCode::NumpadAdd) {
        mode.free_speed = (mode.free_speed + 5.0).min(100.0);
    }
    if keyboard.just_pressed(KeyCode::Minus) || keyboard.just_pressed(KeyCode::NumpadSubtract) {
        mode.free_speed = (mode.free_speed - 5.0).max(1.0);
    }

    transform.rotation =
        Quat::from_rotation_y(mode.free_yaw) * Quat::from_rotation_x(mode.free_pitch);

    let forward = transform.rotation * -Vec3::Z;
    let right = transform.rotation * Vec3::X;

    let mut direction = Vec3::ZERO;
    if keyboard.pressed(KeyCode::KeyW) {
        direction += forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        direction -= forward;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        direction -= right;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        direction += right;
    }
    if keyboard.pressed(KeyCode::Space) {
        direction += Vec3::Y;
    }
    if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
        direction -= Vec3::Y;
    }

    if direction.length_squared() > 0.0 {
        transform.translation += direction.normalize() * mode.free_speed * time.delta_secs();
    }
}

/// Forward direction on XZ plane from camera yaw (used for relative movement).
pub fn forward_from_yaw(yaw: f32) -> Vec3 {
    Vec3::new(-yaw.sin(), 0.0, -yaw.cos()).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_from_yaw_zero_points_neg_z() {
        let fwd = forward_from_yaw(0.0);
        assert!((fwd.x).abs() < 0.001);
        assert!((fwd.y).abs() < 0.001);
        assert!((fwd.z - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn forward_from_yaw_pi_points_pos_z() {
        let fwd = forward_from_yaw(std::f32::consts::PI);
        assert!((fwd.x).abs() < 0.01);
        assert!((fwd.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn forward_from_yaw_half_pi_points_neg_x() {
        let fwd = forward_from_yaw(std::f32::consts::FRAC_PI_2);
        assert!((fwd.x - (-1.0)).abs() < 0.01);
        assert!((fwd.z).abs() < 0.01);
    }

    #[test]
    fn forward_from_yaw_is_unit_vector() {
        for yaw in [0.0, 1.0, 2.5, std::f32::consts::PI] {
            let fwd = forward_from_yaw(yaw);
            assert!((fwd.length() - 1.0).abs() < 0.001, "yaw={yaw}");
        }
    }

    #[test]
    fn forward_from_yaw_has_zero_y() {
        for yaw in [0.0, 1.0, -1.0, std::f32::consts::PI] {
            assert_eq!(forward_from_yaw(yaw).y, 0.0, "yaw={yaw}");
        }
    }

    #[test]
    fn camera_mode_defaults_to_third_person() {
        let mode = CameraMode::default();
        assert!(!mode.free_camera_active);
        assert_eq!(mode.free_speed, 10.0);
    }

    #[test]
    fn third_person_camera_defaults() {
        let cam = ThirdPersonCamera::default();
        assert_eq!(cam.distance, 10.0);
        assert_eq!(cam.height_offset, 1.5);
        assert_eq!(cam.orbit_speed, 0.003);
    }

    #[test]
    fn right_click_state_defaults() {
        let state = RightClickState::default();
        assert_eq!(state.drag_accumulated, Vec2::ZERO);
        assert!(!state.dragging);
    }

    #[test]
    fn free_camera_speed_clamps() {
        let mut mode = CameraMode::default();
        mode.free_speed = 98.0;
        mode.free_speed = (mode.free_speed + 5.0).min(100.0);
        assert_eq!(mode.free_speed, 100.0);

        mode.free_speed = 3.0;
        mode.free_speed = (mode.free_speed - 5.0).max(1.0);
        assert_eq!(mode.free_speed, 1.0);
    }
}
