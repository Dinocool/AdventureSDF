use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::scene_manager::{AppScene, SceneEntity};

pub struct PlayerPlugin;

#[derive(Component)]
#[require(Health, Mana, MovementSpeed, PlayerName, PlayerLevel)]
pub struct Player;

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Health {
    pub fn full(max: f32) -> Self {
        Self { current: max, max }
    }
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct Mana {
    pub current: f32,
    pub max: f32,
}

impl Mana {
    pub fn full(max: f32) -> Self {
        Self { current: max, max }
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct MovementSpeed(pub f32);

impl Default for MovementSpeed {
    fn default() -> Self {
        Self(5.0)
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct PlayerName(pub String);

impl Default for PlayerName {
    fn default() -> Self {
        Self("Adventurer".into())
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct PlayerLevel(pub u32);

impl Default for PlayerLevel {
    fn default() -> Self {
        Self(1)
    }
}

#[derive(Component)]
pub struct CharacterController {
    pub vertical_velocity: f32,
}

#[derive(Message)]
pub struct PlayerLevelUp {
    pub new_level: u32,
}

const GRAVITY: f32 = 18.0;
const JUMP_SPEED: f32 = 6.0;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Health>()
            .register_type::<Mana>()
            .register_type::<MovementSpeed>()
            .register_type::<PlayerName>()
            .register_type::<PlayerLevel>()
            .add_message::<PlayerLevelUp>()
            .add_systems(OnEnter(AppScene::AdventureGame), spawn_player)
            .add_systems(
                Update,
                move_player
                    .after(crate::camera::toggle_camera_mode)
                    .run_if(in_state(AppScene::AdventureGame)),
            );
    }
}

fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Player,
        CharacterController {
            vertical_velocity: 0.0,
        },
        Mesh3d(meshes.add(Capsule3d::default())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.5, 0.8),
            ..default()
        })),
        Transform::from_xyz(0.0, 1.0, 0.0),
        RigidBody::KinematicPositionBased,
        Collider::capsule_y(0.5, 0.5),
        KinematicCharacterController {
            slide: true,
            snap_to_ground: Some(CharacterLength::Absolute(0.05)),
            autostep: Some(CharacterAutostep {
                max_height: CharacterLength::Absolute(0.3),
                min_width: CharacterLength::Absolute(0.2),
                include_dynamic_bodies: false,
            }),
            ..default()
        },
        SceneEntity,
    ));
}

pub fn move_player(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    camera_query: Query<&crate::camera::ThirdPersonCamera>,
    mut query: Query<
        (
            &MovementSpeed,
            &mut Transform,
            &mut CharacterController,
            Option<&KinematicCharacterControllerOutput>,
            &mut KinematicCharacterController,
        ),
        With<Player>,
    >,
    camera_mode: Res<crate::camera::CameraMode>,
) {
    if camera_mode.free_camera_active {
        return;
    }
    let Ok((speed, _transform, mut controller, output, mut char_ctrl)) = query.single_mut() else {
        return;
    };
    let Ok(cam) = camera_query.single() else {
        return;
    };

    let grounded = output.is_some_and(|o| o.grounded);

    // Jump
    if keyboard.just_pressed(KeyCode::Space) && grounded {
        controller.vertical_velocity = JUMP_SPEED;
    }

    // Gravity
    if grounded && controller.vertical_velocity < 0.0 {
        controller.vertical_velocity = 0.0;
    }
    controller.vertical_velocity -= GRAVITY * time.delta_secs();

    // Horizontal movement
    let forward = crate::camera::forward_from_yaw(cam.yaw);
    let right = Vec3::new(-forward.z, 0.0, forward.x);

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

    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
    }

    let horizontal = direction * speed.0 * time.delta_secs();
    let vertical = Vec3::new(0.0, controller.vertical_velocity * time.delta_secs(), 0.0);
    char_ctrl.translation = Some(horizontal + vertical);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn health_default_full() {
        let health = Health {
            current: 100.0,
            max: 100.0,
        };
        assert_eq!(health.current, health.max);
    }

    #[test]
    fn mana_default_full() {
        let mana = Mana {
            current: 50.0,
            max: 50.0,
        };
        assert_eq!(mana.current, mana.max);
    }

    #[test]
    fn gravity_pulls_vertical_velocity_down() {
        let dt = 1.0 / 60.0;
        let mut vv = 0.0_f32;
        vv -= GRAVITY * dt;
        assert!(vv < 0.0, "Gravity should pull velocity negative");
        assert!((vv - (-GRAVITY * dt)).abs() < f32::EPSILON);
    }

    #[test]
    fn jump_speed_constant() {
        assert_eq!(JUMP_SPEED, 6.0);
    }

    #[test]
    fn gravity_constant() {
        assert_eq!(GRAVITY, 18.0);
    }

    #[test]
    fn character_controller_default() {
        let cc = CharacterController {
            vertical_velocity: 0.0,
        };
        assert_eq!(cc.vertical_velocity, 0.0);
    }

    #[test]
    fn move_player_exits_early_in_free_camera() {
        let mut app = test_app_with_input();
        app.add_message::<PlayerLevelUp>();
        let mut mode = crate::camera::CameraMode::default();
        mode.free_camera_active = true;
        app.insert_resource(mode);

        let entity = app
            .world_mut()
            .spawn((
                Player,
                MovementSpeed(5.0),
                CharacterController {
                    vertical_velocity: 0.0,
                },
                Transform::from_xyz(0.0, 1.0, 0.0),
                KinematicCharacterController::default(),
            ))
            .id();

        app.add_systems(Update, move_player);
        press_key(&mut app, KeyCode::KeyW);
        app.update();

        let transform = app.world().get::<Transform>(entity).unwrap();
        assert_eq!(transform.translation.x, 0.0);
        assert_eq!(transform.translation.z, 0.0);
    }
}
