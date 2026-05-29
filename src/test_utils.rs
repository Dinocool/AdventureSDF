use bevy::prelude::*;

pub fn test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app
}

pub fn test_app_with_input() -> App {
    let mut app = test_app();
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.insert_resource(ButtonInput::<MouseButton>::default());
    app
}

pub fn press_key(app: &mut App, key: KeyCode) {
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(key);
}

pub fn release_key(app: &mut App, key: KeyCode) {
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .release(key);
}

pub fn clear_input(app: &mut App) {
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .clear();
}

pub fn spawn_test_player(world: &mut World) -> Entity {
    world
        .spawn((
            crate::player::Player,
            crate::player::Health {
                current: 100.0,
                max: 100.0,
            },
            crate::player::Mana {
                current: 50.0,
                max: 50.0,
            },
            crate::player::MovementSpeed(5.0),
            crate::player::PlayerName("TestPlayer".into()),
            crate::player::PlayerLevel(1),
            crate::player::CharacterController {
                vertical_velocity: 0.0,
            },
            Transform::from_xyz(0.0, 1.0, 0.0),
        ))
        .id()
}
