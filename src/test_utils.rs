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

/// Headless app with state `S` initialized, for testing state-scoped features
/// (`OnEnter`/`OnExit` systems, `run_if(in_state(...))`). Starts in `S::default()`.
pub fn test_app_with_state<S: bevy::state::state::FreelyMutableState + FromWorld>() -> App {
    let mut app = test_app();
    app.init_state::<S>();
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

/// Spawn an NPC quest-giver for world/interaction tests. Mirrors the entity built
/// by the `generate_world_scene` test in `src/world/mod.rs`.
pub fn spawn_test_npc(world: &mut World) -> Entity {
    world
        .spawn((
            crate::world::Npc {
                name: "TestNpc".into(),
                level: 5,
                hostile: false,
            },
            crate::world::QuestGiver {
                quest_name: "Test Quest".into(),
                quest_description: "A test quest.".into(),
            },
            Transform::from_xyz(2.0, 0.0, 2.0),
        ))
        .id()
}
