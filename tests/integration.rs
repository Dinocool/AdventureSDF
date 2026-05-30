use adventure::combat::{CombatPlugin, CombatState, DamageEvent, DamageType};
use adventure::inventory::{InventoryPlugin, ItemType, LootEvent, PlayerInventory, Rarity};
use adventure::networking::{NetworkState, NetworkingPlugin};
use adventure::player::Health;
use adventure::player::Mana;
use adventure::player::MovementSpeed;
use adventure::player::Player;
use adventure::player::PlayerLevel;
use adventure::player::PlayerName;
use adventure::scene_manager::AppScene;
use adventure::world::GameWorld;
use adventure::world::WorldPlugin;
use bevy::prelude::*;

fn integration_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.insert_resource(ButtonInput::<MouseButton>::default());
    app
}

fn app_with_logic_plugins() -> App {
    let mut app = integration_app();
    app.add_plugins(CombatPlugin);
    app.add_plugins(InventoryPlugin);
    app.add_plugins(NetworkingPlugin);
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.init_state::<AppScene>();
    app.world_mut()
        .resource_mut::<NextState<AppScene>>()
        .set(AppScene::AdventureGame);
    app.update();
    app
}

#[test]
fn combat_plugin_registers_resources() {
    let mut app = integration_app();
    app.add_plugins(CombatPlugin);

    assert!(app.world().contains_resource::<CombatState>());
}

#[test]
fn inventory_plugin_registers_resources() {
    let mut app = integration_app();
    app.add_plugins(InventoryPlugin);

    assert!(app.world().contains_resource::<PlayerInventory>());
    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.max_slots, 20);
}

#[test]
fn networking_plugin_registers_resources() {
    let mut app = integration_app();
    app.add_plugins(NetworkingPlugin);

    assert!(app.world().contains_resource::<NetworkState>());
}

#[test]
fn world_plugin_registers_resources() {
    let mut app = integration_app();
    app.add_plugins(WorldPlugin);

    assert!(app.world().contains_resource::<GameWorld>());
    let gw = app.world().resource::<GameWorld>();
    assert_eq!(gw.zone_name, "Elwynn Forest");
}

#[test]
fn damage_then_loot_workflow() {
    let mut app = app_with_logic_plugins();

    let player = app
        .world_mut()
        .spawn((
            Player,
            Health {
                current: 100.0,
                max: 100.0,
            },
            Mana {
                current: 50.0,
                max: 50.0,
            },
            MovementSpeed(5.0),
            PlayerName("Hero".into()),
            PlayerLevel(1),
        ))
        .id();

    app.world_mut()
        .resource_mut::<Messages<DamageEvent>>()
        .write(DamageEvent {
            target: player,
            amount: 40.0,
            damage_type: DamageType::Physical,
        });

    app.update();

    assert_eq!(app.world().get::<Health>(player).unwrap().current, 60.0);

    app.world_mut()
        .resource_mut::<Messages<LootEvent>>()
        .write(LootEvent {
            item: adventure::inventory::Item {
                name: "Health Potion".into(),
                item_type: ItemType::Consumable,
                rarity: Rarity::Common,
                level_requirement: 1,
            },
            quantity: 3,
        });

    app.update();

    let inv = app.world().resource::<PlayerInventory>();
    assert_eq!(inv.items.len(), 1);
    assert_eq!(inv.items[0].0.name, "Health Potion");
    assert_eq!(inv.items[0].1, 3);
}

#[test]
fn multiple_damage_sources_kill_player() {
    let mut app = app_with_logic_plugins();

    let player = app
        .world_mut()
        .spawn((
            Player,
            Health {
                current: 50.0,
                max: 100.0,
            },
            Mana {
                current: 50.0,
                max: 50.0,
            },
            MovementSpeed(5.0),
            PlayerName("Victim".into()),
            PlayerLevel(1),
        ))
        .id();

    let mut msgs = app.world_mut().resource_mut::<Messages<DamageEvent>>();
    msgs.write(DamageEvent {
        target: player,
        amount: 30.0,
        damage_type: DamageType::Fire,
    });
    msgs.write(DamageEvent {
        target: player,
        amount: 30.0,
        damage_type: DamageType::Magical,
    });

    app.update();

    assert_eq!(app.world().get::<Health>(player).unwrap().current, 0.0);
}

// --- B0001 Query Conflict Smoke Tests ---
//
// Bevy's B0001 error is a RUNTIME panic, not a compile error. It fires when
// two &mut Query<T> parameters in the same schedule access overlapping
// components without `Without<Z>` to prove disjointness.
//
// These tests load each app scene with all its plugins and tick a few frames.
// If any B0001 conflict exists, the test panics — catching it in CI instead
// of at the user's desk.

/// Helper: build a minimal app with message types needed by most scenes.
fn scene_test_app() -> App {
    use bevy::input::mouse::{MouseMotion, MouseWheel};
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.insert_resource(ButtonInput::<MouseButton>::default());
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.init_state::<AppScene>();
    app.add_message::<MouseMotion>();
    app.add_message::<MouseWheel>();
    app
}

/// Tick the app through a scene transition + N frames.
/// Panics on B0001 or any other system error.
fn run_scene_frames(app: &mut App, scene: AppScene, frames: usize) {
    app.world_mut()
        .resource_mut::<NextState<AppScene>>()
        .set(scene);
    app.update(); // process transition + first frame
    for _ in 0..frames {
        app.update();
    }
}

#[test]
fn b0001_sdf_editor_scene() {
    use adventure::sdf_render::SdfScenePlugin;

    let mut app = scene_test_app();
    // setup_sdf_scene loads demo materials, so it needs AssetServer (AssetPlugin) plus
    // the MaterialAsset registration + tables/registry that AssetsPlugin sets up.
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.add_plugins(adventure::assets::AssetsPlugin);
    app.add_plugins(SdfScenePlugin);
    run_scene_frames(&mut app, AppScene::SdfEditor, 5);
}

/// Test the UI plugin's systems for B0001 conflicts.
/// The UI plugin queries &mut Node in multiple systems — without Without<T>
/// annotations, Bevy panics at runtime.
#[test]
fn b0001_ui_systems() {
    use adventure::ui::UiPlugin;

    let mut app = scene_test_app();
    app.add_plugins(UiPlugin);
    // Just tick frames — no scene transition needed since we're testing
    // the system query declarations, not their execution on real entities.
    for _ in 0..3 {
        app.update();
    }
}
