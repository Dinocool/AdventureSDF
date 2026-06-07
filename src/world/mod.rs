use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::inventory::{Item, ItemType, Lootable, Rarity};
use crate::scene_manager::{AppScene, SceneEntity};

pub struct WorldPlugin;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Terrain;

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Npc {
    pub name: String,
    pub level: u32,
    pub hostile: bool,
}

#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct QuestGiver {
    pub quest_name: String,
    pub quest_description: String,
}

#[derive(Resource)]
pub struct GameWorld {
    pub zone_name: String,
}

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GameWorld {
            zone_name: "Elwynn Forest".into(),
        })
        .register_type::<Terrain>()
        .register_type::<Npc>()
        .register_type::<QuestGiver>()
        .add_systems(
            OnEnter(AppScene::AdventureGame),
            (
                // TEMP: world terrain disabled during the mesh-bake migration to keep focus on the
                // gallery scene. (Terrain only spawns in the AdventureGame scene anyway, not the SDF
                // gallery; gated here so it's an obvious, reversible one-line toggle.) Re-enable by
                // removing `.run_if(|| false)`.
                spawn_terrain.run_if(|| false),
                spawn_lights,
                spawn_npc_scene,
                spawn_test_chests,
            ),
        )
        .add_systems(
            Update,
            setup_npc_visuals.run_if(in_state(AppScene::AdventureGame)),
        );
    }
}

fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(
            meshes.add(
                Plane3d::new(Vec3::Y, Vec2::splat(100.0))
                    .mesh()
                    .size(200.0, 200.0),
            ),
        ),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.2),
            ..default()
        })),
        Terrain,
        RayCastBackfaces,
        RigidBody::Fixed,
        Collider::halfspace(Vec3::Y).unwrap(),
        SceneEntity,
    ));
}

fn spawn_npc_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        DynamicSceneRoot(asset_server.load("scenes/world.scn.ron")),
        SceneEntity,
    ));
}

fn spawn_lights(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_x(-0.5)),
        SceneEntity,
    ));

    commands.spawn((
        PointLight {
            intensity: 100_000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 8.0, 0.0),
        SceneEntity,
    ));
}

fn setup_npc_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    npcs: Query<Entity, (With<Npc>, Without<Mesh3d>)>,
) {
    for entity in &npcs {
        commands.entity(entity).insert((
            Mesh3d(meshes.add(Capsule3d::default())),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.2, 0.2),
                ..default()
            })),
            RigidBody::Fixed,
            Collider::capsule_y(0.5, 0.5),
            SceneEntity,
        ));
    }
}

fn spawn_test_chests(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let chest_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.35, 0.15),
        ..default()
    });
    let chest_mesh = meshes.add(Cuboid::new(1.0, 0.8, 0.6));

    let chests = [
        (
            Transform::from_xyz(-3.0, 0.4, 2.0),
            vec![
                (
                    Item {
                        name: "Rusty Sword".into(),
                        item_type: ItemType::Weapon,
                        rarity: Rarity::Common,
                        level_requirement: 1,
                    },
                    1,
                ),
                (
                    Item {
                        name: "Health Potion".into(),
                        item_type: ItemType::Consumable,
                        rarity: Rarity::Common,
                        level_requirement: 1,
                    },
                    3,
                ),
            ],
        ),
        (
            Transform::from_xyz(4.0, 0.4, -5.0),
            vec![(
                Item {
                    name: "Leather Boots".into(),
                    item_type: ItemType::Armor,
                    rarity: Rarity::Uncommon,
                    level_requirement: 2,
                },
                1,
            )],
        ),
        (
            Transform::from_xyz(8.0, 0.4, 3.0),
            vec![
                (
                    Item {
                        name: "Mana Crystal".into(),
                        item_type: ItemType::Material,
                        rarity: Rarity::Rare,
                        level_requirement: 5,
                    },
                    2,
                ),
                (
                    Item {
                        name: "Fire Staff".into(),
                        item_type: ItemType::Weapon,
                        rarity: Rarity::Epic,
                        level_requirement: 8,
                    },
                    1,
                ),
            ],
        ),
    ];

    for (transform, items) in chests {
        commands.spawn((
            Mesh3d(chest_mesh.clone()),
            MeshMaterial3d(chest_material.clone()),
            transform,
            RigidBody::Fixed,
            Collider::cuboid(0.5, 0.4, 0.3),
            Lootable { items },
            SceneEntity,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    /// Builds the NPC scene programmatically and writes it to
    /// `assets/scenes/world.scn.ron`. Run with `cargo test -- generate_world_scene --nocapture`
    /// to see the output. The test also validates the file can be loaded back.
    #[test]
    fn generate_world_scene() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .register_type::<Transform>()
            .register_type::<Npc>()
            .register_type::<QuestGiver>();

        let entity = app
            .world_mut()
            .spawn((
                Transform::from_xyz(5.0, 1.0, -3.0),
                Npc {
                    name: "Guard Thomas".into(),
                    level: 5,
                    hostile: false,
                },
                QuestGiver {
                    quest_name: "A Threat Within".into(),
                    quest_description: "Defeat the kobolds threatening the forest.".into(),
                },
            ))
            .id();

        let scene = DynamicSceneBuilder::from_world(app.world())
            .extract_entity(entity)
            .build();

        let registry = app.world().resource::<AppTypeRegistry>();
        let ron = scene.serialize(&registry.read()).unwrap();

        std::fs::create_dir_all("assets/scenes").unwrap();
        std::fs::write("assets/scenes/world.scn.ron", &ron).unwrap();
        eprintln!("Generated assets/scenes/world.scn.ron:\n{ron}");
    }

    #[test]
    fn npc_reflect_registered() {
        let mut app = test_app();
        app.register_type::<Npc>();
        let registry = app.world().resource::<AppTypeRegistry>();
        let reg = registry.read();
        assert!(
            reg.get_type_data::<ReflectComponent>(std::any::TypeId::of::<Npc>())
                .is_some()
        );
    }

    #[test]
    fn quest_giver_reflect_registered() {
        let mut app = test_app();
        app.register_type::<QuestGiver>();
        let registry = app.world().resource::<AppTypeRegistry>();
        let reg = registry.read();
        assert!(
            reg.get_type_data::<ReflectComponent>(std::any::TypeId::of::<QuestGiver>())
                .is_some()
        );
    }

    #[test]
    fn game_world_default_zone() {
        let world = GameWorld {
            zone_name: "Elwynn Forest".into(),
        };
        assert_eq!(world.zone_name, "Elwynn Forest");
    }

    #[test]
    fn scene_serialization_roundtrip() {
        let mut app = test_app();
        app.register_type::<Transform>()
            .register_type::<Npc>()
            .register_type::<QuestGiver>();

        let entity = app
            .world_mut()
            .spawn((
                Transform::from_xyz(1.0, 2.0, 3.0),
                Npc {
                    name: "Test NPC".into(),
                    level: 10,
                    hostile: true,
                },
                QuestGiver {
                    quest_name: "Test Quest".into(),
                    quest_description: "A test.".into(),
                },
            ))
            .id();

        let scene = DynamicSceneBuilder::from_world(app.world())
            .extract_entity(entity)
            .build();

        let registry = app.world().resource::<AppTypeRegistry>();
        let ron = scene.serialize(&registry.read()).unwrap();

        assert!(ron.contains("Test NPC"));
        assert!(ron.contains("Test Quest"));
    }
}
