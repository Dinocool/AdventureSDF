---
name: create-scene
description: Create or update Bevy 0.18 dynamic scene (.scn.ron) files using the generate_world_scene test
---

# Create Bevy Scene

Generate scene files from code. Never hand-write `.scn.ron` — the format is easy to get wrong.

## Validate / Generate Scenes

The `generate_world_scene` test in `src/world/mod.rs` builds the NPC scene programmatically and writes it to `assets/scenes/world.scn.ron`.

```sh
cargo test -- generate_world_scene --nocapture
```

This:
1. Creates entities with desired components in a test world
2. Serializes to canonical RON via `DynamicScene`
3. Writes the file to disk
4. Prints the output for review

**To change the scene**: edit the test (spawn different entities, change positions, add components), then run it.

## Adding New Scenes

1. Add a new test in the relevant module
2. Spawn entities with the components you want in the scene
3. Use `DynamicSceneBuilder` + `serialize` to generate RON
4. Write to `assets/scenes/`

Template:
```rust
#[test]
fn generate_my_scene() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .register_type::<Transform>()
        .register_type::<MyComponent>();

    let entity = app.world_mut().spawn((
        Transform::from_xyz(1.0, 0.0, 2.0),
        MyComponent { field: "value".into() },
    )).id();

    let scene = DynamicSceneBuilder::from_world(app.world())
        .extract_entity(entity)
        .build();

    let registry = app.world().resource::<AppTypeRegistry>();
    let ron = scene.serialize(&registry.read()).unwrap();

    std::fs::create_dir_all("assets/scenes").unwrap();
    std::fs::write("assets/scenes/my_scene.scn.ron", &ron).unwrap();
}
```

## Loading a Scene

```rust
fn load_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(DynamicSceneRoot(asset_server.load("scenes/my_scene.scn.ron")));
}
```

Scenes load **asynchronously** (1+ frames). Entities with immediate physics needs (ground plane) should spawn in code, not scenes.

## What Can / Cannot Go in Scenes

### Can serialize
- Custom components with `#[derive(Component, Reflect)]` + `#[reflect(Component)]` + `register_type::<T>()`
- Components whose fields are all `Reflect` types (String, u32, f32, bool, Vec3, Quat)
- Built-in Bevy components (Transform, Visibility, etc.)

### Cannot serialize
- `Mesh3d` / `MeshMaterial3d` — asset handles
- `RigidBody` / `Collider` / `RayCastBackfaces` — from bevy_rapier3d, not Reflect
- `Handle<T>` — runtime constructs

### Pattern: Scene + Runtime augmentation
Put gameplay data (positions, stats, names) in the scene. Add visuals/physics at runtime:

```rust
fn setup_visuals(
    mut commands: Commands,
    query: Query<Entity, (With<MyMarker>, Without<Mesh3d>)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for entity in &query {
        commands.entity(entity).insert((
            Mesh3d(meshes.add(/* ... */)),
            MeshMaterial3d(materials.add(/* ... */)),
            RigidBody::Fixed,
            Collider::capsule_y(0.5, 0.5),
        ));
    }
}
```

## File Location

Scenes go in `assets/scenes/`. Load with `asset_server.load("scenes/name.scn.ron")`.

## Related

- `/add-feature` — when a feature's entities belong in a saved scene, this is Step 7.
- `/create-component` — components need `Reflect` + `register_type` to serialize.
