# Adventure — Bevy 0.18 ECS Project

## Quick Reference

- Package: `adventure` (Rust, edition 2024)
- Engine: Bevy 0.18
- Physics: bevy_rapier3d 0.34
- BRP: bevy_brp_extras 0.18
- Modules: `camera`, `combat`, `inventory`, `networking`, `player`, `scene_manager`, `sdf_render`, `ui`, `world`

## Build & Run

```sh
cargo build
cargo run
cargo test
cargo test -- generate_world_scene --nocapture   # regenerate scene RON
```

## Architecture

```
src/
  lib.rs              — public module declarations
  main.rs             — App entrypoint (thin)
  camera/mod.rs       — CameraPlugin: third-person orbit + free-fly
  player/mod.rs       — PlayerPlugin: movement, stats (Health, Mana, MovementSpeed, PlayerName, PlayerLevel)
  world/mod.rs        — WorldPlugin: terrain, NPCs, quest givers, scene loading
  combat/mod.rs       — CombatPlugin: damage, abilities
  inventory/mod.rs    — InventoryPlugin: items, loot, equip
  networking/mod.rs   — NetworkingPlugin: chat channels
  ui/mod.rs           — UiPlugin: health/mana bars
  scene_manager.rs    — SceneManagerPlugin: ESC menu, scene switching
  sdf_render/mod.rs   — SdfScenePlugin: SDF voxel editor
  test_utils.rs       — shared test helpers
```

All plugins are structs implementing `Plugin`. Registered in `main.rs`.

---

## Bevy ECS Best Practices

### 1. Keep Components Small and Single-Purpose

One responsibility per component. No god components.

```rust
// BAD
#[derive(Component)]
struct Player { speed: f32, name: String, level: u32, health: f32, max_health: f32, mana: f32, max_mana: f32 }

// GOOD
#[derive(Component)]
struct Player;  // marker

#[derive(Component, Reflect)]
#[reflect(Component)]
struct Health { pub current: f32, pub max: f32 }

#[derive(Component, Reflect)]
#[reflect(Component)]
struct MovementSpeed(pub f32);
```

Other entity types (enemies, NPCs) can reuse `Health`, `MovementSpeed` independently.

### 2. Marker Components (Zero-Sized Types)

Tag entities for query filtering. No memory cost.

```rust
#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Enemy;

// Query with marker
fn move_players(query: Query<&mut Transform, With<Player>>) { ... }
```

### 3. Newtype Components for Type Safety

Wrap primitives so you can't mix up `f32` values.

```rust
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct MovementSpeed(pub f32);
```

### 4. Required Components

Use `#[require(...)]` to auto-insert prerequisites when spawning.

```rust
#[derive(Component)]
#[require(
    Health(|| Health { current: 100.0, max: 100.0 }),
    Mana(|| Mana { current: 50.0, max: 50.0 }),
    MovementSpeed(|| MovementSpeed(5.0)),
)]
pub struct Player;
```

Spawning `Player` automatically adds `Health`, `Mana`, `MovementSpeed` with defaults.

### 5. Composition Over Inheritance

No inheritance in ECS. Entity "types" emerge from component sets.

```rust
// Flying enemy = Enemy + Flying + Health + Transform
// Ground enemy = Enemy + Health + Transform
// Query specifically:
fn update_flying(query: Query<&mut Health, (With<Enemy>, With<Flying>)>) { ... }
```

### 6. Components = Data Only, Systems = Logic

Components hold data. Systems hold logic. No methods that access the World, spawn entities, or mutate other components.

### 7. SparseSet Storage

Use `#[component(storage = "SparseSet")]` for rarely-queried, frequently-toggled components (status effects: `Stunned`, `Invulnerable`, `Poisoned`).

```rust
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct Stunned;
```

Default `Table` storage is best for frequently-queried components.

### 8. System Sets for Ordering

Define ordered execution phases. Never rely on implicit system ordering.

```rust
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GameSet {
    Input,
    Logic,
    Sync,
}

app.configure_sets(Update, (GameSet::Input, GameSet::Logic, GameSet::Sync).chain());
app.add_systems(Update, read_input.in_set(GameSet::Input));
app.add_systems(Update, process_logic.in_set(GameSet::Logic));
```

Cross-plugin ordering: `InventorySet.after(CombatSet)`.

### 9. Run Conditions Over Early Returns

```rust
// GOOD — scheduler skips the system entirely
app.add_systems(Update, play_game.run_if(in_state(GameState::InGame)));

// BAD — system runs every frame just to return early
fn play_game(state: Res<State<GameState>>) {
    if *state != GameState::InGame { return; }
}
```

### 10. Plugin Pattern

Struct plugins (`impl Plugin`) for library crates. Function plugins acceptable for internal code. Keep `main.rs` thin — only `add_plugins`, `insert_resource`, `run`.

### 11. StateScoped for Cleanup

Use `StateScoped(S)` on spawned entities to auto-despawn on state exit, instead of manual cleanup systems.

```rust
commands.spawn((
    Name::new("Player"),
    StateScoped(AppState::InGame),
    Player,
));
```

### 12. Messages vs Observers (Bevy 0.18)

- **Messages** (`#[derive(Message)]`, `MessageReader<T>`): buffered, 2-frame persistence, ordered, one-to-many. Use for cross-system communication.
- **Observers** (`Trigger<T>`, `add_observer`): immediate, same-frame, entity-scoped, propagation. Use for reactive side-effects.

### 13. Reflect on All Serializable Types

Enums and structs used in scenes, messages, or BRP queries must derive `Reflect` and be registered:

```rust
#[derive(Reflect, Clone)]
pub enum DamageType { Physical, Magical, Fire, Frost }

// In plugin:
app.register_type::<DamageType>();
```

### 14. Custom QueryData for Complex Queries

When joining 4+ components, use named query structs to avoid tuple soup:

```rust
#[derive(QueryData)]
struct PlayerQuery {
    entity: Entity,
    health: &'static Health,
    mana: &'static Mana,
    speed: &'static MovementSpeed,
}
```

### 15. Entity IDs Are Not Persistent

`Entity` is a pointer-like ID that can be reused after despawn. For save/load or networking, create your own strong ID types (`QuestId(u32)`).

---

## File Conventions

- One module per directory: `src/{module}/mod.rs` (or flat `src/{module}.rs`)
- Components, resources, messages defined at top of module file
- Plugin struct + `impl Plugin` in same file
- Tests in `#[cfg(test)] mod tests` inside each file
- Shared test helpers in `src/test_utils.rs`
- Integration tests in `tests/`

## Working With Unfamiliar APIs

Do not assume knowledge of any API, framework feature, or render pipeline pattern. When implementing something new or debugging an existing implementation:

1. **Find a concrete working example** in the engine source, official examples, or trusted community code before writing code. Adapt from a known-working pattern rather than writing from memory.
2. **Consult documentation** — use Context7 (`resolve-library-id` then `query-docs`) for Bevy, wgpu, or any library docs. Fetch proactively when library usage is involved.
3. **Verify APIs exist** in the version being used (Bevy 0.18). Don't assume a method/type from a blog post, older version, or `main` branch still works. Grep the source or check docs.

---

## Component Checklist

When creating a new component, use `/create-component` or follow:
1. Choose type: marker / newtype / data / required-component
2. Add derives: `Component` always, `Reflect` + `#[reflect(Component)]` if serializable
3. Choose storage: Table (default) or SparseSet
4. Register in plugin: `app.register_type::<T>()`
5. Place in correct module
6. Add tests (default values, system integration)
