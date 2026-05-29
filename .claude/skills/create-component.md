---
name: create-component
description: Guide for creating well-designed Bevy 0.18 components in the adventure project
---

# Create Bevy Component

Decision tree and checklist for adding a new component to this project.

## Step 1: Choose Component Type

### Marker (ZST)
Use when: tagging entities for query filtering, no data needed.
```rust
#[derive(Component)]
pub struct Player;
```

### Newtype
Use when: single value with type safety.
```rust
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct MovementSpeed(pub f32);
```

### Data Component
Use when: multiple fields, coherent single responsibility.
```rust
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}
```

### Required Component
Use when: component X should always exist when Y is spawned.
```rust
#[derive(Component)]
#[require(Health(|| Health { current: 100.0, max: 100.0 }))]
pub struct Player;
```

## Step 2: Derives

| Need | Add |
|------|-----|
| Always | `#[derive(Component)]` |
| Scene/BRP serialization | `+ Reflect` + `#[reflect(Component)]` |
| Stored in Resources/Messages | `+ Clone` |
| Sensible zero-value | `+ Default` |

Never auto-derive `Clone` or `Default` on components that represent unique runtime state.

## Step 3: Storage

| Default (Table) | SparseSet |
|-----------------|-----------|
| Frequently queried | Rarely queried |
| Stable composition | Frequently added/removed |
| Cache-friendly iteration | Fast insert/remove |

```rust
#[component(storage = "SparseSet")]
```

Use for status effects: `Stunned`, `Invulnerable`, `Poisoned`.

## Step 4: Registration

In the plugin's `build()` method:
```rust
app.register_type::<MyComponent>();
```

Required for any component with `Reflect`. Enables scene serialization and BRP queries.

## Step 5: Module Placement

| Component domain | Module |
|-----------------|--------|
| Player stats/movement | `src/player/mod.rs` |
| Combat mechanics | `src/combat/mod.rs` |
| Items/inventory | `src/inventory/mod.rs` |
| World/NPCs | `src/world/mod.rs` |
| UI markers | `src/ui/mod.rs` |
| Camera | `src/camera/mod.rs` |
| Chat/networking | `src/networking/mod.rs` |

Cross-cutting components: create a new module or place in the primary consumer.

## Step 6: Spawn Pattern

```rust
// In a Startup or OnEnter system:
commands.spawn((
    MyMarker,
    MyDataComponent { field: value },
    Transform::from_xyz(0.0, 0.0, 0.0),
));

// With required components (auto-fills defaults):
commands.spawn(Player);
```

## Step 7: Query Pattern

```rust
fn my_system(
    markers: Query<Entity, With<MyMarker>>,
    data: Query<&MyDataComponent, With<MyMarker>>,
    mut_data: Query<&mut MyDataComponent, With<MyMarker>>,
    filtered: Query<&MyDataComponent, (With<MyMarker>, Without<OtherMarker>)>,
) { ... }
```

## Step 8: Tests

- Unit test for `Default` values (if applicable)
- If `Reflect`: test registration via `app.register_type::<T>()` + `world.contains::<T>()`
- If used in systems: test with `test_utils::test_app()` helper

## Anti-Patterns to Avoid

1. **God components** — one struct holding unrelated data (speed + health + name)
2. **Component on data only stored in Resources** — if it lives in a `Vec` inside a `Resource`, it doesn't need `#[derive(Component)]`
3. **Missing `Reflect`** on types used in scenes or messages
4. **Early returns** instead of run conditions (`.run_if(...)`)
5. **Manual cleanup** instead of `StateScoped<S>`
6. **Logic in components** — keep components as pure data, put logic in systems
7. **Excessive archetype fragmentation** — don't add unique marker to every entity
