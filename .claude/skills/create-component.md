---
name: create-component
description: Create a well-designed Bevy 0.18 component in the adventure project — pick the right kind, derives, storage, and registration. Trigger when adding any new Component, or when unsure whether something should even be a component.
---

# Create Bevy Component

Decision tree for adding a component. Every example below is a real pattern from
this codebase — copy them, don't invent. Part of the `/add-feature` workflow.

## Step 1: Should it even be a component?

A component is data attached to an *entity*. If the data lives in a `Vec` inside a
`Resource`, or is global singleton state, it is NOT a component — use a `Resource`.
Don't derive `Component` just to store something.

## Step 2: Pick the kind

### Marker (zero-sized)
Tag entities for query filtering. No data, no memory cost.
```rust
// src/player/mod.rs
#[derive(Component)]
pub struct Player;

// src/sdf_render/mod.rs:28-35 — markers don't need Reflect unless serialized
#[derive(Component)]
pub struct SdfVolume;
```

### Newtype
Single value with type safety. Note the **manual `Default`** when zero isn't right:
```rust
// src/player/mod.rs:38 — MovementSpeed defaults to 5.0, not 0.0
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct MovementSpeed(pub f32);

impl Default for MovementSpeed {
    fn default() -> Self {
        Self(5.0)
    }
}
```

### Data component
Multiple fields, one responsibility. Provide constructors instead of public-field
churn at call sites:
```rust
// src/player/mod.rs:12 — derives Default (zero is a valid empty bar)
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
```

### Required components
When spawning X must always pull in Y. **Bevy 0.18 syntax** — bare type list using
each type's `Default`:
```rust
// src/player/mod.rs:8 — the ONLY require form used in this repo
#[derive(Component)]
#[require(Health, Mana, MovementSpeed, PlayerName, PlayerLevel)]
pub struct Player;
```
Each required type supplies its own value via `Default` (see the `impl Default`s
above). Spawning `Player` auto-inserts all five.

**To override a required default**, Bevy 0.18 also accepts (verify before use — none
of these appear in this repo yet):
| Form | Meaning |
|---|---|
| `Health` | uses `Health::default()` |
| `Health::full(100.0)` | constructor call |
| `MovementSpeed(8.0)` | tuple-struct literal |
| `Health { current: 50.0, max: 50.0 }` | struct literal |
| `Foo = some_expr()` | arbitrary expression |

The **closure form `Foo(\|\| ...)` was removed** — do not use it.

## Step 3: Derives

| Need | Add |
|------|-----|
| Always | `#[derive(Component)]` |
| Scene / BRP / inspector serialization | `+ Reflect` and `#[reflect(Component)]` |
| Stored in a Resource or Message | `+ Clone` |
| Sensible zero-value | `+ Default` (else write `impl Default` by hand) |

Don't auto-derive `Default`/`Clone` on components holding unique runtime state.

## Step 4: Storage

Default `Table` storage is right for almost everything here (nothing in this repo
uses SparseSet yet). Reach for SparseSet ONLY for components added/removed every few
frames on many entities — status effects are the textbook case:
```rust
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct Stunned;
```

## Step 5: Register if Reflect

Any `Reflect` component must be registered in the owning plugin's `build()`, or
scenes/BRP/inspector won't see it:
```rust
// src/player/mod.rs:83 — register every Reflect type
app.register_type::<Health>()
    .register_type::<Mana>()
    .register_type::<MovementSpeed>();
```

## Step 6: Module placement

| Domain | Module |
|--------|--------|
| Player stats/movement | `src/player/mod.rs` |
| Combat | `src/combat/mod.rs` |
| Items/inventory | `src/inventory/mod.rs` |
| World/NPCs | `src/world/mod.rs` |
| UI | `src/ui/mod.rs` |
| Camera | `src/camera/mod.rs` |
| Chat/networking | `src/networking/mod.rs` |
| SDF editor/render | `src/sdf_render/` |

Cross-cutting types go in their primary consumer, or a new module.

## Step 7: Tests

See `/add-feature` for the per-feature test bar. At minimum for a new component:
- `Default` value test if it has a non-trivial default (`MovementSpeed` → 5.0).
- Registration test if `Reflect`: build the plugin, assert the type is registered.
- If a system reads/writes it, exercise that system via `test_utils::test_app()`.

## Anti-patterns

1. **God component** — unrelated fields (speed + health + name) in one struct.
2. **`Component` on Resource-stored data** — if it lives in a `Vec` in a resource it
   isn't a component.
3. **Missing `Reflect`** on a type used in scenes/messages/BRP.
4. **Closure `#[require]`** — removed in 0.18; use the type-list or `= expr` forms.
5. **Logic in components** — components are data; logic goes in systems.
6. **Marker on every entity** — needless archetype fragmentation.

## Related

- `/add-feature` — the full feature workflow this fits into.
- `design-ecs` — deeper ECS design principles (single-source-of-truth, ordering, guards).
- `/create-scene` — if the component goes in a `.scn.ron`.
