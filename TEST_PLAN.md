# Test Plan — Adventure

## Strategy

Three-tier testing model adapted for Bevy 0.18:

| Tier | Scope | Tool | Speed |
|------|-------|------|-------|
| Unit | Single function/component/constant | `#[test]` in module | Instant |
| System | Single ECS system with mocked input | `World::run_system_once` or `App` + `MinimalPlugins` | Fast |
| Integration | Multi-plugin workflows | `App` + `MinimalPlugins` + multiple plugins | Fast |

All tests use `MinimalPlugins` (no GPU/windowing). Rendering and physics plugin tests are deferred to manual verification.

## Test Coverage Map

### Unit Tests (36 tests in `src/`)

| Module | Tests | What's Covered |
|--------|-------|----------------|
| `camera` | 8 | `forward_from_yaw` (4 directions, unit vector, zero-Y), `CameraMode` defaults, `ThirdPersonCamera` defaults, speed clamping |
| `combat` | 5 | Damage reduces health, health clamps at 0, multiple damage accumulates, nonexistent entity no-panic, `CombatState` defaults |
| `inventory` | 4 | Loot adds items, bag-full rejection, multiple loot same frame, `PlayerInventory` defaults |
| `player` | 6 | Player struct defaults, gravity formula, jump/gravity constants, `CharacterController` defaults, free camera early-exit |
| `world` | 4 | Scene generation, Reflect registration for Npc/QuestGiver, `GameWorld` zone, serialization roundtrip |
| `networking` | 3 | `NetworkState` defaults, chat processing no-panic, channel variants |
| `ui` | 5 | Health bar width at full/half/zero, mana bar at full/partial |

### Integration Tests (6 tests in `tests/`)

| Test | What's Covered |
|------|----------------|
| `combat_plugin_registers_resources` | Plugin init creates `CombatState` |
| `inventory_plugin_registers_resources` | Plugin init creates `PlayerInventory` with 20 slots |
| `networking_plugin_registers_resources` | Plugin init creates `NetworkState` |
| `world_plugin_registers_resources` | Plugin init creates `GameWorld` with zone name |
| `damage_then_loot_workflow` | Damage → verify health → loot item → verify inventory (cross-plugin) |
| `multiple_damage_sources_kill_player` | Two damage events same frame → health = 0 |

## Architecture Decisions

### Why `lib.rs` + `main.rs`

`src/lib.rs` exposes all modules as public. `src/main.rs` is the thin binary entry point. This lets integration tests (`tests/`) import `adventure::*` directly.

### Why `MinimalPlugins` over `DefaultPlugins`

- No GPU context needed for logic tests
- No window creation (headless CI compatible)
- ~100x faster than `DefaultPlugins`
- Must manually insert `ButtonInput<KeyCode>`, `ButtonInput<MouseButton>` for input-dependent systems

### Test Utilities (`src/test_utils.rs`)

Shared helpers: `test_app()`, `test_app_with_input()`, `press_key()`, `spawn_test_player()`. Available via `use crate::test_utils::*` in any `#[cfg(test)]` module.

## Best Practices for Bevy Testing

### 1. Test pure logic separately from rendering

Systems doing math (damage, inventory, movement direction) should be testable without `Mesh`, `Material`, or `Camera`. This project follows that pattern — combat/inventory logic is decoupled from visuals.

### 2. Use `Messages<T>` (Bevy 0.18) for event testing

```rust
// Write a message
app.world_mut().resource_mut::<Messages<MyEvent>>().write(MyEvent { ... });

// Run systems
app.update();

// Assert on component state
assert_eq!(app.world().get::<Health>(entity).unwrap().0, 70);
```

### 3. Mock keyboard input via `ButtonInput`

```rust
app.insert_resource(ButtonInput::<KeyCode>::default());
app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(KeyCode::Space);
app.update();
// just_pressed(KeyCode::Space) returns true in systems
```

### 4. `app.update()` is required

- Startup systems run on first `app.update()`
- `Commands` (spawn/despawn/insert) flush on `app.update()`
- Messages written before `app.update()` are visible to systems during that update

### 5. Plugin isolation

Each plugin (`CombatPlugin`, `InventoryPlugin`, etc.) is independently testable. Integration tests compose multiple plugins.

### 6. What not to test with `MinimalPlugins`

Systems requiring `Assets<Mesh>`, `Assets<StandardMaterial>`, `MeshRayCast`, or Rapier physics output need `DefaultPlugins` or manual resource setup. These are best tested manually or with a future visual regression setup.

## Future Test Improvements

- [ ] **Physics tests**: Add `RapierPhysicsPlugin` to test gravity, collision, character controller
- [ ] **Movement tests**: Full `move_player` with mocked `KinematicCharacterControllerOutput`
- [ ] **Camera orbit tests**: Mock `MeshRayCast` and `MouseMotion` messages
- [ ] **UI system tests**: Spawn `HealthBar`/`ManaBar` entities, run `update_ui`, verify `Node` width
- [ ] **Visual regression**: Screenshot comparison via Pixel Eagle or custom harness
- [ ] **Scene loading tests**: Load `world.scn.ron` with `AssetServer` in `DefaultPlugins` test
- [ ] **Stress tests**: Max inventory (20 items), many NPCs, many damage events per frame
- [ ] **WASM target**: Add `wasm32-unknown-unknown` build check to CI

## Running Tests

```bash
# All tests
cargo test

# Specific module
cargo test -- camera
cargo test -- combat

# Scene generator (writes to assets/)
cargo test -- generate_world_scene --nocapture

# With output
cargo test -- --show-output
```
