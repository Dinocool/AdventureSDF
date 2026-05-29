# Adventure — Bevy 0.18 ECS Project

## Quick Reference

- Package: `adventure` (Rust, edition 2024)
- Engine: Bevy 0.18
- Physics: bevy_rapier3d 0.34
- BRP: bevy_brp_extras 0.18
- Modules: `camera`, `combat`, `inventory`, `networking`, `player`, `scene_manager`, `sdf_render`, `ui`, `world` (+ `debug_toolkit`, feature-gated)

## Build & Run

```sh
cargo build
cargo build --features debug_toolkit                # dev tooling build
cargo run
cargo test
cargo test -- generate_world_scene --nocapture      # regenerate scene RON
```

## Architecture

```
src/
  lib.rs              — public module declarations
  main.rs             — App entrypoint (thin: add_plugins, insert_resource, run)
  camera/mod.rs       — CameraPlugin: third-person orbit + free-fly
  player/mod.rs       — PlayerPlugin: movement, stats (Health, Mana, MovementSpeed, PlayerName, PlayerLevel)
  world/mod.rs        — WorldPlugin: terrain, NPCs, quest givers, scene loading
  combat/mod.rs       — CombatPlugin: damage, abilities
  inventory/mod.rs    — InventoryPlugin: items, loot, equip
  networking/mod.rs   — NetworkingPlugin: chat channels
  ui/mod.rs           — UiPlugin: health/mana bars
  scene_manager.rs    — SceneManagerPlugin: ESC menu, scene switching
  sdf_render/mod.rs   — SdfScenePlugin: SDF voxel editor
  sdf_render/render.rs— SdfRenderPlugin: render-graph node (registered SEPARATELY in main.rs)
  debug_toolkit/      — DebugToolkitPlugin: dev panels + shader debug (feature "debug_toolkit" only)
  test_utils.rs       — shared test helpers
```

All plugins are structs implementing `Plugin`, registered in `main.rs`. Render
sub-plugins (`SdfRenderPlugin`) and feature-gated plugins (`DebugToolkitPlugin`) are
added in their own blocks — see `main.rs`.

---

## Invariants

Non-negotiables. These hold even if the skills below are unavailable:

1. **Zero warnings.** A build/clippy warning is a failure. Fix all before reporting done.
2. **Build both feature configs.** `cargo build` AND `cargo build --features debug_toolkit`.
3. **Never hand-write `.scn.ron`.** Generate scenes from code (see `/create-scene`).
4. **Register every `Reflect` type** with `app.register_type::<T>()` in its plugin.

## Workflow — use the skills

Don't reinvent these; the detailed, code-verified guidance lives in the skills:

| Task | Skill |
|---|---|
| **Add any new feature** (the hub) | `/add-feature` |
| Create a component | `/create-component` |
| ECS design (SSOT, ordering, guards, feature-gating) | `design-ecs` |
| Create/update a scene `.scn.ron` | `/create-scene` |
| Debug panels / shader visualization | `debug-shader` |
| Pre-done verification (mirrors CI) | `/verify-build` |
| Rust language rules (179 rules, 14 categories) | `/rust-skills` |

## File Conventions

- One module per directory: `src/{module}/mod.rs` (or flat `src/{module}.rs`)
- Components, resources, messages defined at top of the module file
- Plugin struct + `impl Plugin` in the same file
- Tests in `#[cfg(test)] mod tests` inside each file
- Shared test helpers in `src/test_utils.rs`
- Cross-plugin integration tests in `tests/`
