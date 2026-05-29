---
name: add-feature
description: End-to-end workflow for adding a new gameplay or render feature to the adventure Bevy project — module, plugin, registration, systems, tests, and verification. Trigger whenever asked to add a feature, system, plugin, message, or render pass, or when starting any non-trivial change that introduces new behavior.
---

# Add a Feature

The canonical sequence for adding a feature. This is the hub — each step defers to a
focused skill rather than repeating it. Follow in order.

## Step 0: Working with unfamiliar APIs

Before writing code against any Bevy/wgpu/library API you're not certain of:
1. **Find a working example** in this repo, the engine source, or official examples.
   Adapt a known-good pattern; don't write from memory.
2. **Fetch docs via Context7** (`resolve-library-id` → `query-docs`, or `/bevyengine/bevy`
   directly). Do this proactively, not after a failed build.
3. **Verify the API exists in Bevy 0.18** — methods/types from blog posts, older
   versions, or `main` may not exist. Grep the source or check docs.

## Step 1: Decide the shape

| Feature kind | Shape |
|---|---|
| New domain (a new system of behavior) | New module `src/<name>/mod.rs` + a `Plugin` struct |
| Extends existing domain | Add to the existing module/plugin |
| Render pass / GPU work | Lives in `src/sdf_render/` or a new render module + a render sub-plugin |
| Dev/debug tooling | Behind `debug_toolkit` feature → see `debug-shader` |

Module → domain mapping is in `/create-component` Step 6.

## Step 2: Data — components, resources, messages

- Components → `/create-component` (kinds, derives, `#[require]`, storage).
- Design questions (shared state, ordering, guards) → `design-ecs`.
- **Messages** for buffered cross-system comms: `#[derive(Message)]`, read with
  `MessageReader<T>`, register with `app.add_message::<T>()`.
- **Observers** for immediate same-frame reactions: `Trigger<T>` + `add_observer`.
- Every `Reflect` type gets `app.register_type::<T>()` in `build()`.

## Step 3: Plugin

Struct + `impl Plugin`. The plugin owns its types, resources, and systems (`design-ecs`
rule 4):
```rust
pub struct MyFeaturePlugin;

impl Plugin for MyFeaturePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<MyComponent>()
            .add_message::<MyEvent>()
            .add_systems(Update, my_system.run_if(in_state(AppScene::SdfEditor)));
    }
}
```

## Step 4: Register in main.rs

Add the plugin in `src/main.rs` (see existing block, `main.rs:22-31`):
```rust
.add_plugins(adventure::my_feature::MyFeaturePlugin)
```
Notes:
- **Render sub-plugins register separately** — `sdf_render` adds both `SdfScenePlugin`
  AND `render::SdfRenderPlugin` (`main.rs:23-24`).
- **Feature-gated plugins** go in the `#[cfg(feature = "debug_toolkit")]` block
  (`main.rs:34-37`), not the main chain.
- Declare the module in `src/lib.rs`.

## Step 5: Systems

- Explicit ordering with `SystemSet`/`.chain()`; never rely on insertion order
  (`design-ecs` rule 2).
- Gate with `run_if(...)` instead of early-return (CLAUDE.md §9).
- Guard resources that may be absent in tests (`design-ecs` rule 3).

## Step 6: Tests (the per-feature bar)

Every feature ships at least:
- **(a)** Default-value test for any new component with a non-trivial default.
- **(b)** At least one **system-level** test via `test_utils::test_app()` that
  exercises the feature's main system (spawn entity → send message / `app.update()` →
  assert resulting component/resource state). See `combat/mod.rs` damage tests for the
  pattern; reuse `spawn_test_player` / `spawn_test_npc` helpers.
- **(c)** Registration test if the type is `Reflect`.

Tests live inline in `#[cfg(test)] mod tests` in the module file. Cross-plugin flows go
in `tests/integration.rs`.

## Step 7: Scenes & shaders (if relevant)

- Entities that belong in a saved scene → `/create-scene` (never hand-write `.scn.ron`).
- New shader / debug visualization → `debug-shader`. WGSL has sharp edges; the
  `tests/shader_validation.rs` rig parses every `.wgsl` — run it.

## Step 8: Verify

Run `/verify-build` before reporting done — it mirrors CI (build both feature configs,
test, clippy `-D warnings`, fmt). For render/ECS features, also verify at **runtime**
via the `brp` MCP server: launch the app, query entities, take a screenshot to confirm
the feature actually appears. (The `chrome-devtools` MCP is for web apps — not relevant
to this native game.)

## Related skills

- `/create-component` · `design-ecs` · `/create-scene` · `debug-shader` · `/verify-build`
