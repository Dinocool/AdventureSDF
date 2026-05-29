---
name: bevy-ecs-design
description: Design principles for Bevy 0.18 ECS code in this project — single source of truth, data/logic separation, render-graph ordering, and deliberate design over rushed edits. Trigger before adding or refactoring any plugin, system, component, gizmo/overlay, or render pass.
---

# Bevy ECS Design Principles

Apply these when designing or refactoring ECS code in this project. They
complement the per-rule guidance in `rust-skills/` and the project `CLAUDE.md`.
Design deliberately: confirm the abstraction before editing, then validate with
`cargo build` (both feature configs) and `cargo test`. Avoid rushed edits.

## 1. Single source of truth

Any value, geometry, or constant used in two places must be defined ONCE and
shared. Two copies always drift.

- If drawing and picking (or CPU and GPU, or two systems) need the same shape,
  size, layout, or constant, factor it into one struct/function/`const` that all
  consumers call. Do not re-encode it per call site.
- Worked example: editor gizmo handles live in `src/sdf_render/gizmo.rs` as a
  `Handle` value exposing both `draw(&mut Gizmos)` and `sdf(p) -> f32` from the
  same `const` dimensions. Drawing and CPU picking both build handles via the
  same constructor. (They had drifted before — changing the drawn radii silently
  broke picking. See memory `gizmo-single-source-geometry`.)
- Same rule for CPU/GPU mirrored layouts: a uniform struct's field order/types
  must match its WGSL counterpart; atlas tile layout used by upload AND debug
  view must come from one helper.

## 2. Components = data, systems = logic

One responsibility per component. No methods that touch the `World`, spawn, or
mutate other components. Logic lives in systems. (Reinforces CLAUDE.md §6.)

## 3. Keep components small, compose with markers

Prefer small single-purpose components + marker types for query filtering over
god components. Entity "types" emerge from component sets, not inheritance.

## 4. Run conditions over early returns

Gate systems with `run_if(...)` rather than an `if … { return; }` at the top.
When a system needs a resource that may be absent in some app configs (e.g.
`Assets<GizmoAsset>` is missing under `MinimalPlugins` in tests), guard it:
`run_if(resource_exists::<T>)`, or only register it when the resource is present
at build time. This is how SDF overlay systems stay test-safe.

## 5. Explicit ordering; never rely on implicit order

Use `SystemSet`s / `.chain()` for system order, and explicit render-graph edges
for passes. Document WHY an order matters.
- Worked example: the SDF fullscreen node runs between `Node3d::MainOpaquePass`
  and `Node3d::MainTransparentPass` so gizmos (drawn in the `Transparent3d`
  phase) composite on top. Moving it to `EndMainPass` makes overlays invisible.

## 6. Plugins own their data

A plugin registers its own resources, types, and systems. Cross-cutting tools
(debug toolkit, future BVH viz) are *consumers* that register into a generic
framework — they do not reach into another plugin's internals. (See the
`debug_toolkit` panel-registry pattern and memory `sdf-editor-overlays`.)

## 7. Reflect + register serializable types

Types used in scenes, messages, BRP, or the inspector derive `Reflect` and are
registered with `app.register_type::<T>()`. (Reinforces CLAUDE.md §13.)

## 8. Feature-gate optional subsystems cleanly

Debug/dev-only code sits behind a cargo feature (`debug_toolkit`) and must not
change runtime behavior or break the build when the feature is off. Keep core
resource *types* in the core module if a gated system references them, so the
core build still compiles.

## Checklist before implementing

1. Is any value/geometry/layout duplicated? → factor to one source (rule 1).
2. Does ordering matter? → make it explicit and comment why (rule 5).
3. Could a required resource be absent in tests/other configs? → guard (rule 4).
4. Does this belong to this plugin, or is it a consumer of a framework? (rule 6).
5. Build both feature configs + run tests before reporting done.
