---
name: design-ecs
description: Project-specific Bevy 0.18 ECS design rules for the adventure project — single-source-of-truth, render-graph ordering, test-safety guards, and feature-gating. Trigger before designing or refactoring any plugin, system, render pass, gizmo/overlay, or anything shared between CPU and GPU. Complements CLAUDE.md (general ECS rules) and `rust-skills`.
---

# ECS Design (project-specific)

CLAUDE.md covers the general ECS rules (small components, markers, data-vs-logic,
run-conditions, StateScoped, Messages-vs-Observers, Reflect). This skill covers the
*non-obvious, project-specific* lessons that aren't in CLAUDE.md — the ones this
codebase learned the hard way. Validate with `cargo build` (both feature configs) and
`cargo test` before reporting done; see `/verify-build`.

## 1. Single source of truth (the gizmo lesson)

Any value, geometry, constant, or layout used in two places must be defined ONCE and
shared. Two copies always drift.

Worked example — editor gizmo handles in `src/sdf_render/gizmo.rs`. One `Handle`
struct exposes both:
```rust
// draw and pick read the SAME const dimensions (TRANSLATE_LEN, ROTATE_RADIUS, …)
pub fn draw(&self, gizmos: &mut Gizmos<SdfOverlayGizmos>, color: Color) { ... }
pub fn sdf(&self, p: Vec3, pad: f32) -> f32 { ... }
```
Drawing and CPU picking both build handles via the same constructor. They had drifted
before — changing a drawn radius silently broke clicking. (Memory:
`gizmo-single-source-geometry`.)

Same rule for CPU/GPU mirror layouts: a uniform struct's field order/types must match
its WGSL counterpart; atlas tile layout used by upload AND the debug viewer comes from
one helper. If you change one side, you must change the other — so make it one source.

## 2. Render-graph ordering is explicit and load-bearing

Render passes are ordered with explicit graph edges, never implicit order. The SDF
fullscreen node sits between opaque and transparent so gizmos (drawn in the
`Transparent3d` phase) composite on top:
```rust
// src/sdf_render/render.rs:326
.add_render_graph_edges(
    Core3d,
    (Node3d::MainOpaquePass, SdfLabel, Node3d::MainTransparentPass),
);
```
Move `SdfLabel` to `EndMainPass` and overlays vanish. Document WHY any ordering edge
exists. Same discipline applies to CPU systems: use `SystemSet` + `.chain()`, never
rely on insertion order.

## 3. Two test-safety guard styles — know which to use

Systems that need a resource absent under `MinimalPlugins` (test harness) must be
guarded. This repo uses **two different mechanisms** depending on whether the resource
can appear/disappear at runtime:

**Run-condition** — resource may come and go while the app runs:
```rust
// src/sdf_render/debug.rs:197 — EguiUserTextures only exists with the debug toolkit
.add_systems(Update,
    update_atlas_textures
        .run_if(in_state(AppScene::SdfEditor))
        .run_if(resource_exists::<EguiUserTextures>),
)
```

**Build-time conditional registration** — resource presence is fixed at app-build
time (e.g. whether `GizmoPlugin` was added); don't pay a per-frame `run_if` check:
```rust
// src/sdf_render/debug.rs:203 — Assets<GizmoAsset> is missing under MinimalPlugins
if app.world().get_resource::<Assets<GizmoAsset>>().is_some() {
    app.add_systems(Update,
        (draw_bounds, draw_bvh, live_ray_capture).run_if(in_state(AppScene::SdfEditor)),
    );
}
```
Rule of thumb: runtime-variable presence → `run_if(resource_exists::<T>)`; fixed-at-
build presence → `if app.world().get_resource::<T>().is_some()` in `build()`.

## 4. Plugins own their data; tools are consumers

A plugin registers its own resources, types, and systems. Cross-cutting tools (the
`debug_toolkit`, future visualizers) register INTO a generic framework via its public
API — they never reach into another plugin's internals. The debug toolkit's
`register_panel` pattern is the model (see `debug-shader`). A panel
reads resources through `&mut World`; it doesn't own them.

## 5. Feature-gate optional subsystems cleanly

Dev-only code sits behind a cargo feature and must not change runtime behavior or break
either build config when toggled.
```rust
// src/sdf_render/mod.rs:3 — the debug module only compiles with the feature
#[cfg(feature = "debug_toolkit")]
pub mod debug;
```
Keep any core resource *type* that a gated system references in the core module, so the
non-feature build still compiles. Always build BOTH `cargo build` and
`cargo build --features debug_toolkit`.

## Checklist before implementing

1. Is any value/geometry/layout duplicated across draw/pick, CPU/GPU, or two systems? →
   factor to one source (rule 1).
2. Does pass or system ordering matter? → explicit edge/`SystemSet`, comment why (rule 2).
3. Could a required resource be absent in tests/other configs? → pick the right guard
   style (rule 3).
4. Does this belong to this plugin, or is it a consumer of a framework? (rule 4).
5. Behind a feature? Build both configs + run tests before done (rule 5, `/verify-build`).

## Related

- `/add-feature` — the workflow this design guidance feeds into.
- `/create-component` — concrete component-creation patterns.
- `debug-shader` — the debug-toolkit consumer pattern referenced in rule 4.
