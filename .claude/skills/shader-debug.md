---
name: shader-debug
description: Add debug panels and shader debug modes via the debug_toolkit framework. Trigger when adding debug/inspection tooling to any pipeline, registering a debug UI panel, or adding shader visualization modes.
---

# Debug Toolkit

Feature-gated behind the `debug_toolkit` cargo feature. A generic, extensible debug
framework: the toolkit provides only a panel-registration API + dock layout and a
shader-debug-mode registry. **Each plugin owns its own debug data and registers its
own panels** — the toolkit never reaches into a pipeline's internals.

Run with: `cargo run --features debug_toolkit`.

## Architecture

```
src/debug_toolkit/            — generic framework only
  panels.rs        — DockSide, DebugPanel, DebugPanelRegistry, register_panel()
  registry.rs      — ShaderDebugMode/State, debug_modes_ui, active_defines_for_prefix
  config.rs        — DebugToolkitConfig (master enable)
  profiling.rs     — FPS/frame-time panel
  hot_reload.rs    — shader hot-reload counter panel
  uniform_inspector.rs — generic closure-based uniform inspection
  mod.rs           — DebugToolkitPlugin + data-driven dock_layout

src/sdf_render/debug.rs       — SDF consumer (example). Behind #[cfg(feature)].
```

`DebugToolkitPlugin` registers framework panels (world inspector, uniforms, perf,
hot-reload) and runs `dock_layout`, which iterates the `DebugPanelRegistry` and
renders each panel into its `DockSide` (Left/Right side panels = collapsing
sections; Bottom = horizontal columns), sorted by `order`.

## Adding a Debug Panel (any plugin)

In your plugin's `build`, call `register_panel`. The render closure gets exclusive
`&mut World` — read any resource/query you need.

```rust
use crate::debug_toolkit::panels::{register_panel, DockSide};

register_panel(app, "mything/stats", "My Stats", DockSide::Left, 0, |world, ui| {
    let r = world.resource::<MyResource>();
    ui.label(format!("count: {}", r.count));
});
```

`register_panel` inits the registry if absent, so it works regardless of plugin
build order relative to `DebugToolkitPlugin`. Keep panel ids namespaced:
`<prefix>/<name>`. Gate your whole debug module behind `#[cfg(feature = "debug_toolkit")]`.

## Adding Shader Debug Modes

### 1. Register modes (in your debug module)

```rust
use crate::debug_toolkit::registry::{DebugModeKind, ShaderDebugMode, ShaderDebugRegistry};

app.init_resource::<ShaderDebugRegistry>(); // build-order safety
let mut reg = app.world_mut().resource_mut::<ShaderDebugRegistry>();
reg.register(ShaderDebugMode {
    id: "myshader/normals".into(),
    label: "Normals".into(),
    shader_define: "MY_DEBUG_NORMALS".into(),
    kind: DebugModeKind::Exclusive { group: "myshader_overlay".into() }, // or Toggle
    description: "Surface normals as RGB".into(),
});
```

Expose the selector with a panel that calls `debug_modes_ui(world, ui)`.

### 2. Bridge state → shader defs (in your render plugin)

```rust
#[cfg(feature = "debug_toolkit")]
fn sync_my_shader_defs(
    registry: Res<crate::debug_toolkit::registry::ShaderDebugRegistry>,
    state: Res<crate::debug_toolkit::registry::ShaderDebugState>,
    mut defs: ResMut<MyShaderDefs>,
) {
    let active = state.active_defines_for_prefix(&registry, "myshader/");
    if defs.defs != active { defs.defs = active; }
}
```

Extract `MyShaderDefs` to the render world and rebuild the pipeline when it changes
(see `sdf_render/render.rs`: `extract_shader_defs` → `rebuild_pipeline_on_def_change`).

### 3. Gate output in WGSL

```wgsl
#ifdef MY_DEBUG_NORMALS
    return FragmentOutput(vec4(normal * 0.5 + 0.5, 1.0), depth);
#endif
```

## Live params via uniform (no pipeline recompile)

For numeric tuning (not on/off modes), pass values through an existing uniform
rather than shader_defs. Example: `SdfRaymarchParams` (owned by `sdf_render`, not the
toolkit) is written into `SdfCameraData.debug_params` each frame by
`prepare_sdf_camera_data`; the shader reads `camera.debug_params.xyz`. The toolkit
panel just edits the resource. This avoids a pipeline rebuild per slider tick.

## Existing SDF Tools (`src/sdf_render/debug.rs`)

| Tool | Kind |
|------|------|
| Steps / Normals / Obj ID / Bricks | Exclusive shader overlay modes (`sdf_overlay`) |
| Atlas stats + texture viewer | Left panel (distance + object-id images via `EguiUserTextures`) |
| Gizmo / camera state | Left panel |
| Wireframe bounds toggle | Left panel → `WireframeBoundsVisible` → spawn/despawn system |
| Raymarch params | Bottom panel → `SdfRaymarchParams` → camera uniform |
| CPU ray-step inspector | Bottom panel → `picking::debug_capture_march` (no GPU readback) |

## Choosing Toggle vs Exclusive

- **Toggle**: independent on/off (additive overlays: wireframe, grid).
- **Exclusive { group }**: one active per group (replacement views: normals, heatmaps).
