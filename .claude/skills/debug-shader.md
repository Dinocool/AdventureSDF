---
name: debug-shader
description: Add debug panels and shader debug modes via the debug_toolkit framework, AND debug shaders that compile but render WRONG (GPU compute harness + round-trip overlays). Trigger when adding debug/inspection tooling, registering a debug UI panel, adding shader visualization modes, OR diagnosing a shader producing wrong output at runtime (misplaced/garbage/flickering values, suspected GPU-specific math divergence).
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

## Writing the WGSL

WGSL has sharp edges that the compiler error messages don't make obvious (no tuples,
no `{field: val}` struct init, no implicit u32/f32 mixing, etc.). Before editing any
`.wgsl`, check the `wgsl-gotchas` memory note, and after, run `cargo test` — the
`tests/shader_validation.rs` rig parses every shader and catches these without a GPU.

## Debugging a shader that COMPILES but renders WRONG

`shader_validation.rs` only proves a shader *parses + type-checks*. It says nothing
about wrong math, misplaced data, GPU-specific numeric divergence, or cross-frame
state bugs. For those, use these two tools — and **do not iterate on the user's eyes**.
Overlays *localize* (which stage / which axis); the GPU harness *proves* the exact
failing math against a CPU reference and *locks it* with a regression test.

### 1. GPU compute harness — the decisive tool (`tests/sdf_gpu_rig.rs`, `sdf_bake_gpu.rs`)

Runs the ACTUAL (or extracted) WGSL on a headless wgpu device, feeds input storage
buffers, reads back output buffers, and asserts against a Rust reference. This catches
**GPU-specific divergence the CPU and the naga validator cannot reproduce**, on the
real target GPU. Pattern (see `gpu_scalar_ops_vs_rust`, `gpu_fib_dir_stays_on_unit_sphere`):

```rust
// self-contained: inline the function under test, or compose_entry(...) to pull real sdf::* modules.
const WGSL: &str = r#"
struct In { /* one case */ }; struct Out { /* results */ };
@group(0) @binding(0) var<storage, read> ins: array<In>;
@group(0) @binding(1) var<storage, read_write> outs: array<Out>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) g: vec3<u32>) { outs[g.x] = ...; }
"#;
// device_queue() → buffer_init(ins) → empty out_buf (STORAGE|COPY_SRC) + readback (MAP_READ|COPY_DST)
// → auto layout (layout: None) → dispatch(n,1,1) → copy_buffer_to_buffer → map_async + poll(wait) → assert.
```
Make the test **self-incriminating**: run BOTH the old (buggy) and new (fixed) form and
assert the old one fails the invariant (a guard that the test actually exercises the bug).

### 2. Round-trip overlay — to localize before harnessing

Store a KNOWN ground-truth value in the pipeline's output and display the DIFFERENCE
from what it should be at the consumer. **Black = correct.** E.g. to verify a probe
volume's world↔texel addressing, the compute stores each probe's own world centre and
the lit pass shows `|sampled_centre − surface_world_pos|`. Two rules that make or break it:
- **Exercise the SAME code path as the real data.** A single-frame round-trip will pass
  while a cross-frame bug (moving average / ping-pong / reset) hides — route the debug
  value through the same blend so the overlay tests it too.
- **Don't let invalid/zero sentinels poison interpolation.** If invalid cells store 0 and
  the read trilinearly filters, near-boundary samples look broken even when addressing is
  fine — store the real value (or validity-weight) in debug mode.

Gate both behind `#ifdef` defs (compute pipeline must rebuild on def change too — see
`rebuild_pipeline_on_def_change`) so they're permanent, toggleable verification tools.

### Known GPU numeric hazards — test these explicitly

- **Native signed `%` and `/` are WRONG for negative operands** on real GPUs (return
  unsigned-ish results). Never use them on values that can go negative (world/texel/grid
  coords routinely do). Use `euclid_mod` / `floor_div` from `sdf/bindings.wgsl`. See the
  `wgsl-integer-ops-gpu` memory and `gpu_scalar_ops_vs_rust`. This one silently scrambles
  addressing — the symptom is data landing in the wrong place, not a crash.
- **Keep generated directions ON the unit sphere.** Spherical-Fibonacci `z` must stay in
  `[-1,1]`; jitter the AZIMUTH per frame, never add an unbounded value into the index (it
  shoves `z` out of range → `r=sqrt(1-z²)=0` → all rays collapse to ±Z). Assert `length≈1`
  on-GPU (`gpu_fib_dir_stays_on_unit_sphere`).
- **Freshly-created storage textures have UNDEFINED contents.** Clear (queue.write_texture
  zeros) before any read-modify-write (temporal accumulators, ping-pong history) or frame 1
  blends garbage.
- **Assert value RANGES on-GPU** (unit length, in-bounds indices, validity ∈ {0,1}). A
  number that's "plausible" on the CPU can be garbage after a GPU op.

## Related

- `/add-feature` — Step 7 covers shader/debug work.
- `design-ecs` — rule 4 (plugins own data; tools are consumers) is the model this
  toolkit follows.
