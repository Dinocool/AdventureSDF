# GPU preview raymarch — plan

Move the per-node 3D preview raymarch from the CPU (`march_surface`) onto the GPU.

## The fork: how to evaluate a *dynamic* graph on the GPU

**Option A — full WGSL graph interpreter.** Upload the compiled flat `Graph` as a storage buffer and
evaluate it per-pixel in WGSL (a stack machine over `NodeKind`).
- ✗ Requires porting the **entire** eval stack to WGSL: the bit-portable value-noise basis,
  `fbm_height_grad`, the monotone Hermite spline, the ridge fold, every `NodeKind` — **with gradients**.
- ✗ Two implementations of the noise basis = **SSOT drift**; the GPU preview would slowly diverge from
  the CPU bake (misleading — the whole point of the preview is to mirror the real terrain).
- ✗ Largest, most error-prone, hardest to verify.

**Option B — CPU bakes a heightfield texture, GPU raymarches it (RECOMMENDED).**
- The CPU evaluates the graph over an N×N grid **once** (only when graph/zoom changes — already the
  cache key), producing a `height + analytic-normal` texture (e.g. `Rgba32Float`: r = height,
  gba = normal). This reuses the EXACT `Graph::eval` → **no WGSL noise port, no SSOT drift**.
- A GPU fullscreen fragment shader raymarches a ray through that heightfield (bilinear height samples),
  finds the intersection, and shades with the baked normal + the absolute-height/sea-level ramp.
- The camera (yaw/pitch/zoom) is a **uniform** → rotating the preview just updates the uniform and the
  GPU re-marches; **no CPU rebake on rotate**. The heightfield rebakes only when the graph/window
  changes. This is exactly the perf win we want (smooth high-res rotation), at far lower risk.
- The raymarch — the part that's actually slow at high res / on rotate — runs on the GPU.

→ **Go with Option B.**

## Stages (each a runnable checkpoint — user verifies visually)

1. **Offscreen → egui plumbing** (mirror `editor/material_preview.rs`): one offscreen `Image`
   (`RenderTarget::Image`), a fullscreen pass, registered as an egui texture. Shader = a trivial UV
   gradient. Checkpoint: a popped preview window shows the gradient (proves the pipeline + egui wiring).
2. **Heightfield bake** (`bake_heightfield(g, half, res) -> Image` of height+normal, cached by the
   existing `preview_key`). Upload as a texture bound to the raymarch shader. CPU side is unit-tested
   (values vs `Graph::eval`).
3. **WGSL heightfield raymarch**: ray–heightfield intersection via texture samples + adaptive steps,
   shaded by baked normal + `height_color_rgb` equivalent in WGSL, camera/zoom uniforms, sky bg.
   Checkpoint: the 3D preview matches the CPU one but is GPU-fast + smooth on rotate.
4. **Wire into the editor**: replace the CPU `render_surface_preview` path for nodes in 3D mode with the
   GPU target (keep CPU 2D heatmap; keep CPU 3D as a fallback if the GPU target isn't ready). Per-preview
   + per-popped-window targets, allocated lazily, freed when the preview closes.

## Constraints carried
- Editor-gated (`feature = "editor"`). Zero warnings, both build configs. No auto-run — user verifies.
- SSOT: the heightfield is the single `Graph::eval`; WGSL only marches/share-shades, never re-evaluates
  the graph or noise.
- Bevy 0.18 render-graph + bevy_egui 0.39 texture registration.
