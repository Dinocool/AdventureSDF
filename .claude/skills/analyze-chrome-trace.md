---
name: analyze-chrome-trace
description: Analyze CPU frame time and per-system cost from a Bevy chrome trace (the editor's tracing-chrome capture → trace-*.json). Trigger for "CPU frame time", "which system is slow", "what's eating the main thread", a hitch / freeze / stutter / lag spike / main-thread stall, "why does scene load / streaming stutter", per-system CPU cost, the per-frame breakdown of a slow frame, or any trace-*.json. CPU-side only (systems + render-graph spans); for GPU/shader/per-pass cost see profile-shaders.
---

# Analyzing a Bevy chrome trace (CPU frame time / hitches)

The closed loop: **editor captures a `tracing-chrome` trace (`trace-<micros>.json`) → I run the
`rdoc/scripts/trace/` perfetto scripts → per-frame freeze breakdown + global hot systems**, all
by self-time (a span's `dur` minus its children's), naming the real Bevy systems.

This is the **CPU** counterpart to `profile-shaders` (the GPU/Nsight path). A chrome trace sees
CPU system + render-graph spans on the main + render threads. It does **not** see GPU fragment
cost — GPU work shows up only indirectly as longer `queue_submit` / `present_frames` / vsync
waits or a fat render `sub app`. If the CPU spans are cheap but frames are still long, the cost
is on the GPU → switch to `profile-shaders` (Nsight per-pass timing).

| Question | Tool |
|---|---|
| **What froze this frame?** per-frame dist + slowest-frame breakdown by self-time | `frames.py` |
| Which systems cost the most CPU over the whole session? | `hotspots.py` |
| Distribution + worst instances of ONE named system | `span.py <name>` |
| HUGE trace (multi-GB) perfetto can't load | `stream_hotspots.py` (stdlib, streaming) |
| GPU per-pass / per-WGSL-line cost | **not here** → `profile-shaders` |

## Prerequisite (one-time)
`pip install perfetto` — the first four scripts use perfetto's `trace_processor`. `stream_hotspots.py`
is pure-stdlib (no perfetto) for traces too big to load.

## Capturing a trace (in-app)
1. Editor → **Performance panel → "Chrome trace capture"** checkbox (or **F6**). It writes
   `./trace-<micros>.json` in the worktree root and shows the path. Off by default.
2. Reproduce the slow thing (load the scene, fly around, trigger the hitch).
3. **Stop the capture / exit the app CLEANLY.** The trace only flushes on the FlushGuard's
   `Drop` at clean shutdown — **a force-kill / crash leaves a 0-byte file**. If you find empty
   `trace-*.json`, that's why; re-capture and exit normally.

Source: `src/editor/chrome_trace.rs` (the gated `tracing-chrome` layer, `.include_args(true)`).

## The recipe (I run these)
```sh
python rdoc/scripts/trace/frames.py    trace-<micros>.json            # per-frame freeze breakdown
python rdoc/scripts/trace/frames.py    trace-<micros>.json --steady   # ignore early startup spikes
python rdoc/scripts/trace/hotspots.py  trace-<micros>.json            # global hot systems, self + total
python rdoc/scripts/trace/span.py      stream_voxel_rt trace-<micros>.json  # one system's distribution
python rdoc/scripts/trace/stream_hotspots.py trace-<micros>.json --frames   # for GB-scale traces
```
Pass the trace path as `argv[1]`; omit it to auto-pick the newest `trace-*.json` in the worktree
root (`_lib.newest_trace()`, `__file__`-relative — works in any worktree, not a hardcoded one).

## How the scripts read a current Bevy trace (the two non-obvious bits)
- **Per-frame span name.** The repeated top-level frame span is `update` (older Bevy traces used
  `update: ` with a trailing colon-space). `frames.py` detects it robustly via
  `_lib.frame_span_name()` (accepts either, else falls back to the most-frequent depth-0 root) —
  don't hardcode a literal.
- **Real system names live in `args.message`.** Bevy names every per-system / per-function span
  the generic literal **`function_scope`**; the actual name (e.g. `egui::context::Context::run_dyn`)
  is in that slice's `args.message` arg. So every breakdown resolves the name with
  `_lib.resolved_name()` →
  `CASE WHEN name='function_scope' THEN EXTRACT_ARG(arg_set_id,'args.message') ELSE name END`.
  Without this, all systems collapse into one meaningless `function_scope` bucket.
  (`stream_hotspots.py` does the same by reading the `args.message` field off each `B` event.)

## Reading the output
- `frames.py` prints `n / min / avg / MAX` frame duration + the 10 slowest frames, then the
  **slowest frame's spans ranked by self-time**. A system at the top of that list with large
  self-time = the thing that froze the frame. **Caveat:** if a heavy system has *no* nested
  tracing span, its cost rolls up into its parent's self-time — you'll see a generic container
  (`schedule`, `sub app`) at the top rather than a named system. That means the hot work is an
  **uninstrumented main-world / render system** (add a `#[instrument]`/`info_span!` to attribute
  it, or it's GPU — check `profile-shaders`).
- `hotspots.py` gives the whole-session ranking (self vs total). Use `--steady` on `frames.py`
  (and tune the ts cutoff) to drop the startup allocation spikes and see steady-state cost.

Cross-reference: **`profile-shaders`** — the GPU/Nsight half of the CPU/GPU split (`analyze-rdoc`
for a single-frame RenderDoc deep-dive). Same `rdoc/scripts/` tree.
