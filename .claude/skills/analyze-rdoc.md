---
name: analyze-rdoc
description: Analyze RenderDoc GPU captures (.rdc) and Bevy chrome traces to find rendering performance bottlenecks. Trigger when investigating GPU/frame cost, "which draw is slow", shader perf, a .rdc capture, or a trace-*.json. Toolkit lives in rdoc/scripts/.
---

# Analyzing RenderDoc captures & Bevy traces

Two profiling data sources, two toolkits under `rdoc/scripts/`. Pick by the question:

| Question | Source | Tool dir |
|---|---|---|
| **Which draw/dispatch is slow on the GPU?** | `.rdc` capture | `rdoc/scripts/rdoc/` |
| Which CPU system / render-graph node costs most? | `trace-*.json` | `rdoc/scripts/trace/` |
| What did the GPU actually see (uniforms, textures)? | `.rdc` capture | `rdoc/scripts/rdoc/` |

**Critical distinction:** a Bevy chrome trace canNOT see GPU fragment cost — heavy shaders
show only as longer `prepare_windows` / vsync wait. To attribute frame cost to a *draw* you
MUST use the `.rdc` GPU-timing path. Conversely, CPU hitches (bake spikes, system stalls)
only show in the trace. Use both.

## Capturing (the user does this)

Editor build has F5 RenderDoc capture (`editor/renderdoc_capture.rs`). Press F5 in the
running editor; `.rdc` writes to `rdoc/` (older captures may be in
`%LOCALAPPDATA%/Temp/RenderDoc/`). Overlay is disabled, captures go to `rdoc/`.
Requires a static build — `default = []` (NOT `fast`/dynamic_linking, which RenderDoc
can't hook). Run: `cargo run --features editor`.

Chrome traces: any `cargo run --features editor` writes `trace-<ts>.json` on exit
(bevy/trace_chrome). `main::prune_old_traces` keeps the newest few.

## Running the .rdc tools (IMPORTANT — read this)

GPU counters need a **replay device**, which ONLY qrenderdoc provides. Run scripts with
the capture path in the `RDOC_CAPTURE` **env var** (NOT as an argument — see gotcha below),
launched from the **repo root** (cwd is how scripts find the toolkit + repo):

```sh
RDOC_CAPTURE="<abs path to .rdc>" \
  "C:/Program Files/RenderDoc/qrenderdoc.exe" --python rdoc/scripts/rdoc/<script>.py
# then read rdoc/<script>_out.txt
```

- **A window flashes briefly** then auto-closes (every script ends in `_lib.finish()` →
  `os._exit`). Expected, the user has OK'd it. Do NOT try to suppress it.
- **The USER runs this, not me** (default). Per [[feedback-no-launch-renderdoc]] I don't
  launch qrenderdoc unless the user explicitly tells me to in the moment. I write/edit the
  scripts, hand the user the command, then **read `rdoc/<script>_out.txt`** — qrenderdoc
  swallows stdout on Windows, so the `_out.txt` mirror is the ONLY real output.
- **Replay is slow:** an 80MB capture takes 60–150s to init the replay device before the
  first counter row appears. The `_out.txt` header line appears early; the data lines come
  after. Don't assume failure until ~3min with no growth AND qrenderdoc has exited.
- Capture optional: omit `RDOC_CAPTURE` → newest `.rdc` in `rdoc/` then the temp dir.

### qrenderdoc embedded-Python gotchas (THE silent-failure traps)

qrenderdoc embeds **Python 3.6** with a stripped `sys` — learned by losing a capture to a
silent empty file. Two names are MISSING; using either kills the script before its Tee log
opens, producing NO output at all:

- **`__file__` is undefined** → never `os.path.dirname(__file__)`. Bootstrap `sys.path`
  from `os.getcwd()` (qrenderdoc runs from the repo root): `os.path.join(os.getcwd(),
  "rdoc","scripts","rdoc")`. This is also why there's no hardcoded worktree path — cwd is
  portable across worktrees/clones.
- **`sys.argv` is absent** → can't pass args positionally. Inputs come via env vars:
  `RDOC_CAPTURE` (the .rdc), `RDOC_TEXTURES` (space-separated resource names for
  save_textures). `_lib.capture_arg` reads the env var first, with a `getattr(sys,'argv',
  [])` fallback for system-python use.

Any NEW `.rdc` script MUST follow both rules or it fails silently.

### .rdc tools (`rdoc/scripts/rdoc/`)
- **`gpu_timings.py`** — THE perf tool. Every draw/dispatch/copy by GPU µs, descending.
  Top row = bottleneck. For the SDF renderer the fullscreen `vkCmdDraw` is the raymarch;
  if it dominates, cost is in the fragment shader (texture taps, march steps, reflections).
- **`list_passes.py`** — frame structure: every draw/dispatch/marker with eventId. Run
  first to orient and get the eventId for other tools.
- **`dump_camera_ubo.py`** — decodes `SdfCameraData` bound to the SDF fragment shader
  (camera_pos, num_chunks, lod_params, debug_params). For "GPU saw wrong data" bugs.
- **`save_textures.py [names...]`** — saves named GPU textures to `rdoc/<base>_<name>.png`
  (default `sdf_dist_atlas`, `sdf_cone_seed`). Visual atlas/bake diffing.
- **`_lib.py`** — shared: `open_capture`, `walk`, `action_names`, `find_action`,
  `counter_seconds` (robust counter decode), `Tee` (stdout+file), `finish` (hard-exit).

## Running the trace tools (system python, no qrenderdoc)

```sh
python rdoc/scripts/trace/<script>.py [trace.json]
```

Needs `python -m pip install perfetto`. I CAN run these myself (no GUI). Trace arg
optional → newest `trace-*.json` in the repo root.

**Huge-trace caveat:** `bevy/trace_chrome` traces grow to **6–18GB** for a long editor
session. perfetto's `trace_processor` loads the whole file into RAM — a >8GB trace can take
>5min to load or OOM. If a trace tool times out, ask the user for a SHORT capture session
(start the editor, repro, exit within ~30s) which yields a far smaller trace. The GPU-timing
`.rdc` path has no such limit — prefer it for single-frame perf questions anyway.

### trace tools (`rdoc/scripts/trace/`)
- **`hotspots.py`** — top CPU spans by self-time (children subtracted) + total. First tool.
- **`frames.py [--steady]`** — per-frame distribution + slowest-frame breakdown. `--steady`
  ignores the first ~2.2s of startup allocation spikes to find real in-game hitches.
- **`span.py <name-substring>`** — stats + worst occurrences of one named span (e.g.
  `schedule_bakes`, `init_texture_streaming`) once hotspots points at a suspect.
- **`_lib.py`** — `trace_arg`, `processor`, `SELF_TIME_CTE`.

## Best practices (learned)

- **GPU µs ≠ CPU record µs.** `renderdoccmd convert -c chrome.json` is headless but its
  `vkCmdDraw` durations are CPU *record* times (read ~0µs), NOT GPU execution. Useless for
  "which draw is slow." Only `FetchCounters(EventGPUDuration)` under replay gives real GPU
  time. (This is why the GUI runner is unavoidable — see Dead ends.)
- **Always read `_out.txt`,** never trust the console for `.rdc` scripts.
- **Wrap script bodies in try/except** that logs to the Tee before `finish()` — a bare
  exception under qrenderdoc produces a SILENT empty file otherwise.
- **One frame at a time:** `ctrl.SetFrameEvent(eid, True)` before reading pipeline state /
  buffers / textures, or you get the wrong event's bindings.
- **SDF perf reading:** fullscreen raymarch `vkCmdDraw` dominating ⇒ fragment cost. Check
  texture tap count, march steps, and the reflection gate (a per-pixel 2nd march) before
  blaming the CPU.

## Dead ends (do NOT re-attempt)

- **Offscreen qrenderdoc** (`QT_QPA_PLATFORM=offscreen`): deadlocks at Qt init before the
  python script runs (no out file ever created). The Vulkan replay wants a real surface.
  Tried a matched Qt 5.15.2 `qoffscreen.dll` from the PyQt5-Qt5 wheel — still hangs.
- **`renderdoccmd convert`**: headless but GPU-blind (CPU record times only, above).
- **Rust replay util binding `renderdoc.dll`**: only `renderdoc_app.h` (in-app capture, flat
  C) ships — NO replay header, no import lib. The replay API (`ICaptureFile`,
  `IReplayController`) is C++ vtable-based; FFI would mean hand-replicating the MSVC vtable
  ABI, version-locked and fragile. Not worth it. The supported replay interface is the
  Python bindings inside qrenderdoc.

Conclusion: **windowed `qrenderdoc --python`, user-run, read `_out.txt`** is the path.

## Keeping this skill current

When I learn a new RenderDoc API trick, a new analysis recipe, or a new gotcha, ADD it
here and (if reusable) add a script to `rdoc/scripts/`. The toolkit + this skill are meant
to grow. `rdoc/scripts/` is git-tracked (the rest of `rdoc/` — captures, dumps, `_out.txt`
— is gitignored).

## Related
- [[feedback-no-launch-renderdoc]] — never launch qrenderdoc myself; hand the user the cmd.
- [[feedback-no-auto-run]] — same principle for `cargo run`.
- `debug-shader` — in-engine shader debug overlays (the other half of GPU debugging).
- `wgsl-gotchas` memory — WGSL pitfalls when acting on findings.
