---
name: profile-shaders
description: Profile SDF shader / GPU performance with NVIDIA Nsight Graphics (headless, AI-runnable) and Bevy chrome traces. Trigger for GPU/frame cost, "which pass is slow", shader optimization, per-stage GPU time, bottleneck (compute vs memory bound), or per-WGSL-line cost. The primary GPU-perf path; RenderDoc (see analyze-rdoc) is now an optional deep-dive.
---

# Profiling SDF shader / GPU performance

The closed loop: **`ngfx` headless GPU Trace → `parse.py` → `.soul/ngfx/perf.json`**, which I
read to attribute frame cost to each render pass and tell what it's BOUND on (compute /
texture / L2 / DRAM). I can run this whole flow myself — no GUI, no out-file dance.

| Question | Path | Who runs it |
|---|---|---|
| **Which pass is slow + what's it bound on?** (headless, fixed frame N) | Nsight GPU Trace → `perf.json` | **me** (`capture.ps1` + `parse.py`) |
| Same, but **the live editor frame** the user is looking at | launch under Nsight (F11 trigger) → `perf.json` | user presses F11; **me** reads `perf.json` |
| Per-WGSL-line cost inside a hot shader | the `.ngfx-gputrace` binary, in Nsight UI | user (UI deep-dive) |
| Which CPU system / render-graph node costs most? | `trace-*.json` (F6) | me (`rdoc/scripts/trace/`) |
| Inspect a single frame's textures / UBOs / disasm | `.rdc` RenderDoc capture | see `analyze-rdoc` (optional) |

There are **two ways into the same `perf.json` loop**: the headless `capture.ps1` (a fresh
instance, fixed frame, fully AI-runnable end-to-end) and the interactive F11 launch (the live
editor frame, user-triggered). Both auto-export the same `BASE/*.xls` that `parse.py` reads.

## The Nsight loop (primary)

### Prerequisites (one-time)
1. **Nsight Graphics** installed (`ngfx.exe` under `C:\Program Files\NVIDIA Corporation\Nsight Graphics*\host\...`).
2. **GPU performance counters enabled for all users** — NVIDIA gates these to admin by
   default. Set once (admin): `reg add "HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak" /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f`
   (or NVIDIA Control Panel → Developer → Manage GPU Performance Counters → allow all users).
   Without this the capture fails with `TARGET ERROR: GPU Performance Counters unavailable`.
3. Build **static, with shader debug info**:
   `cargo build --no-default-features --features editor,shader-debug`. `--no-default-features`
   drops `fast`/dynamic_linking (Nsight's injector can't attach to a dynamically-linked Bevy —
   see Gotchas); `shader-debug` enables wgpu `InstanceFlags::DEBUG` **AND** pulls
   `bevy_render/decoupled_naga` (both needed) → naga emits SPIR-V `OpSource` (embedded composed
   WGSL) + `OpLine` (line mapping). Per-pass timing works without it; per-WGSL-line **source**
   needs it. (Before decoupled_naga, Bevy handed wgpu a bare `Naga` module with an empty source
   string, so naga emitted *neither* OpSource nor OpLine regardless of the DEBUG flag — wgpu
   Discussion #7761. That's why earlier `shader-debug` captures had no source mapping.)

### Capture + parse (I run these)
```sh
cargo build --no-default-features --features editor,shader-debug
powershell -ExecutionPolicy Bypass -File rdoc/scripts/ngfx/capture.ps1   # [-Frames 240] [-Out .soul/ngfx]
python rdoc/scripts/ngfx/parse.py .soul/ngfx                            # -> .soul/ngfx/perf.json + summary
```
`capture.ps1` injects via `ngfx --activity "GPU Trace Profiler"`, waits `-Frames` frames so
the scene settles, traces 1 frame with the SM hardware sampling profiler, auto-exports the
`BASE/*.xls` TSVs, and the app self-terminates (`ADVENTURE_EXIT_AFTER_FRAMES`, set by the
script). It forces `WGPU_BACKEND=vulkan` (SPIR-V → OpLine source mapping) and
`BEVY_ASSET_ROOT` (the exe run directly can't find `assets/` otherwise).

### Interactive capture — the live editor frame (F11)
When you need the **exact frame the user is editing** (a specific camera/scene state), launch
the editor *under* Nsight instead of headless: run the root `run-worktree.ps1`, press **S** to
turn on "Nsight GPU-Trace profiling", pick the worktree, Enter. That builds
`editor,shader-debug` and launches the exe under `ngfx --activity "GPU Trace Profiler"
--start-after-hotkey` (Vulkan forced). In-game, **F11** triggers Nsight to capture the live
frame and auto-export to the worktree's `.soul/ngfx`; the editor shows a toast acknowledging
the trigger (`src/editor/nsight_capture.rs`, `shader-debug` builds only — Nsight does the
capture out-of-process, the app just confirms it). On exit the launcher runs `parse.py`
automatically. I then read `.soul/ngfx/perf.json` the same way. Same trace, same loop — only
the trigger (a hotkey on the live process) and frame selection differ from `capture.ps1`.

### Reading `perf.json`
One object per pass (`pass` = render-graph node label): `gpu_time_us`, `bottleneck`
(`SM`/`L1TEX`/`L2`/`DRAM` — the highest unit throughput), the four `*_throughput_pct`,
`inst_executed` + `inst_alu`/`inst_fma`/`inst_transcendental` (compute mix), `tex_hit_rate_pct`,
`ps_warp_occupancy_pct`/`cs_warp_occupancy_pct`, `draws`/`dispatches`.

The SDF passes to look for: **`sdf_brick_bake`** (compute, only on frames with bake jobs),
**`sdf_cone_prepass`** (compute), **`sdf_gbuffer_pass`** (the heavy fullscreen raymarch),
**`sdf_combine_pass`** (deferred lit). Shader sources: `assets/shaders/sdf_raymarch.wgsl` →
`sdf/march.wgsl` (the loop) + `sdf/brick.wgsl` (field eval) for the gbuffer pass.

### Richer per-marker metrics — the vendored analyzer
`parse.py` gives the per-pass table; for a **per-marker bottleneck verdict + 120+ metrics** use the
vendored [`nsight-graphics-analyzer`](rdoc/scripts/ngfx/analyzer/VENDORED.md) (MIT, stdlib-only).
It reads the **same `BASE/*.xls` bundle** `capture.ps1` already exported (it can't parse the
`.ngfx-gputrace` binary — NVIDIA dropped the offline parser in 2026.1), so no re-capture needed:
```sh
NS=rdoc/scripts/ngfx/analyzer/nsight.py ; TR=.soul/ngfx/<capture>.ngfx-gputrace
python $NS gputrace              "$TR"                                 # summary/stages/actions JSON
python $NS gputrace-shader-bound "$TR" --in-marker sdf_probe_trace     # SM/occupancy diagnosis, scoped
python $NS gputrace-metric       "$TR" --name <regex> --in-marker <pass>  # drill ANY metric, per-marker
```
The decisive occupancy signals it surfaces (per marker): **`cs_warps_active`** (compute warp
occupancy as % of peak — single digits = severe register starvation), `sm_throughput`,
`warps_inactive_sm_active`, the ALU/FMA/XU pipe mix, and `texin_cycles_stalled_on_tsl1_miss`
(texture-cache stalls). Example verdict that settled an occupancy-vs-split debate: the probe trace
at `cs_warps_active 1.98%`, all pipes <5%, texture stall 0.06% → purely **occupancy-starved**
(register-limited), not memory- or pipe-bound. (`gputrace-stalls` is frame-level idle; it reports
0% coverage here because our markers are D3DPERF spans, not NVTX ranges.)

### Interpreting bottlenecks
- `bottleneck=SM` **with low `sm_throughput_pct`** (e.g. 20%) → **occupancy / latency bound**,
  NOT ALU-throughput bound. The SMs are stalling (memory latency, warp divergence in the
  march, low occupancy). Fixes: reduce divergence, improve atlas locality, lower register
  pressure to raise occupancy — NOT "do fewer FLOPs".
- `bottleneck=SM` with high `sm_throughput_pct` → genuinely ALU bound; cut math (esp. high
  `inst_transcendental` = trig/exp/rsqrt in the shader).
- `bottleneck=DRAM`/`L2` → memory bound; shrink/repack atlas reads, raise `tex_hit_rate_pct`.
- Low `tex_hit_rate_pct` on the gbuffer pass → brick-atlas access pattern is thrashing cache.

### The A/B measurement loop
1. Fix a camera/scene (deterministic frame). 2. Capture + parse → baseline `perf.json`.
3. Make the shader change + rebuild. 4. Re-capture → diff `gpu_time_us` + throughputs +
step/inst counts per pass. `--set-gpu-clocks base` (in `capture.ps1`) locks clocks so numbers
are comparable across runs.

### Per-WGSL-line cost — source mapping WORKS (with decoupled_naga); export is one GUI click
With the `shader-debug` build (now incl. `decoupled_naga`, see prereq 3) the `.ngfx-gputrace`
embeds the composed WGSL (`OpSource`) + line mapping (`OpLine`), so Nsight's Shader Profiler shows
real per-line WGSL cost — **registers, samples, and stall reasons per line** (this is what was
broken before: no decoupled_naga → no OpSource/OpLine at all). The **capture** is headless; only
the per-line **export** is a GUI click — there's no CLI shader-source export flag (verified;
`--auto-export` writes only the per-pass `BASE/*.xls`, the report is a proprietary `WRPV` binary).

Workflow: open the `.ngfx-gputrace` in `ngfx-ui.exe` → **Shader Profiler / Shader Source** for the
hot pass → **right-click the table → Export to CSV** → hand me the CSV. I parse it with
`python rdoc/scripts/ngfx/shader_lines.py <export>.csv`, which reconstructs the embedded WGSL,
maps each SPIR-V instruction back to its `OpLine` WGSL line, and ranks lines by **GPU samples**
(time) and **peak Live Registers** (the occupancy limiter). Note: `Samples` is 0 on a settled/idle
frame (e.g. a dormant probe pass) — capture an **actively-converging** frame (low `-Frames`, e.g.
90) for time; `Live Registers` is present regardless and pinpoints what's pinning occupancy.

Fully-headless alternative — **ablation A/B**: attribute cost to march REGIONS (not lines) by
gating suspected hot blocks behind shader `#define`s (the field already has
`SDF_DISABLE_CHUNK_CACHE` / `SDF_DISABLE_LOD`) and diffing the pass's `gpu_time_us`/`inst_executed`
in `perf.json` across F11 captures. Coarser than per-line, but AI-runnable end-to-end. The
in-shader step histogram (`march.wgsl` `result.steps`/`.fate`) is the other closed-loop signal.

## CPU side — chrome traces (F6)
`cargo run --features editor` + **F6** toggles our custom chrome layer (off by default) →
`trace-<ts>.json`. Captures CPU systems + render-graph node spans (NOT GPU fragment cost —
that shows only as longer `prepare_windows`/vsync). Analyze with `rdoc/scripts/trace/` (system
python + `perfetto`): `hotspots.py`, `frames.py [--steady]`, `span.py <name>`,
`stream_hotspots.py` (for multi-GB traces, no perfetto). I CAN run these myself.

## Gotchas (learned the hard way)
- **Dynamic linking breaks injection.** A `fast`/`dynamic_linking` build fails to attach:
  `Launch process exited. Searching for attachable child processes... Failed to connect`.
  Build `--no-default-features` (drops `fast`) for any injected capture — both `capture.ps1`
  and the interactive F11 launch. The `run-worktree.ps1` profiling launch already does this.
  Worse, a failed capture leaves the **previous** `BASE/*.xls` in place, so `parse.py` happily
  prints STALE numbers — clear `.soul/ngfx` (the launcher does) or distrust a `perf.json` whose
  numbers didn't move after a "failed to connect".
- **GPU counters disabled** → `TARGET ERROR`. Enable `RmProfilingAdminOnly=0` (above).
- **`--platform "..."` collides with `ngfx`'s Qt `-platform`** → "Could not find Qt platform
  plugin". OMIT `--platform`; the default is correct.
- **Assets not found** (`target/debug/assets/...`) when the exe is launched directly (not via
  `cargo run`): set `BEVY_ASSET_ROOT=<repo root>` (the script does).
- **Source mapping needs Vulkan** (SPIR-V/OpLine), not DX12 — `WGPU_BACKEND=vulkan` (script
  sets it) + the `shader-debug` build.
- **`sdf_brick_bake` is absent** in a capture if no bake jobs ran that frame — fly the camera
  to dirty bricks, or capture later, if you need bake timing.
- The `BASE/*.xls` exports are **tab-separated text** despite the `.xls` extension.

## Related
- `analyze-rdoc` — RenderDoc single-frame deep-dive (textures, UBO decode, disasm). Optional,
  behind the `renderdoc` feature (F7). Superseded for per-pass timing by this skill.
- `debug-shader` — in-engine shader debug overlays (the other half of GPU debugging).
- `wgsl-gotchas` memory — WGSL pitfalls when acting on findings.
- [[feedback-no-auto-run]] — but the Nsight capture is explicitly AI-runnable here.

## Keeping this current
The toolkit lives in `rdoc/scripts/ngfx/` (git-tracked); captures land in `.soul/ngfx/`
(gitignored):
- `capture.ps1` — headless GPU-Trace capture + auto-export.
- `parse.py` — quick per-pass table → `perf.json`.
- `analyzer/` — vendored `nsight-graphics-analyzer` (per-marker metrics + bottleneck verdicts;
  update per `analyzer/VENDORED.md`, don't edit the vendored files).
- `shader_lines.py` — parse a GUI-exported Shader-Profiler CSV → hot WGSL lines (samples + live
  registers).

When a new ngfx flag, export field, or interpretation recipe is learned, ADD it here and extend
`parse.py`.
