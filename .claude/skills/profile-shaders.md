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
| **Which pass is slow + what's it bound on?** | Nsight GPU Trace → `perf.json` | **me** (`capture.ps1` + `parse.py`) |
| Per-WGSL-line cost inside a hot shader | the `.ngfx-gputrace` binary, in Nsight UI | user (UI deep-dive) |
| Which CPU system / render-graph node costs most? | `trace-*.json` (F6) | me (`rdoc/scripts/trace/`) |
| Inspect a single frame's textures / UBOs / disasm | `.rdc` RenderDoc capture | see `analyze-rdoc` (optional) |

## The Nsight loop (primary)

### Prerequisites (one-time)
1. **Nsight Graphics** installed (`ngfx.exe` under `C:\Program Files\NVIDIA Corporation\Nsight Graphics*\host\...`).
2. **GPU performance counters enabled for all users** — NVIDIA gates these to admin by
   default. Set once (admin): `reg add "HKLM\SOFTWARE\NVIDIA Corporation\Global\NVTweak" /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f`
   (or NVIDIA Control Panel → Developer → Manage GPU Performance Counters → allow all users).
   Without this the capture fails with `TARGET ERROR: GPU Performance Counters unavailable`.
3. Build with shader debug info so source correlation works:
   `cargo build --features editor,shader-debug` (enables wgpu `InstanceFlags::DEBUG` → naga
   `OpLine`). Needed for per-line; per-pass timing works without it.

### Capture + parse (I run these)
```sh
cargo build --features editor,shader-debug
powershell -ExecutionPolicy Bypass -File rdoc/scripts/ngfx/capture.ps1   # [-Frames 240] [-Out .soul/ngfx]
python rdoc/scripts/ngfx/parse.py .soul/ngfx                            # -> .soul/ngfx/perf.json + summary
```
`capture.ps1` injects via `ngfx --activity "GPU Trace Profiler"`, waits `-Frames` frames so
the scene settles, traces 1 frame with the SM hardware sampling profiler, auto-exports the
`BASE/*.xls` TSVs, and the app self-terminates (`ADVENTURE_EXIT_AFTER_FRAMES`, set by the
script). It forces `WGPU_BACKEND=vulkan` (SPIR-V → OpLine source mapping) and
`BEVY_ASSET_ROOT` (the exe run directly can't find `assets/` otherwise).

### Reading `perf.json`
One object per pass (`pass` = render-graph node label): `gpu_time_us`, `bottleneck`
(`SM`/`L1TEX`/`L2`/`DRAM` — the highest unit throughput), the four `*_throughput_pct`,
`inst_executed` + `inst_alu`/`inst_fma`/`inst_transcendental` (compute mix), `tex_hit_rate_pct`,
`ps_warp_occupancy_pct`/`cs_warp_occupancy_pct`, `draws`/`dispatches`.

The SDF passes to look for: **`sdf_brick_bake`** (compute, only on frames with bake jobs),
**`sdf_cone_prepass`** (compute), **`sdf_gbuffer_pass`** (the heavy fullscreen raymarch),
**`sdf_combine_pass`** (deferred lit). Shader sources: `assets/shaders/sdf_raymarch.wgsl` →
`sdf/march.wgsl` (the loop) + `sdf/brick.wgsl` (field eval) for the gbuffer pass.

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

### Per-WGSL-line cost (deep-dive, UI for now)
The 2.2MB `.ngfx-gputrace` binary holds the source-level shader profiler (PC sampling mapped
to WGSL lines via the `shader-debug` OpLine info). It's not yet parsed headless — open it in
`ngfx-ui.exe` → Shader Profiler view. If we later find a headless export of that table, add a
parser here. Until then, `perf.json` per-pass bottleneck + the in-shader step histogram (see
`march.wgsl` `result.steps`/`.fate`) are the closed-loop signals.

## CPU side — chrome traces (F6)
`cargo run --features editor` + **F6** toggles our custom chrome layer (off by default) →
`trace-<ts>.json`. Captures CPU systems + render-graph node spans (NOT GPU fragment cost —
that shows only as longer `prepare_windows`/vsync). Analyze with `rdoc/scripts/trace/` (system
python + `perfetto`): `hotspots.py`, `frames.py [--steady]`, `span.py <name>`,
`stream_hotspots.py` (for multi-GB traces, no perfetto). I CAN run these myself.

## Gotchas (learned the hard way)
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
(gitignored). When a new ngfx flag, export field, or interpretation recipe is learned, ADD it
here and extend `parse.py`.
