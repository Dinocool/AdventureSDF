<#
.SYNOPSIS
  Headless NVIDIA Nsight Graphics GPU-Trace + source-level (per-WGSL-line) shader-profiler
  capture of the adventure editor. Auto-exports a report under -Out for parsing.

.DESCRIPTION
  Runs `ngfx.exe --activity "GPU Trace Profiler"` to inject into the app, wait -Frames frames
  (so the scene settles), trace 1 frame with the SM hardware sampling profiler on, and
  auto-export. `--real-time-shader-profiler` collects per-source-line GPU cost; for that cost
  to map back to WGSL lines the app MUST be built with `shader-debug` (naga OpLine) and run on
  the Vulkan backend (SPIR-V), which this script forces via WGPU_BACKEND=vulkan.

.PREREQUISITES
  cargo build --no-default-features --features editor,shader-debug
  (--no-default-features drops `fast`/dynamic_linking; Nsight's injector cannot attach to a
   dynamically-linked Bevy -- the launch fails with "process exited / searching for child
   processes". A static build is required for any injected capture, same as RenderDoc.)

.USAGE
  powershell -ExecutionPolicy Bypass -File rdoc/scripts/ngfx/capture.ps1
  powershell -ExecutionPolicy Bypass -File rdoc/scripts/ngfx/capture.ps1 -Frames 300 -Out .soul/ngfx
#>
param(
  [int]$Frames = 240,
  [string]$Out = ".soul/ngfx",
  [string]$Exe = "target/debug/adventure.exe",
  # Optional: load a specific scene at startup instead of the default (project-root-relative path,
  # e.g. "assets/scenes/cornell8.scene"). Sets ADVENTURE_STARTUP_SCENE for the captured run.
  [string]$Scene = "",
  # Optional: pin a FIXED in-scene camera for the capture so perf numbers reflect a representative viewpoint
  # (e.g. inside the Sponza atrium) instead of the cheap default boot camera. Six comma-separated floats
  # "ex,ey,ez,lx,ly,lz" (eye + look_at) — grab them by pressing F8 in the editor. Activates the bench harness
  # (`ADVENTURE_BENCH_BISTRO=1`) booting the in-RAM Sponza (`ADVENTURE_BENCH_SCENE=sponza`) with the pin.
  [string]$Cam = ""
)
$ErrorActionPreference = "Stop"

$ngfx = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Graphics*\host\windows-desktop-nomad-x64\ngfx.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
if (-not $ngfx) { Write-Error "ngfx.exe not found - is Nsight Graphics installed?"; exit 1 }

if (-not (Test-Path $Exe)) {
  Write-Error "App exe not found at '$Exe'. Build first: cargo build --features editor,shader-debug"
  exit 1
}
$exePath = (Resolve-Path $Exe).Path
$proj    = (Get-Location).Path
New-Item -ItemType Directory -Force $Out | Out-Null
$outAbs  = (Resolve-Path $Out).Path

Write-Host "ngfx : $($ngfx.FullName)"
Write-Host "exe  : $exePath"
Write-Host "out  : $outAbs"
Write-Host "Tracing 1 frame at frame $Frames with the source shader profiler on..."

# Self-terminate the app a bit after the trace point so ngfx finishes + exports hands-free.
$exitAt = $Frames + 180

# App env passed through the injector. Optional ADVENTURE_STARTUP_SCENE picks the scene to capture.
$envStr = "ADVENTURE_EXIT_AFTER_FRAMES=$exitAt; WGPU_BACKEND=vulkan; BEVY_ASSET_ROOT=$proj;"
if ($Scene) { $envStr += " ADVENTURE_STARTUP_SCENE=$Scene;"; Write-Host "scene: $Scene" }
# -Cam pins a representative in-Sponza viewpoint via the bench harness (camera stays fixed each frame). We still
# rely on ADVENTURE_EXIT_AFTER_FRAMES (above) for the capture+exit timing, not the bench's secs-based exit.
if ($Cam) {
  $envStr += " ADVENTURE_BENCH_BISTRO=1; ADVENTURE_BENCH_SCENE=sponza; ADVENTURE_CAM=$Cam;"
  Write-Host "cam  : $Cam (Sponza bench pin)"
}

# Lock GPU clocks to base for repeatable numbers; metric-set 0 = Throughput Metrics.
& $ngfx.FullName `
  --activity "GPU Trace Profiler" `
  --exe $exePath `
  --dir $proj `
  --output-dir $outAbs `
  --env $envStr `
  --start-after-frames $Frames `
  --limit-to-frames 1 `
  --auto-export `
  --real-time-shader-profiler `
  --set-gpu-clocks base `
  --metric-set-id 0 `
  --collect-screenshot 1 `
  --no-timeout

Write-Host ""
Write-Host "Capture finished. Exported files:"
Get-ChildItem $outAbs -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 20 FullName, Length
