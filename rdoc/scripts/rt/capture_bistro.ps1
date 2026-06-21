# Headless Nsight GPU-Trace capture of the BISTRO raytrace bench (per-pass GPU time + occupancy/bottleneck).
# Mirrors rdoc/scripts/ngfx/capture.ps1 but drives the Bistro-alone bench env (NOT the sponza -Cam path).
# Needs a STATIC build: cargo build --no-default-features --features editor,physics,dlss
# Output: .soul/ngfx/*.xls (per-pass) -> parse with rdoc/scripts/ngfx/parse.py .soul/ngfx
param(
  [int]$Frames = 400,            # frames to settle (stream Bistro) before the 1-frame trace
  [string]$Out = ".soul/ngfx",
  [string]$Exe = "target/debug/adventure.exe",
  [string]$Cam = "-52,12,-40,-12.8,13.6,-32.8",  # eye + look_at, interior street view (inside geom_aabb)
  [int]$ClipHalf = 64,
  [switch]$Heavy                 # add --real-time-shader-profiler (per-line); default off = stable timing
)
$ErrorActionPreference = "Stop"
$ngfx = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Graphics*\host\windows-desktop-nomad-x64\ngfx.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
if (-not $ngfx) { Write-Error "ngfx.exe not found"; exit 1 }
if (-not (Test-Path $Exe)) { Write-Error "exe not found: $Exe"; exit 1 }
$exePath = (Resolve-Path $Exe).Path
$proj = (Get-Location).Path
New-Item -ItemType Directory -Force $Out | Out-Null
$outAbs = (Resolve-Path $Out).Path
$exitAt = $Frames + 180

$envStr = "ADVENTURE_EXIT_AFTER_FRAMES=$exitAt; WGPU_BACKEND=vulkan; BEVY_ASSET_ROOT=$proj; ADVENTURE_BENCH_BISTRO=1; ADVENTURE_CAM=$Cam; ADVENTURE_CLIP_HALF=$ClipHalf; ADVENTURE_EXIT_AFTER_SECS=9999;"
Write-Host "ngfx : $($ngfx.FullName)"
Write-Host "exe  : $exePath"
Write-Host "out  : $outAbs"
Write-Host "cam  : $Cam  clip=$ClipHalf  settle-frames=$Frames"

$ngfxArgs = @(
  "--activity", "GPU Trace Profiler",
  "--exe", $exePath,
  "--dir", $proj,
  "--output-dir", $outAbs,
  "--env", $envStr,
  "--start-after-frames", $Frames,
  "--limit-to-frames", "1",
  "--auto-export",
  "--set-gpu-clocks", "base",
  "--metric-set-id", "0",
  "--collect-screenshot", "1",
  "--no-timeout"
)
if ($Heavy) { $ngfxArgs += "--real-time-shader-profiler" }
# ngfx forwards the injected app's tracing logs on STDERR. Under $ErrorActionPreference=Stop, PowerShell 5.1
# wraps each native stderr line as a terminating NativeCommandError and aborts the script. Switch to Continue
# and capture ngfx's combined output to a log file so the app's stderr can't kill the capture.
$ngfxLog = Join-Path $outAbs "ngfx_run.log"
$ErrorActionPreference = "Continue"
& $ngfx.FullName @ngfxArgs *>&1 | Tee-Object -FilePath $ngfxLog | Out-Null
$ErrorActionPreference = "Stop"
Write-Host ""
Write-Host "=== ngfx tail ==="
Get-Content $ngfxLog -Tail 8 -ErrorAction SilentlyContinue
Write-Host "=== counter / export check ==="
Select-String -Path $ngfxLog -Pattern "TARGET ERROR|Performance Counters|Permission|export|Export|\.xls|error" -ErrorAction SilentlyContinue | Select-Object -Last 8 | ForEach-Object { $_.Line }
Write-Host "=== exported files ==="
Get-ChildItem $outAbs -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 12 FullName, Length
