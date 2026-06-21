# Headless, self-exiting Bistro raytrace bench for autonomous A/B perf tuning of voxel_raytrace.wgsl.
# Launches the (already built, STATIC) editor exe with the bench harness, waits for self-exit, then
# extracts BENCH RESULT (whole-frame ms), the per-pass GPU breakdown (bench-gpu lines), and geom_aabb.
# WGSL is runtime-loaded, so shader-only edits need NO rebuild between runs -- just edit the .wgsl and re-run.
# Usage: powershell -File rdoc/scripts/rt/bench_rt.ps1 -Label baseline -Secs 30 -Cam "ex,ey,ez,lx,ly,lz" -ClipHalf 96
param(
  [string]$Label = "run",
  [int]$Secs = 30,
  [string]$Cam = "",
  [int]$ClipHalf = 0,
  [int]$Budget = 0,
  [int]$DebugView = -1,
  [int]$GiRays = -1,
  [switch]$GpuResidency,
  [string]$Exe = "target/debug/adventure.exe"
)
$ErrorActionPreference = "Stop"
$proj = (Get-Location).Path
if (-not (Test-Path $Exe)) { Write-Error "exe not found: $Exe (build: cargo build --no-default-features --features editor,physics,dlss)"; exit 1 }
$exeAbs = (Resolve-Path $Exe).Path
$logDir = "D:\tmp_test\rtbench"
New-Item -ItemType Directory -Force $logDir | Out-Null
$outLog = Join-Path $logDir ($Label + ".log")
$errLog = Join-Path $logDir ($Label + ".err")
if (Test-Path $outLog) { Remove-Item $outLog -Force }
if (Test-Path $errLog) { Remove-Item $errLog -Force }

$env:ADVENTURE_BENCH_BISTRO = "1"
$env:ADVENTURE_EXIT_AFTER_SECS = "$Secs"
$env:BEVY_ASSET_ROOT = $proj
$env:WGPU_BACKEND = "vulkan"
Remove-Item Env:\ADVENTURE_CAM -ErrorAction SilentlyContinue
Remove-Item Env:\ADVENTURE_CLIP_HALF -ErrorAction SilentlyContinue
Remove-Item Env:\ADVENTURE_DEBUG_VIEW -ErrorAction SilentlyContinue
Remove-Item Env:\ADVENTURE_GI_RAYS -ErrorAction SilentlyContinue
Remove-Item Env:\ADVENTURE_GPU_RESIDENCY -ErrorAction SilentlyContinue
Remove-Item Env:\ADVENTURE_STREAM_BUDGET -ErrorAction SilentlyContinue
if ($Cam) { $env:ADVENTURE_CAM = $Cam }
if ($ClipHalf -gt 0) { $env:ADVENTURE_CLIP_HALF = "$ClipHalf" }
if ($Budget -gt 0) { $env:ADVENTURE_STREAM_BUDGET = "$Budget" }
if ($DebugView -ge 0) { $env:ADVENTURE_DEBUG_VIEW = "$DebugView" }
if ($GiRays -ge 0) { $env:ADVENTURE_GI_RAYS = "$GiRays" }
if ($GpuResidency) { $env:ADVENTURE_GPU_RESIDENCY = "1" }

Write-Host "=== bench '$Label' secs=$Secs cam='$Cam' clip=$ClipHalf dv=$DebugView gi=$GiRays gpuRes=$GpuResidency ==="
Write-Host "exe=$exeAbs"
Write-Host "out=$outLog"
$proc = Start-Process -FilePath $exeAbs -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$deadline = $Secs + 90
if (-not $proc.WaitForExit($deadline * 1000)) {
  Write-Host "TIMEOUT killing"
  try { $proc.Kill() } catch {}
}
Start-Sleep -Milliseconds 300
if (Test-Path $errLog) { Get-Content $errLog | Add-Content $outLog }
Write-Host "--- BENCH RESULT / per-pass GPU / geom ---"
Select-String -Path $outLog -Pattern "BENCH RESULT|bench-gpu|geom_aabb|resident_bricks|panicked|error\[|TARGET ERROR" | Select-Object -Last 14 | ForEach-Object { $_.Line }
Write-Host "--- full log: $outLog ---"
