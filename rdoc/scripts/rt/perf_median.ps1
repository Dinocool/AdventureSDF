# Median-of-N Nsight per-dispatch capture for reliable A/B (single captures swing +/-15% due to stochastic GI +
# convergence-state variance). Runs capture_bistro.ps1 N times, parses each trace's per-dispatch ms via the
# analyzer, and reports the MEDIAN + min/max per marker. Use a SMALL clip_half + high Frames so the resident set
# is converged (reproducible) across runs.
# Usage: powershell -File rdoc/scripts/rt/perf_median.ps1 -Label baseline -N 3 -Frames 500 -ClipHalf 48
param(
  [string]$Label = "median",
  [int]$N = 3,
  [int]$Frames = 500,
  [int]$ClipHalf = 48,
  [int]$Wc = -1
)
# Continue (NOT Stop): the ngfx capture + the python analyzer both write progress to STDERR, which PowerShell
# 5.1 wraps as a terminating NativeCommandError under Stop. We check exit/outputs explicitly instead.
$ErrorActionPreference = "Continue"
$proj = (Get-Location).Path
$store = "D:\tmp_test\rtbench\median"
New-Item -ItemType Directory -Force $store | Out-Null
$collected = @()
for ($i = 1; $i -le $N; $i++) {
  Write-Host "=== capture $i / $N ($Label) ==="
  $capArgs = @{ Frames = $Frames; ClipHalf = $ClipHalf }
  if ($Wc -ge 0) { $capArgs['Wc'] = $Wc }
  & "rdoc/scripts/rt/capture_bistro.ps1" @capArgs | Out-Null
  # Parse THIS capture's per-dispatch JSON NOW, before the next capture overwrites .soul/ngfx/BASE/*.xls
  # (the analyzer reads the .xls bundle, not the .ngfx-gputrace binary). Save the actions.json per run.
  $tr = Get-ChildItem ".soul/ngfx/*.ngfx-gputrace" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  python rdoc/scripts/ngfx/analyzer/nsight.py gputrace $tr.FullName *>$null
  $acts = $tr.FullName -replace '\.ngfx-gputrace$', '.gputrace.actions.json'
  $dst = Join-Path $store ("{0}_{1}.actions.json" -f $Label, $i)
  Copy-Item $acts $dst -Force
  $collected += $dst
}
# Median per marker across the saved per-run actions.json files.
python "rdoc/scripts/rt/median_actions.py" $Label @collected
