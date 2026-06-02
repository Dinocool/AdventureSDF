$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Get-Worktrees {
    $wts = @()
    foreach ($line in (git worktree list --porcelain)) {
        if ($line -like "worktree *") {
            $wts += ($line -replace "^worktree ", "") -replace "/", "\"
        }
    }
    return $wts
}

function Show-Menu($items, $selected, $nsight) {
    Clear-Host
    if ($nsight) {
        Write-Host "Select worktree to run (Nsight GPU-Trace: editor,shader-debug under ngfx)" -ForegroundColor Cyan
    } else {
        Write-Host "Select worktree to run (cargo run --features editor)" -ForegroundColor Cyan
    }
    Write-Host "Up/Down to move, Enter to launch, S to toggle Nsight profiling, Q to quit" -ForegroundColor DarkGray
    $sdState = if ($nsight) { "ON  (press F11 in-game to capture the live frame)" } else { "OFF" }
    $sdColor = if ($nsight) { "Green" } else { "DarkGray" }
    Write-Host "Nsight GPU-Trace profiling: " -ForegroundColor DarkGray -NoNewline
    Write-Host $sdState -ForegroundColor $sdColor
    Write-Host ""
    for ($i = 0; $i -lt $items.Count; $i++) {
        if ($i -eq $selected) {
            Write-Host ("  > " + $items[$i]) -ForegroundColor Black -BackgroundColor Green
        } else {
            Write-Host ("    " + $items[$i])
        }
    }
}

while ($true) {
    $worktrees = Get-Worktrees
    if ($worktrees.Count -eq 0) {
        Write-Host "No worktrees found." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    $sel = 0
    $nsight = $false
    $done = $false
    while (-not $done) {
        Show-Menu $worktrees $sel $nsight
        $key = [System.Console]::ReadKey($true)
        switch ($key.Key) {
            "UpArrow"   { $sel = ($sel - 1 + $worktrees.Count) % $worktrees.Count }
            "DownArrow" { $sel = ($sel + 1) % $worktrees.Count }
            "S"         { $nsight = -not $nsight }
            "Enter"     { $done = $true }
            "Q"         { exit 0 }
        }
    }

    $target = $worktrees[$sel]
    Write-Host ""

    if ($nsight) {
        # Profiling launch: build the editor with shader-debug (naga OpLine -> WGSL source
        # mapping), then launch the built exe UNDER Nsight Graphics GPU Trace. The trace is
        # armed with --start-after-hotkey, so pressing F11 in-game captures the live frame.
        # On exit, parse.py turns the auto-exported trace into .soul/ngfx/perf.json.
        Write-Host "Launching under Nsight GPU Trace (editor,shader-debug)" -ForegroundColor Yellow
        Write-Host "Dir: $target" -ForegroundColor Yellow
        $inner = @"
Set-Location '$target'
Write-Host 'Building static (cargo build --no-default-features --features editor,shader-debug)...' -ForegroundColor Yellow
Write-Host '(no-default-features drops fast/dynamic_linking, which Nsight cannot inject into)' -ForegroundColor DarkGray
cargo build --no-default-features --features editor,shader-debug
if (`$LASTEXITCODE -ne 0) { Write-Host ''; Write-Host 'Build failed.' -ForegroundColor Red; Read-Host 'Press Enter to close'; exit }
`$exe  = Join-Path '$target' 'target\debug\adventure.exe'
`$ngfx = (Get-ChildItem 'C:\Program Files\NVIDIA Corporation\Nsight Graphics*\host\windows-desktop-nomad-x64\ngfx.exe' -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1).FullName
if (-not `$ngfx) { Write-Host 'ngfx.exe not found - is Nsight Graphics installed?' -ForegroundColor Red; Read-Host 'Press Enter to close'; exit }
`$out = Join-Path '$target' '.soul\ngfx'
# Clear any prior export so a failed/aborted capture can't leave stale numbers for parse.py.
if (Test-Path `$out) { Remove-Item -Recurse -Force `$out }
New-Item -ItemType Directory -Force `$out | Out-Null
Write-Host ''
Write-Host 'Nsight GPU Trace armed. Press F11 in-game to capture the live frame; close the app when done.' -ForegroundColor Cyan
& `$ngfx --activity 'GPU Trace Profiler' --exe `$exe --dir '$target' --output-dir `$out --env 'WGPU_BACKEND=vulkan; BEVY_ASSET_ROOT=$target;' --start-after-hotkey --limit-to-frames 1 --auto-export --real-time-shader-profiler --set-gpu-clocks base --metric-set-id 0 --collect-screenshot 1 --no-timeout
Write-Host ''
`$parse = Join-Path '$target' 'rdoc\scripts\ngfx\parse.py'
if (Test-Path `$parse) { Write-Host 'Parsing trace -> perf.json...' -ForegroundColor Yellow; python `$parse `$out }
Read-Host 'Capture session ended. Press Enter to close'
"@
        Start-Process powershell -ArgumentList "-NoProfile", "-Command", $inner
    } else {
        Write-Host "Launching: cargo run --features editor" -ForegroundColor Yellow
        Write-Host "Dir: $target" -ForegroundColor Yellow
        $inner = "Set-Location '$target'; cargo run --features editor; if (`$LASTEXITCODE -ne 0) { Write-Host ''; Write-Host 'Exited with error code '`$LASTEXITCODE -ForegroundColor Red; Read-Host 'Press Enter to close' }"
        Start-Process powershell -ArgumentList "-NoProfile", "-Command", $inner
    }

    Start-Sleep -Milliseconds 600
}
