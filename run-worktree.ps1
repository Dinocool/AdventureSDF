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

function Show-Menu($items, $selected) {
    Clear-Host
    Write-Host "Select worktree to run (cargo run --features editor)" -ForegroundColor Cyan
    Write-Host "Up/Down to move, Enter to launch, Q to quit" -ForegroundColor DarkGray
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
    $done = $false
    while (-not $done) {
        Show-Menu $worktrees $sel
        $key = [System.Console]::ReadKey($true)
        switch ($key.Key) {
            "UpArrow"   { $sel = ($sel - 1 + $worktrees.Count) % $worktrees.Count }
            "DownArrow" { $sel = ($sel + 1) % $worktrees.Count }
            "Enter"     { $done = $true }
            "Q"         { exit 0 }
        }
    }

    $target = $worktrees[$sel]
    Write-Host ""
    Write-Host "Launching: cargo run --features editor" -ForegroundColor Yellow
    Write-Host "Dir: $target" -ForegroundColor Yellow

    $inner = "Set-Location '$target'; cargo run --features editor; if (`$LASTEXITCODE -ne 0) { Write-Host ''; Write-Host 'Exited with error code '`$LASTEXITCODE -ForegroundColor Red; Read-Host 'Press Enter to close' }"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $inner

    Start-Sleep -Milliseconds 600
}
