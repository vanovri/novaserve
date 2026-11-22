param(
    [string]$FinalMarker = "ALL_MODULES_VERIFIED_100_PERCENT",
    [int]$MaxSessions = 25,
    [int]$CooldownSeconds = 5
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "CLAUDE.md") -or -not (Test-Path "PROMPT_MASTER.md")) {
    Write-Error "Missing CLAUDE.md or PROMPT_MASTER.md in current workspace directory."
    exit 1
}

if (-not (Test-Path "PROGRESS.md")) {
    Set-Content -Path "PROGRESS.md" -Value "# Autonomous Build Progress Ledger`r`nStatus: INITIALIZED`r`n" -Encoding UTF8
}

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

for ($session = 1; $session -le $MaxSessions; $session++) {
    $progressContent = Get-Content -Path "PROGRESS.md" -Raw
    if ($progressContent -match $FinalMarker) {
        Write-Host "[SUCCESS] Final verification marker '$FinalMarker' detected in PROGRESS.md!" -ForegroundColor Green
        exit 0
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path "logs" ("session_{0:D2}_{1}.log" -f $session, $timestamp)

    $resumePrompt = @"
Read CLAUDE.md, PROGRESS.md, and PROMPT_MASTER.md completely.
1. Inspect git log -n 5, git status, and PROGRESS.md to identify the exact current module state.
2. Verify all previously completed modules still compile cleanly without warnings.
3. Continue autonomous implementation, fuzzing, benchmarking, and git commits strictly module-by-module through the final module.
4. ONLY AFTER running the full final end-to-end verification gate (make verify-all or cargo test --workspace --release + studio build) with EXIT CODE 0 and confirming zero todo!()/unimplemented!() remain anywhere, write the exact token '$FinalMarker' at the bottom of PROGRESS.md and commit it.
"@

    Write-Host "=== Starting Autonomous Session $session / $MaxSessions ($timestamp) ===" -ForegroundColor Cyan

    & claude -p $resumePrompt 2>&1 | Tee-Object -FilePath $logFile
    $exitCode = $LASTEXITCODE

    Write-Host "Session $session exited with code $exitCode. Cooling down ${CooldownSeconds}s before checking ledger..." -ForegroundColor Yellow
    Start-Sleep -Seconds $CooldownSeconds
}

Write-Error "Reached MaxSessions ($MaxSessions) without hitting '$FinalMarker'. Check logs/ and PROGRESS.md."
exit 1
