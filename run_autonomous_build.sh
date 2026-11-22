#!/usr/bin/env bash
set -uo pipefail

FINAL_MARKER="${1:-ALL_MODULES_VERIFIED_100_PERCENT}"
MAX_SESSIONS="${2:-25}"
mkdir -p logs

[[ -f CLAUDE.md && -f PROMPT_MASTER.md ]] || { echo "Missing CLAUDE.md or PROMPT_MASTER.md"; exit 1; }
[[ -f PROGRESS.md ]] || printf "# Autonomous Build Progress Ledger\nStatus: INITIALIZED\n" > PROGRESS.md

for ((i = 1; i <= MAX_SESSIONS; i++)); do
    if grep -q "$FINAL_MARKER" PROGRESS.md; then
        echo "[SUCCESS] Marker '$FINAL_MARKER' confirmed in PROGRESS.md!"
        exit 0
    fi

    TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="logs/session_$(printf '%02d' "$i")_${TS}.log"
    echo "=== Autonomous Session $i / $MAX_SESSIONS ($TS) ==="

    PROMPT="Read CLAUDE.md, PROGRESS.md, and PROMPT_MASTER.md completely. Inspect git status/log, verify the last checkpoint, and continue executing the remaining modules strictly in order. Only after 'make verify-all' passes 100% with exit code 0 and zero todo!/unimplemented! stubs exist in crates/, append '$FINAL_MARKER' to PROGRESS.md and commit."

    claude -p "$PROMPT" 2>&1 | tee "$LOG_FILE"
    sleep 5
done

echo "[STOP] Reached $MAX_SESSIONS sessions without '$FINAL_MARKER'. Inspect PROGRESS.md."
exit 1
