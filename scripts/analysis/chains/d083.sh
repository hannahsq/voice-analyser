#!/usr/bin/env bash
# D-083 (registered 2026-11-06): the aVTL head with a predicted iVTL track. Skips trunks already scored.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
say "d083: start"
./.venv/Scripts/python.exe scripts/analysis/d083_vtl_link.py v31_s1 v31_s2 >> runs/d083.log 2>&1 < /dev/null || { say "FAILED: d083; stopping"; exit 1; }
say "d083: $(grep '^DECISION' reports/analysis/d083/decide_d083.txt)"
say "d083: done"
