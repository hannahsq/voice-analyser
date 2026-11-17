#!/usr/bin/env bash
# D-123's registered check: the reconstruction loss's candidates scanned on the stores, the synthetic control and
# PTDB-TUG dev, branch-point models. No training. Skips when its report exists. Run a copy (runs/chains/), from main.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
export MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
mkdir -p reports/analysis/d123 runs/d123

say "d123: start"
if [ ! -s reports/analysis/d123/check.txt ]; then
  VA_DEVICE=cuda $PY -u scripts/analysis/d123_check.py > runs/d123/check.out 2> runs/d123/check.err < /dev/null || fail "d123 check"
  cp runs/d123/check.out reports/analysis/d123/check.txt
fi
say "d123: done"
