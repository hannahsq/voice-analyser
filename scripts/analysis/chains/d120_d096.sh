#!/usr/bin/env bash
# D-120 (D-071's stage B on the dev gold sets), then D-096 (repeatability and minimal detectable change). Inference only,
# teacher v3.1's two seeds. Steps skip when their output exists. Run a copy (runs/chains/), from the main checkout.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
export MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
mkdir -p reports/analysis/d120 reports/analysis/d096 runs/d120 runs/d096

say "d120: start"
if [ ! -s reports/analysis/d120/d120_stage_b.txt ]; then
  VA_DEVICE=cuda $PY -u $SA/d120_calibration_gold.py analyse >> runs/d120/analyse.log 2>&1 || fail "d120 analyse"
  $PY -u $SA/d120_calibration_gold.py report > runs/d120/report.out 2> runs/d120/report.err || fail "d120 report"
  cp runs/d120/report.out reports/analysis/d120/d120_stage_b.txt
fi
say "d120: done"

say "d096: start"
if [ ! -s reports/analysis/d096/d096.txt ]; then
  VA_DEVICE=cuda $PY -u $SA/d096_repeatability.py score >> runs/d096/score.log 2>&1 || fail "d096 score"
  $PY -u $SA/d096_repeatability.py report > runs/d096/report.out 2> runs/d096/report.err || fail "d096 report"
  cp runs/d096/report.out reports/analysis/d096/d096.txt
fi
say "d096: done"
