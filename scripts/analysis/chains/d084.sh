#!/usr/bin/env bash
# D-084 (registered 2026-11-06): formant links. S (the chain re-run, with its self-check) first, then M, then decide.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d084 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
P=scripts/analysis/d084_formant_links.py
say "d084: start"
$PY $P S v31_s1=runs/d040_5b_s1_b20k/ckpt_last.pt v31_s2=runs/d040_5b_s2_b20k/ckpt_last.pt >> runs/d084.log 2>&1 < /dev/null || fail "S"
say "d084: M"
$PY $P M v31_s1 v31_s2 >> runs/d084.log 2>&1 < /dev/null || fail "M"
$PY $P decide v31_s1 v31_s2 >> runs/d084.log 2>&1 < /dev/null || fail "decide"
say "d084: $(grep '^DECISION' reports/analysis/d084/decide_d084.txt)"
say "d084: done"
