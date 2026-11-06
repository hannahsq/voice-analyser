#!/usr/bin/env bash
# D-082 (registered 2026-11-06): the closed-F1 shift on aspiration-free cascade audio. Render (CPU), v3.1 features,
# heads, decision. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d082 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
P=scripts/analysis/d082_probe.py
say "d082: start"
$PY $P render 24 >> runs/d082.log 2>&1 < /dev/null || fail "render"
say "d082: features"
$PY $P features v31_s1=runs/d040_5b_s1_b20k/ckpt_last.pt v31_s2=runs/d040_5b_s2_b20k/ckpt_last.pt >> runs/d082.log 2>&1 < /dev/null || fail "features"
say "d082: heads"
$PY $P heads v31_s1 v31_s2 >> runs/d082.log 2>&1 < /dev/null || fail "heads"
$PY $P decide v31_s1 v31_s2 >> runs/d082.log 2>&1 < /dev/null || fail "decide"
say "d082: $(grep '^DECISION' reports/analysis/d082/decide_d082.txt)"
say "d082: done"
