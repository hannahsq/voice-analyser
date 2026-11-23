#!/usr/bin/env bash
# D-130 (registered 2026-11-23): the vowel-plane sweep. Renders on the CPU at once; waits for D-128's chain to finish
# before using the GPU; scores teacher v3.1 (both seeds, the control) and D-128's four students; then the analysis.
# Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d130 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
S=scripts/analysis/d130_sweep.py
R=reports/probes/vowelplane
say "d130: start"
[ -f data/probes/vowelplane_v1/meta.json ] || $PY -I $S render > runs/d130_render.log 2>&1 || fail "render"
say "d130: rendered"
until grep -qE "d128: done|FAILED: d128" "$LOG"; do sleep 60; done
grep -q "FAILED: d128" "$LOG" && fail "D-128's chain failed first"
for m in d040_5b_s1_b20k d040_5b_s2_b20k d128_L0_s0 d128_L0_s1 d128_L2_s0 d128_L2_s1; do
  [ -s $R/${m}_raw.npz ] || VA_DEVICE=cuda $PY -I $S score $m runs/$m/ckpt_last.pt >> runs/d130_score.log 2>&1 || fail "score $m"
done
say "d130: scored"
[ -s reports/analysis/d130/d130.txt ] || $PY -I $S analyse d040_5b_s1_b20k,d040_5b_s2_b20k \
  L0=d128_L0_s0,d128_L0_s1 L2=d128_L2_s0,d128_L2_s1 > runs/d130_analyse.log 2>&1 || fail "analyse"
say "d130: done"
