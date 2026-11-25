#!/usr/bin/env bash
# D-132 (registered 2026-11-25): the length head against f0 on fixed synthetic tracts. Renders on the CPU, scores
# teacher v3.1 (both seeds) and D-128's four students on the GPU, then the analysis. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d132 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
S=scripts/analysis/d132_length_f0.py
R=reports/probes/length_f0
say "d132: start"
[ -f data/probes/length_f0_v1/meta.json ] || $PY -I $S render > runs/d132_render.log 2>&1 || fail "render"
say "d132: rendered"
for m in d040_5b_s1_b20k d040_5b_s2_b20k d128_L0_s0 d128_L0_s1 d128_L2_s0 d128_L2_s1; do
  [ -s $R/${m}.npz ] || VA_DEVICE=cuda $PY -I $S score $m runs/$m/ckpt_last.pt >> runs/d132_score.log 2>&1 || fail "score $m"
done
say "d132: scored"
[ -s reports/analysis/d132/d132.txt ] || $PY -I $S analyse > runs/d132_analyse.log 2>&1 || fail "analyse"
say "d132: done"
