#!/usr/bin/env bash
# D-133 (registered 2026-11-25): noise-only gaps. Renders on the CPU, scores
# teacher v3.1 (both seeds) and D-128's four students on the GPU, then the analysis. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d133 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
S=scripts/analysis/d133_noise_gaps.py
R=reports/probes/noise_gaps
say "d133: start"
[ -f data/probes/noise_gaps_v1/meta.json ] || $PY -I $S render > runs/d133_render.log 2>&1 || fail "render"
say "d133: rendered"
for m in d040_5b_s1_b20k d040_5b_s2_b20k d128_L0_s0 d128_L0_s1 d128_L2_s0 d128_L2_s1; do
  [ -s $R/${m}.npz ] || VA_DEVICE=cuda $PY -I $S score $m runs/$m/ckpt_last.pt >> runs/d133_score.log 2>&1 || fail "score $m"
done
say "d133: scored"
[ -s reports/analysis/d133/d133.txt ] || $PY -I $S analyse > runs/d133_analyse.log 2>&1 || fail "analyse"
say "d133: done"
