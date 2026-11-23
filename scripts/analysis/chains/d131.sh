#!/usr/bin/env bash
# D-131 (registered 2026-11-23): the context and source checks on the teacher's low-f0 F1 overread. Renders on the CPU,
# scores teacher v3.1 (both seeds) and D-128's four students on the GPU, then the analysis. Every step skips if its
# output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d131 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
S=scripts/analysis/d131_context.py
R=reports/probes/vowelplane_d131
say "d131: start"
[ -f data/probes/vowelplane_d131/jit/meta.json ] || $PY -I $S render > runs/d131_render.log 2>&1 || fail "render"
say "d131: rendered"
for m in d040_5b_s1_b20k d040_5b_s2_b20k d128_L0_s0 d128_L0_s1 d128_L2_s0 d128_L2_s1; do
  [ -s $R/${m}_jit_raw.npz ] || VA_DEVICE=cuda $PY -I $S score $m runs/$m/ckpt_last.pt >> runs/d131_score.log 2>&1 || fail "score $m"
done
say "d131: scored"
[ -s reports/analysis/d131/d131.txt ] || $PY -I $S analyse > runs/d131_analyse.log 2>&1 || fail "analyse"
say "d131: done"
