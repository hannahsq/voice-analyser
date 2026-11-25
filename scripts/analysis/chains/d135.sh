#!/usr/bin/env bash
# D-135 (registered 2026-11-25): CHIMERA renders (Story's shapes, CHIMERA's postures). Renders on the CPU, waits for the
# D-132 rerun's chain, scores teacher v3.1 (both seeds) and D-128's four students on the GPU, then the analysis.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d135 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
S=scripts/analysis/d135_chimera.py
R=reports/probes/chimera
say "d135: start"
[ -f data/probes/chimera_v1/posture/meta.json ] || $PY -I $S render > runs/d135_render.log 2>&1 || fail "render"
say "d135: rendered"
until grep -qE "d132v2: d134 re-read|FAILED: d132v2" "$LOG"; do sleep 30; done
for m in d040_5b_s1_b20k d040_5b_s2_b20k d128_L0_s0 d128_L0_s1 d128_L2_s0 d128_L2_s1; do
  [ -s $R/${m}.npz ] || VA_DEVICE=cuda $PY -I $S score $m runs/$m/ckpt_last.pt >> runs/d135_score.log 2>&1 || fail "score $m"
done
say "d135: scored"
[ -s reports/analysis/d135/d135.txt ] || $PY -I $S analyse > runs/d135_analyse.log 2>&1 || fail "analyse"
say "d135: done"
