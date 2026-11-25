#!/usr/bin/env bash
# D-134 (registered 2026-11-25): the length head by vowel; KL renders. Renders on the CPU, scores
# teacher v3.1 (both seeds) and D-128's four students on the GPU, then the analysis. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d134 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
S=scripts/analysis/d134_vowel_length.py
R=reports/probes/vowel_length_kl
say "d134: start"
[ -f data/probes/vowel_length_kl_v1/meta.json ] || $PY -I $S render > runs/d134_render.log 2>&1 || fail "render"
say "d134: rendered"
for m in d040_5b_s1_b20k d040_5b_s2_b20k d128_L0_s0 d128_L0_s1 d128_L2_s0 d128_L2_s1; do
  [ -s $R/${m}.npz ] || VA_DEVICE=cuda $PY -I $S score $m runs/$m/ckpt_last.pt >> runs/d134_score.log 2>&1 || fail "score $m"
done
say "d134: scored"
[ -s reports/analysis/d134/d134.txt ] || $PY -I $S analyse > runs/d134_analyse.log 2>&1 || fail "analyse"
say "d134: done"
