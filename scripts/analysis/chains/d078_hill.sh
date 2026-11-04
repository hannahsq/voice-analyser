#!/usr/bin/env bash
# D-078 addendum (2026-11-04): the registered Hillenbrand dev per-head bias the main chain left out. Each head paired
# against v3.1 of the same seed (m2_paired_sub.py, dev tiers), then signed per-group error from the caches.
# Arg: windowed scratch dir. Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d078_hill $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
F=reports/analysis/d078/paired
mkdir -p $F
say "d078_hill: start"
for s in 1 2; do
  for l in closed acoustic; do
    r=d078_${l}_s$s
    [ -s $F/d040_5b_s${s}_b20k_vs_$r.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" \
      d040_5b_s${s}_b20k=runs/d040_5b_s${s}_b20k/ckpt_last.pt $r=runs/$r/ckpt_last.pt > $F/d040_5b_s${s}_b20k_vs_$r.txt 2>&1 || fail "paired $r"
  done
done
$PY -I $SA/hill_signed.py "$SW/m2cache_sub" d040_5b_s1_b20k d040_5b_s2_b20k d078_closed_s1 d078_closed_s2 d078_acoustic_s1 d078_acoustic_s2 \
  > reports/analysis/d078/hillenbrand_signed.txt 2>&1 || fail "signed"
say "d078_hill: done"
