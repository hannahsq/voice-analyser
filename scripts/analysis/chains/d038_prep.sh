#!/usr/bin/env bash
# D-038 (draft) preparation: the mixed-kind training cache (d037_cache.py's mix part) for arm 3's trunk, F's three seeds on it
# for the noise floor (a registration check: the control only), then arm 1's mix cache. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d038 prep $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
C=cache/d037
R=runs/d038/arm3
say "d038 prep: start"
VA_DEVICE=cuda $PY -u scripts/analysis/d037_cache.py runs/d036_arm3/ckpt_last.pt $C/arm3 mix >> $C/arm3/mix.log 2>&1 || fail "mix arm3"
mkdir -p $R
pids=()
for s in 0 1 2; do
  VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -u scripts/analysis/d038_head.py $C/arm3 F $s $R/F_s$s > $R/F_s$s.log 2>&1 &
  pids+=($!)
done
VA_DEVICE=cuda $PY -u scripts/analysis/d037_cache.py runs/d036_arm1/ckpt_last.pt $C/arm1 mix >> $C/arm1/mix.log 2>&1 || fail "mix arm1"
for p in "${pids[@]}"; do wait $p || fail "F seeds"; done
[ -s reports/analysis/d038/noise_floor_F.txt ] || VA_DEVICE=cuda $PY -u scripts/analysis/d038_score.py $C/arm3 $R reports/analysis/d038/noise_floor_F.txt \
  > reports/analysis/d038/noise_floor_F.out 2>&1 || fail "noise floor scoring"
say "d038 prep: done"
