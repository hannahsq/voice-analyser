#!/usr/bin/env bash
# D-040 amendment: 5a-R on 5a seed 1 (d031_rA_b20k, already trained): D-037's feature cache, then D-037's refit, three seeds.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d040 5a-R s1 $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
C=cache/d037/rA_s1
say "d040 5a-R s1: caching"
VA_DEVICE=cuda $PY -u scripts/analysis/d037_cache.py runs/d031_rA_b20k/ckpt_last.pt $C train val probe hb >> $C/cache.log 2>&1 || fail "cache"
pids=()
for s in 0 1 2; do
  mkdir -p runs/d040/5aR_s1_s$s
  VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -u scripts/analysis/d037_head.py $C h $s runs/d040/5aR_s1_s$s >> runs/d040/5aR_s1_s$s/train.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || fail "refit"; done
say "d040 5a-R s1: done"
