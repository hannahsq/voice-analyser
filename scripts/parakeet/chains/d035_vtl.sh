#!/usr/bin/env bash
# D-035 VTL probe (pre-registered 2026-10-26): waits for D-031's chain to finish (its "d031: done" or a FAILED line), then
# teacher v3's features on the labelled head-cache crops (GPU), the T and Tc heads (three seeds each; P's are trained
# beforehand on CPU), and the scoring with the decision rule. Every step is skipped if its output exists.
# Run from anywhere: bash d035_vtl.sh
set -u
cd /c/github/voice-analyser
W=../voice-analyser-parakeet
LOG=runs/orchestrator.log
V=$W/cache/vtl
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d035 $*"; exit 1; }
export PYTHONPATH=/c/github/voice-analyser VA_DEVICE=cuda MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
PY=$W/.venv/Scripts/python.exe
P=$W/scripts/parakeet

until grep -q "d031: done\|FAILED: d031\|FAILED: runs/d031" "$LOG"; do sleep 60; done
say "d035: start (teacher features on the labelled crops)"
[ -f $V/train_T/done ] || $PY -u $P/vtl_cache.py $V train_T val probe hb > $V/cache_gpu.out 2>&1 || fail "vtl_cache"
for arm in T Tc P; do
  for s in 0 1 2; do
    [ -f $V/heads/${arm}_s$s/head.pt ] && continue
    say "d035: head $arm seed $s"
    VA_DEVICE=cuda $PY -u $P/vtl_head.py $arm $s $V/heads/${arm}_s$s >> $V/heads/$arm.out 2>&1 || fail "head $arm $s"
  done
done
mkdir -p $W/reports/analysis/vtl
say "d035: scoring"
VA_DEVICE=cpu $PY -u $P/vtl_score.py $V $W/reports/analysis/vtl/d035_probe.txt > $V/score.out 2>&1 || fail "score"
say "d035: $(grep '^DECISION' $W/reports/analysis/vtl/d035_probe.txt)"
say "d035: done"
