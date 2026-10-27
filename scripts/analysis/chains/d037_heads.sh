#!/usr/bin/env bash
# D-037 (pre-registered 2026-10-27): the VTL head refits, on the GPU alongside the caching (d037_prep.sh; small MLPs, so the two share it). Each
# trunk's heads start once its cache is complete; three seeds at a time, 2 CPU threads each (the machine has 32). Order: R3 and R3c (decisive),
# then RT, R1, R1c, R2. Heads resume from their checkpoints and skip when done. Then the scoring.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d037 heads $*; stopping"; exit 1; }
export VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=./.venv/Scripts/python.exe
C=cache/d037
R=runs/d037

wait_cache() { until [ -f $C/$1/hb/done ]; do grep -q "FAILED: d037 prep" <(tail -5 "$LOG") && [ ! -f $C/$1/hb/done ] && sleep 600; sleep 60; done; }

arm() {  # name, trunk, inputs
  wait_cache $2
  say "d037 heads: $1 ($2, $3)"
  local pids=()
  for s in 0 1 2; do
    mkdir -p $R/${1}_s$s
    $PY -u scripts/analysis/d037_head.py $C/$2 $3 $s $R/${1}_s$s >> $R/${1}_s$s/train.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail "$1"; done
}

mkdir -p $R
arm R3 arm3 h
arm R3c arm3 hF
arm RT tv3 h
arm R1 arm1 h
arm R1c arm1 hF
arm R2 arm2 h
say "d037 heads: scoring"
mkdir -p reports/analysis/d037
$PY -u scripts/analysis/d037_score.py reports/analysis/d037/d037.txt R3=$C/arm3:$R/R3 R3c=$C/arm3:$R/R3c RT=$C/tv3:$R/RT \
  R1=$C/arm1:$R/R1 R1c=$C/arm1:$R/R1c R2=$C/arm2:$R/R2 J3=$C/arm3 J1=$C/arm1 J2=$C/arm2 > reports/analysis/d037/d037.out 2>&1 || fail "scoring"
say "d037: $(grep '^  DECISION Q1' reports/analysis/d037/d037.txt)"
say "d037: $(grep '^  DECISION Q3' reports/analysis/d037/d037.txt)"
say "d037 heads: done"
