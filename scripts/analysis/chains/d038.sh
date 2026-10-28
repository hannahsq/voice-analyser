#!/usr/bin/env bash
# D-038 (pre-registered 2026-10-28): the VTL -> formants cascade. Arm 3's trunk decides (F's three seeds already trained by
# d038_prep.sh with the same code), arm 1's is reported. Three heads at a time on the GPU, 2 CPU threads each; then the scoring.
# Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d038 $*; stopping"; exit 1; }
export VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=./.venv/Scripts/python.exe
C=cache/d037
arm() {  # trunk, arm
  local pids=() R=runs/d038/$1
  mkdir -p $R
  for s in 0 1 2; do
    $PY -u scripts/analysis/d038_head.py $C/$1 $2 $s $R/${2}_s$s >> $R/${2}_s$s.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail "$1 $2"; done
}
say "d038: start"
for t in arm3 arm1; do
  [ -f $C/$t/mix/done ] && [ -f $C/$t/shift/done ] && [ -f $C/$t/cp/done ] || fail "$t caches missing"
  for a in F Fv Fi; do arm $t $a; done
  say "d038: scoring $t"
  [ -s reports/analysis/d038/d038_$t.json ] || $PY -u scripts/analysis/d038_score.py $C/$t runs/d038/$t reports/analysis/d038/d038_$t.txt \
    > reports/analysis/d038/d038_$t.out 2>&1 || fail "scoring $t"
  say "d038 $t: $(grep '^  DECISION Fv' reports/analysis/d038/d038_$t.txt)"
done
say "d038: done"
