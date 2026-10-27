#!/usr/bin/env bash
# D-039 (pre-registered 2026-10-28; exploratory, reported only): the fused VTL heads, four arms x three seeds on the GPU
# (minutes each, alongside other GPU work), then the scoring. Every step is skipped if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d039 $*; stopping"; exit 1; }
export VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=./.venv/Scripts/python.exe
R=runs/d039
mkdir -p $R reports/analysis/d039
say "d039: start"
for a in E ED F FD; do
  for s in 0 1 2; do
    [ -f $R/${a}_s$s/done ] && continue
    mkdir -p $R/${a}_s$s
    $PY -u scripts/analysis/d039_fusion.py $a $s $R/${a}_s$s >> $R/${a}_s$s/train.log 2>&1 || fail "$a s$s"
  done
done
say "d039: scoring"
[ -s reports/analysis/d039/d039.json ] || VA_DEVICE=cpu $PY -u scripts/analysis/d039_score.py reports/analysis/d039/d039.txt $R \
  > reports/analysis/d039/d039.out 2>&1 || fail "scoring"
say "d039: done"
