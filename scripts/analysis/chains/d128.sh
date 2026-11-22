#!/usr/bin/env bash
# D-128 (registered 2026-11-22): teacher v3.1's causal copy by plain distillation. Arms L = 0 and L = 2 frames, seeds 0
# and 1, two runs at a time (L0 and L2 of one seed together); then the dev measures: closed-phase frames for each
# student (add_model.py), the paired windowed comparison against teacher v3.1 (m2_paired_sub.py, VA_TIERS=dev), and the
# closed-phase F1 bias by sex (cp_dev.py). Every step skips if its output exists; training resumes from ckpt_last.pt.
# Arg: windowed scratch dir holding ptdb_egg_sub.npz and m2cache_sub [cache/m3w].
set -u
SW="${1:-cache/m3w}"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d128 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
O=$A/d128
T=d040_5b_s2_b20k  # teacher v3.1
mkdir -p $F $O

train() {  # lookahead, seed
  local r=runs/d128_L$1_s$2
  [ -f $r/ckpt_008000.pt ] && return 0
  mkdir -p $r
  $VA train student --out $r --steps 8000 --workers 6 --lookahead $1 --attn-window 400 --train-set seed=$2 \
    >> $r/train.log 2>&1
}
closed() {  # run name
  grep -q "'$1'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || VA_DEVICE=cuda $PY -u $SA/add_model.py "$SW" $1 runs/$1/ckpt_last.pt _sub > "$SW/add_sub_$1.log" 2>&1 || fail "closed phase $1"
}
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt.tmp 2>&1 && mv $F/$1_vs_$2.txt.tmp $F/$1_vs_$2.txt || fail "paired $2"
}

say "d128: start"
for s in 0 1; do
  train 0 $s & P0=$!
  train 2 $s & P2=$!
  wait $P0 || fail "training L0 seed $s"
  wait $P2 || fail "training L2 seed $s"
  say "d128: seed $s trained"
done
RUNS="d128_L0_s0 d128_L2_s0 d128_L0_s1 d128_L2_s1"
closed $T
for r in $RUNS; do closed $r; done
for r in $RUNS; do paired $T $r; done
[ -s $O/cp_dev.txt ] || $PY -I $SA/cp_dev.py "$SW" $T,${RUNS// /,} $(for r in $RUNS; do printf "%s-%s " $r $T; done) \
  > $O/cp_dev.txt.tmp 2>&1 && mv $O/cp_dev.txt.tmp $O/cp_dev.txt || fail "cp_dev"
say "d128: done"
