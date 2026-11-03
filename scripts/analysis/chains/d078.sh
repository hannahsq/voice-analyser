#!/usr/bin/env bash
# D-078 (registered 2026-11-04): two F1 heads (closed-glottis, cycle-mean acoustic labels) on teacher v3.1's frozen trunk,
# trained on train_v2k, head seeds 1 and 2; then scoring (held-out val_v2k, uncoupled and resynthesised validation sets),
# PTDB-TUG closed-phase frames, KL (D-060's renders) and the decision. Waits for train_v2k / val_v2k and for D-072 and
# D-073 to finish. Arg: windowed scratch dir. Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d078 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
O=reports/analysis/d078
mkdir -p $O

say "d078: waiting for the render and for d072 / d073"
until [ -f data/synth/train_v2k/flat/done ] && [ -f data/synth/val_v2k/flat/done ] \
      && grep -q "d072: done" $LOG && grep -q "d073: done" $LOG; do sleep 300; done
say "d078: start"
RUNS=""
for s in 1 2; do
  for l in closed acoustic; do
    r=d078_${l}_s$s; RUNS="$RUNS $r"
    [ -f runs/$r/ckpt_008000.pt ] || { say "d078: training $r"; $PY $SA/d078_heads.py $l $s > runs/$r.log 2>&1 || fail "train $r"; }
  done
done
say "d078: training done"
$PY $SA/d078_score.py $RUNS >> runs/d078_score.log 2>&1 || fail "score"
for r in $RUNS; do
  grep -q "'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
[ -s $O/kl/main_per_file.tsv ] || $PY $SA/d060_kl_gap.py main $O/kl $(for r in $RUNS; do printf '%s=runs/%s/ckpt_last.pt ' $r $r; done) \
  > runs/d078_kl.log 2>&1 || fail "kl"
say "d078: scoring done"
$PY -I $SA/decide_d078.py "$SW" > $O/decide_d078.txt 2>&1 || fail "decide"
say "d078: $(grep '^DECISION' $O/decide_d078.txt)"
say "d078: done"
