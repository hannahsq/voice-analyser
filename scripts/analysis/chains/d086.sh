#!/usr/bin/env bash
# D-086 (registered 2026-11-06): the formant chain with standardised fed means (std) and a truth ramp (ramp) in 5b's decay branch, seeds 2 and 1,
# std then ramp; then per-run scoring in parallel (score_run.sh),
# closed-phase frames and paired measures serially, the length measures (vtl_score_joint.py) and the decision.
# Arg: windowed scratch dir (closed-phase frames, m2cache_sub). Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d086 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d086
mkdir -p $F $O

say "d086: start"
for arm in std ramp; do
  for s in 2 1; do
    R=runs/d086_${arm}_s$s
    [ -f $R/ckpt_020000.pt ] || { say "d086: $arm, seed $s"; mkdir -p $R
      $VA train teacher --branch-from runs/d040_5b_s$s/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $R \
        --train-set fed_standardised=true --train-set fed_lr=0.001 $( [ $arm = ramp ] && echo --train-set fed_truth_steps=2000 ) >> $R/train.log 2>&1 || fail "train $arm s$s"; }
  done
done
say "d086: training done"
RUNS="d086_std_s2 d086_std_s1 d086_ramp_s2 d086_ramp_s1"
for r in $RUNS; do bash $SA/chains/score_run.sh $r & done
wait
for r in $RUNS; do
  grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
for s in 2 1; do
  paired d040_5b_s${s}_b20k d086_std_s$s
  paired d040_5b_s${s}_b20k d086_ramp_s$s
  paired d086_std_s$s d086_ramp_s$s
done
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k
for arm in std ramp; do
  C1=d086_${arm}_s1; C2=d086_${arm}_s2
  [ -s $O/vtl_joint_$arm.json ] || VA_SEED_MEANS="$arm=$C1+$C2;5b=$B1+$B2" VA_SEED_PAIRS="$arm-5b=$C1+$C2:$B1+$B2;s1=$C1:$B1;s2=$C2:$B2" \
    $PY -u $SA/vtl_score_joint.py $O/vtl_joint_$arm.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
    $C1=runs/$C1/ckpt_last.pt $C2=runs/$C2/ckpt_last.pt > $O/vtl_joint_$arm.out 2>&1 || fail "vtl scoring $arm"
done
say "d086: scoring done"
[ -f $SA/decide_d086.py ] || { say "d086: decide script missing; stopping before the decision"; exit 0; }
$PY -I $SA/decide_d086.py "$SW" > $O/decide_d086.txt 2>&1 || fail "decide"
say "d086: $(grep '^DECISION' $O/decide_d086.txt)"
say "d086: done"
