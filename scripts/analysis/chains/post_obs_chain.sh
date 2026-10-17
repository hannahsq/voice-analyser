#!/usr/bin/env bash
# After the D-027 head arm ("obs: done"), 2026-10-17 (roadmap items 1 and 5).
# 1. Replica pair: mix_conformer_s0's command for 50 steps twice on the same code (mixrep_a / mixrep_b); replica_check
#    a against b says how far GPU nondeterminism alone moves the step-50 loss parts (the D-027 check found bandwidth
#    -7.8 % against the old log).
# 2. D-028 long inputs on the GPU (fp32): long_test.py on data/synth/val_long_v2, cp_long.py on PTDB-TUG dev, for
#    teacher v3 (conformer_v2r2_s1_d16k), conformer_v2_s1_d16k and the GRU teacher_v2_d16k; decide_long.py on v3.
# Arg: scratch dir (closed-phase frames).
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
A=reports/analysis
SA=scripts/analysis
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
NF=$(grep -c FAILED $LOG)
until grep -q "obs: done" $LOG; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "an earlier chain failed; post-obs not started"; exit 1; }
  sleep 300
done
say "post-obs: replica pair"
for r in a b; do
  R=runs/small/mixrep_$r
  rm -rf $R; mkdir -p $R
  $VA train teacher --data data/synth/train_v2 --steps 50 --out $R $S2 --workers 4 --train-set seed=0 --train-set comb_truth_steps=1400 \
    --train-set crop_frames=32,400 --train-set eval_every=100000 --train-set ckpt_every=100000 >> $R/train.log 2>&1 || fail "$R"
done
$PY -I $SA/replica_check.py runs/small/mixrep_a runs/small/mixrep_b > $A/observability/replica_pair.txt 2>&1 || fail "replica pair"
say "post-obs: replica pair $(sed -n 's/^REPLICA: //p' $A/observability/replica_pair.txt)"
mkdir -p $A/long
M="v3=runs/conformer_v2r2_s1_d16k/ckpt_last.pt v2conf=runs/conformer_v2_s1_d16k/ckpt_last.pt v2gru=runs/teacher_v2_d16k/ckpt_last.pt"
say "post-obs: D-028 long inputs"
VA_DEVICE=cuda $PY -I $SA/long_test.py $M > $A/long/long_test.txt 2>&1 || fail "long_test"
VA_DEVICE=cuda $PY -I $SA/cp_long.py "$S" $M > $A/long/cp_long.txt 2>&1 || fail "cp_long"
$PY -I $SA/decide_long.py v3 $A/long/long_test.txt $A/long/cp_long.txt > $A/long/decide_long.txt 2>&1 || fail "decide_long"
say "post-obs: D-028 DECISION $(sed -n 's/^DECISION: //p' $A/long/decide_long.txt)"
say "post-obs: done"
