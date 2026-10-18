#!/usr/bin/env bash
# 2026-10-18 (pre-registered before running; D-028 and roadmap 3).
# 0. D-028 windowed analysis: teacher v3 whole against 400-frame windows (m2_paired_sub with "<ckpt>@w400", cp_long.py),
#    decide_window.py.
# 1. Training recipe screen, S2 Conformer on train_v2, seeds 0-2 per arm against mixr_conformer_s0-2 (same command
#    otherwise: 2000 steps constant then 1500 decay, crops 32-400), decide_recipe.py per arm. Arms: learning rate x2,
#    x0.5, x3 (quality); warmup 250 (cost); decay over 20 % of the run, 2800 + 700 (cost); schedule-free AdamW, 3500 steps,
#    no decay (quality). "grey" arms get seeds 3-5 at the end (baselines mixr_conformer_s3-5 trained once), then n = 6.
# 2026-10-18 afternoon: $ARMS (e.g. "lr5_conformer lr8_conformer") runs only those arms, same rule (D-029).
# Arg: scratch dir (closed-phase frames, prediction caches).
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
F=$A/paired/fp32
SA=scripts/analysis
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
COMMON="--data data/synth/train_v2 $S2 --workers 4 --train-set comb_truth_steps=1400 --train-set crop_frames=32,400"
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }
run_arm_seed() {  # arm, seed
  local ARM=$1 SEED=$2 N=$1_s$2 R=runs/small/$1_s$2
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "recipe: $N"
    case $ARM in
      mixr_conformer) EX=""; ST=2000; DS=1500 ;;
      lr2_conformer) EX="--train-set lr=0.002"; ST=2000; DS=1500 ;;
      lr05_conformer) EX="--train-set lr=0.0005"; ST=2000; DS=1500 ;;
      lr3_conformer) EX="--train-set lr=0.003"; ST=2000; DS=1500 ;;
      lr5_conformer) EX="--train-set lr=0.005"; ST=2000; DS=1500 ;;
      lr8_conformer) EX="--train-set lr=0.008"; ST=2000; DS=1500 ;;
      wu250_conformer) EX="--train-set warmup=250"; ST=2000; DS=1500 ;;
      dec20_conformer) EX="--train-set ckpt_every=1400"; ST=2800; DS=700 ;;
      sf_conformer) EX="--optimizer schedulefree"; ST=0; DS=0 ;;
      *) fail "unknown arm $ARM" ;;
    esac
    if [ $ST = 0 ]; then  # schedule-free: one run of 3500 steps, no decay branch
      $VA train teacher $COMMON --steps 3500 --out ${R}_k35 $EX --train-set seed=$SEED >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    else
      CK=$(printf "%s/ckpt_%06d.pt" $R $ST)
      [ -f $CK ] || $VA train teacher $COMMON --steps $ST --out $R $EX --train-set seed=$SEED >> $R/train.log 2>&1 || fail "$R"
      [ -f $CK ] || fail "no $CK"
      $VA train teacher --branch-from $CK --decay-steps $DS --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    fi
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
  closed $N $(k35 $N)
  [ $ARM = mixr_conformer ] || paired mixr_conformer_s$SEED $(k35 mixr_conformer_s$SEED) $N $(k35 $N)
}
NF=$(grep -c FAILED $LOG)
# 0. windowed analysis
V3=runs/conformer_v2r2_s1_d16k/ckpt_last.pt
if [ -z "${ARMS:-}" ] && [ ! -f $A/long/decide_window.txt ]; then
  say "recipe: D-028 windowed analysis check"
  VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" v3=$V3 v3w=$V3@w400 > $F/v3_vs_v3w.txt 2>&1 || fail "paired v3 window"
  VA_DEVICE=cuda $PY -I $SA/cp_long.py "$S" v3=$V3 v3w=$V3@w400 > $A/long/cp_window.txt 2>&1 || fail "cp_long window"
  $PY -I $SA/decide_window.py $F/v3_vs_v3w.txt $A/long/cp_window.txt > $A/long/decide_window.txt 2>&1 || fail "decide_window"
  say "recipe: D-028 windowed DECISION $(sed -n 's/^DECISION: //p' $A/long/decide_window.txt)"
fi
# 1. arms, seeds 0-2
GREY=""
for ARM in ${ARMS:-lr2_conformer lr05_conformer lr3_conformer wu250_conformer dec20_conformer sf_conformer}; do
  for SEED in 0 1 2; do run_arm_seed $ARM $SEED; done
  $PY -I $SA/decide_recipe.py "$S" $ARM 3 > $A/recipe/decide_${ARM}_n3.txt 2>&1 || fail "decide $ARM"
  D=$(sed -n 's/^DECISION: //p' $A/recipe/decide_${ARM}_n3.txt)
  say "recipe: $ARM (3 seeds) DECISION $D"
  [ "$D" = grey ] && GREY="$GREY $ARM"
done
# 2. grey arms: seeds 3-5
if [ -n "$GREY" ]; then
  for SEED in 3 4 5; do run_arm_seed mixr_conformer $SEED; done
  for ARM in $GREY; do
    for SEED in 3 4 5; do run_arm_seed $ARM $SEED; done
    $PY -I $SA/decide_recipe.py "$S" $ARM 6 > $A/recipe/decide_${ARM}_n6.txt 2>&1 || fail "decide $ARM n6"
    say "recipe: $ARM (6 seeds) DECISION $(sed -n 's/^DECISION: //p' $A/recipe/decide_${ARM}_n6.txt)"
  done
fi
say "recipe: done"
