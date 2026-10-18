#!/usr/bin/env bash
# D-030 (pre-registered 2026-10-18 evening, before running): Muon, Adan and Muon + Adan on the S2 Conformer, against
# the tuned AdamW recipe. Waits for the D-029 combination arm ("recipe: done" a third time).
# Reference: combo_conformer (8e-3, warmup 250, decay over 20 %) if its decision is "better" and it is not worse than
# lr8_conformer (compare_arms.py); otherwise lr8_conformer (8e-3, warmup 1000, 2000 + 1500).
# Arms (seeds 0-2, the reference's schedule; amended 2026-10-18 18:20 before any arm ran, Hannah: Adan tolerates larger
# rates): Adan at x1 / x2 / x5 of 8e-3; Muon (+ AdamW) at 8e-3 and 4e-3; Muon + Adan with Muon at 8e-3 / 4e-3 and its
# Adan part at Adan's best rate (rest_lr_scale). Best arm per optimiser: pick_best_arm.py. decide_recipe.py against the
# reference (its rule; grey counts as not established, no extra seeds).
# Speed: for each optimiser whose better-decided rate is "better" or "same" (the reference rate on a tie), a
# half-budget arm (<name>_half: every step count halved: warmup, comb teacher forcing, constant and decay phases)
# against the reference at full budget, same
# rule: "same" or "better" means it reaches the reference's quality in half the steps. Wall-clock (audio s/s over the
# constant phase) reported per arm.
# Arg: scratch dir.
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
COMMON="--data data/synth/train_v2 $S2 --workers 4 --train-set crop_frames=32,400"
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }
train_arm() {  # name, seed, lr, constant steps, decay steps, optimizer, warmup, comb-truth steps, rest_lr_scale
  local N=$1_s$2 R=runs/small/$1_s$2
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "opt: $N (lr $3, $4 + $5, $6, warmup $7, comb truth $8, rest x${9:-1})"
    local CK=$(printf "%s/ckpt_%06d.pt" $R $4)
    [ -f $CK ] || $VA train teacher $COMMON --steps $4 --out $R --optimizer $6 --train-set lr=$3 --train-set warmup=$7 \
      --train-set ckpt_every=$4 --train-set seed=$2 --train-set comb_truth_steps=$8 --train-set rest_lr_scale=${9:-1} >> $R/train.log 2>&1 || fail "$R"
    [ -f $CK ] || fail "no $CK"
    $VA train teacher --branch-from $CK --decay-steps $5 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
  fi
  closed $N $(k35 $N)
  paired ${REF}_s$2 $(k35 ${REF}_s$2) $N $(k35 $N)
}
until [ "$(grep -c 'recipe: done' $LOG)" -ge 3 ]; do sleep 300; done
[ -f $A/recipe/decide_combo_conformer_n3.txt ] || fail "no combination decision"
$PY -I $SA/compare_arms.py combo_conformer lr8_conformer > $A/recipe/combo_vs_lr8.txt 2>&1 || fail "compare combo"
if [ "$(sed -n 's/^DECISION: //p' $A/recipe/decide_combo_conformer_n3.txt)" = better ] && grep -q "NOTWORSE: yes" $A/recipe/combo_vs_lr8.txt; then
  REF=combo_conformer; WU=250; ST=2800; DS=700
else
  REF=lr8_conformer; WU=1000; ST=2000; DS=1500
fi
say "opt: reference $REF (warmup $WU, $ST + $DS)"
run_set() {  # arm, lr, optimizer, rest_lr_scale: seeds 0-2 at full budget, decide against the reference
  for SEED in 0 1 2; do train_arm $1 $SEED $2 $ST $DS $3 $WU 1400 $4; done
  $PY -I $SA/decide_recipe.py "$S" $1 3 $REF > $A/recipe/decide_$1_n3.txt 2>&1 || fail "decide $1"
  say "opt: $1 (lr $2, $3, rest x$4) DECISION $(sed -n 's/^DECISION: //p' $A/recipe/decide_$1_n3.txt)"
}
lr_of() { case $1 in adan_conformer|muon_conformer|muonadan_conformer) echo 0.008 ;; adan2_conformer) echo 0.016 ;;
  adan5_conformer) echo 0.04 ;; muonh_conformer|muonadanh_conformer) echo 0.004 ;; esac; }
speed() {  # optimizer, rest_lr_scale for the given best arm's rate, arms...
  local O=$1; shift
  read BEST BD <<< "$($PY -I $SA/pick_best_arm.py $A/recipe "$@")"
  say "opt: best $O arm $BEST ($BD)"
  if [ "$BD" = better ] || [ "$BD" = same ]; then
    local LR=$(lr_of $BEST) RS=1
    [ $O = muon_adan ] && RS=$($PY -c "print($ADAN_LR / $LR)")
    local H=${BEST%_conformer}_half_conformer
    for SEED in 0 1 2; do train_arm $H $SEED $LR $((ST / 2)) $((DS / 2)) $O $((WU / 2)) 700 $RS; done
    $PY -I $SA/decide_recipe.py "$S" $H 3 $REF > $A/recipe/decide_${H}_n3.txt 2>&1 || fail "decide $H"
    say "opt: $H (half budget) DECISION $(sed -n 's/^DECISION: //p' $A/recipe/decide_${H}_n3.txt)"
  fi
}
# 1. Adan alone: x1, x2, x5 the AdamW rate (Adan tolerates larger rates; Xie et al. use 5-10x Adam's)
run_set adan_conformer 0.008 adan 1
run_set adan2_conformer 0.016 adan 1
run_set adan5_conformer 0.04 adan 1
read ADAN_BEST _ <<< "$($PY -I $SA/pick_best_arm.py $A/recipe adan_conformer adan2_conformer adan5_conformer)"
ADAN_LR=$(lr_of $ADAN_BEST)
# 2. Muon (+ AdamW for the rest): 8e-3 and 4e-3
run_set muon_conformer 0.008 muon 1
run_set muonh_conformer 0.004 muon 1
# 3. Muon + Adan: Muon at 8e-3 / 4e-3, its Adan part at Adan's best rate
run_set muonadan_conformer 0.008 muon_adan $($PY -c "print($ADAN_LR / 0.008)")
run_set muonadanh_conformer 0.004 muon_adan $($PY -c "print($ADAN_LR / 0.004)")
# 4. speed: half budget for each optimiser's best arm, if it is better or same
speed adan adan_conformer adan2_conformer adan5_conformer
speed muon muon_conformer muonh_conformer
speed muon_adan muonadan_conformer muonadanh_conformer
$PY - <<'PY' > $A/recipe/opt_speed.txt 2>&1
import json, glob, numpy as np
for arm in ["lr8_conformer", "combo_conformer", "adan_conformer", "adan2_conformer", "adan5_conformer", "muon_conformer", "muonh_conformer", "muonadan_conformer", "muonadanh_conformer"]:
    v = []
    for k in range(3):
        try:
            recs = [json.loads(x) for x in open(f"runs/small/{arm}_s{k}/log.jsonl")]
        except FileNotFoundError:
            continue
        v += [r["audio_s_per_s"] for r in recs if "audio_s_per_s" in r and r["step"] > 200]
    if v:
        print(f"{arm:22s} audio s/s over the constant phase: median {np.median(v):.0f}")
PY
say "opt: done"
