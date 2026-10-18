#!/usr/bin/env bash
# D-030 (pre-registered 2026-10-18 evening, before running): Muon, Adan and Muon + Adan on the S2 Conformer, against
# the tuned AdamW recipe. Waits for the D-029 combination arm ("recipe: done" a third time).
# Reference: combo_conformer (8e-3, warmup 250, decay over 20 %) if its decision is "better" and it is not worse than
# lr8_conformer (compare_arms.py); otherwise lr8_conformer (8e-3, warmup 1000, 2000 + 1500).
# Arms (seeds 0-2, the reference's schedule): adan / muon / muonadan at the reference rate and at half of it
# (<name>h). decide_recipe.py against the reference (its rule; grey counts as not established, no extra seeds).
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
train_arm() {  # name, seed, lr, constant steps, decay steps, optimizer, warmup, comb-truth steps
  local N=$1_s$2 R=runs/small/$1_s$2
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "opt: $N (lr $3, $4 + $5, $6, warmup $7, comb truth $8)"
    local CK=$(printf "%s/ckpt_%06d.pt" $R $4)
    [ -f $CK ] || $VA train teacher $COMMON --steps $4 --out $R --optimizer $6 --train-set lr=$3 --train-set warmup=$7 \
      --train-set ckpt_every=$4 --train-set seed=$2 --train-set comb_truth_steps=$8 >> $R/train.log 2>&1 || fail "$R"
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
declare -A BEST
for O in adan muon muon_adan; do
  NAME=$(echo $O | tr -d _)
  for SC in 1 h; do
    [ $SC = 1 ] && LR=0.008 && ARM=${NAME}_conformer || { LR=0.004; ARM=${NAME}h_conformer; }
    for SEED in 0 1 2; do train_arm $ARM $SEED $LR $ST $DS $O $WU 1400; done
    $PY -I $SA/decide_recipe.py "$S" $ARM 3 $REF > $A/recipe/decide_${ARM}_n3.txt 2>&1 || fail "decide $ARM"
    say "opt: $ARM DECISION $(sed -n 's/^DECISION: //p' $A/recipe/decide_${ARM}_n3.txt)"
  done
  D1=$(sed -n 's/^DECISION: //p' $A/recipe/decide_${NAME}_conformer_n3.txt)
  DH=$(sed -n 's/^DECISION: //p' $A/recipe/decide_${NAME}h_conformer_n3.txt)
  rank() { case $1 in better) echo 3 ;; same) echo 2 ;; grey) echo 1 ;; *) echo 0 ;; esac; }
  if [ $(rank $DH) -gt $(rank $D1) ]; then BL=0.004; BD=$DH; else BL=0.008; BD=$D1; fi
  if [ "$BD" = better ] || [ "$BD" = same ]; then
    HALF=${NAME}_half_conformer
    for SEED in 0 1 2; do train_arm $HALF $SEED $BL $((ST / 2)) $((DS / 2)) $O $((WU / 2)) 700; done
    $PY -I $SA/decide_recipe.py "$S" $HALF 3 $REF > $A/recipe/decide_${HALF}_n3.txt 2>&1 || fail "decide $HALF"
    say "opt: $HALF (half budget, lr $BL) DECISION $(sed -n 's/^DECISION: //p' $A/recipe/decide_${HALF}_n3.txt)"
  fi
done
$PY - <<'PY' > $A/recipe/opt_speed.txt 2>&1
import json, glob, numpy as np
for arm in ["lr8_conformer", "combo_conformer", "adan_conformer", "adanh_conformer", "muon_conformer", "muonh_conformer", "muonadan_conformer", "muonadanh_conformer"]:
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
