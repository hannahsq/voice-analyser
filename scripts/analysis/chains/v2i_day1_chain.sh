#!/usr/bin/env bash
# D-026 day 1 (pre-registered 2026-10-13, before any v2t result was read), after "v2i d16k: done":
# 1. Three more S2 seeds (3-5) per arm: mix_conformer (train_v2), v2i_conformer (train_v2i), v2t_conformer (train_v2t);
#    paired against mix_conformer and v2t - v2i, closed phase; cp_dev over all 18; decide_v2i.py (six seeds per arm).
# 2. On DECISION v2i or v2t: full-size Conformer (mixed crops) on that set, constant to 16000 then d16k (decay 4000),
#    against conformer_v2_d16k (same architecture and recipe, train_v2); closed phase; identity resynthesis.
#    On "fix" or "v2" the chain stops after step 1 (logged).
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
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }

NF=$(grep -c FAILED $LOG)
until grep -q "v2i d16k: done" $LOG; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "earlier chain failed; day 1 not started"; exit 1; }
  sleep 300
done

# 1. seeds 3-5
for SEED in 3 4 5; do
  for spec in "mix train_v2" "v2i train_v2i" "v2t train_v2t"; do
    set -- $spec; N=$1_conformer_s$SEED; R=runs/small/$N
    mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "day1: $R (constant to 2000, decay 1500; crop_frames 32-400)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/$2 --steps 2000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
    fi
    closed $N $(k35 $N)
  done
  for a in v2i v2t; do paired mix_conformer_s$SEED $(k35 mix_conformer_s$SEED) ${a}_conformer_s$SEED $(k35 ${a}_conformer_s$SEED); done
  paired v2t_conformer_s$SEED $(k35 v2t_conformer_s$SEED) v2i_conformer_s$SEED $(k35 v2i_conformer_s$SEED)
  say "day1: seed $SEED measured"
done
M=$(for a in mix v2i v2t; do for k in 0 1 2 3 4 5; do echo -n "${a}_conformer_s$k,"; done; done)
$PY -I $SA/cp_dev.py "$S" ${M%,} $(for k in 0 1 2 3 4 5; do echo -n "v2i_conformer_s$k-mix_conformer_s$k v2t_conformer_s$k-mix_conformer_s$k v2i_conformer_s$k-v2t_conformer_s$k "; done) \
  > $A/closed_phase/cp_small_v2i_6seed_dev.txt 2>&1 || fail "cp_dev 6 seeds"
$PY -I $SA/decide_v2i.py "$S" $F > $A/decide_v2i.txt 2>&1 || fail "decide_v2i"
DEC=$(sed -n 's/^DECISION: //p' $A/decide_v2i.txt)
say "day1: six seeds done; DECISION: $DEC"
case "$DEC" in v2i|v2t) ;; *) say "day1: no full-size run (decision $DEC)"; say "day1: done"; exit 0 ;; esac

# 2. full-size on the chosen set
R=runs/conformer_$DEC; RD=${R}_d16k
mkdir -p $R $RD
if [ ! -f $RD/ckpt_last.pt ]; then
  if [ ! -f $R/ckpt_016000.pt ]; then
    say "day1: $R full-size Conformer (mixed crops) on train_$DEC, constant to 16000"
    $VA train teacher --data data/synth/train_$DEC --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
  fi
  [ -f $R/ckpt_016000.pt ] || fail "no ckpt_016000 in $R"
  say "day1: $RD (decay 4000)"
  $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
fi
paired conformer_v2_d16k runs/conformer_v2_d16k/ckpt_last.pt conformer_${DEC}_d16k $RD/ckpt_last.pt
closed conformer_${DEC}_d16k $RD/ckpt_last.pt
$PY -I $SA/cp_dev.py "$S" d16k,conformer_v2_d16k,conformer_m3_d16k,conformer_${DEC}_d16k conformer_${DEC}_d16k-conformer_v2_d16k \
  conformer_${DEC}_d16k-conformer_m3_d16k > $A/closed_phase/cp_conformer_${DEC}_d16k_dev.txt 2>&1 || fail "cp_dev conformer_$DEC"
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt conformer_v2_d16k=runs/conformer_v2_d16k/ckpt_last.pt \
  conformer_m3_d16k=runs/conformer_m3_d16k/ckpt_last.pt conformer_${DEC}_d16k=$RD/ckpt_last.pt \
  > $A/sensitivity/resynth_identity_conformer_${DEC}.txt 2>&1 || say "day1: identity test failed (not fatal)"
say "day1: done"
