#!/usr/bin/env bash
# Night of 2026-10-12/13 (55 W GPU cap), in order:
# 1. Small M3 to d16k: the best and worst S2 M3 seeds by dev Hillenbrand F1 MAE (s2 34.2, s0 37.3 Hz) continue their
#    constant runs to 16000, then decay 4000 (as the full-size d16k); measured against their own k35.
#    train_v2c2 renders alongside (CPU; D-024 pre-registered test).
# 2. v2c2: three S2 mixed-crop Conformer seeds (k35) on train_v2c2, against mix_conformer and v2c_conformer.
#    "night2: small done" in the log marks the end of the small runs.
# 3. Full-size Conformer on train_v2 alone (conformer_v2): constant to 16000, d16k branch (decay 4000), against
#    teacher_v2_d16k (architecture) and conformer_m3_d16k (data).
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
M3="--resynth data/resynth/vocalset_train_d16k --resynth-val data/resynth/vocalset_val_d16k"
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }

# train_v2c2 render, in the background while stage 1 trains
D=data/synth/train_v2c2
RENDER=
if [ ! -f $D/index.json ]; then
  say "night2: generating $D (100k clips, seed 2, p_coupling 1, coupling_full_poles 1, 24 processes)"
  $PY -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('$D', 100_000, replace(SynthConfig(), p_coupling=1.0, coupling_full_poles=1), seed=2, workers=24,
          extra_index={'note': 'train_v2c with full-strength F1 / B1 coupling (D-024 v2c2 test)'})
" > "$S/gen_v2c2.log" 2>&1 &
  RENDER=$!
fi

# 1. small M3 to d16k
for SEED in 2 0; do
  N=m3_conformer_s$SEED; R=runs/small/$N; RD=${R}_d16k
  mkdir -p $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    if [ ! -f $R/ckpt_016000.pt ]; then
      say "night2: $R constant 2000 -> 16000"
      $VA train teacher --data data/synth/train_v2 $M3 --steps 16000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
    fi
    [ -f $R/ckpt_016000.pt ] || fail "no ckpt_016000 in $R"
    say "night2: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
  closed ${N}_d16k $RD/ckpt_last.pt
  say "night2: ${N}_d16k measured"
done
$PY -I $SA/cp_dev.py "$S" m3_conformer_s0,m3_conformer_s2,m3_conformer_s0_d16k,m3_conformer_s2_d16k \
  m3_conformer_s0_d16k-m3_conformer_s0 m3_conformer_s2_d16k-m3_conformer_s2 > $A/closed_phase/cp_small_m3_d16k_dev.txt 2>&1 || fail "cp_dev small m3 d16k"

# 2. v2c2
if [ -n "$RENDER" ]; then wait $RENDER || fail "generate train_v2c2"; fi
[ -f $D/index.json ] || fail "train_v2c2 missing"
cp $D/index.json docs/synth_sets/train_v2c2.index.json
say "night2: train_v2c2 ready"
for SEED in 0 1 2; do
  N=v2c2_conformer_s$SEED; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "night2: $R (constant to 2000, decay 1500; crop_frames 32-400)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data $D --steps 2000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
  paired mix_conformer_s$SEED $(k35 mix_conformer_s$SEED) $N $(k35 $N)
  paired v2c_conformer_s$SEED $(k35 v2c_conformer_s$SEED) $N $(k35 $N)
  closed $N $(k35 $N)
  say "night2: $N measured"
done
$PY -I $SA/cp_dev.py "$S" mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,v2c_conformer_s0,v2c_conformer_s1,v2c_conformer_s2,v2c2_conformer_s0,v2c2_conformer_s1,v2c2_conformer_s2 \
  v2c2_conformer_s0-mix_conformer_s0 v2c2_conformer_s1-mix_conformer_s1 v2c2_conformer_s2-mix_conformer_s2 \
  v2c2_conformer_s0-v2c_conformer_s0 v2c2_conformer_s1-v2c_conformer_s1 v2c2_conformer_s2-v2c_conformer_s2 \
  > $A/closed_phase/cp_small_v2c2_dev.txt 2>&1 || fail "cp_dev v2c2"
say "night2: small done"

# 3. full-size Conformer on train_v2 alone
R=runs/conformer_v2; RD=${R}_d16k
FULL="--model-set temporal=conformer"
mkdir -p $R $RD
if [ ! -f $RD/ckpt_last.pt ]; then
  if [ ! -f $R/ckpt_016000.pt ]; then
    say "night2: $R full-size Conformer (mixed crops) on train_v2, constant to 16000"
    $VA train teacher --data data/synth/train_v2 --steps 16000 --out $R $FULL --workers 8 \
      --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
  fi
  [ -f $R/ckpt_016000.pt ] || fail "no ckpt_016000 in $R"
  say "night2: $RD (decay 4000)"
  $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
fi
paired teacher_v2_d16k runs/teacher_v2_d16k/ckpt_last.pt conformer_v2_d16k $RD/ckpt_last.pt
paired conformer_v2_d16k $RD/ckpt_last.pt conformer_m3_d16k runs/conformer_m3_d16k/ckpt_last.pt
closed conformer_v2_d16k $RD/ckpt_last.pt
$PY -I $SA/cp_dev.py "$S" d16k,conformer_v2_d16k,conformer_m3_d16k conformer_v2_d16k-d16k conformer_m3_d16k-conformer_v2_d16k \
  > $A/closed_phase/cp_conformer_v2_d16k_dev.txt 2>&1 || fail "cp_dev conformer_v2"
say "night2: done"
