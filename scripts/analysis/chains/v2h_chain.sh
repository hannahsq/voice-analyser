#!/usr/bin/env bash
# D-026 dose arm (pre-registered 2026-10-14, before rendering): train_v2h = train_v2i with p_interactive 0.5. Renders
# now (CPU); GPU work waits for "v2s1: done". Three S2 seeds (k35) against mix_conformer_s0-2; s0 / s1 to d16k;
# identity test; decide_v2h.py. On "v2h": full-size Conformer on train_v2h to d16k against conformer_v2_d16k and
# conformer_v2_s1_d16k, then "v2h: done" (m3_speech_chain.sh waits for it). Arg: scratch dir.
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

D=data/synth/train_v2h
if [ ! -f $D/index.json ]; then
  say "v2h: generating $D (100k clips, seed 2, p_interactive 0.5, 24 processes)"
  $PY -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('$D', 100_000, replace(SynthConfig(), p_interactive=0.5), seed=2, workers=24,
          extra_index={'note': 'train_v2 with the interactive source on half the tube clips (D-026 dose arm)'})
" > "$S/gen_v2h.log" 2>&1 || fail "generate train_v2h"
fi
cp $D/index.json docs/synth_sets/train_v2h.index.json
say "v2h: train_v2h ready"
NF=$(grep -c FAILED $LOG)
until grep -q "v2s1: done" $LOG; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "earlier chain failed"; exit 1; }
  sleep 300
done
for SEED in 0 1 2; do
  N=v2h_conformer_s$SEED; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "v2h: $R (constant to 2000, decay 1500; crop_frames 32-400)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data $D --steps 2000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
  fi
  paired mix_conformer_s$SEED $(k35 mix_conformer_s$SEED) $N $(k35 $N)
  closed $N $(k35 $N)
  say "v2h: $N measured"
done
for SEED in 0 1; do
  N=v2h_conformer_s$SEED; R=runs/small/$N; RD=${R}_d16k
  mkdir -p $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    say "v2h: $R constant 2000 -> 16000"
    [ -f $R/ckpt_016000.pt ] || $VA train teacher --data $D --steps 16000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
    say "v2h: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
  closed ${N}_d16k $RD/ckpt_last.pt
  say "v2h: ${N}_d16k measured"
done
R=runs/small
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  $(for s in 0 1 2; do echo -n "v2h_s$s=$R/v2h_conformer_s${s}_k35/ckpt_last.pt "; done) \
  v2h_s0_d16k=$R/v2h_conformer_s0_d16k/ckpt_last.pt v2h_s1_d16k=$R/v2h_conformer_s1_d16k/ckpt_last.pt \
  v2i_s0_d16k=$R/v2i_conformer_s0_d16k/ckpt_last.pt v2i_s1_d16k=$R/v2i_conformer_s1_d16k/ckpt_last.pt \
  > $A/sensitivity/resynth_identity_v2h_d16k.txt 2>&1 || fail "identity v2h"
M=mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,v2i_conformer_s0,v2i_conformer_s1,v2i_conformer_s2,v2h_conformer_s0,v2h_conformer_s1,v2h_conformer_s2,v2i_conformer_s0_d16k,v2i_conformer_s1_d16k,v2h_conformer_s0_d16k,v2h_conformer_s1_d16k
$PY -I $SA/cp_dev.py "$S" $M v2h_conformer_s0-mix_conformer_s0 v2h_conformer_s1-mix_conformer_s1 v2h_conformer_s2-mix_conformer_s2 \
  v2h_conformer_s0-v2i_conformer_s0 v2h_conformer_s1-v2i_conformer_s1 v2h_conformer_s2-v2i_conformer_s2 \
  v2h_conformer_s0_d16k-v2i_conformer_s0_d16k v2h_conformer_s1_d16k-v2i_conformer_s1_d16k > $A/closed_phase/cp_small_v2h_dev.txt 2>&1 || fail "cp_dev v2h"
$PY -I $SA/decide_v2h.py "$S" $F $A/sensitivity/resynth_identity_v2h_d16k.txt > $A/decide_v2h.txt 2>&1 || fail "decide_v2h"
DEC=$(sed -n 's/^DECISION: //p' $A/decide_v2h.txt)
say "v2h: DECISION: $DEC"
if [ "$DEC" = v2h ]; then
  R=runs/conformer_v2h; RD=${R}_d16k; mkdir -p $R $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    say "v2h: $R full-size Conformer (mixed crops) on train_v2h, constant to 16000"
    [ -f $R/ckpt_016000.pt ] || $VA train teacher --data $D --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    say "v2h: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  paired conformer_v2_d16k runs/conformer_v2_d16k/ckpt_last.pt conformer_v2h_d16k $RD/ckpt_last.pt
  closed conformer_v2h_d16k $RD/ckpt_last.pt
  $PY -I $SA/cp_dev.py "$S" d16k,conformer_v2_d16k,conformer_v2_s1_d16k,conformer_m3_d16k,conformer_v2h_d16k \
    conformer_v2h_d16k-conformer_v2_d16k conformer_v2h_d16k-conformer_v2_s1_d16k > $A/closed_phase/cp_conformer_v2h_d16k_dev.txt 2>&1 || fail "cp_dev conformer_v2h"
  VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt conformer_v2_d16k=runs/conformer_v2_d16k/ckpt_last.pt \
    conformer_v2_s1_d16k=runs/conformer_v2_s1_d16k/ckpt_last.pt conformer_v2h_d16k=$RD/ckpt_last.pt \
    > $A/sensitivity/resynth_identity_conformer_v2h.txt 2>&1 || say "v2h: identity test failed (not fatal)"
fi
say "v2h: done"
