#!/usr/bin/env bash
# D-026 dose arm (pre-registered 2026-10-14, before rendering): train_v2m = train_v2i with p_interactive_coupled 0.5 (feedback on for half the interactive clips). Renders
# now (CPU); GPU work waits for "v2s1: done". Three S2 seeds (k35) against mix_conformer_s0-2; s0 / s1 to d16k;
# identity test; decide_v2m.py. On "v2m": full-size Conformer on train_v2m to d16k against conformer_v2_d16k and
# conformer_v2_s1_d16k, then "v2m: done" (m3_speech_chain.sh waits for it). Arg: scratch dir.
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

D=data/synth/train_v2m
if [ ! -f $D/index.json ]; then
  say "v2m: generating $D (100k clips, seed 2, p_interactive 1, p_interactive_coupled 0.5, 24 processes)"
  $PY -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('$D', 100_000, replace(SynthConfig(), p_interactive=1.0, p_interactive_coupled=0.5), seed=2, workers=24,
          extra_index={'note': 'train_v2i with the feedback on for half the interactive clips (D-026 dose arm)'})
" > "$S/gen_v2m.log" 2>&1 || fail "generate train_v2m"
fi
cp $D/index.json docs/synth_sets/train_v2m.index.json
say "v2m: train_v2m ready"
NF=$(grep -c FAILED $LOG)
until grep -q "v2s1: done" $LOG; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "earlier chain failed"; exit 1; }
  sleep 300
done
for SEED in 0 1 2; do
  N=v2m_conformer_s$SEED; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "v2m: $R (constant to 2000, decay 1500; crop_frames 32-400)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data $D --steps 2000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
  fi
  paired mix_conformer_s$SEED $(k35 mix_conformer_s$SEED) $N $(k35 $N)
  closed $N $(k35 $N)
  say "v2m: $N measured"
done
for SEED in 0 1; do
  N=v2m_conformer_s$SEED; R=runs/small/$N; RD=${R}_d16k
  mkdir -p $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    say "v2m: $R constant 2000 -> 16000"
    [ -f $R/ckpt_016000.pt ] || $VA train teacher --data $D --steps 16000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
    say "v2m: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
  closed ${N}_d16k $RD/ckpt_last.pt
  say "v2m: ${N}_d16k measured"
done
R=runs/small
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  $(for s in 0 1 2; do echo -n "v2m_s$s=$R/v2m_conformer_s${s}_k35/ckpt_last.pt "; done) \
  v2m_s0_d16k=$R/v2m_conformer_s0_d16k/ckpt_last.pt v2m_s1_d16k=$R/v2m_conformer_s1_d16k/ckpt_last.pt \
  v2i_s0_d16k=$R/v2i_conformer_s0_d16k/ckpt_last.pt v2i_s1_d16k=$R/v2i_conformer_s1_d16k/ckpt_last.pt \
  > $A/sensitivity/resynth_identity_v2m_d16k.txt 2>&1 || fail "identity v2m"
M=mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,v2i_conformer_s0,v2i_conformer_s1,v2i_conformer_s2,v2m_conformer_s0,v2m_conformer_s1,v2m_conformer_s2,v2i_conformer_s0_d16k,v2i_conformer_s1_d16k,v2m_conformer_s0_d16k,v2m_conformer_s1_d16k
$PY -I $SA/cp_dev.py "$S" $M v2m_conformer_s0-mix_conformer_s0 v2m_conformer_s1-mix_conformer_s1 v2m_conformer_s2-mix_conformer_s2 \
  v2m_conformer_s0-v2i_conformer_s0 v2m_conformer_s1-v2i_conformer_s1 v2m_conformer_s2-v2i_conformer_s2 \
  v2m_conformer_s0_d16k-v2i_conformer_s0_d16k v2m_conformer_s1_d16k-v2i_conformer_s1_d16k > $A/closed_phase/cp_small_v2m_dev.txt 2>&1 || fail "cp_dev v2m"
$PY -I $SA/decide_v2m.py "$S" $F $A/sensitivity/resynth_identity_v2m_d16k.txt > $A/decide_v2m.txt 2>&1 || fail "decide_v2m"
DEC=$(sed -n 's/^DECISION: //p' $A/decide_v2m.txt)
say "v2m: DECISION: $DEC"
if [ "$DEC" = v2m ]; then
  R=runs/conformer_v2m; RD=${R}_d16k; mkdir -p $R $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    say "v2m: $R full-size Conformer (mixed crops) on train_v2m, constant to 16000"
    [ -f $R/ckpt_016000.pt ] || $VA train teacher --data $D --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    say "v2m: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  paired conformer_v2_d16k runs/conformer_v2_d16k/ckpt_last.pt conformer_v2m_d16k $RD/ckpt_last.pt
  closed conformer_v2m_d16k $RD/ckpt_last.pt
  $PY -I $SA/cp_dev.py "$S" d16k,conformer_v2_d16k,conformer_v2_s1_d16k,conformer_m3_d16k,conformer_v2m_d16k \
    conformer_v2m_d16k-conformer_v2_d16k conformer_v2m_d16k-conformer_v2_s1_d16k > $A/closed_phase/cp_conformer_v2m_d16k_dev.txt 2>&1 || fail "cp_dev conformer_v2m"
  VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt conformer_v2_d16k=runs/conformer_v2_d16k/ckpt_last.pt \
    conformer_v2_s1_d16k=runs/conformer_v2_s1_d16k/ckpt_last.pt conformer_v2m_d16k=$RD/ckpt_last.pt \
    > $A/sensitivity/resynth_identity_conformer_v2m.txt 2>&1 || say "v2m: identity test failed (not fatal)"
fi
say "v2m: done"
