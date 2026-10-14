#!/usr/bin/env bash
# D-026 (pre-registered 2026-10-14, before train_v2s was rendered): replaces v2m_chain.sh after its three S2 seeds.
# Renders train_v2s now (CPU): train_v2m with Rosenberg closure sharpness log-uniform 1-3 per interactive clip.
# Finishes v2m (s0 / s1 to d16k, identity, decide_v2m.py); then v2s the same way; the shortcut probe on mix / v2i /
# v2m / v2s; select_base.py picks the base; a full-size Conformer on it to d16k against conformer_v2_d16k and
# conformer_v2_s1_d16k; "base: done" for m3_speech_chain.sh (auto). Arg: scratch dir.
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

D=data/synth/train_v2s
RENDER=
if [ ! -f $D/index.json ]; then
  say "sharp: generating $D (100k clips, seed 2, p_interactive 1, p_interactive_coupled 0.5, interactive_closure 1-3, 20 processes)"
  $PY -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('$D', 100_000, replace(SynthConfig(), p_interactive=1.0, p_interactive_coupled=0.5, interactive_closure=(1.0, 3.0)),
          seed=2, workers=20, extra_index={'note': 'train_v2m with Rosenberg closure sharpness 1-3 per interactive clip (D-026)'})
" > "$S/gen_v2s.log" 2>&1 &
  RENDER=$!
fi

arm() {  # name, data dir: three S2 seeds (k35), s0 / s1 to d16k, identity test, decision
  local N R RD
  for SEED in 0 1 2; do
    N=$1_conformer_s$SEED; R=runs/small/$N; mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "sharp: $R (constant to 2000, decay 1500; crop_frames 32-400)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data $2 --steps 2000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      paired mix_conformer_s$SEED $(k35 mix_conformer_s$SEED) $N $(k35 $N)
      closed $N $(k35 $N)
      say "sharp: $N measured"
    fi
  done
  for SEED in 0 1; do
    N=$1_conformer_s$SEED; R=runs/small/$N; RD=${R}_d16k; mkdir -p $RD
    if [ ! -f $RD/ckpt_last.pt ]; then
      say "sharp: $R constant 2000 -> 16000"
      [ -f $R/ckpt_016000.pt ] || $VA train teacher --data $2 --steps 16000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
      say "sharp: $RD (decay 4000)"
      $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
    fi
    paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
    closed ${N}_d16k $RD/ckpt_last.pt
    say "sharp: ${N}_d16k measured"
  done
  local Rs=runs/small
  VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
    $(for s in 0 1 2; do echo -n "$1_s$s=$Rs/$1_conformer_s${s}_k35/ckpt_last.pt "; done) \
    $1_s0_d16k=$Rs/$1_conformer_s0_d16k/ckpt_last.pt $1_s1_d16k=$Rs/$1_conformer_s1_d16k/ckpt_last.pt \
    v2i_s0_d16k=$Rs/v2i_conformer_s0_d16k/ckpt_last.pt v2i_s1_d16k=$Rs/v2i_conformer_s1_d16k/ckpt_last.pt \
    > $A/sensitivity/resynth_identity_$1_d16k.txt 2>&1 || fail "identity $1"
  local M=mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,v2i_conformer_s0,v2i_conformer_s1,v2i_conformer_s2,$1_conformer_s0,$1_conformer_s1,$1_conformer_s2,v2i_conformer_s0_d16k,v2i_conformer_s1_d16k,$1_conformer_s0_d16k,$1_conformer_s1_d16k
  $PY -I $SA/cp_dev.py "$S" $M $(for k in 0 1 2; do echo -n "$1_conformer_s$k-mix_conformer_s$k $1_conformer_s$k-v2i_conformer_s$k "; done) \
    $1_conformer_s0_d16k-v2i_conformer_s0_d16k $1_conformer_s1_d16k-v2i_conformer_s1_d16k > $A/closed_phase/cp_small_$1_dev.txt 2>&1 || fail "cp_dev $1"
  $PY -I $SA/decide_v2m.py "$S" $F $A/sensitivity/resynth_identity_$1_d16k.txt $1 > $A/decide_$1.txt 2>&1 || fail "decide $1"
  say "sharp: $1 DECISION: $(sed -n 's/^DECISION: //p' $A/decide_$1.txt)"
}

arm v2m data/synth/train_v2m
if [ -n "$RENDER" ]; then wait $RENDER || fail "generate train_v2s"; fi
[ -f $D/index.json ] || fail "train_v2s missing"
cp $D/index.json docs/synth_sets/train_v2s.index.json
arm v2s $D
Rs=runs/small
VA_DEVICE=cuda $PY -u $SA/shortcut_probe.py $A/sensitivity/shortcut_probe_sharp.txt mix \
  $(for a in mix v2i v2m v2s; do for s in 0 1 2; do echo -n "${a}_s$s=$Rs/${a}_conformer_s${s}_k35/ckpt_last.pt "; done; done) \
  > "$S/shortcut_probe_sharp.log" 2>&1 || fail "shortcut probe"
$PY -I $SA/select_base.py $A > $A/decide_base.txt 2>&1 || fail "select_base"
BASE=$(sed -n 's/^DECISION: //p' $A/decide_base.txt)
say "sharp: base DECISION: $BASE"
case "$BASE" in v2m|v2s) ;; *) say "sharp: no base carried forward"; say "base: done"; exit 0 ;; esac
R=runs/conformer_$BASE; RD=${R}_d16k; mkdir -p $R $RD
if [ ! -f $RD/ckpt_last.pt ]; then
  say "sharp: $R full-size Conformer (mixed crops) on train_$BASE, constant to 16000"
  [ -f $R/ckpt_016000.pt ] || $VA train teacher --data data/synth/train_$BASE --steps 16000 --out $R --model-set temporal=conformer \
    --workers 8 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
  say "sharp: $RD (decay 4000)"
  $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
fi
paired conformer_v2_d16k runs/conformer_v2_d16k/ckpt_last.pt conformer_${BASE}_d16k $RD/ckpt_last.pt
paired conformer_v2_s1_d16k runs/conformer_v2_s1_d16k/ckpt_last.pt conformer_${BASE}_d16k $RD/ckpt_last.pt
closed conformer_${BASE}_d16k $RD/ckpt_last.pt
$PY -I $SA/cp_dev.py "$S" d16k,conformer_v2_d16k,conformer_v2_s1_d16k,conformer_m3_d16k,conformer_${BASE}_d16k \
  conformer_${BASE}_d16k-conformer_v2_d16k conformer_${BASE}_d16k-conformer_v2_s1_d16k > $A/closed_phase/cp_conformer_${BASE}_d16k_dev.txt 2>&1 || fail "cp_dev conformer_$BASE"
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt conformer_v2_d16k=runs/conformer_v2_d16k/ckpt_last.pt \
  conformer_v2_s1_d16k=runs/conformer_v2_s1_d16k/ckpt_last.pt conformer_${BASE}_d16k=$RD/ckpt_last.pt \
  > $A/sensitivity/resynth_identity_conformer_${BASE}.txt 2>&1 || say "sharp: identity test failed (not fatal)"
say "base: done"
