#!/usr/bin/env bash
# D-026 pre-registered test: train_v2i (interactive source on train_v2's tube clips) and its control train_v2t (feedback
# off), three S2 mixed-crop Conformer seeds each against mix_conformer_s0-2, dev speakers. train_v2t renders while the
# v2i seeds train. Arg: scratch dir (closed-phase frames, prediction caches).
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
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$(k35 $1) $2=$(k35 $2) > $F/$1_vs_$2.txt 2>&1 || fail "paired $1 vs $2"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $(k35 $1) _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }
gen() {  # name, coupled (True / False), workers
  [ -f data/synth/$1/index.json ] && return 0
  $PY -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('data/synth/$1', 100_000, replace(SynthConfig(), p_interactive=1.0, interactive_coupled=$2), seed=2, workers=$3,
          extra_index={'note': 'train_v2 with the interactive source on tube clips (coupled=$2; D-026 pre-registered test)'})
" > "$S/gen_$1.log" 2>&1
}
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
seeds() {  # set name, model prefix
  for SEED in 0 1 2; do
    N=$2_conformer_s$SEED; R=runs/small/$N
    mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "v2i: $R (constant to 2000, decay 1500; crop_frames 32-400)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/$1 --steps 2000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
    fi
    paired mix_conformer_s$SEED $N
    closed $N
    say "v2i: $N measured"
  done
}
say "v2i: generating data/synth/train_v2i (100k clips, seed 2, p_interactive 1, 32 processes)"
gen train_v2i True 32 || fail "generate train_v2i"
cp data/synth/train_v2i/index.json docs/synth_sets/train_v2i.index.json
say "v2i: train_v2i ready; generating train_v2t (feedback off, 16 processes) alongside the v2i seeds"
gen train_v2t False 16 &
GEN=$!
seeds train_v2i v2i
wait $GEN || fail "generate train_v2t"
[ -f data/synth/train_v2t/index.json ] || fail "train_v2t missing"
cp data/synth/train_v2t/index.json docs/synth_sets/train_v2t.index.json
say "v2i: train_v2t ready"
seeds train_v2t v2t
for SEED in 0 1 2; do paired v2t_conformer_s$SEED v2i_conformer_s$SEED; done
M=mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,v2i_conformer_s0,v2i_conformer_s1,v2i_conformer_s2,v2t_conformer_s0,v2t_conformer_s1,v2t_conformer_s2
$PY -I $SA/cp_dev.py "$S" $M v2i_conformer_s0-mix_conformer_s0 v2i_conformer_s1-mix_conformer_s1 v2i_conformer_s2-mix_conformer_s2 \
  v2t_conformer_s0-mix_conformer_s0 v2t_conformer_s1-mix_conformer_s1 v2t_conformer_s2-mix_conformer_s2 \
  v2i_conformer_s0-v2t_conformer_s0 v2i_conformer_s1-v2t_conformer_s1 v2i_conformer_s2-v2t_conformer_s2 \
  > $A/closed_phase/cp_small_v2i_dev.txt 2>&1 || fail "cp_dev v2i"
R=runs/small
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  $(for s in 0 1 2; do echo -n "mix_s$s=$R/mix_conformer_s${s}_k35/ckpt_last.pt v2i_s$s=$R/v2i_conformer_s${s}_k35/ckpt_last.pt v2t_s$s=$R/v2t_conformer_s${s}_k35/ckpt_last.pt "; done) \
  > $A/sensitivity/resynth_identity_v2i.txt 2>&1 || say "v2i: identity test failed (not fatal)"
say "v2i: done"
