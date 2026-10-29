#!/usr/bin/env bash
# D-051 (registered 2026-10-30): roadmap 5c = 5b (D-040: rA + joint VTL head) + 30 % long articulation clips + the
# instantaneous-length head, seeds 1 and 2, each 16,000 constant steps then the 4,000-step decay branch, exactly as
# chains/d040.sh's 5b apart from the three --train-set additions. Prepares the long set and the labels first.
# Training only; scoring follows in its own script. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d051 $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
L=data/synth/train_long_v1
RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
V="--train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=1000 --train-set vtl_warmup=300
   --train-set vtl_gate_ramp=700 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001 --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0"
X="--train-set vtl_inst=true --train-set synth_extra=$L --train-set synth_extra_frac=0.3"

say "d051: start (data)"
$PY scripts/analysis/make_long_set.py $L 25000 5051 12 >> runs/make_long_v1.log 2>&1 || fail "long set"
[ -f $L/flat/done ] || $PY -c "from vocal_analysis.train.data import consolidate; consolidate('$L')" >> runs/make_long_v1.log 2>&1 || fail "consolidate"
[ -f $L/flat/label_length_cm.npy ] || $PY scripts/analysis/vtl_labels.py $L 12 >> runs/make_long_v1.log 2>&1 || fail "vtl labels long"
[ -f $L/flat/label_length_inst_cm.npy ] || $PY scripts/analysis/inst_labels.py $L 12 >> runs/make_long_v1.log 2>&1 || fail "inst labels long"
[ -f data/synth/train_v2/flat/label_length_inst_cm.npy ] || $PY scripts/analysis/inst_labels.py data/synth/train_v2 12 \
  >> runs/inst_labels_train_v2.log 2>&1 || fail "inst labels train_v2"
run() {  # name, seed
  local R=runs/d051_$1 B=runs/d051_$1_b20k
  mkdir -p $R $B
  if [ ! -f $R/ckpt_016000.pt ]; then
    say "d051: $R (5c, seed $2)"
    $VA train teacher --data data/synth/train_v2 $R2 $RV --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --optimizer muon_adan --train-set lr=0.002 --train-set rest_lr_scale=5 --train-set weight_decay=0.007 --train-set warmup=1000 \
      --train-set crop_frames=32,400 --train-set seed=$2 $V $X >> $R/train.log 2>&1 || fail "$R"
    [ -f $R/ckpt_016000.pt ] || fail "no step-16000 checkpoint in $R"
  fi
  if [ ! -f $B/ckpt_last.pt ]; then
    say "d051: $B (decay 4000 from 16000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B >> $B/train.log 2>&1 || fail "$B"
  fi
}
run 5c_s1 1
run 5c_s2 2
say "d051: training done"
