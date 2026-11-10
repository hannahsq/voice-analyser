#!/usr/bin/env bash
# D-085 (registered 2026-11-08; amendments 1 and 2, 2026-11-10): the screening run. Base and heads arms from scratch,
# seed 1, side by side on the GPU: D-040's 5b recipe at 1.5x width (16000 steps, then a 4000-step decay branch), D-080's
# coupled share from train_v2ku (strength drawn on [0, 1]), long crops from train_long_v3u. Then per-run scoring in
# parallel (score_run.sh), closed-phase frames and paired measures against 5b serially, the length measures, and the
# decision (decide_d085.py; the chain stops before it if the script is missing).
# Waits for the render (d085_render_draw.py). Arg: windowed scratch dir. Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d085 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d085
mkdir -p $F $O

say "d085: waiting for the render"
until grep -q "train_v2ku: done" runs/d085_render_draw.log 2>/dev/null; do sleep 300; done
for d in data/synth/train_long_v3u data/synth/train_v2ku; do
  [ -f $d/flat/clip_tube.npy ] || $PY -u $SA/vtl_labels.py $d 8 > runs/d085_labels.log 2>&1 < /dev/null || fail "vtl labels $d"
  [ -f $d/flat/label_length_inst_cm.npy ] || $PY -u $SA/inst_labels.py $d 8 >> runs/d085_labels.log 2>&1 < /dev/null || fail "inst labels $d"
done
say "d085: start"

RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
BASE="--model-set temporal=conformer --model-set conformer_dim=192 --model-set embed=576 --workers 8
  --optimizer muon_adan --train-set lr=0.001333 --train-set rest_lr_scale=5 --train-set weight_decay=0.007
  --train-set warmup=1000 --train-set crop_frames=32,400 --train-set seed=1
  --train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=1000
  --train-set vtl_warmup=300 --train-set vtl_gate_ramp=700 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001
  --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0 --train-set vtl_inst=true --train-set aux_budget=0.25
  --train-set synth_extra=data/synth/train_v2ku --train-set synth_extra_frac=0.42 --train-set resynth_frac=0.517
  --train-set long_set=data/synth/train_long_v3u --train-set long_batch_frac=0.2"
HEADS="--train-set acoustic=true --train-set vtl_link=true --train-set upper=3 --train-set upper_lr=0.001"

arm() {  # name, extra flags
  local R=runs/d085_$1 B=runs/d085_$1_b20k
  mkdir -p $R $B
  [ -f $R/ckpt_016000.pt ] || $VA train teacher --data data/synth/train_v2 $R2 $RV --steps 16000 --out $R $BASE $2 \
    >> $R/train.log 2>&1 < /dev/null || return 1
  [ -f $B/ckpt_020000.pt ] || $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B \
    >> $B/train.log 2>&1 < /dev/null || return 1
}
say "d085: training base and heads side by side"
arm base "" & P1=$!
arm heads "$HEADS" & P2=$!
wait $P1 || fail "train base"
wait $P2 || fail "train heads"
say "d085: training done"

RUNS="d085_base_b20k d085_heads_b20k"
for r in $RUNS; do bash $SA/chains/score_run.sh $r & done
wait
for r in $RUNS; do
  grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
for s in 1 2; do
  paired d040_5b_s${s}_b20k d085_base_b20k
  paired d040_5b_s${s}_b20k d085_heads_b20k
done
paired d085_base_b20k d085_heads_b20k
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k; CB=d085_base_b20k; CH=d085_heads_b20k
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="5b=$B1+$B2" VA_SEED_PAIRS="base-5b=$CB:$B1+$B2;heads-5b=$CH:$B1+$B2;heads-base=$CH:$CB" \
  $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
  $CB=runs/$CB/ckpt_last.pt $CH=runs/$CH/ckpt_last.pt > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
say "d085: scoring done"
[ -f $SA/decide_d085.py ] || { say "d085: decide script missing; stopping before the decision"; exit 0; }
$PY -I $SA/decide_d085.py "$SW" > $O/decide_d085.txt 2>&1 || fail "decide"
say "d085: done"
