#!/usr/bin/env bash
# D-085 smoke test (open item 6): (1) peak memory and step time of the heads arm's model at each crop length up to
# 1600 frames; (2) 400 steps of the heads arm from scratch, on the strength-1 sets (train_v2kb, train_long_v3: these
# test the code path only, not the run's data), to check every head's loss moves and the main tasks hold.
# Every step skips if its output exists.
set -u
cd "$(dirname "$0")/../../.."
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
fail() { say "FAILED: $*"; exit 1; }
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
R=runs/d085_smoke
mkdir -p $R
say "d085 smoke: start"
[ -s $R/memory.txt ] || $PY -u scripts/analysis/d085_probe_memory.py > $R/memory.txt 2>&1 < /dev/null || fail "d085 smoke memory"
say "d085 smoke: memory done"
if ! grep -q '"step": 400' $R/log.jsonl 2>/dev/null; then
  $VA train teacher --data data/synth/train_v2 \
    --resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k \
    --steps 400 --out $R --model-set temporal=conformer --model-set conformer_dim=192 --model-set embed=576 --workers 8 \
    --optimizer muon_adan --train-set lr=0.001333 --train-set rest_lr_scale=5 --train-set weight_decay=0.007 \
    --train-set warmup=1000 --train-set crop_frames=32,400 --train-set seed=1 \
    --train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=100 \
    --train-set vtl_warmup=100 --train-set vtl_gate_ramp=100 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001 \
    --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0 --train-set vtl_inst=true --train-set vtl_link=true \
    --train-set vtl_link_truth_steps=200 \
    --train-set acoustic=true --train-set upper=3 --train-set upper_lr=0.001 --train-set opening_detach=100 \
    --train-set opening_gate_ramp=100 --train-set aux_budget=0.25 \
    --train-set synth_extra=data/synth/train_v2kb --train-set synth_extra_frac=0.42 --train-set resynth_frac=0.517 \
    --train-set long_set=data/synth/train_long_v3 --train-set long_batch_frac=0.2 \
    --train-set eval_every=200 --train-set ckpt_every=200 --train-set val_clips=64 \
    >> $R/train.log 2>&1 < /dev/null || fail "d085 smoke train"
fi
say "d085 smoke: done"
