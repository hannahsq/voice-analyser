#!/usr/bin/env bash
# D-065 (registered 2026-11-01): two arms against 5b, each exactly chains/d040.sh's 5b (rA + joint VTL head, 16,000
# constant steps then the 4,000-step decay branch) apart from one change, seeds 1 and 2:
#   gain: --train-set channel_gain_db=-40,0 (training crops; validation keeps -20 to 0)
#   bw:   train_bw_v1 (train_v2 re-rendered with bandwidths decoupled from the formants) in train_v2's place
#         (synth_extra_replace; train_v2 then gives only the synthetic validation, as for 5b)
# Order: gain s1, gain s2 ("d065: gain trained", for the scoring chain), then bw s1, bw s2. The bw set renders in
# its own process from the start (make_bw_set.py, 6 workers) and is waited for before bw's first run.
# Training only; scoring is d065_score.sh. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d065 $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
W=data/synth/train_bw_v1
RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
V="--train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=1000 --train-set vtl_warmup=300
   --train-set vtl_gate_ramp=700 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001 --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0"
run() {  # name, seed, extra --train-set flags
  local R=runs/d065_$1 B=runs/d065_$1_b20k
  mkdir -p $R $B
  if [ ! -f $R/ckpt_016000.pt ]; then
    say "d065: $R (seed $2)"
    $VA train teacher --data data/synth/train_v2 $R2 $RV --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --optimizer muon_adan --train-set lr=0.002 --train-set rest_lr_scale=5 --train-set weight_decay=0.007 --train-set warmup=1000 \
      --train-set crop_frames=32,400 --train-set seed=$2 $V $3 >> $R/train.log 2>&1 || fail "$R"
    [ -f $R/ckpt_016000.pt ] || fail "no step-16000 checkpoint in $R"
  fi
  if [ ! -f $B/ckpt_last.pt ]; then
    say "d065: $B (decay 4000 from 16000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B >> $B/train.log 2>&1 || fail "$B"
  fi
}

say "d065: start"
BWREADY=0; [ -f $W/flat/label_length_cm.npy ] && BWREADY=1
if [ $BWREADY = 0 ]; then
  rm -f runs/make_bw_v1.log
  ( { $PY scripts/analysis/make_bw_set.py $W 6 \
      && $PY -c "from vocal_analysis.train.data import consolidate
if __name__ == '__main__': consolidate('$W')" \
      && $PY scripts/analysis/vtl_labels.py $W 6; echo "exit $?"; } >> runs/make_bw_v1.log 2>&1 < /dev/null ) &
fi
G="--train-set channel_gain_db=-40,0"
run gain_s1 1 "$G"
run gain_s2 2 "$G"
say "d065: gain trained"
if [ $BWREADY = 0 ]; then
  until grep -q "^exit" runs/make_bw_v1.log 2>/dev/null; do sleep 300; done
  grep -q "^exit 0" runs/make_bw_v1.log || fail "bw set (runs/make_bw_v1.log)"
fi
X="--train-set synth_extra=$W --train-set synth_extra_replace=true"
run bw_s1 1 "$X"
run bw_s2 2 "$X"
say "d065: training done"
