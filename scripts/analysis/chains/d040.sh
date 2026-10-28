#!/usr/bin/env bash
# D-040 (pre-registered 2026-10-28): items 5a (rA control, seed 2) and 5b (rA + joint VTL head from scratch, seeds 1 and 2),
# each D-031's rA path: 16,000 constant steps, then the 4,000-step decay branch. Waits for D-038 to finish. Training only;
# the scoring follows in its own script. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d040 $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
V="--train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=1000 --train-set vtl_warmup=300
   --train-set vtl_gate_ramp=700 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001 --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0"
until grep -q "d038: done\|FAILED: d038" "$LOG"; do sleep 120; done
run() {  # name, seed, extra args
  local R=runs/d040_$1 B=runs/d040_$1_b20k
  mkdir -p $R $B
  if [ ! -f $R/ckpt_016000.pt ]; then
    say "d040: $R (rA, seed $2${3:+, + VTL head})"
    $VA train teacher --data data/synth/train_v2 $R2 $RV --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --optimizer muon_adan --train-set lr=0.002 --train-set rest_lr_scale=5 --train-set weight_decay=0.007 --train-set warmup=1000 \
      --train-set crop_frames=32,400 --train-set seed=$2 $3 >> $R/train.log 2>&1 || fail "$R"
    [ -f $R/ckpt_016000.pt ] || fail "no step-16000 checkpoint in $R"
  fi
  if [ ! -f $B/ckpt_last.pt ]; then
    say "d040: $B (decay 4000 from 16000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B >> $B/train.log 2>&1 || fail "$B"
  fi
}
say "d040: start"
run 5b_s1 1 "$V"
run 5a_s2 2 ""
run 5b_s2 2 "$V"
say "d040: training done"
