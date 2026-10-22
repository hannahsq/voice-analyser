#!/usr/bin/env bash
# D-026 (2026-10-14): second full-size seed of the train_v2 baseline (conformer_v2_s1, constant to 16000, d16k), from the
# day-3 plan; useful whatever the fix. Paired and closed-phase reports against conformer_v2_d16k (dev). Arg: scratch dir.
set -u
S="$1"; cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe; PY=./.venv/Scripts/python.exe; SA=scripts/analysis; A=reports/analysis
R=runs/conformer_v2_s1; RD=${R}_d16k; mkdir -p $R $RD
if [ ! -f $RD/ckpt_last.pt ]; then
  if [ ! -f $R/ckpt_016000.pt ]; then
    say "v2s1: $R full-size Conformer (mixed crops, seed 1) on train_v2, constant to 16000"
    $VA train teacher --data data/synth/train_v2 --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
      --train-set crop_frames=32,400 --train-set seed=1 >> $R/train.log 2>&1 || fail "$R"
  fi
  say "v2s1: $RD (decay 4000)"
  $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
fi
VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" conformer_v2_d16k=runs/conformer_v2_d16k/ckpt_last.pt \
  conformer_v2_s1_d16k=$RD/ckpt_last.pt > $A/paired/fp32/conformer_v2_d16k_vs_conformer_v2_s1_d16k.txt 2>&1 || fail "paired v2s1"
VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" conformer_v2_s1_d16k $RD/ckpt_last.pt _sub > "$S/add_sub_conformer_v2_s1_d16k.log" 2>&1 || fail "closed v2s1"
$PY -I $SA/cp_dev.py "$S" conformer_v2_d16k,conformer_v2_s1_d16k conformer_v2_s1_d16k-conformer_v2_d16k \
  > $A/closed_phase/cp_conformer_v2_seeds_dev.txt 2>&1 || fail "cp_dev v2s1"
say "v2s1: done"
