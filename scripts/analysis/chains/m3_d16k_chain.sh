#!/usr/bin/env bash
# conformer_m3's d16k branch (decay 4000 from ckpt_016000), measured on dev speakers against teacher_v2_d16k:
# paired Hillenbrand / PTDB-TUG measures, closed-phase frames on the GPU, and cp_dev (d16k and d06k pairs).
# Args: scratch dir (closed-phase frames, prediction caches).
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
SA=scripts/analysis
R=runs/conformer_m3_d16k
mkdir -p $R
if [ ! -f $R/ckpt_last.pt ]; then
  say "m3 d16k: $R (branch from runs/conformer_m3/ckpt_016000.pt, decay 4000)"
  $VA train teacher --branch-from runs/conformer_m3/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $R >> $R/train.log 2>&1 || fail "$R"
fi
VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" teacher_v2_d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  conformer_m3_d16k=$R/ckpt_last.pt > $A/paired/fp32/teacher_v2_d16k_vs_conformer_m3_d16k.txt 2>&1 || fail "paired d16k"
VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" conformer_m3_d16k $R/ckpt_last.pt _sub > "$S/add_sub_conformer_m3_d16k.log" 2>&1 \
  || fail "closed phase d16k"
$PY -I $SA/cp_dev.py "$S" d06k,d16k,conformer_m3_d06k,conformer_m3_d16k conformer_m3_d16k-d16k conformer_m3_d06k-d06k \
  conformer_m3_d16k-conformer_m3_d06k > $A/closed_phase/cp_m3_d16k_dev.txt 2>&1 || fail "cp_dev d16k"
say "m3 d16k: done"
