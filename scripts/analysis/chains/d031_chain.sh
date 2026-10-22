#!/usr/bin/env bash
# D-031 (pre-registered 2026-10-22): the D-030 recipe at full size, a rate bracket from scaling rules. Teacher v3's data
# (train_v2 + resynthesised VocalSet / Zeroth-Korean), seed 1, Muon + Adan (rest x5):
#   rA Muon 2e-3, wd 0.007; rB 4e-3, wd 0.0035; rC 8e-3, wd 0.00175: each constant to 16000, decay branches 4000 + 1000,
#   8000 + 2000, 16000 + 4000; control rB's rates with wd 0.01: constant to 8000, branches 4000 + 1000, 8000 + 2000.
# Reference conformer_v2r2_s1_d16k. Dev measures for every branch (paired, closed phase); decide_d031.py applies the rules.
# Arg: scratch dir (closed-phase frames, m2cache_sub). Every step is skipped if its output exists.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d031 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/fp32
mkdir -p $A/d031
RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
REF=conformer_v2r2_s1_d16k
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $REF=runs/$REF/ckpt_last.pt $1=runs/$1/ckpt_last.pt > $F/${REF}_vs_$1.txt 2>&1 || fail "paired $1"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 runs/$1/ckpt_last.pt _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }
trunk() {  # name, lr, wd, steps
  local R=runs/d031_$1
  mkdir -p $R
  [ -f $R/$(printf "ckpt_%06d.pt" $4) ] && return 0
  say "d031: $R (Muon $2, Adan x5, wd $3), constant to $4"
  $VA train teacher --data data/synth/train_v2 $R2 $RV --steps $4 --out $R --model-set temporal=conformer --workers 8 \
    --optimizer muon_adan --train-set lr=$2 --train-set rest_lr_scale=5 --train-set weight_decay=$3 --train-set warmup=1000 \
    --train-set crop_frames=32,400 --train-set seed=1 >> $R/train.log 2>&1 || fail "$R"
  [ -f $R/$(printf "ckpt_%06d.pt" $4) ] || fail "no step-$4 checkpoint in $R"
}
branch() {  # name, at, decay
  local P=runs/d031_$1 B=d031_$1_b$(printf "%02d" $(( ($2 + $3) / 1000 )))k
  if [ ! -f runs/$B/ckpt_last.pt ]; then
    say "d031: runs/$B (decay $3 from $2)"
    mkdir -p runs/$B
    $VA train teacher --branch-from $P/$(printf "ckpt_%06d.pt" $2) --decay-steps $3 --workers 8 --out runs/$B >> runs/$B/train.log 2>&1 || fail "runs/$B"
  fi
  [ -s $F/${REF}_vs_$B.txt ] || paired $B
  $PY -c "import numpy as np, sys; sys.exit(0 if '$B' in np.load(r'$S/ptdb_egg_sub.npz', allow_pickle=True).files else 1)" || closed $B
}
for arm in "rA 0.002 0.007" "rB 0.004 0.0035" "rC 0.008 0.00175"; do
  set -- $arm
  trunk $1 $2 $3 16000
  branch $1 4000 1000
  branch $1 8000 2000
  branch $1 16000 4000
done
trunk ctl 0.004 0.01 8000
branch ctl 4000 1000
branch ctl 8000 2000
$PY -I $SA/decide_d031.py "$S" > $A/d031/decide_d031.txt 2>&1 || fail "decide"
say "d031: DECISION $(grep '^DECISION' $A/d031/decide_d031.txt | tr '\n' ' ')"
say "d031: done"
