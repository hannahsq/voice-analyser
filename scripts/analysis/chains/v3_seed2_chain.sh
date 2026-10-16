#!/usr/bin/env bash
# D-025 (2026-10-17): third full-size seed pair for the teacher v3 candidate, then its semi-private confirmation.
# 1. conformer_v2_s2 (train_v2) and conformer_v2r2_s2 (train_v2 + VocalSet + Zeroth), seed 2, constant to 16000, d16k;
#    dev measures as for seeds 0 / 1; identity on all six full-size models.
# 2. semi-private (a logged use, D-024): the three pairs' paired reports and closed phase with VA_TIERS=semi;
#    decide_v3_semi.py (rule fixed before this chain ran). Arg: scratch dir.
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
FS=$A/paired/fp32_semi
SA=scripts/analysis
RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
mkdir -p $FS
paired() { VA_TIERS=${T:-dev} $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > ${OUT:-$F}/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }
full() {  # name, seed, data args... (as m3_speech_chain.sh)
  local N=$1 SEED=$2; shift 2
  local R=runs/$N RD=runs/${N}_d16k
  mkdir -p $R $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    if [ ! -f $R/ckpt_016000.pt ]; then
      say "v3s2: $R full-size Conformer (mixed crops, seed $SEED), constant to 16000"
      $VA train teacher "$@" --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
        --train-set crop_frames=32,400 --train-set seed=$SEED >> $R/train.log 2>&1 || fail "$R"
    fi
    [ -f $R/ckpt_016000.pt ] || fail "no ckpt_016000 in $R"
    say "v3s2: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  closed ${N}_d16k $RD/ckpt_last.pt
}
ck() { echo runs/$1/ckpt_last.pt; }

full conformer_v2_s2 2 --data data/synth/train_v2
full conformer_v2r2_s2 2 --data data/synth/train_v2 $R2 $RV
paired conformer_v2_s2_d16k $(ck conformer_v2_s2_d16k) conformer_v2r2_s2_d16k $(ck conformer_v2r2_s2_d16k)
M=conformer_v2_d16k,conformer_v2r2_d16k,conformer_v2_s1_d16k,conformer_v2r2_s1_d16k,conformer_v2_s2_d16k,conformer_v2r2_s2_d16k
PAIRS="conformer_v2r2_d16k-conformer_v2_d16k conformer_v2r2_s1_d16k-conformer_v2_s1_d16k conformer_v2r2_s2_d16k-conformer_v2_s2_d16k"
$PY -I $SA/cp_dev.py "$S" $M $PAIRS > $A/closed_phase/cp_full_seeds3_conformer_v2r2_dev.txt 2>&1 || fail "cp_dev"
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  $(for m in ${M//,/ }; do echo -n "$m=$(ck $m) "; done) > $A/sensitivity/resynth_identity_full_seeds3.txt 2>&1 || say "v3s2: identity test failed (not fatal)"
say "v3s2: seed 2 pair measured (dev)"

# 2. semi-private
say "v3s2: semi-private use (D-024): PTDB-TUG and Hillenbrand semi-private speakers, the three v3 candidate pairs"
for k in "" _s1 _s2; do
  T=semi OUT=$FS paired conformer_v2${k}_d16k $(ck conformer_v2${k}_d16k) conformer_v2r2${k}_d16k $(ck conformer_v2r2${k}_d16k)
done
VA_TIERS=semi $PY -I $SA/cp_dev.py "$S" $M $PAIRS > $A/closed_phase/cp_full_seeds3_conformer_v2r2_semi.txt 2>&1 || fail "cp semi"
$PY -I $SA/decide_v3_semi.py "$S" $F $FS > $A/decide_v3_semi.txt 2>&1 || fail "decide_v3_semi"
say "v3s2: semi-private DECISION: $(sed -n 's/^DECISION: //p' $A/decide_v3_semi.txt)"
say "v3s2: done"
