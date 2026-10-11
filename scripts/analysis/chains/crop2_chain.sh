#!/usr/bin/env bash
# Crop-length follow-up (D-024), all measures in fp32 (main 69ccf9a), closed phase on the GPU, one thing at a time.
# A: re-measure the six S2 models the crop runs are compared with, in fp32 (paired vs v2_s0, probe).
# B: Conformer, mixed crops 32-400, seeds 1 and 2 (seed spread with this morning's seed 0); long-weighted mix
#    (crop_long_frac 2/3), seed 0. Each with paired (vs v2_s0, and vs mix_conformer_s0), closed phase; then the probe,
#    the length test and the closed-phase comparison over all. Arg: scratch dir.
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
SA=../voice-analyser-m3/scripts/analysis
F=$A/paired/fp32
mkdir -p $F
ck() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { $PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" $1=$(ck $1) $2=$(ck $2) > $F/$1_vs_$2.txt 2>&1 || fail "paired $1 vs $2"; }
probe() {  # name list -> invariance report
  local out=$1; shift
  (cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/invariance_run.py \
    $(for n in "$@"; do echo -n "$n=small/${n}_k35/ckpt_last.pt "; done) > ../voice-analyser/$A/invariance/$out 2>&1) || fail "probe $out"
}
say "crop2: start (A: fp32 re-measure)"
for n in v2_s1 v2_s2 conf_v2_s0 mix_gru_s0 mix_conformer_s0; do paired v2_s0 $n; done
probe small_fp32.txt v2_s0 v2_s1 v2_s2 conf_v2_s0 mix_gru_s0 mix_conformer_s0
say "crop2: A done"
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
for spec in "mix_conformer_s1 1 none" "mix_conformer_s2 2 none" "longw_conformer_s0 0 0.6667"; do
  set -- $spec; N=$1; SEED=$2; LF=$3; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "crop2: $R (constant to 2000, decay 1500; crop_frames 32-400, crop_long_frac $LF)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/train_v2 --steps 2000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 --train-set crop_long_frac=$LF >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
  paired v2_s0 $N
  paired mix_conformer_s0 $N
  (cd ../voice-analyser-m3 && VA_DEVICE=cuda PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" $N "../voice-analyser/$(ck $N)" _sub > "$S/add_sub_$N.log" 2>&1) || fail "closed phase $N"
done
probe small_crop2.txt v2_s0 conf_v2_s0 mix_conformer_s0 mix_conformer_s1 mix_conformer_s2 longw_conformer_s0
$PY -I "$SA/len_test.py" $(for n in v2_s0 conf_v2_s0 mix_conformer_s0 mix_conformer_s1 mix_conformer_s2 longw_conformer_s0; do echo -n "$n=$(ck $n) "; done) > $A/sensitivity/len_test_crop2.txt 2>&1 || fail "length test"
(cd ../voice-analyser-m3 && CP_SUFFIX=_sub ../voice-analyser/.venv/Scripts/python.exe -I scripts/analysis/cp_cmp.py "$S" v2_s0,v2_s1,v2_s2,conf_v2_s0,mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,longw_conformer_s0 \
  mix_conformer_s1-mix_conformer_s0 mix_conformer_s2-mix_conformer_s0 longw_conformer_s0-mix_conformer_s0 longw_conformer_s0-conf_v2_s0 > ../voice-analyser/$A/closed_phase/cp_small_crop2.txt 2>&1) || fail "cp_cmp crop2"
say "crop2: done"
