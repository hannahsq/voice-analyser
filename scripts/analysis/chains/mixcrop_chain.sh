#!/usr/bin/env bash
# Mixed crop lengths (crop_frames 32-400, main fd1adeb) on S2, v2 recipe, seed 0: GRU and Conformer. Compared with the
# same-seed 200-frame runs (v2_s0, conf_v2_s0): length test, subset measures, probe, closed phase. Arg: scratch dir.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
A=reports/analysis
SA=../voice-analyser-m3/scripts/analysis
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64"
say "mixcrop: start"
for T in gru conformer; do
  N=mix_${T}_s0; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "mixcrop: $R (constant to 2000, decay 1500; crop_frames 32-400)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/train_v2 --steps 2000 --out $R $S2 --model-set temporal=$T --workers 16 \
      --train-set seed=0 --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 16 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
done
G=runs/small/mix_gru_s0_k35/ckpt_last.pt; C=runs/small/mix_conformer_s0_k35/ckpt_last.pt
$PY -I "$SA/len_test.py" v2_s0=runs/small/v2_s0_k35/ckpt_last.pt mix_gru_s0=$G conf_v2_s0=runs/small/conf_v2_s0_k35/ckpt_last.pt mix_conformer_s0=$C > $A/sensitivity/len_test_mixcrop.txt 2>&1 || fail "length test"
$PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" v2_s0=runs/small/v2_s0_k35/ckpt_last.pt mix_gru_s0=$G > $A/paired/small_mix_gru_s0.txt 2>&1 || fail "paired mix_gru_s0"
$PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" v2_s0=runs/small/v2_s0_k35/ckpt_last.pt mix_conformer_s0=$C > $A/paired/small_mix_conformer_s0.txt 2>&1 || fail "paired mix_conformer_s0"
$PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" mix_gru_s0=$G mix_conformer_s0=$C > $A/paired/small_mix_conformer_vs_gru.txt 2>&1 || fail "paired mix conformer vs gru"
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/invariance_run.py v2_s0=small/v2_s0_k35/ckpt_last.pt mix_gru_s0=small/mix_gru_s0_k35/ckpt_last.pt conf_v2_s0=small/conf_v2_s0_k35/ckpt_last.pt mix_conformer_s0=small/mix_conformer_s0_k35/ckpt_last.pt > ../voice-analyser/$A/invariance/small_mixcrop.txt 2>&1) || fail "mixcrop probe"
say "mixcrop: GPU work done"
for N in mix_gru_s0 mix_conformer_s0; do
  (cd ../voice-analyser-m3 && VA_THREADS=8 PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" $N "../voice-analyser/runs/small/${N}_k35/ckpt_last.pt" _sub > "$S/add_sub_$N.log" 2>&1) || fail "closed phase $N"
done
(cd ../voice-analyser-m3 && CP_SUFFIX=_sub ../voice-analyser/.venv/Scripts/python.exe -I scripts/analysis/cp_cmp.py "$S" v2_s0,v2_s1,v2_s2,mix_gru_s0,conf_v2_s0,mix_conformer_s0 mix_gru_s0-v2_s0 mix_conformer_s0-conf_v2_s0 mix_conformer_s0-mix_gru_s0 > ../voice-analyser/$A/closed_phase/cp_small_mixcrop.txt 2>&1) || fail "cp_cmp mixcrop"
say "mixcrop: done"
