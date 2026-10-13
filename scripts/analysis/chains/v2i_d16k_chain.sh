#!/usr/bin/env bash
# After the v2i chain ("v2i: done"): the best and worst S2 v2i seeds (dev Hillenbrand F1 MAE s0 34.3, s1 36.5 Hz; also
# the lowest / highest women's closed-phase bias) continue their constant runs to 16000, then decay 4000, as the small
# M3 d16k runs. Measured against their own k35 (paired, closed phase), cp_dev alongside the small M3 d16k runs, and the
# identity-resynthesis test. Arg: scratch dir (closed-phase frames, prediction caches).
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
SA=scripts/analysis
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }

# wait for the v2i chain; stop if it logs a new failure
NF=$(grep -c FAILED $LOG)
until grep -q "v2i: done" $LOG; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "v2i chain failed; v2i d16k not started"; exit 1; }
  sleep 300
done
for SEED in 0 1; do
  N=v2i_conformer_s$SEED; R=runs/small/$N; RD=${R}_d16k
  mkdir -p $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    if [ ! -f $R/ckpt_016000.pt ]; then
      say "v2i d16k: $R constant 2000 -> 16000"
      $VA train teacher --data data/synth/train_v2i --steps 16000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
    fi
    [ -f $R/ckpt_016000.pt ] || fail "no ckpt_016000 in $R"
    say "v2i d16k: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
  closed ${N}_d16k $RD/ckpt_last.pt
  say "v2i d16k: ${N}_d16k measured"
done
$PY -I $SA/cp_dev.py "$S" v2i_conformer_s0,v2i_conformer_s1,v2i_conformer_s0_d16k,v2i_conformer_s1_d16k,m3_conformer_s0_d16k,m3_conformer_s2_d16k \
  v2i_conformer_s0_d16k-v2i_conformer_s0 v2i_conformer_s1_d16k-v2i_conformer_s1 \
  > $A/closed_phase/cp_small_v2i_d16k_dev.txt 2>&1 || fail "cp_dev v2i d16k"
R=runs/small
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  v2i_s0=$R/v2i_conformer_s0_k35/ckpt_last.pt v2i_s1=$R/v2i_conformer_s1_k35/ckpt_last.pt \
  v2i_s0_d16k=$R/v2i_conformer_s0_d16k/ckpt_last.pt v2i_s1_d16k=$R/v2i_conformer_s1_d16k/ckpt_last.pt \
  m3_s0_d16k=$R/m3_conformer_s0_d16k/ckpt_last.pt m3_s2_d16k=$R/m3_conformer_s2_d16k/ckpt_last.pt \
  > $A/sensitivity/resynth_identity_v2i_d16k.txt 2>&1 || say "v2i d16k: identity test failed (not fatal)"
say "v2i d16k: done"
