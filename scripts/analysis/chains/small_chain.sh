#!/usr/bin/env bash
# Small-model proxy (replaces the full-size 3.5k / 5k study), S2 (reports/analysis/bench_small.txt): 3 seeds per arm at 3.5k (2000 constant + 1500 decay) with the subset measures.
# Runs: runs/small/<arm>_s<seed>[_k35]. Reports: reports/analysis/{paired/small_*,
# invariance/small.txt}. v1's arm is skipped if runs/small/skip_v1 exists when it is reached.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
A=reports/analysis  # a junction to the m3 worktree's tracked copy
say "small: seed study (S2) start"

# benchmark done separately (reports/analysis/bench_small.txt): S2, 16 loader workers
ARCH="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64"
SA=../voice-analyser-m3/scripts/analysis

declare -A DATA=([v2]=data/synth/train_v2 [v3b]=data/synth/train_v3b [v1]=data/synth/train_v1)
for ARM in v2 v3b v1; do
  if [ $ARM = v1 ] && [ -f runs/small/skip_v1 ]; then say "small: v1 arm skipped"; continue; fi
  for SEED in 0 1 2; do
    R=runs/small/${ARM}_s$SEED
    mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "small: $R (constant to 2000, decay 1500)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data ${DATA[$ARM]} --steps 2000 --out $R $ARCH --workers 16 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 16 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
    fi
    N=${ARM}_s$SEED
    [ "$N" = v2_s0 ] || $PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" v2_s0=runs/small/v2_s0_k35/ckpt_last.pt $N=${R}_k35/ckpt_last.pt \
      > $A/paired/small_$N.txt 2>&1 || fail "paired $N"
    echo "$N ${R}_k35/ckpt_last.pt" >> "$S/small_cpu_queue.txt"  # closed phase, on the CPU queue
  done
done
echo DONE >> "$S/small_cpu_queue.txt"
ARGS=$(for d in runs/small/*_k35; do n=$(basename $d); echo -n "${n%_k35}=small/$n/ckpt_last.pt "; done)
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u "scripts/analysis/invariance_run.py" $ARGS > ../voice-analyser/$A/invariance/small.txt 2>&1) || fail "small probe"
say "small: GPU work done"
