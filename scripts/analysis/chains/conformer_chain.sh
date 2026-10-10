#!/usr/bin/env bash
# Conformer swap (D-021 item 1) at S2 size on train_v2, the v2 arm's recipe (2000 constant + 1500 decay, 16 workers).
# Args: scratch dir, mode. "train": seeds 0-2 with the subset measures against the S2 GRU v2 arm (paired against
# v2_s0; closed phase queued for after training). "bench": seed 0 only, no real-voice measures: throughput and
# synthetic validation at 3.5k against the v2 arm's logs.
set -u
S="$1"; MODE="$2"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
A=reports/analysis
SA=../voice-analyser-m3/scripts/analysis
ARCH="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
case "$MODE" in train) SEEDS="0 1 2";; bench) SEEDS="0";; *) fail "conformer: unknown mode $MODE";; esac
say "conformer ($MODE): waiting for the EMA chain's GPU work"
until grep -q "ema: GPU work done" "$LOG"; do sleep 60; done
say "conformer ($MODE): start"
for SEED in $SEEDS; do
  R=runs/small/conf_v2_s$SEED
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "conformer: $R (constant to 2000, decay 1500)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/train_v2 --steps 2000 --out $R $ARCH --workers 16 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 16 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
  [ "$MODE" = train ] || continue
  N=conf_v2_s$SEED
  $PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" v2_s0=runs/small/v2_s0_k35/ckpt_last.pt $N=${R}_k35/ckpt_last.pt > $A/paired/small_$N.txt 2>&1 || fail "paired $N"
  echo "$N ${R}_k35/ckpt_last.pt" >> "$S/conf_cpu_queue.txt"
done
if [ "$MODE" = train ]; then
  echo DONE >> "$S/conf_cpu_queue.txt"
  ARGS=$(for SEED in 0 1 2; do echo -n "v2_s$SEED=small/v2_s${SEED}_k35/ckpt_last.pt conf_v2_s$SEED=small/conf_v2_s${SEED}_k35/ckpt_last.pt "; done)
  (cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/invariance_run.py $ARGS > ../voice-analyser/$A/invariance/small_conformer.txt 2>&1) || fail "conformer probe"
  # closed phase after training, one model at a time (CPU load during training lowers the GPU clock)
  until grep -q "ema: closed phase done" "$LOG"; do sleep 60; done  # same npz files
  n=0
  while line=$(sed -n "$((n + 1))p" "$S/conf_cpu_queue.txt"); [ "$line" != DONE ]; do
    set -- $line
    (cd ../voice-analyser-m3 && VA_THREADS=8 PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" "$1" "../voice-analyser/$2" _sub > "$S/add_sub_$1.log" 2>&1) || fail "closed phase $1"
    n=$((n + 1))
  done
fi
say "conformer ($MODE): done"
