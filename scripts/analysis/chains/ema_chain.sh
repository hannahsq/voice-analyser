#!/usr/bin/env bash
# EMA test: each small run that kept a moving average (all but v2_s0, v2_s1), its averaged weights against its own
# raw weights on the subset measures. Runs after the small study's GPU work. Reports: reports/analysis/paired/ema_*,
# invariance/small_ema.txt; closed phase (<name>_ema columns) on the CPU queue.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
A=reports/analysis
SA=../voice-analyser-m3/scripts/analysis
say "ema: waiting for the small study's GPU work"
since() { tail -n +"$(grep -n "ema: waiting for the small study's GPU work" "$LOG" | tail -1 | cut -d: -f1)" "$LOG"; }
until since | grep -q "small: GPU work done\|FAILED"; do sleep 60; done
since | grep -q "FAILED" && fail "an earlier chain failed; ema test not started"
say "ema: start"
ARGS=""
for d in runs/small/*_k35; do
  n=$(basename $d); N=${n%_k35}
  $PY -c "import torch,sys; sys.exit(0 if torch.load('$d/ckpt_last.pt', weights_only=False).get('ema') is not None else 1)" || continue
  $PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" $N=$d/ckpt_last.pt ${N}_ema=$d/ckpt_last.pt#ema > $A/paired/ema_$N.txt 2>&1 || fail "paired ema $N"
  echo "${N}_ema $d/ckpt_last.pt#ema" >> "$S/small_cpu_queue_ema.txt"
  ARGS="$ARGS $N=small/$n/ckpt_last.pt ${N}_ema=small/$n/ckpt_last.pt#ema"
done
echo DONE >> "$S/small_cpu_queue_ema.txt"
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/invariance_run.py $ARGS > ../voice-analyser/$A/invariance/small_ema.txt 2>&1) || fail "ema probe"
say "ema: GPU work done"
