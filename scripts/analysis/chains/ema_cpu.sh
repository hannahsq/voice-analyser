#!/usr/bin/env bash
# Closed-phase F1 (subset) for each EMA model as its paired measures finish, one at a time (they share the npz files).
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
Q="$S/small_cpu_queue_ema.txt"
# the small study's closed-phase queue writes the same files: wait for it
until grep -q "small: closed phase done" "$LOG"; do sleep 60; done
n=0
while true; do
  line=$(sed -n "$((n + 1))p" $Q 2>/dev/null)
  [ -z "$line" ] && { sleep 60; continue; }
  [ "$line" = DONE ] && break
  set -- $line
  (cd ../voice-analyser-m3 && VA_THREADS=4 PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" "$1" "../voice-analyser/$2" _sub > "$S/add_sub_$1.log" 2>&1) \
    || say "FAILED: ema closed phase $1 (continuing)"
  n=$((n + 1))
done
say "ema: closed phase done"
