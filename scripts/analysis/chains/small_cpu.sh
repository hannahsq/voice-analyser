#!/usr/bin/env bash
# Closed-phase F1 (subset) for each small model, after the study's training, one at a time (they share the npz files).
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
Q="$S/small_cpu_queue.txt"
# CPU work during training costs GPU clock (shared power budget): wait until the study's training is done
until grep -q "small: GPU work done" "$LOG"; do sleep 60; done  # add_model.py skips models already added
n=0
while true; do
  line=$(sed -n "$((n + 1))p" $Q 2>/dev/null)
  [ -z "$line" ] && { sleep 60; continue; }
  [ "$line" = DONE ] && break
  set -- $line
  (cd ../voice-analyser-m3 && VA_THREADS=8 PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" "$1" "../voice-analyser/$2" _sub > "$S/add_sub_$1.log" 2>&1) \
    || say "FAILED: small closed phase $1 (continuing)"
  n=$((n + 1))
done
say "small: closed phase done"
