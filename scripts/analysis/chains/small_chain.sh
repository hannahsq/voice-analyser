#!/usr/bin/env bash
# Small-model proxy (replaces the full-size 3.5k / 5k study): data-loader and model speed benchmark (full, S1, S2);
# pick a size; then 3 seeds per arm at 3.5k (2000 constant + 1500 decay) with the subset measures.
# Runs: runs/bench/<size>, runs/small/<arm>_s<seed>[_k35]. Reports: reports/analysis/{bench_small.txt,paired/small_*,
# invariance/small.txt}. v1's arm is skipped if runs/small/skip_v1 exists when it is reached.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
A=reports/analysis
# benchmarks need a quiet CPU: wait for the v1 branch's closed-phase step
say "small: waiting for the v1 branch measures (CPU)"
since() { tail -n +"$(grep -n "small: waiting for the v1 branch measures (CPU)" "$LOG" | tail -1 | cut -d: -f1)" "$LOG"; }
until since | grep -q "v1 branch measures done\|FAILED"; do sleep 60; done
since | grep -q "FAILED" && fail "an earlier chain failed; small study not started"

declare -A ARCH=([full]="" [S1]="--model-set channels=16,32,48 --model-set embed=256 --model-set gru_hidden=128 --model-set f0_gru_hidden=96"
                 [S2]="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64")
say "small: speed benchmark"
mkdir -p runs/bench
$PY -I "$S/bench_loader.py" > runs/bench/loader.txt 2>&1 || fail "loader bench"
for X in full S1 S2; do
  rm -rf runs/bench/$X; mkdir -p runs/bench/$X
  $VA train teacher --data data/synth/train_v2 --steps 600 --out runs/bench/$X ${ARCH[$X]} --train-set eval_every=100000 --train-set ckpt_every=600 \
    > runs/bench/$X/train.log 2>&1 || fail "bench $X"
  (cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe "$S/bench_infer.py" ../voice-analyser/runs/bench/$X/ckpt_last.pt) \
    2>/dev/null | grep "infer" > runs/bench/$X/infer.txt
done
$PY - > $A/bench_small.txt <<'PYEOF'
import json, numpy as np
sp = {}
for x in ("full", "S1", "S2"):
    rows = [r for r in map(json.loads, open(f"runs/bench/{x}/log.jsonl")) if "audio_s_per_s" in r]
    sp[x] = np.median([r["audio_s_per_s"] for r in rows if r["step"] > 300])
    cfg = json.load(open(f"runs/bench/{x}/config.json"))
    print(f"{x}: {cfg['n_params']:,} params, {sp[x]:.1f} audio s/s training (median, steps 300-600), x{sp[x] / sp['full']:.2f}")
    print(open(f"runs/bench/{x}/infer.txt").read().rstrip())
print("data loader alone (no model):")
print(open("runs/bench/loader.txt").read().rstrip())
pick = "S2" if sp["S2"] >= 1.25 * sp["S1"] and sp["S2"] >= 1.2 * sp["full"] else "S1" if sp["S1"] >= 1.2 * sp["full"] else "none"
print(f"pick: {pick}")
PYEOF
PICK=$(grep "^pick: " $A/bench_small.txt | cut -d' ' -f2)
case "$PICK" in S1|S2|none) ;; *) fail "benchmark summary failed ($A/bench_small.txt)";; esac
say "small: benchmark done ($A/bench_small.txt), pick $PICK"
[ "$PICK" = none ] && fail "no small size is 1.2x faster than full; small study not started"

declare -A DATA=([v2]=data/synth/train_v2 [v3b]=data/synth/train_v3b [v1]=data/synth/train_v1)
for ARM in v2 v3b v1; do
  if [ $ARM = v1 ] && [ -f runs/small/skip_v1 ]; then say "small: v1 arm skipped"; continue; fi
  for SEED in 0 1 2; do
    R=runs/small/${ARM}_s$SEED
    mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "small: $R (constant to 2000, decay 1500)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data ${DATA[$ARM]} --steps 2000 --out $R ${ARCH[$PICK]} \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
    fi
    N=${ARM}_s$SEED
    [ "$N" = v2_s0 ] || $PY -I "$S/m2_paired_sub.py" "$S/m2cache_sub" v2_s0=runs/small/v2_s0_k35/ckpt_last.pt $N=${R}_k35/ckpt_last.pt \
      > $A/paired/small_$N.txt 2>&1 || fail "paired $N"
    echo "$N ${R}_k35/ckpt_last.pt" >> "$S/small_cpu_queue.txt"  # closed phase, on the CPU queue
  done
done
echo DONE >> "$S/small_cpu_queue.txt"
ARGS=$(for d in runs/small/*_k35; do n=$(basename $d); echo -n "${n%_k35}=small/$n/ckpt_last.pt "; done)
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u "$S/invariance_run.py" $ARGS > ../voice-analyser/$A/invariance/small.txt 2>&1) || fail "small probe"
say "small: GPU work done"
