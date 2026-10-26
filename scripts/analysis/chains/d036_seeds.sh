#!/usr/bin/env bash
# D-036 amendment (2026-10-27, before any length result): head-seed replicates of arm 1 (vtl_seed 1, 2), after the main
# D-036 chain. The GPU is free until DEADLINE: a watchdog stops this chain's whole process tree then. Training resumes
# from ckpt_last.pt on a rerun (checkpoints every 250 steps); a measure step only starts if it can finish before the
# deadline, and writes through a temporary file, so a stop never leaves a partial output that a rerun would skip.
# Args: windowed scratch dir (as d036_chain.sh), deadline as HH:MM today [11:30].
set -u
SW="$1"
DEADLINE=$(date -d "today ${2:-11:30}" +%s)
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d036 seeds $*; stopping"; exit 1; }
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }  # minutes
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
PARENT=runs/conformer_v2r2_s1/ckpt_016000.pt

WINPID=$(cat /proc/$$/winpid)
( sleep $(( DEADLINE - $(date +%s) )); say "d036 seeds: deadline, stopping the queue"; taskkill //T //F //PID $WINPID > /dev/null 2>&1 ) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

until grep -qE "d036: done|FAILED: d036 " <(sed -n '/d036: restarted/,$p' "$LOG"); do sleep 30; done
grep -q "FAILED: d036 " <(sed -n '/d036: restarted/,$p' "$LOG") && fail "the main chain failed first"
say "d036 seeds: start ($(left) min left)"

seed_arm() {  # vtl seed
  local n=d036_arm1s$1 R=runs/d036_arm1s$1
  if [ ! -f $R/ckpt_last.pt ] || ! $PY -c "import torch,sys; sys.exit(0 if torch.load('$R/ckpt_last.pt', map_location='cpu', weights_only=False)['step'] >= 20000 else 1)"; then
    [ "$(left)" -gt 10 ] || { say "d036 seeds: no time for $n"; return 1; }
    say "d036 seeds: $R"
    mkdir -p $R
    $VA train teacher --branch-from $PARENT --decay-steps 4000 --workers 8 --out $R --train-set decay_hold=300 \
      --train-set vtl=new --train-set vtl_seed=$1 --train-set ckpt_every=250 >> $R/train.log 2>&1 || fail "$R"
  fi
  [ "$(left)" -gt 16 ] || { say "d036 seeds: $n trained; measures left for a rerun"; return 1; }
  VA_DEVICE=cuda $PY -u $SA/add_model.py "$SW" $n $R/ckpt_last.pt _sub > "$SW/add_sub_$n.log" 2>&1 || fail "closed phase $n"
  if [ ! -s $F/d036_arm0_vs_$n.txt ]; then
    VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" d036_arm0=runs/d036_arm0/ckpt_last.pt $n=$R/ckpt_last.pt \
      > $F/d036_arm0_vs_$n.tmp 2>&1 || fail "paired $n"
    mv $F/d036_arm0_vs_$n.tmp $F/d036_arm0_vs_$n.txt
  fi
}

done_seeds=()
for s in 1 2; do
  seed_arm $s && done_seeds+=("d036_arm1s$s=runs/d036_arm1s$s/ckpt_last.pt") || break
done
if [ ${#done_seeds[@]} -gt 0 ] && [ "$(left)" -gt 20 ]; then
  say "d036 seeds: length measures (${#done_seeds[@]} seeds)"
  $PY -u $SA/vtl_score_joint.py $A/d036/vtl_joint_seeds.txt "${done_seeds[@]}" > $A/d036/vtl_joint_seeds.out 2>&1 || fail "vtl scoring"
fi
say "d036 seeds: done (${#done_seeds[@]} complete)"
