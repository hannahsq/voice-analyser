#!/usr/bin/env bash
# D-117: the VTL head refitted on each branch arm's frozen trunk (D-037's recipe) against 5b's existing refits (D-040's 5b-R).
# Steps skip when their output exists. Run a copy (runs/chains/), from the main checkout.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d117 $*; stopping"; exit 1; }
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
O=reports/analysis/d117
export MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
mkdir -p $O runs/d117
say "d117: start"

# precondition: 5b's cached training crops rebuilt under today's code (first 200), against the 29 Oct cache
if [ ! -s $O/check.txt ]; then
  mkdir -p cache/d117/check_5b_s1
  VA_LIMIT=200 VA_DEVICE=cuda $PY -u $SA/d037_cache.py runs/d040_5b_s1_b20k/ckpt_last.pt cache/d117/check_5b_s1 train \
    > cache/d117/check_5b_s1/cache.log 2>&1 || fail "check cache"
  $PY -u $SA/d117_refit.py check cache/d037/5b_s1 cache/d117/check_5b_s1 > $O/check.txt 2>&1 || fail "check"
fi
say "d117: reproduction check written"

cache() {  # $1 run name
  local C=cache/d037/$1
  mkdir -p $C
  [ -f $C/hb/done ] && return 0
  VA_WORKERS=14 VA_DEVICE=cuda $PY -u $SA/d037_cache.py runs/$1/ckpt_last.pt $C train val probe hb >> $C/cache.log 2>&1  # 2 x 14 = 28 loader threads (vtl-mri sweep)
}
for pair in "d114_s1 d114_s2" "d109_s1 d109_s2" "d090_f5_s1 d090_f5_s2"; do
  set -- $pair
  cache $1 & p1=$!
  cache $2 & p2=$!
  wait $p1 || fail "cache $1"
  wait $p2 || fail "cache $2"
  say "d117: cached $1 $2"
done

for t in d114_s1 d114_s2 d109_s1 d109_s2 d090_f5_s1 d090_f5_s2; do
  pids=()
  for s in 0 1 2; do
    D=runs/d117/R_${t}_s$s
    [ -f $D/done ] && continue
    mkdir -p $D
    VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -u $SA/d037_head.py cache/d037/$t h $s $D >> $D/train.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail "refit $t"; done
  say "d117: refitted $t"
done

# joint heads (no head prefix) go to cache/d117/joint; refits to cache/d117/refit
$PY -u $SA/d117_refit.py export $(for t in d040_5b_s1_b20k:5b_s1 d040_5b_s2_b20k:5b_s2 d114_s1:d114_s1 d114_s2:d114_s2 d109_s1:d109_s1 \
  d109_s2:d109_s2 d090_f5_s1:d090_f5_s1 d090_f5_s2:d090_f5_s2; do echo "${t%%:*}=cache/d037/${t##*:}"; done) > $O/export_joint.out 2>&1 || fail "export joint"
$PY -u $SA/d117_refit.py export d040_5b_s1_b20k=cache/d037/5b_s1:runs/d040/5bR_s1 d040_5b_s2_b20k=cache/d037/5b_s2:runs/d040/5bR_s2 \
  $(for t in d114_s1 d114_s2 d109_s1 d109_s2 d090_f5_s1 d090_f5_s2; do echo "$t=cache/d037/$t:runs/d117/R_$t"; done) > $O/export_refit.out 2>&1 || fail "export refit"
[ -s $O/d117.txt ] || $PY -u $SA/d117_refit.py report > $O/d117.txt 2> $O/d117.err || fail "report"
say "d117: done"
