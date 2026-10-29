#!/usr/bin/env bash
# D-040's remaining parts (registered 2026-10-28). Stage A, 5a only: closed phase and windowed paired measures for 5a seed 2,
# 5a-R on 5a seed 2, the shift part of 5a seed 1's cache, and decide_d040.py --check on the two 5a seeds (the registered
# requirement: the scoring is checked on 5a before any 5b result). Stage B waits for runs/d040/score_checked (written by hand
# after reading the check): 5b seed 1's VTL validation backfill, closed phase and paired measures for 5b, the length measures
# (vtl_score_joint.py, seed means), 5b-R refits, the refit report (5b-R against 5b and 5a-R), the f0 / shift probes for every
# arm, and the decision. Every step skips if its output exists.
# Arg: windowed scratch dir (closed-phase frames, m2cache_sub; as d036_chain.sh).
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d040 score $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
O=$A/d040
mkdir -p $F $O runs/d040
A1=d031_rA_b20k; A2=d040_5a_s2_b20k; B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k

closed() {  # name
  grep -q "\"$1\"\|'$1'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || VA_DEVICE=cuda $PY -u $SA/add_model.py "$SW" $1 runs/$1/ckpt_last.pt _sub > "$SW/add_sub_$1.log" 2>&1 || fail "closed phase $1"
}
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
cache() {  # checkpoint, cache dir
  mkdir -p $2
  VA_DEVICE=cuda $PY -u $SA/d037_cache.py $1 $2 train val probe hb shift >> $2/cache.log 2>&1 || fail "cache $2"
}
refit() {  # cache dir, head dir prefix
  local pids=()
  for s in 0 1 2; do
    [ -f $2_s$s/head.pt ] && continue
    mkdir -p $2_s$s
    VA_DEVICE=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -u $SA/d037_head.py $1 h $s $2_s$s >> $2_s$s/train.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || fail "refit $2"; done
}

say "d040 score: stage A (5a)"
closed $A2
paired $A1 $A2
cache runs/$A1/ckpt_last.pt cache/d037/rA_s1
cache runs/$A2/ckpt_last.pt cache/d037/rA_s2
refit cache/d037/rA_s2 runs/d040/5aR_s2
$PY -I $SA/decide_d040.py --check "$SW" $A/d036/vtl_joint.json d036_arm1 $A1:$A2:d036_arm1 > $O/check_5a.txt 2>&1 || fail "check"
say "d040 score: stage A done; check in $O/check_5a.txt; waiting for runs/d040/score_checked"
until [ -f runs/d040/score_checked ]; do sleep 300; done

say "d040 score: stage B (5b)"
$PY -u $SA/vtl_val_backfill.py runs/d040_5b_s1 >> runs/d040/backfill_5b_s1.log 2>&1 || fail "backfill"
closed $B1; closed $B2
paired $A1 $B1
paired $A2 $B2
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="5b=$B1+$B2" $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt \
  $B2=runs/$B2/ckpt_last.pt > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
cache runs/$B1/ckpt_last.pt cache/d037/5b_s1
cache runs/$B2/ckpt_last.pt cache/d037/5b_s2
refit cache/d037/5b_s1 runs/d040/5bR_s1
refit cache/d037/5b_s2 runs/d040/5bR_s2
[ -s $O/refits.txt ] || VA_PAIRS="R5b_s1:J5b_s1,R5b_s2:J5b_s2,R5b_s1:R5a_s1,R5b_s2:R5a_s2" $PY -u $SA/d037_score.py $O/refits.txt \
  R5a_s1=cache/d037/rA_s1:runs/d040/5aR_s1 R5a_s2=cache/d037/rA_s2:runs/d040/5aR_s2 R5b_s1=cache/d037/5b_s1:runs/d040/5bR_s1 \
  R5b_s2=cache/d037/5b_s2:runs/d040/5bR_s2 J5b_s1=cache/d037/5b_s1 J5b_s2=cache/d037/5b_s2 > $O/refits.out 2>&1 || fail "refit scoring"
[ -s $O/invariance.txt ] || VA_REF_PREFIX=5a_ $PY -u $SA/d041_invariance.py $O/invariance.txt 5a_s1=cache/d037/rA_s1 5a_s2=cache/d037/rA_s2 \
  5b_s1=cache/d037/5b_s1 5b_s2=cache/d037/5b_s2 > $O/invariance.out 2>&1 || fail "invariance"
$PY -I $SA/decide_d040.py "$SW" $O/vtl_joint.json 5b $A1:$B1:$B1 $A2:$B2:$B2 > $O/decide_d040.txt 2>&1 || fail "decide"
say "d040: $(grep '^DECISION' $O/decide_d040.txt)"
say "d040 score: done"
