#!/usr/bin/env bash
# D-056 (registered 2026-10-31): D-054's film seed 2 on the frozen trunk (follow-up, reported), then per seed (2, then 1)
# the anchored read-out's phase 1 (frozen 16k branch point) and phase 2 (5b's decay branch with it added); then per-run
# scoring in parallel (score_run.sh), closed-phase frames and paired measures serially, and the decision.
# Arg: windowed scratch dir (closed-phase frames, m2cache_sub). Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d056 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d056
mkdir -p $F $O

say "d056: start"
[ -f runs/d054_film_s2/ckpt_008000.pt ] || { say "d056: training d054_film_s2"; $PY $SA/d054_readout.py film 2 > runs/d054_film_s2.log 2>&1 || fail "train d054_film_s2"; }
for s in 2 1; do
  P=runs/d056_p1_s$s; B=runs/d056_s$s
  [ -f $P/ckpt_004000.pt ] || { say "d056: phase 1, seed $s"; $PY $SA/d056_anchored.py $s > $P.log 2>&1 || fail "phase 1 s$s"; }
  [ -f $B/ckpt_020000.pt ] || { say "d056: phase 2, seed $s"; mkdir -p $B
    $VA train teacher --branch-from runs/d040_5b_s$s/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B \
      --train-set readout=film_residual --train-set add_from=$P/ckpt_last.pt --train-set readout_lr=0.001 >> $B/train.log 2>&1 || fail "phase 2 s$s"; }
done
say "d056: training done"
RUNS="d054_film_s2 d056_p1_s2 d056_s2 d056_p1_s1 d056_s1"
for r in $RUNS; do bash $SA/chains/score_run.sh $r & done
wait
for r in $RUNS; do
  grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
paired d054_mlp_s2 d054_film_s2
paired d040_5b_s2_b20k d056_s2
paired d040_5b_s1_b20k d056_s1
say "d056: scoring done"
[ -f $SA/decide_d056.py ] || { say "d056: decide script missing; stopping before the decision"; exit 0; }
$PY -I $SA/decide_d056.py "$SW" > $O/decide_d056.txt 2>&1 || fail "decide"
say "d056: $(grep '^DECISION' $O/decide_d056.txt)"
say "d056: done"
