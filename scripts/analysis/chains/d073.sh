#!/usr/bin/env bash
# D-073 (registered 2026-11-03): learned formant queries as the read-out on teacher v3.1's frozen trunk (D-054's recipe),
# arms queries (content-only keys) and queries_abs, head seeds 1 then 2 (seed 2 skipped by the futility stop); then
# D-054's scoring per run (D-053 frames, probes 11 and 12, closed-phase frames), paired against D-054's mlp, and the
# decision. Arg: windowed scratch dir (closed-phase frames, m2cache_sub). Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d073 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d073
mkdir -p $F $O
ARMS="queries queries_abs"

futile() {  # as D-054 amendment 1: the arm's seed 1 synthetic F1-F4 MAE over 1.5 x mlp seed 1's
  $PY -c "import json, sys
m = lambda r: [json.loads(x)['eval'] for x in open(f'runs/{r}/log.jsonl') if '\"eval\"' in x][-1]
f = lambda r: sum(m(r)[f'F{n}_mae_hz'] for n in range(1, 5)) / 4
sys.exit(0 if f('d073_$1_s1') > 1.5 * f('d054_mlp_s1') else 1)"
}
trained() { [ -f runs/$1/ckpt_008000.pt ]; }

say "d073: start"
for s in 1 2; do
  for a in $ARMS; do
    r=d073_${a}_s$s
    if [ $s = 2 ] && futile $a; then
      [ -f $O/futility_stop_$a ] || { touch $O/futility_stop_$a; say "d073: futility stop: $a seed 2 not trained"; }
      continue
    fi
    trained $r || { say "d073: training $r"; $PY $SA/d073_queries.py $a $s > runs/$r.log 2>&1 || fail "train $r"; }
  done
done
say "d073: training done"
for s in 1 2; do
  for a in $ARMS; do
    r=d073_${a}_s$s; ck=runs/$r/ckpt_last.pt
    trained $r || continue
    ls cache/d053/vocalset_$r.npz > /dev/null 2>&1 || $PY -c "import sys; sys.path.insert(0, '$SA'); import d053_iseli_alwan as D
if __name__ == '__main__': D.score('$ck')" >> runs/d073_score_$r.log 2>&1 || fail "d053 frames $r"
    [ -s reports/probes/scale/$r.json ] || $PY -m vocal_analysis.eval.probes.scale score $ck >> runs/d073_score_$r.log 2>&1 || fail "probe scale $r"
    for p in bwhold counterfactual bandwidth; do  # probe 12 reads these probes' reports (arms A and B)
      [ -s reports/probes/$p/$r.json ] || $PY -m vocal_analysis.eval.probes.run score $p $ck >> runs/d073_score_$r.log 2>&1 || fail "probe $p $r"
    done
    [ -s reports/probes/bw_evidence/$r.json ] || $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> runs/d073_score_$r.log 2>&1 || fail "probe bw_evidence $r"
    grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
      || $PY -u $SA/add_model.py "$SW" $r $ck _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
    [ -s $F/d054_mlp_s${s}_vs_$r.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" \
      d054_mlp_s$s=runs/d054_mlp_s$s/ckpt_last.pt $r=$ck > $F/d054_mlp_s${s}_vs_$r.txt 2>&1 || fail "paired $r"
  done
done
say "d073: scoring done"
[ -f $SA/decide_d073.py ] || { say "d073: decide script missing; stopping before the decision"; exit 0; }
$PY -I $SA/decide_d073.py "$SW" > $O/decide_d073.txt 2>&1 || fail "decide"
say "d073: $(grep '^DECISION' $O/decide_d073.txt)"
say "d073: done"
