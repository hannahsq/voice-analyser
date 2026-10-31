#!/usr/bin/env bash
# D-054 (registered 2026-10-31): frozen-trunk read-out arms (flat, mlp, film; head seeds 1, 2) on teacher v3.1's trunk,
# then per run D-053's bandwidth and corrected-measure frames (score only: D-053's committed report is not rewritten),
# probes 11 and 12, closed-phase frames and windowed paired measures against the mlp arm, and the decision.
# Arg: windowed scratch dir (closed-phase frames, m2cache_sub; as d051_score.sh). Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d054 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d054
mkdir -p $F $O
ARMS="mlp film flat"

say "d054: start"
for s in 1 2; do
  for a in $ARMS; do
    r=d054_${a}_s$s
    [ -f runs/$r/ckpt_008000.pt ] || { say "d054: training $r"; $PY $SA/d054_readout.py $a $s > runs/$r.log 2>&1 || fail "train $r"; }
  done
done
say "d054: training done"
for s in 1 2; do
  for a in $ARMS; do
    r=d054_${a}_s$s; ck=runs/$r/ckpt_last.pt
    ls cache/d053/vocalset_$r.npz > /dev/null 2>&1 || $PY -c "import sys; sys.path.insert(0, '$SA'); import d053_iseli_alwan as D
if __name__ == '__main__': D.score('$ck')" >> runs/d054_score_$r.log 2>&1 || fail "d053 frames $r"
    [ -s reports/probes/scale/$r.json ] || $PY -m vocal_analysis.eval.probes.scale score $ck >> runs/d054_score_$r.log 2>&1 || fail "probe scale $r"
    [ -s reports/probes/bw_evidence/$r.json ] || $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> runs/d054_score_$r.log 2>&1 || fail "probe bw_evidence $r"
    grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
      || $PY -u $SA/add_model.py "$SW" $r $ck _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
  done
  for a in film flat; do
    [ -s $F/d054_mlp_s${s}_vs_d054_${a}_s$s.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" \
      d054_mlp_s$s=runs/d054_mlp_s$s/ckpt_last.pt d054_${a}_s$s=runs/d054_${a}_s$s/ckpt_last.pt \
      > $F/d054_mlp_s${s}_vs_d054_${a}_s$s.txt 2>&1 || fail "paired $a s$s"
  done
done
say "d054: scoring done"
[ -f $SA/decide_d054.py ] || { say "d054: decide script missing; stopping before the decision"; exit 0; }
$PY -I $SA/decide_d054.py "$SW" > $O/decide_d054.txt 2>&1 || fail "decide"
say "d054: $(grep '^DECISION' $O/decide_d054.txt)"
say "d054: done"
