#!/usr/bin/env bash
# D-051 scoring (registered 2026-10-30): waits for "d051: training done", then closed-phase frames and windowed paired
# measures for both 5c seeds against their 5b seeds, the length measures (vtl_score_joint.py: 5b and 5c seed means and the
# paired 5c - 5b interval), D-050's probe suite on both 5c seeds (probe 10 is reported by the decision), and the decision.
# The decision script was checked on D-040's inputs first (reports/analysis/d051/check_5b.txt). Every step skips if its
# output exists.
# Arg: windowed scratch dir (closed-phase frames, m2cache_sub; as d040_score.sh).
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d051 score $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
O=$A/d051
mkdir -p $F $O
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k; C1=d051_5c_s1_b20k; C2=d051_5c_s2_b20k

closed() {  # name
  grep -q "\"$1\"\|'$1'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || VA_DEVICE=cuda $PY -u $SA/add_model.py "$SW" $1 runs/$1/ckpt_last.pt _sub > "$SW/add_sub_$1.log" 2>&1 || fail "closed phase $1"
}
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
probes() {  # run
  local ck=runs/$1/ckpt_last.pt
  for p in counterfactual bandwidth polezero chain articulation shift timing voice stretch bwhold; do
    [ -s reports/probes/$p/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.run score $p $ck >> runs/d051_probes_$1.log 2>&1 || fail "probe $p $1"
  done
  [ -s reports/probes/channel/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.channel score $ck >> runs/d051_probes_$1.log 2>&1 || fail "probe channel $1"
  [ -s reports/probes/scale/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.scale score $ck >> runs/d051_probes_$1.log 2>&1 || fail "probe scale $1"
  [ -s reports/probes/reversal/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.reversal $ck >> runs/d051_probes_$1.log 2>&1 || fail "probe reversal $1"
  [ -s reports/probes/bw_evidence/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> runs/d051_probes_$1.log 2>&1 || fail "probe bw_evidence $1"
}

START=$(wc -l < $LOG)  # only lines written after this point count (the log holds the morning's restarted failure)
until tail -n +$((START + 1)) $LOG | grep -q "d051: training done"; do
  tail -n +$((START + 1)) $LOG | grep -q "FAILED: d051" && { say "d051 score: training chain failed; not scoring"; exit 1; }
  sleep 300
done
say "d051 score: start"
closed $C1; closed $C2
paired $B1 $C1
paired $B2 $C2
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="5c=$C1+$C2;5b=$B1+$B2" VA_SEED_PAIRS="5c-5b=$C1+$C2:$B1+$B2;s1=$C1:$B1;s2=$C2:$B2" $PY -u $SA/vtl_score_joint.py \
  $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt $C1=runs/$C1/ckpt_last.pt $C2=runs/$C2/ckpt_last.pt \
  > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
probes $C1
probes $C2
$PY -I $SA/decide_d051.py "$SW" $O/vtl_joint.json 5c-5b $B1:$C1:s1 $B2:$C2:s2 > $O/decide_d051.txt 2>&1 || fail "decide"
say "d051: $(grep '^DECISION' $O/decide_d051.txt)"
say "d051 score: done"
