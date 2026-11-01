#!/usr/bin/env bash
# D-065 scoring (registered 2026-11-01): for each arm once both its seeds are trained ("d065: gain trained", then
# "d065: training done"): closed-phase frames and windowed paired measures against 5b's seeds, the length measures
# (vtl_score_joint.py), D-050's probe suite (probe 13, gain, included) and the decision (decide_d065.py, checked on
# D-051's files first: reports/analysis/d065/check_5c_*.txt). The gain arm's scoring overlaps bw's training.
# Every step skips if its output exists.
# Arg: windowed scratch dir (closed-phase frames, m2cache_sub; as d051_score.sh).
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d065 score $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
O=$A/d065
mkdir -p $F $O
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k

closed() {  # name
  grep -q "\"$1\"\|'$1'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || VA_DEVICE=cuda $PY -u $SA/add_model.py "$SW" $1 runs/$1/ckpt_last.pt _sub > "$SW/add_sub_$1.log" 2>&1 || fail "closed phase $1"
}
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
probes() {  # run
  local ck=runs/$1/ckpt_last.pt L=runs/d065_probes_$1.log
  for p in counterfactual bandwidth polezero chain articulation shift timing voice stretch bwhold gain valley; do
    [ -s reports/probes/$p/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.run score $p $ck >> $L 2>&1 || fail "probe $p $1"
  done
  [ -s reports/probes/channel/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.channel score $ck >> $L 2>&1 || fail "probe channel $1"
  [ -s reports/probes/scale/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.scale score $ck >> $L 2>&1 || fail "probe scale $1"
  [ -s reports/probes/reversal/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.reversal $ck >> $L 2>&1 || fail "probe reversal $1"
  [ -s reports/probes/bw_evidence/$1.json ] || VA_DEVICE=cuda $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> $L 2>&1 || fail "probe bw_evidence $1"
}
arm() {  # arm name: gain or bw
  local C1=d065_$1_s1_b20k C2=d065_$1_s2_b20k
  say "d065 score: arm $1"
  closed $C1; closed $C2
  paired $B1 $C1
  paired $B2 $C2
  [ -s $O/vtl_joint_$1.json ] || VA_SEED_MEANS="$1=$C1+$C2;5b=$B1+$B2" VA_SEED_PAIRS="$1-5b=$C1+$C2:$B1+$B2;s1=$C1:$B1;s2=$C2:$B2" \
    $PY -u $SA/vtl_score_joint.py $O/vtl_joint_$1.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
    $C1=runs/$C1/ckpt_last.pt $C2=runs/$C2/ckpt_last.pt > $O/vtl_joint_$1.out 2>&1 || fail "vtl scoring $1"
  probes $C1
  probes $C2
  $PY -I $SA/decide_d065.py $1 "$SW" $O/vtl_joint_$1.json $1-5b $B1:$C1:s1 $B2:$C2:s2 > $O/decide_d065_$1.txt 2>&1 || fail "decide $1"
  say "d065: $(grep '^DECISION' $O/decide_d065_$1.txt)"
}
wait_for() {  # log line
  until tail -n +$((START + 1)) $LOG | grep -q "$1"; do
    tail -n +$((START + 1)) $LOG | grep -q "FAILED: d065 [^s]" && { say "d065 score: training chain failed; not scoring"; exit 1; }
    sleep 300
  done
}

START=$(wc -l < $LOG)  # only lines written after this point count
wait_for "d065: gain trained"
arm gain
wait_for "d065: training done"
arm bw
say "d065 score: done"
