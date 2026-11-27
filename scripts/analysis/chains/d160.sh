#!/usr/bin/env bash
# D-160 (registered 2026-11-27): decay branches from d040_5b_s{1,2}/ckpt_016000.pt, 4000 steps, with train_v2w in
# train_v2's place (wide low-f0 bandwidths, off-line LF return phase); then the guards (D-051's, through D-065's scoring:
# closed phase, windowed paired measures, length, probes, decide_d065.py) on the GPU, and the CPU scoring and report
# (d160_score.py). Waits for train_v2w (runs/make_v2w.log "exit 0"). Every step skips if its output exists.
# Arg: windowed scratch dir holding ptdb_egg_sub.npz and m2cache_sub [cache/m3w].
set -u
SW="${1:-cache/m3w}"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d160 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
O=$A/d160
W=data/synth/train_v2w
mkdir -p $F $O
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k; C1=d160_s1; C2=d160_s2

say "d160: waiting for train_v2w"
until grep -q "^exit" runs/make_v2w.log 2>/dev/null; do sleep 300; done
grep -q "^exit 0" runs/make_v2w.log || fail "train_v2w (runs/make_v2w.log)"
[ -f $W/flat/done ] || fail "train_v2w not consolidated"
say "d160: start"
X="--train-set synth_extra=$W --train-set synth_extra_replace=true"
for s in 1 2; do
  r=d160_s$s
  if [ ! -f runs/$r/ckpt_last.pt ] || ! grep -q '"step": 20000' runs/$r/log.jsonl; then
    say "d160: training $r"
    mkdir -p runs/$r
    $VA train teacher --branch-from runs/d040_5b_s$s/ckpt_016000.pt --decay-steps 4000 --workers 12 --out runs/$r $X \
      >> runs/$r/train.log 2>&1 || fail "train $r"
  fi
done
say "d160: training done"

closed() {  # name
  grep -q "\"$1\"\|'$1'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $1 runs/$1/ckpt_last.pt _sub > "$SW/add_sub_$1.log" 2>&1 || fail "closed phase $1"
}
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
probes() {  # run
  local ck=runs/$1/ckpt_last.pt L=runs/d160_probes_$1.log
  for p in counterfactual bandwidth polezero chain articulation shift timing voice stretch bwhold gain valley; do
    [ -s reports/probes/$p/$1.json ] || $PY -m vocal_analysis.eval.probes.run score $p $ck >> $L 2>&1 || fail "probe $p $1"
  done
  [ -s reports/probes/channel/$1.json ] || $PY -m vocal_analysis.eval.probes.channel score $ck >> $L 2>&1 || fail "probe channel $1"
  [ -s reports/probes/scale/$1.json ] || $PY -m vocal_analysis.eval.probes.scale score $ck >> $L 2>&1 || fail "probe scale $1"
  [ -s reports/probes/reversal/$1.json ] || $PY -m vocal_analysis.eval.probes.reversal $ck >> $L 2>&1 || fail "probe reversal $1"
  [ -s reports/probes/bw_evidence/$1.json ] || $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> $L 2>&1 || fail "probe bw_evidence $1"
}

# the CPU scoring overlaps the GPU guards
( VA_DEVICE=cpu $PY $SA/d160_score.py run $C1,$C2,$B1,$B2 > runs/d160_score.log 2>&1 && echo "exit 0" >> runs/d160_score.log \
  || echo "exit 1" >> runs/d160_score.log ) &
say "d160: guards"
closed $C1; closed $C2
paired $B1 $C1
paired $B2 $C2
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="d160=$C1+$C2;5b=$B1+$B2" VA_SEED_PAIRS="d160-5b=$C1+$C2:$B1+$B2;s1=$C1:$B1;s2=$C2:$B2" \
  $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
  $C1=runs/$C1/ckpt_last.pt $C2=runs/$C2/ckpt_last.pt > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
probes $C1
probes $C2
$PY -I $SA/decide_d065.py bw "$SW" $O/vtl_joint.json d160-5b $B1:$C1:s1 $B2:$C2:s2 > $O/decide_guards.txt 2>&1 || fail "guards"
say "d160: guards done"
until grep -q "^exit" runs/d160_score.log 2>/dev/null; do sleep 120; done
grep -q "^exit 0" runs/d160_score.log || fail "cpu scoring (runs/d160_score.log)"
$PY $SA/d160_score.py report > runs/d160_report.log 2>&1 || fail "report"
say "d160: $(grep '^Reading' $O/d160.txt)"
say "d160: done"
