#!/usr/bin/env bash
# D-036 (pre-registered 2026-10-27): the VTL head trained jointly in a decay branch. Five arms branched from teacher
# v3's constant phase (conformer_v2r2_s1/ckpt_016000), 4000 steps each, in order 0, 1 (decisive), 2, 3, 4; after each,
# windowed dev measures against arm 0 (paired: Hillenbrand, PTDB-TUG RPA; closed phase), then the length measures for
# the head arms and decide_d036.py. Every step is skipped if its output exists.
# Args: windowed scratch dir (closed-phase frames, m2cache_sub; windowed_ref.sh's).
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d036 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
mkdir -p $F $A/d036
PARENT=runs/conformer_v2r2_s1/ckpt_016000.pt
HEAD=../voice-analyser-parakeet/cache/vtl/heads/T_s0/head.pt
REF=d036_arm0

arm() {  # name, train-set arguments...
  local R=runs/$1; shift
  if [ ! -f $R/ckpt_last.pt ] || ! $PY -c "import torch,sys; sys.exit(0 if torch.load('$R/ckpt_last.pt', map_location='cpu', weights_only=False)['step'] >= 20000 else 1)"; then
    say "d036: $R ($*)"
    mkdir -p $R
    $VA train teacher --branch-from $PARENT --decay-steps 4000 --workers 8 --out $R "$@" >> $R/train.log 2>&1 || fail "$R"
  fi
  local n=$(basename $R)
  VA_DEVICE=cuda $PY -u $SA/add_model.py "$SW" $n $R/ckpt_last.pt _sub > "$SW/add_sub_$n.log" 2>&1 || fail "closed phase $n"
  local ref=$REF
  [ "$n" = "$REF" ] && ref=conformer_v2r2_s1_d16k
  [ -s $F/${ref}_vs_$n.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $ref=runs/$ref/ckpt_last.pt $n=$R/ckpt_last.pt \
    > $F/${ref}_vs_$n.txt 2>&1 || fail "paired $n"
}

H="--train-set decay_hold=300"
arm d036_arm0 $H
arm d036_arm1 $H --train-set vtl=new --train-set vtl_seed=0
arm d036_arm2 $H --train-set vtl=frozen:$HEAD
arm d036_arm3 --train-set vtl=new --train-set vtl_seed=0
arm d036_arm4 $H --train-set vtl=tepid:$HEAD --train-set vtl_seed=0
say "d036: length measures"
[ -s $A/d036/vtl_joint.json ] || $PY -u $SA/vtl_score_joint.py $A/d036/vtl_joint.txt d036_arm1=runs/d036_arm1/ckpt_last.pt \
  d036_arm2=runs/d036_arm2/ckpt_last.pt d036_arm3=runs/d036_arm3/ckpt_last.pt d036_arm4=runs/d036_arm4/ckpt_last.pt \
  > $A/d036/vtl_joint.out 2>&1 || fail "vtl scoring"
$PY -I $SA/decide_d036.py "$SW" $A/d036/vtl_joint.json > $A/d036/decide_d036.txt 2>&1 || fail "decide"
say "d036: $(grep '^DECISION' $A/d036/decide_d036.txt)"
say "d036: done"
