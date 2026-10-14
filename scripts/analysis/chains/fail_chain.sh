#!/usr/bin/env bash
# D-026 fail branch (pre-registered 2026-10-15, before select_base.py ran): runs only if select_base.py says "fix"
# (neither v2m nor v2s carried forward). The CPU checks ran beforehand (render_check.txt, cascade_probe.txt,
# hill_signed_v2m.txt); none of them changes the pre-registered rules. GPU, in order:
#   1. control: mix_conformer_s0 (train_v2) constant 2000 -> 16000, then d16k: does a small model that never saw the
#      interactive source also move down from k35 to d16k?
#   2. dose arm, unless render_check.py said "mismatch" (then the renderer is looked at first, no arm): the better of
#      v2m / v2s (smaller |d16k women's closed-phase F3 bias|) with the feedback on for 25 % of interactive clips
#      (train_v2q from v2m, train_v2sq from v2s), through the same rule (decide_v2m.py).
# Waits for this run's "sharp: base DECISION" line. Arg: scratch dir.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
A=reports/analysis
F=$A/paired/fp32
SA=scripts/analysis
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }

START=$(wc -l < $LOG)
until tail -n +$((START + 1)) $LOG | grep -q "sharp: base DECISION"; do
  tail -n +$((START + 1)) $LOG | grep -q "FAILED" && { say "fail: the sharp chain failed; not run"; exit 0; }
  sleep 120
done
BASE=$(sed -n 's/^DECISION: //p' $A/decide_base.txt)
[ "$BASE" = fix ] || { say "fail: base $BASE carried forward; fail branch not needed"; exit 0; }
until tail -n +$((START + 1)) $LOG | grep -q "base: done"; do sleep 60; done
say "fail: neither v2m nor v2s carried forward; running the fail branch (D-026)"

# which interactive set the dose arm builds on: smaller |d16k women's closed-phase F3 bias|
f3() { sed -n "s/^d16k women's closed-phase F1 bias .*, F3 \([+-][0-9.]*\).*/\1/p" $A/decide_$1.txt | tr -d '+-'; }
if $PY -c "import sys; sys.exit(0 if float('$(f3 v2s)') < float('$(f3 v2m)') else 1)"; then
  FROM=v2s; DOSE=v2sq; EXTRA="interactive_closure=(1.0, 3.0), "
else
  FROM=v2m; DOSE=v2q; EXTRA=""
fi
D=data/synth/train_$DOSE
RENDER=
VERDICT=$(sed -n 's/^DECISION: //p' $A/sensitivity/render_check.txt 2>/dev/null)
if [ "$VERDICT" = mismatch ]; then
  say "fail: render_check says mismatch; no dose arm (the renderer first)"
elif [ ! -f $D/index.json ]; then
  say "fail: generating $D (100k clips, seed 2, train_$FROM with p_interactive_coupled 0.25, 20 processes)"
  $PY -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('$D', 100_000, replace(SynthConfig(), p_interactive=1.0, p_interactive_coupled=0.25, $EXTRA),
          seed=2, workers=20, extra_index={'note': 'train_$FROM with the feedback on for 25 % of interactive clips (D-026 fail branch)'})
" > "$S/gen_$DOSE.log" 2>&1 &
  RENDER=$!
fi

# 1. control: mix to d16k
N=mix_conformer_s0; R=runs/small/$N; RD=${R}_d16k; mkdir -p $RD
if [ ! -f $RD/ckpt_last.pt ]; then
  say "fail: $R constant 2000 -> 16000 (control)"
  [ -f $R/ckpt_016000.pt ] || $VA train teacher --data data/synth/train_v2 --steps 16000 --out $R $S2 --workers 4 \
    --train-set seed=0 --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
  say "fail: $RD (decay 4000)"
  $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
fi
paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
closed ${N}_d16k $RD/ckpt_last.pt
$PY -I $SA/cp_dev.py "$S" mix_conformer_s0,mix_conformer_s0_d16k,v2m_conformer_s0,v2m_conformer_s0_d16k,conformer_v2_d16k \
  mix_conformer_s0_d16k-mix_conformer_s0 v2m_conformer_s0_d16k-v2m_conformer_s0 > $A/closed_phase/cp_small_mix_d16k_dev.txt 2>&1 || fail "cp_dev mix d16k"
say "fail: control mix_conformer_s0_d16k measured"

# 2. dose arm
if [ -n "$RENDER" ]; then
  wait $RENDER || fail "generate train_$DOSE"
  cp $D/index.json docs/synth_sets/train_$DOSE.index.json
fi
if [ "$VERDICT" != mismatch ]; then
  [ -f $D/index.json ] || fail "train_$DOSE missing"
  for SEED in 0 1 2; do
    N=${DOSE}_conformer_s$SEED; R=runs/small/$N; mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "fail: $R (constant to 2000, decay 1500; crop_frames 32-400)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data $D --steps 2000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      paired mix_conformer_s$SEED $(k35 mix_conformer_s$SEED) $N $(k35 $N)
      closed $N $(k35 $N)
      say "fail: $N measured"
    fi
  done
  for SEED in 0 1; do
    N=${DOSE}_conformer_s$SEED; R=runs/small/$N; RD=${R}_d16k; mkdir -p $RD
    if [ ! -f $RD/ckpt_last.pt ]; then
      say "fail: $R constant 2000 -> 16000"
      [ -f $R/ckpt_016000.pt ] || $VA train teacher --data $D --steps 16000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
      say "fail: $RD (decay 4000)"
      $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 4 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
    fi
    paired $N $(k35 $N) ${N}_d16k $RD/ckpt_last.pt
    closed ${N}_d16k $RD/ckpt_last.pt
    say "fail: ${N}_d16k measured"
  done
  Rs=runs/small
  VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
    $(for s in 0 1 2; do echo -n "${DOSE}_s$s=$Rs/${DOSE}_conformer_s${s}_k35/ckpt_last.pt "; done) \
    ${DOSE}_s0_d16k=$Rs/${DOSE}_conformer_s0_d16k/ckpt_last.pt ${DOSE}_s1_d16k=$Rs/${DOSE}_conformer_s1_d16k/ckpt_last.pt \
    ${FROM}_s0_d16k=$Rs/${FROM}_conformer_s0_d16k/ckpt_last.pt ${FROM}_s1_d16k=$Rs/${FROM}_conformer_s1_d16k/ckpt_last.pt \
    > $A/sensitivity/resynth_identity_${DOSE}_d16k.txt 2>&1 || fail "identity $DOSE"
  M=$(for a in mix v2i $FROM $DOSE; do for k in 0 1 2; do echo -n "${a}_conformer_s$k,"; done; done)${FROM}_conformer_s0_d16k,${FROM}_conformer_s1_d16k,${DOSE}_conformer_s0_d16k,${DOSE}_conformer_s1_d16k
  $PY -I $SA/cp_dev.py "$S" $M $(for k in 0 1 2; do echo -n "${DOSE}_conformer_s$k-mix_conformer_s$k ${DOSE}_conformer_s$k-${FROM}_conformer_s$k "; done) \
    ${DOSE}_conformer_s0_d16k-${FROM}_conformer_s0_d16k ${DOSE}_conformer_s1_d16k-${FROM}_conformer_s1_d16k > $A/closed_phase/cp_small_${DOSE}_dev.txt 2>&1 || fail "cp_dev $DOSE"
  $PY -I $SA/decide_v2m.py "$S" $F $A/sensitivity/resynth_identity_${DOSE}_d16k.txt $DOSE > $A/decide_$DOSE.txt 2>&1 || fail "decide $DOSE"
  say "fail: $DOSE DECISION: $(sed -n 's/^DECISION: //p' $A/decide_$DOSE.txt)"
fi
say "fail: done"
