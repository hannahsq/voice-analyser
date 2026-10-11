#!/usr/bin/env bash
# Night of 2026-10-11/12 (stop by 07:00; a watchdog kills training at 06:50, runs resume from ckpt_last).
# 1. Wait for the resynthesised VocalSet sets (train / val, tracked by teacher_v2_d16k, rendered separately).
# 2. M3 at S2: three mixed-crop Conformer seeds on train_v2 + 30 % resynthesised VocalSet (m3_conformer_s0-2), measured
#    on dev speakers against mix_conformer_s0-2 (same seeds, same recipe, no resynthesised data).
# 3. Pre-registered rule (decide_m3_s2.py): no degradation -> the full-size Conformer trains on the M3 mix; otherwise on
#    train_v2 alone (the M2 Conformer baseline every later full-size comparison needs anyway).
# 4. Full size: constant to 6000, the d06k branch (decay 4000), measures; then the constant run carries on toward 16000
#    until the watchdog stops it (d16k branches later from ckpt_016000).
# Args: scratch dir (closed-phase frames, prediction caches), render-done marker file.
set -u
S="$1"; MARK="$2"
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
RS=data/resynth/vocalset_train_d16k
RV=data/resynth/vocalset_val_d16k
M3="--resynth $RS --resynth-val $RV"
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }

until grep -q "render done" "$MARK" 2>/dev/null; do sleep 30; done
grep -q FAILED "$MARK" && fail "resynth render ($(grep FAILED "$MARK"))"
[ -f $RS/index.json ] && [ -f $RV/index.json ] || fail "resynth sets missing"
say "m3: resynthesised VocalSet ready ($RS, $RV)"

S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
for SEED in 0 1 2; do
  N=m3_conformer_s$SEED; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "m3: $R (S2 Conformer, mixed crops, train_v2 + 30 % resynthesised VocalSet; constant to 2000, decay 1500)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/train_v2 $M3 --steps 2000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
  paired mix_conformer_s$SEED runs/small/mix_conformer_s${SEED}_k35/ckpt_last.pt $N ${R}_k35/ckpt_last.pt
  closed $N ${R}_k35/ckpt_last.pt
  say "m3: $N measured"
done
M="v2_s0,mix_conformer_s0,mix_conformer_s1,mix_conformer_s2,m3_conformer_s0,m3_conformer_s1,m3_conformer_s2"
$PY -I $SA/cp_dev.py "$S" $M m3_conformer_s0-mix_conformer_s0 m3_conformer_s1-mix_conformer_s1 m3_conformer_s2-mix_conformer_s2 \
  > $A/closed_phase/cp_small_m3_dev.txt 2>&1 || fail "cp_dev m3"
$PY -I $SA/decide_m3_s2.py "$S" $F > $A/paired/fp32/decide_m3_s2.txt 2>&1 || fail "decide m3"
DEC=$(grep "^DECISION:" $A/paired/fp32/decide_m3_s2.txt | awk '{print $2}')
say "m3: S2 rule -> $DEC ($(grep -v DECISION $A/paired/fp32/decide_m3_s2.txt | tr '\n' ';' | tr -s ' '))"

if [ "$DEC" = m3 ]; then N=conformer_m3; EXTRA="$M3"; else N=conformer_v2; EXTRA=""; fi
R=runs/$N
mkdir -p $R ${R}_d06k
FULL="--model-set temporal=conformer"
if [ ! -f ${R}_d06k/ckpt_last.pt ]; then
  say "m3: $R full-size Conformer (mixed crops) on train_v2 ${EXTRA:+(+ resynthesised VocalSet)}, constant to 6000"
  [ -f $R/ckpt_006000.pt ] || $VA train teacher --data data/synth/train_v2 $EXTRA --steps 6000 --out $R $FULL --workers 8 \
    --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
  say "m3: ${R}_d06k (decay 4000)"
  $VA train teacher --branch-from $R/ckpt_006000.pt --decay-steps 4000 --workers 8 --out ${R}_d06k >> ${R}_d06k/train.log 2>&1 || fail "${R}_d06k"
fi
paired teacher_v2_d06k runs/teacher_v2_d06k/ckpt_last.pt ${N}_d06k ${R}_d06k/ckpt_last.pt
closed ${N}_d06k ${R}_d06k/ckpt_last.pt
say "m3: ${N}_d06k measured; constant run carries on toward 16000 until the watchdog"
$VA train teacher --data data/synth/train_v2 $EXTRA --steps 16000 --out $R $FULL --workers 8 \
  --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R to 16000"
say "m3: done"
