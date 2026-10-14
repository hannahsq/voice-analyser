#!/usr/bin/env bash
# Days 2-3 (pre-registered 2026-10-13 in D-026, before any v2t result was read), after "day1: done" and "zeroth: resynthesis ready".
# Base set from decide_v2i.txt: train_v2i / train_v2t, or train_v2 on "v2"; stops on "fix" (high-frequency loss first).
# 1. M3 speech, S2 seeds 0-2: base + resynthesised VocalSet ("r1"; on train_v2 that is m3_conformer, already run) and
#    base + VocalSet + Zeroth-Korean ("r2"), 30 % resynthesised items split by set size (about half speech, half singing).
#    Against the base seeds (paired, closed phase); decide_m3_s2.py (the M3 no-degradation rule) on r2 against base.
# 2. On "m3": full-size Conformer on base + both (the teacher v3 candidate, conformer_<base>r2), to d16k, against the
#    full-size base. Otherwise no candidate.
# 3. Second full-size seed (seed 1) of the main comparison: conformer_v2 and the candidate (v3 if made, else the base
#    set's full-size run, else conformer_m3), both to d16k, paired; identity test on all full-size models.
# Args: scratch dir (closed-phase frames, prediction caches); optionally a base set (v2i / v2t / v2m / v2x / v2, or auto) and a log line
# to wait for instead of the day-1 decision (2026-10-14: v2m after the dose arm, D-026).
set -u
S="$1"
WAIT="${3:-day1: done}"
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
RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R1="--resynth data/resynth/vocalset_train_d16k"
R2="$R1 --resynth data/resynth/zeroth_korean_train_d16k"
k35() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$S/m2cache_sub" $1=$2 $3=$4 > $F/$1_vs_$3.txt 2>&1 || fail "paired $1 vs $3"; }
closed() { VA_DEVICE=cuda $PY -u $SA/add_model.py "$S" $1 $2 _sub > "$S/add_sub_$1.log" 2>&1 || fail "closed phase $1"; }
full() {  # name, seed, data args...
  local N=$1 SEED=$2; shift 2
  local R=runs/$N RD=runs/${N}_d16k
  mkdir -p $R $RD
  if [ ! -f $RD/ckpt_last.pt ]; then
    if [ ! -f $R/ckpt_016000.pt ]; then
      say "m3s: $R full-size Conformer (mixed crops, seed $SEED), constant to 16000"
      $VA train teacher "$@" --steps 16000 --out $R --model-set temporal=conformer --workers 8 \
        --train-set crop_frames=32,400 --train-set seed=$SEED >> $R/train.log 2>&1 || fail "$R"
    fi
    [ -f $R/ckpt_016000.pt ] || fail "no ckpt_016000 in $R"
    say "m3s: $RD (decay 4000)"
    $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $RD >> $RD/train.log 2>&1 || fail "$RD"
  fi
  closed ${N}_d16k $RD/ckpt_last.pt
}

NF=$(grep -c FAILED $LOG)
until grep -q "$WAIT" $LOG && grep -q "zeroth: resynthesis" $LOG; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "earlier chain failed; days 2-3 not started"; exit 1; }
  sleep 300
done
grep -q "zeroth: resynthesis ready" $LOG || fail "zeroth resynthesis missing"
DEC=${2:-$(sed -n 's/^DECISION: //p' $A/decide_v2i.txt)}
# "auto": the base select_base.py chose after the dose arms (D-026)
[ "$DEC" = auto ] && DEC=$(sed -n 's/^DECISION: //p' $A/decide_base.txt)
case "$DEC" in
  v2i|v2t|v2m|v2x) BASE=${DEC}_conformer; DATA=data/synth/train_$DEC; ARM1=${DEC}r1_conformer; FBASE=conformer_$DEC ;;
  v2) BASE=mix_conformer; DATA=data/synth/train_v2; ARM1=m3_conformer; FBASE=conformer_v2 ;;
  *) say "m3s: day-1 decision '$DEC'; days 2-3 wait for the high-frequency fix"; exit 0 ;;
esac
P=${DEC}r2
say "m3s: base $DATA ($DEC); arms $ARM1 (+ VocalSet), ${P}_conformer (+ VocalSet + Zeroth)"

# 1. S2 seeds
for SEED in 0 1 2; do
  for spec in "$ARM1 R1" "${P}_conformer R2"; do
    set -- $spec; N=$1_s$SEED; R=runs/small/$N
    [ "$2" = R1 ] && RS=$R1 || RS=$R2
    mkdir -p $R ${R}_k35
    if [ ! -f ${R}_k35/ckpt_last.pt ]; then
      say "m3s: $R (constant to 2000, decay 1500; crop_frames 32-400)"
      [ -f $R/ckpt_002000.pt ] || $VA train teacher --data $DATA $RS $RV --steps 2000 --out $R $S2 --workers 4 \
        --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
      $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
      [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
    fi
    closed $N $(k35 $N)
    paired ${BASE}_s$SEED $(k35 ${BASE}_s$SEED) $N $(k35 $N)
  done
  paired ${ARM1}_s$SEED $(k35 ${ARM1}_s$SEED) ${P}_conformer_s$SEED $(k35 ${P}_conformer_s$SEED)
  say "m3s: seed $SEED measured"
done
M=$(for a in $BASE $ARM1 ${P}_conformer; do for k in 0 1 2; do echo -n "${a}_s$k,"; done; done)
$PY -I $SA/cp_dev.py "$S" ${M%,} $(for k in 0 1 2; do echo -n "${ARM1}_s$k-${BASE}_s$k ${P}_conformer_s$k-${BASE}_s$k ${P}_conformer_s$k-${ARM1}_s$k "; done) \
  > $A/closed_phase/cp_small_${P}_dev.txt 2>&1 || fail "cp_dev $P"
$PY -I $SA/decide_m3_s2.py "$S" $F $BASE ${P}_conformer > $A/decide_${P}_s2.txt 2>&1 || fail "decide $P"
D2=$(sed -n 's/^DECISION: //p' $A/decide_${P}_s2.txt)
say "m3s: S2 done; DECISION ($P against $BASE): $D2"

# 2. teacher v3 candidate
CAND=$FBASE
[ "$CAND" = conformer_v2 ] && CAND=conformer_m3
if [ "$D2" = m3 ]; then
  full conformer_$P 0 --data $DATA $R2 $RV
  paired ${FBASE}_d16k runs/${FBASE}_d16k/ckpt_last.pt conformer_${P}_d16k runs/conformer_${P}_d16k/ckpt_last.pt
  $PY -I $SA/cp_dev.py "$S" d16k,conformer_v2_d16k,conformer_m3_d16k,${FBASE}_d16k,conformer_${P}_d16k \
    conformer_${P}_d16k-${FBASE}_d16k conformer_${P}_d16k-conformer_v2_d16k > $A/closed_phase/cp_conformer_${P}_d16k_dev.txt 2>&1 || fail "cp_dev conformer_$P"
  CAND=conformer_$P
  say "m3s: teacher v3 candidate conformer_${P}_d16k measured"
fi

# 3. second full-size seed of conformer_v2 and the candidate
full conformer_v2_s1 1 --data data/synth/train_v2
case $CAND in
  conformer_m3) full ${CAND}_s1 1 --data data/synth/train_v2 $R1 --resynth-val data/resynth/vocalset_val_d16k ;;
  conformer_$P) full ${CAND}_s1 1 --data $DATA $R2 $RV ;;
  *) full ${CAND}_s1 1 --data $DATA ;;
esac
paired conformer_v2_s1_d16k runs/conformer_v2_s1_d16k/ckpt_last.pt ${CAND}_s1_d16k runs/${CAND}_s1_d16k/ckpt_last.pt
$PY -I $SA/cp_dev.py "$S" conformer_v2_d16k,${CAND}_d16k,conformer_v2_s1_d16k,${CAND}_s1_d16k \
  ${CAND}_d16k-conformer_v2_d16k ${CAND}_s1_d16k-conformer_v2_s1_d16k > $A/closed_phase/cp_full_seeds_${CAND}_dev.txt 2>&1 || fail "cp_dev full seeds"
VA_DEVICE=cuda $PY -u $SA/resynth_identity.py 120 d16k=runs/teacher_v2_d16k/ckpt_last.pt \
  conformer_v2_d16k=runs/conformer_v2_d16k/ckpt_last.pt conformer_v2_s1_d16k=runs/conformer_v2_s1_d16k/ckpt_last.pt \
  ${CAND}_d16k=runs/${CAND}_d16k/ckpt_last.pt ${CAND}_s1_d16k=runs/${CAND}_s1_d16k/ckpt_last.pt \
  > $A/sensitivity/resynth_identity_full_seeds.txt 2>&1 || say "m3s: identity test failed (not fatal)"
say "m3s: done"
