#!/usr/bin/env bash
# D-027 (pre-registered 2026-10-17, before any run): the observability head, S2 Conformer seeds 0-2 on train_v2
# (obs_conformer_sN = mix_conformer_sN's settings plus `obs_head=true`; the head is built last, so initial weights and
# data match the baseline's), each to k35, against the same-seed mix_conformer; decide_obs_s2.py.
# 0. Waits for the v3 seed-2 chain ("v3s2: done") and train_v2's observability store (its log's "done in").
# 1. Replica check: 50 steps of mix_conformer_s0's command on current code against its log (replica_check.py). If the
#    loss parts differ by more than 0.5 %, fresh baselines (mixr_conformer_s0-2, the same command now) replace it.
# Args: scratch dir (closed-phase frames, prediction caches), the store's log.
set -u
S="$1"
STORE_LOG="$2"
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
small() {  # name, seed, extra args...
  local N=$1_s$2 SEED=$2; shift 2
  local R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "obs: $R (constant to 2000, decay 1500; crop_frames 32-400)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/train_v2 --steps 2000 --out $R $S2 "$@" --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
}
NF=$(grep -c FAILED $LOG)
until grep -q "v3s2: done" $LOG && grep -q "^done in" "$STORE_LOG"; do
  [ "$(grep -c FAILED $LOG)" -gt "$NF" ] && { echo "an earlier chain failed; obs not started"; exit 1; }
  sleep 300
done
[ -f data/synth/train_v2/flat/obs_sig.npy ] || fail "train_v2 store missing"
say "obs: start (D-027 head arm)"
# 1. replica check
R=runs/small/mixrep_conformer_s0
if [ ! -f $A/observability/replica_mix_conformer_s0.txt ]; then
  rm -rf $R; mkdir -p $R
  $VA train teacher --data data/synth/train_v2 --steps 50 --out $R $S2 --workers 4 --train-set seed=0 --train-set comb_truth_steps=1400 \
    --train-set crop_frames=32,400 --train-set eval_every=100000 --train-set ckpt_every=100000 >> $R/train.log 2>&1 || fail "$R"
  $PY -I $SA/replica_check.py runs/small/mix_conformer_s0 $R > $A/observability/replica_mix_conformer_s0.txt 2>&1 || fail "replica check"
fi
REP=$(sed -n 's/^REPLICA: //p' $A/observability/replica_mix_conformer_s0.txt)
BASE=mix_conformer
if [ "$REP" != yes ]; then
  BASE=mixr_conformer
  say "obs: current code does not reproduce mix_conformer_s0's start; fresh baselines $BASE"
fi
say "obs: replica $REP; base $BASE"
# 2. seeds
for SEED in 0 1 2; do
  [ $BASE = mixr_conformer ] && { small mixr_conformer $SEED; closed mixr_conformer_s$SEED $(k35 mixr_conformer_s$SEED); }
  small obs_conformer $SEED --model-set obs_head=true
  closed obs_conformer_s$SEED $(k35 obs_conformer_s$SEED)
  paired ${BASE}_s$SEED $(k35 ${BASE}_s$SEED) obs_conformer_s$SEED $(k35 obs_conformer_s$SEED)
  say "obs: seed $SEED measured"
done
# 3. decision
$PY -I $SA/decide_m3_s2.py "$S" $F $BASE obs_conformer > $A/observability/noharm_obs_s2.txt 2>&1 || fail "no-degradation rule"
$PY -I $SA/decide_obs_s2.py "$S" $A/observability/noharm_obs_s2.txt > $A/observability/decide_obs_s2.txt 2>&1 || fail "decide obs"
say "obs: DECISION $(sed -n 's/^DECISION: //p' $A/observability/decide_obs_s2.txt)"
say "obs: done"
