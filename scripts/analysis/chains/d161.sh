#!/usr/bin/env bash
# D-161 (registered 2026-11-28): warm restarts from d040_5b_s{1,2}/ckpt_016000.pt on train_v2w (4000 constant steps,
# then the 4000-step decay: --decay-steps 8000, decay_hold 4000), with a stage-1 source head and a fourth stage-2 input
# channel: arm E (source=pred, the predicted source envelope, truth ramp on synthetic clips) and arm Z (source=zero, the
# matched control), on the re-rendered resynthesised sets (40 source harmonics, beta*). First a 200-step smoke run (arm E,
# seed 1); then the four runs; then D-160's guards on the GPU for both arms against 5b. The CPU scoring and the reading
# are d161_score.py's, run after this chain. Every step skips if its output exists.
# Arg: windowed scratch dir holding ptdb_egg_sub.npz and m2cache_sub [cache/m3w].
set -u
SW="${1:-cache/m3w}"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d161 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
A=reports/analysis
F=$A/paired/w400
O=$A/d161
R=data/resynth
mkdir -p $F $O
for s in vocalset_train_d16k zeroth_korean_train_d16k vocalset_val_d16k zeroth_korean_val_d16k; do
  [ -f $R/${s}_h40/flat/label_beta_star.npy ] || fail "beta* missing for $s (d161_residue.py assemble)"
done
[ -f data/synth/train_v2w/flat/label_source_h_db40.npy ] && [ -f data/synth/train_v2/flat/label_source_h_db40.npy ] \
  || fail "synthetic 40-harmonic labels missing (d161_source_labels.py)"
X="--train-set synth_extra=data/synth/train_v2w --train-set synth_extra_replace=true --train-set decay_hold=4000"
X="$X --train-set resynth=$R/vocalset_train_d16k_h40,$R/zeroth_korean_train_d16k_h40"
X="$X --train-set resynth_val=$R/vocalset_val_d16k_h40,$R/zeroth_korean_val_d16k_h40"
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k

train() {  # run, seed, source, [extra]
  local r=$1 s=$2 src=$3 last=${4:-24000}
  if [ ! -f runs/$r/ckpt_last.pt ] || ! grep -q "\"step\": $last" runs/$r/log.jsonl; then
    say "d161: training $r"
    mkdir -p runs/$r
    $VA train teacher --branch-from runs/d040_5b_s$s/ckpt_016000.pt --decay-steps 8000 --workers 12 --out runs/$r $X \
      --train-set source=$src ${5:-} >> runs/$r/train.log 2>&1 || fail "train $r"
  fi
}

say "d161: start (smoke)"
train d161_smoke 1 pred 16200 "--train-set steps=16200"
VA_DEVICE=cpu $PY $SA/d161_smoke_check.py > runs/d161_smoke_check.log 2>&1 || fail "smoke checks (reports/analysis/d161/smoke.txt)"
say "d161: smoke checks pass"
for s in 1 2; do
  train d161_env_s$s $s pred
  train d161_zero_s$s $s zero
done
say "d161: training done"

closed() {  # name
  grep -q "\"$1\"\|'$1'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $1 runs/$1/ckpt_last.pt _sub > "$SW/add_sub_$1.log" 2>&1 || fail "closed phase $1"
}
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
probes() {  # run
  local ck=runs/$1/ckpt_last.pt L=runs/d161_probes_$1.log
  for p in counterfactual bandwidth polezero chain articulation shift timing voice stretch bwhold gain valley; do
    [ -s reports/probes/$p/$1.json ] || $PY -m vocal_analysis.eval.probes.run score $p $ck >> $L 2>&1 || fail "probe $p $1"
  done
  [ -s reports/probes/channel/$1.json ] || $PY -m vocal_analysis.eval.probes.channel score $ck >> $L 2>&1 || fail "probe channel $1"
  [ -s reports/probes/scale/$1.json ] || $PY -m vocal_analysis.eval.probes.scale score $ck >> $L 2>&1 || fail "probe scale $1"
  [ -s reports/probes/reversal/$1.json ] || $PY -m vocal_analysis.eval.probes.reversal $ck >> $L 2>&1 || fail "probe reversal $1"
  [ -s reports/probes/bw_evidence/$1.json ] || $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> $L 2>&1 || fail "probe bw_evidence $1"
}

say "d161: guards"
for a in env zero; do
  closed d161_${a}_s1; closed d161_${a}_s2
  paired $B1 d161_${a}_s1
  paired $B2 d161_${a}_s2
done
E1=d161_env_s1; E2=d161_env_s2; Z1=d161_zero_s1; Z2=d161_zero_s2
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="env=$E1+$E2;zero=$Z1+$Z2;5b=$B1+$B2" \
  VA_SEED_PAIRS="env-5b=$E1+$E2:$B1+$B2;zero-5b=$Z1+$Z2:$B1+$B2;env-zero=$E1+$E2:$Z1+$Z2;s1=$E1:$B1;s2=$E2:$B2;z1=$Z1:$B1;z2=$Z2:$B2" \
  $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
  $E1=runs/$E1/ckpt_last.pt $E2=runs/$E2/ckpt_last.pt $Z1=runs/$Z1/ckpt_last.pt $Z2=runs/$Z2/ckpt_last.pt > $O/vtl_joint.out 2>&1 \
  || fail "vtl scoring"
for r in $E1 $E2 $Z1 $Z2; do probes $r; done
$PY -I $SA/decide_d065.py bw "$SW" $O/vtl_joint.json env-5b $B1:$E1:s1 $B2:$E2:s2 > $O/decide_guards_env.txt 2>&1 || fail "guards env"
$PY -I $SA/decide_d065.py bw "$SW" $O/vtl_joint.json zero-5b $B1:$Z1:z1 $B2:$Z2:z2 > $O/decide_guards_zero.txt 2>&1 || fail "guards zero"
say "d161: guards done"
say "d161: done"
