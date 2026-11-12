#!/usr/bin/env bash
# D-099 (registered 2026-11-12): the width control and base's second seed. Two runs side by side, both D-098's base arm
# (same data, recipe, fixed length weights) with one change each: narrow = v3.1's width (conformer_dim 128, embed 384,
# lr 0.002 by the width rule), seed 1; base_s2 = seed 2. D-098's stop rule (d098_watch.py at step 3000) guards both.
# Arg: windowed scratch dir. Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d099 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d099
mkdir -p $F $O
[ -f runs/d098_base_b20k/ckpt_last.pt ] || fail "D-098's base arm missing"
for d in data/synth/train_long_v3u data/synth/train_v2ku; do
  [ -f $d/flat/clip_tube.npy ] && [ -f $d/flat/label_length_inst_cm.npy ] || fail "length labels missing in $d"
done
say "d099: start"

RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
COMMON="--model-set temporal=conformer --workers 8
  --optimizer muon_adan --train-set rest_lr_scale=5 --train-set weight_decay=0.007
  --train-set warmup=1000 --train-set crop_frames=32,400
  --train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=1000
  --train-set vtl_warmup=300 --train-set vtl_gate_ramp=700 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001
  --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0 --train-set vtl_inst=true
  --train-set synth_extra=data/synth/train_v2ku --train-set synth_extra_frac=0.42 --train-set resynth_frac=0.517
  --train-set long_set=data/synth/train_long_v3u --train-set long_batch_frac=0.2"
WIDE="--model-set conformer_dim=192 --model-set embed=576 --train-set lr=0.001333"   # D-098's base
NARROW="--model-set conformer_dim=128 --model-set embed=384 --train-set lr=0.002"   # v3.1's width and rate

arm() {  # name, extra flags
  local R=runs/d099_$1 B=runs/d099_$1_b20k
  mkdir -p $R $B
  [ -f $R/ckpt_016000.pt ] || $VA train teacher --data data/synth/train_v2 $R2 $RV --steps 16000 --out $R $COMMON $2 \
    >> $R/train.log 2>&1 < /dev/null || return 1
  [ -f $B/ckpt_020000.pt ] || $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B \
    >> $B/train.log 2>&1 < /dev/null || return 1
}
kill_arms() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='va.exe'\" | Where-Object { \$_.CommandLine -match 'd099_' } | ForEach-Object { taskkill /F /T /PID \$_.ProcessId }" >> $O/watch.txt 2>&1
}
say "d099: training narrow (seed 1) and base_s2 side by side"
arm narrow "$NARROW --train-set seed=1" & P1=$!
arm base_s2 "$WIDE --train-set seed=2" & P2=$!
if [ ! -f runs/d099_narrow/ckpt_016000.pt ] || [ ! -f runs/d099_base_s2/ckpt_016000.pt ]; then
  if ! $PY -u $SA/d098_watch.py runs/d099_narrow runs/d099_base_s2 >> $O/watch.txt 2>&1 < /dev/null; then
    say "d099: stop rule met at step 3000 ($(tail -2 $O/watch.txt | tr '\n' ' '))"
    kill_arms; kill $P1 $P2 2>/dev/null; wait
    fail "stop rule (gradient norm), see $O/watch.txt"
  fi
  say "d099: stop rule passed ($(tail -2 $O/watch.txt | tr '\n' ' '))"
fi
wait $P1 || fail "train narrow"
wait $P2 || fail "train base_s2"
say "d099: training done"

N=d099_narrow_b20k; S2=d099_base_s2_b20k; S1=d098_base_b20k; B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k
RUNS="$N $S2"
for r in $RUNS; do bash $SA/chains/score_run.sh $r & done
wait
for r in $RUNS; do
  grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
for s in 1 2; do
  paired d040_5b_s${s}_b20k $N
  paired d040_5b_s${s}_b20k $S2
done
paired $N $S1
paired $N $S2
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="5b=$B1+$B2;base=$S1+$S2" \
  VA_SEED_PAIRS="base-narrow=$S1+$S2:$N;base_s2-5b=$S2:$B1+$B2;narrow-5b=$N:$B1+$B2" \
  $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
  $S1=runs/$S1/ckpt_last.pt $S2=runs/$S2/ckpt_last.pt $N=runs/$N/ckpt_last.pt > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
mkdir -p cache/d099 && cp -n cache/d098/*_d040_5b_s?_b20k.* cache/d098/*_$S1.* cache/d098/*_d080_a0_s?.* cache/d099/ 2>/dev/null
ls cache/d099/v2kn_$N.npz cache/d099/v2kn_$S2.npz > /dev/null 2>&1 || $PY -u $SA/d085_score.py $N=runs/$N/ckpt_last.pt $S2=runs/$S2/ckpt_last.pt \
  --out cache/d099 >> runs/d099_post.log 2>&1 < /dev/null || fail "d085_score"
for r in $RUNS; do
  [ -s reports/probes/articulation/$r.json ] || $PY -m vocal_analysis.eval.probes.run score articulation runs/$r/ckpt_last.pt \
    >> runs/d099_post.log 2>&1 < /dev/null || fail "probe 10 $r"
done
[ -s $O/kl/main_per_file.tsv ] || $PY -u $SA/d060_kl_gap.py main $O/kl $(for r in $B1 $B2 $S1 $S2 $N; do printf '%s=runs/%s/ckpt_last.pt ' $r $r; done) \
  >> runs/d099_post.log 2>&1 < /dev/null || fail "kl"
say "d099: scoring done"
[ -f $SA/decide_d099.py ] || fail "decide_d099.py not written yet (scores are complete)"
$PY -I $SA/decide_d099.py "$SW" > $O/decide_d099.txt 2>&1 || fail "decide"
say "d099: $(grep '^DECISION' $O/decide_d099.txt)"
say "d099: done"
