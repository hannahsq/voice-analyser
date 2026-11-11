#!/usr/bin/env bash
# D-098 (registered 2026-11-11): D-085 rerun with D-036's fixed length weights. As d085.sh (same data, recipe, seed,
# arms, scoring and decision script), with one change: no aux_budget, so aVTL and iVTL train at losses.WEIGHTS' 0.0125
# each. Stop rule: d098_watch.py at step 3000 (gradient norm after the VTL gate opens <= 2x before it); on a stop both
# arms' process trees are killed and nothing is scored. Arg: windowed scratch dir. Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d098 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda VA_D085_TAG=d098
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d098
mkdir -p $F $O

for d in data/synth/train_long_v3u data/synth/train_v2ku; do
  [ -f $d/flat/clip_tube.npy ] && [ -f $d/flat/label_length_inst_cm.npy ] || fail "length labels missing in $d (D-085 made them)"
done
say "d098: start"

RV="--resynth-val data/resynth/vocalset_val_d16k --resynth-val data/resynth/zeroth_korean_val_d16k"
R2="--resynth data/resynth/vocalset_train_d16k --resynth data/resynth/zeroth_korean_train_d16k"
BASE="--model-set temporal=conformer --model-set conformer_dim=192 --model-set embed=576 --workers 8
  --optimizer muon_adan --train-set lr=0.001333 --train-set rest_lr_scale=5 --train-set weight_decay=0.007
  --train-set warmup=1000 --train-set crop_frames=32,400 --train-set seed=1
  --train-set vtl=new --train-set vtl_start=0 --train-set vtl_schedule=model --train-set vtl_detach=1000
  --train-set vtl_warmup=300 --train-set vtl_gate_ramp=700 --train-set vtl_optimizer=adamw --train-set vtl_lr=0.001
  --train-set vtl_weight_decay=0.01 --train-set vtl_seed=0 --train-set vtl_inst=true
  --train-set synth_extra=data/synth/train_v2ku --train-set synth_extra_frac=0.42 --train-set resynth_frac=0.517
  --train-set long_set=data/synth/train_long_v3u --train-set long_batch_frac=0.2"
HEADS="--train-set acoustic=true --train-set vtl_link=true --train-set upper=3 --train-set upper_lr=0.001"

arm() {  # name, extra flags
  local R=runs/d098_$1 B=runs/d098_$1_b20k
  mkdir -p $R $B
  [ -f $R/ckpt_016000.pt ] || $VA train teacher --data data/synth/train_v2 $R2 $RV --steps 16000 --out $R $BASE $2 \
    >> $R/train.log 2>&1 < /dev/null || return 1
  [ -f $B/ckpt_020000.pt ] || $VA train teacher --branch-from $R/ckpt_016000.pt --decay-steps 4000 --workers 8 --out $B \
    >> $B/train.log 2>&1 < /dev/null || return 1
}
kill_arms() {  # the va.exe trees training d098 runs (loader workers included)
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='va.exe'\" | Where-Object { \$_.CommandLine -match 'd098_' } | ForEach-Object { taskkill /F /T /PID \$_.ProcessId }" >> $O/watch.txt 2>&1
}
say "d098: training base and heads side by side"
arm base "" & P1=$!
arm heads "$HEADS" & P2=$!
if [ ! -f runs/d098_base/ckpt_016000.pt ]; then
  if ! $PY -u $SA/d098_watch.py runs/d098_base runs/d098_heads >> $O/watch.txt 2>&1 < /dev/null; then
    say "d098: stop rule met at step 3000 ($(tail -2 $O/watch.txt | tr '\n' ' '))"
    kill_arms; kill $P1 $P2 2>/dev/null; wait
    fail "stop rule (gradient norm), see $O/watch.txt"
  fi
  say "d098: stop rule passed ($(tail -2 $O/watch.txt | tr '\n' ' '))"
fi
wait $P1 || fail "train base"
wait $P2 || fail "train heads"
say "d098: training done"

RUNS="d098_base_b20k d098_heads_b20k"
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
  paired d040_5b_s${s}_b20k d098_base_b20k
  paired d040_5b_s${s}_b20k d098_heads_b20k
done
paired d098_base_b20k d098_heads_b20k
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k; CB=d098_base_b20k; CH=d098_heads_b20k
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="5b=$B1+$B2" VA_SEED_PAIRS="base-5b=$CB:$B1+$B2;heads-5b=$CH:$B1+$B2;heads-base=$CH:$CB" \
  $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
  $CB=runs/$CB/ckpt_last.pt $CH=runs/$CH/ckpt_last.pt > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
ALL="$B1 $B2 $CB $CH"
mkdir -p cache/d098 && cp -n cache/d085/*_d040_5b_s?_b20k.* cache/d098/ 2>/dev/null  # 5b's scores are the same as D-085's
ls cache/d098/v2kn_$CH.npz cache/d098/v2kn_$CB.npz > /dev/null 2>&1 || $PY -u $SA/d085_score.py $(for r in $ALL; do printf '%s=runs/%s/ckpt_last.pt ' $r $r; done) --out cache/d098 \
  >> runs/d098_post.log 2>&1 < /dev/null || fail "d085_score"
for r in $RUNS; do
  [ -s reports/probes/articulation/$r.json ] || $PY -m vocal_analysis.eval.probes.run score articulation runs/$r/ckpt_last.pt \
    >> runs/d098_post.log 2>&1 < /dev/null || fail "probe 10 $r"
done
[ -s $O/kl/main_per_file.tsv ] || $PY -u $SA/d060_kl_gap.py main $O/kl $(for r in $ALL; do printf '%s=runs/%s/ckpt_last.pt ' $r $r; done) \
  >> runs/d098_post.log 2>&1 < /dev/null || fail "kl"
say "d098: scoring done"
$PY -I $SA/decide_d085.py "$SW" > $O/decide_d098.txt 2>&1 || fail "decide"
say "d098: $(grep '^DECISION' $O/decide_d098.txt)"
say "d098: done"
