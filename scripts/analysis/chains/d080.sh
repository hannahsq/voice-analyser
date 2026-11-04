#!/usr/bin/env bash
# D-080 (registered 2026-11-04): v3.1's final stage (decay branch from d040_5b_s{1,2}/ckpt_016000.pt, 4000 steps) with
# train_v2kb at 42 % of items (A0), plus a glottal-opening head (A1); branch seeds 1 and 2. Then scoring (val_v2ko,
# uncoupled and resynthesised validation), PTDB-TUG closed-phase frames, Hillenbrand (paired against v3.1), KL and the
# decision. Waits for train_v2kb. Arg: windowed scratch dir. Every step skips if its output exists.
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d080 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
O=reports/analysis/d080
mkdir -p $O
say "d080: waiting for train_v2kb"
until [ -f data/synth/train_v2kb/flat/done ] && grep -q "train_v2kb: done" runs/d080_render.log; do sleep 300; done
say "d080: start"
X="--train-set synth_extra=data/synth/train_v2kb --train-set synth_extra_frac=0.42 --train-set resynth_frac=0.517"
OP="--train-set opening=true --train-set opening_lr=0.001"
RUNS=""
for s in 1 2; do
  for arm in a0 a1; do
    r=d080_${arm}_s$s; RUNS="$RUNS $r"
    extra="$X"; [ $arm = a1 ] && extra="$X $OP"
    if [ ! -f runs/$r/ckpt_last.pt ] || ! grep -q '"step": 20000' runs/$r/log.jsonl; then
      say "d080: training $r"
      mkdir -p runs/$r
      $VA train teacher --branch-from runs/d040_5b_s$s/ckpt_016000.pt --decay-steps 4000 --workers 8 --out runs/$r $extra \
        >> runs/$r/train.log 2>&1 || fail "train $r"
    fi
  done
done
say "d080: training done"
$PY $SA/d080_score.py $RUNS d040_5b_s1_b20k d040_5b_s2_b20k >> runs/d080_score.log 2>&1 || fail "score"
for r in $RUNS; do
  grep -q "'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
mkdir -p $O/paired
for s in 1 2; do
  for arm in a0 a1; do
    r=d080_${arm}_s$s
    [ -s $O/paired/d040_5b_s${s}_b20k_vs_$r.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" \
      d040_5b_s${s}_b20k=runs/d040_5b_s${s}_b20k/ckpt_last.pt $r=runs/$r/ckpt_last.pt > $O/paired/d040_5b_s${s}_b20k_vs_$r.txt 2>&1 || fail "paired $r"
  done
done
$PY -I $SA/hill_signed.py "$SW/m2cache_sub" d040_5b_s1_b20k d040_5b_s2_b20k $RUNS > $O/hillenbrand_signed.txt 2>&1 || fail "signed"
[ -s $O/kl/main_per_file.tsv ] || $PY $SA/d060_kl_gap.py main $O/kl $(for r in $RUNS; do printf '%s=runs/%s/ckpt_last.pt ' $r $r; done) \
  > runs/d080_kl.log 2>&1 || fail "kl"
say "d080: scoring done"
$PY -I $SA/decide_d080.py "$SW" > runs/d080_decide.log 2>&1 || fail "decide"
say "d080: $(grep '^DECISION' $O/decide_d080.txt)"
say "d080: done"
