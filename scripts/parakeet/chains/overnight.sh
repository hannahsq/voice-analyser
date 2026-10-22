#!/usr/bin/env bash
# D-033 overnight (2026-10-22, Hannah): (A) seeds 1-5 for the four Parakeet heads, decisions on six-seed means (recipe,
# then layer set; H3456 retrained on the S2 recipe if that recipe wins); (B) part 5, DAC: scoring cache, 63 ridge
# combinations, head cache of the F1 winner's taps, six DAC heads on the winning recipe, bias analysis with the winning
# Parakeet head group and the three-way ensemble per seed pair; (C) hands over to D-031's chain in the main repo.
# Every step is skipped if its output exists. Run from anywhere: bash overnight.sh <scratch dir with m3/ptdb_egg_sub.npz>
set -u
SP="$1"
cd /c/github/voice-analyser
W=../voice-analyser-parakeet
R=$W/reports/analysis/parakeet
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d033 $*; stopping"; exit 1; }
export PYTHONPATH=/c/github/voice-analyser VA_DEVICE=cuda MIOPEN_FIND_MODE=FAST
PY=$W/.venv/Scripts/python.exe
PYD=$W/.venv-dac/Scripts/python.exe
PC=$W/cache/heads_l3_6

dir() { [ "$2" = 0 ] && echo "$W/runs/$1" || echo "$W/runs/$1_s$2"; }  # seed 0 runs predate the seed suffix
train() {  # arm, cache, layers, steps, recipe, seed
  local out; out=$(dir "$1" "$6")
  [ -f "$out/ckpt_last.pt" ] && return 0
  say "d033: train $1 seed $6 ($3, $4 steps, $5)"
  $PY -u $W/scripts/parakeet/train_cached.py "$2" "$out" "$3" "$4" "$5" "$6" > "$out.out" 2>&1 || fail "train $1 seed $6"
}
heads_args() {  # label, arm -> "label#k=ckpt" for seeds 0-5
  for k in 0 1 2 3 4 5; do echo -n "$1#$k=$(dir "$2" $k)/ckpt_last.pt "; done
}

# ---- (A) Parakeet heads, six seeds
for k in 1 2 3 4 5; do
  train h5 $PC 5 6000 adamw $k
  train h3456 $PC 3,4,5,6 6000 adamw $k
  train h5_s2 $PC 5 6000 s2 $k
  train h5_s2s $PC 5 3500 s2 $k
done
if [ ! -s $R/heads_seeds.txt ]; then
  say "d033: six-seed evaluation of the Parakeet heads"
  $PY -u $W/scripts/parakeet/eval_heads.py $W/cache/emb_v1 $R/heads_seeds.txt $(heads_args H5 h5) $(heads_args H3456 h3456) \
    $(heads_args H5S2 h5_s2) $(heads_args H5S2s h5_s2s) col:s2_s0 col:s2_s1 col:s2_s2 > $R/heads_seeds.out 2>&1 || fail "eval heads_seeds"
fi
if grep -q "part-4b trigger on means.*S2 recipe for heads" $R/heads_seeds.txt; then
  RECIPE=s2; P5=h5_s2
  for k in 0 1 2 3 4 5; do train h3456_s2r $PC 3,4,5,6 6000 s2 $k; done
  if [ ! -s $R/heads_seeds_s2.txt ]; then
    $PY -u $W/scripts/parakeet/eval_heads.py $W/cache/emb_v1 $R/heads_seeds_s2.txt $(heads_args H5 h5_s2) $(heads_args H3456 h3456_s2r) \
      > $R/heads_seeds_s2.out 2>&1 || fail "eval heads_seeds_s2"
  fi
  LAYERFILE=$R/heads_seeds_s2.txt; P3456=h3456_s2r
else
  RECIPE=adamw; P5=h5; P3456=h3456; LAYERFILE=$R/heads_seeds.txt
fi
grep -q "part-4 rule on means.*H3456 better" $LAYERFILE && PARM=$P3456 || PARM=$P5
say "d033: recipe $RECIPE; Parakeet partner $PARM"

# ---- (B) part 5: DAC
if [ ! -f $W/cache/dac_v1/meta.json ]; then
  say "d033: DAC scoring cache"
  $PYD -u $W/scripts/parakeet/dac_cache.py "$SP" $W/cache/dac_v1 > $W/cache/dac_v1.out 2>&1 || fail "dac cache"
fi
if [ ! -f $R/dac_combo/combo_results.json ] || ! grep -q '"winners"' $R/dac_combo/combo_results.json; then
  say "d033: DAC ridge combinations"
  $PY -u $W/scripts/parakeet/combo.py $W/cache/dac_v1 $R/dac_combo > $R/dac_combo.out 2>&1 || fail "dac combos"
fi
TAPS=$($PY -c "import json; w = json.load(open('$R/dac_combo/combo_results.json'))['winners']['F1']; print(','.join(t[1:] for t in w.split('+')))")
say "d033: DAC F1 winner taps $TAPS"
if [ ! -f $W/cache/dac_heads/done ]; then
  $PYD -u $W/scripts/parakeet/dac_cache.py "$SP" $W/cache/dac_heads --heads "$TAPS" 40000 > $W/cache/dac_heads.out 2>&1 || fail "dac head cache"
fi
for k in 0 1 2 3 4 5; do train dac_head $W/cache/dac_heads "$TAPS" 6000 $RECIPE $k; done
if [ ! -s $R/dac_part5.txt ]; then
  THREE=""
  for k in 0 1 2 3 4 5; do THREE="$THREE three:P#$k,DAC#$k"; done
  $PY -u $W/scripts/parakeet/eval_heads.py $W/cache/emb_v1 $R/dac_part5.txt $(heads_args P "$PARM") $(heads_args DAC dac_head) \
    --dac=$W/cache/dac_v1 $THREE > $R/dac_part5.out 2>&1 || fail "eval part 5"
fi
say "d033: overnight done"

# ---- (C) D-031 in the main repo (its own chain, its own copy)
if [ -f scripts/analysis/chains/d031_chain.sh ]; then
  cp scripts/analysis/chains/d031_chain.sh "$SP/d031_chain_run.sh"
  bash "$SP/d031_chain_run.sh" "$SP/m3"
fi
