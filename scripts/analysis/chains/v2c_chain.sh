#!/usr/bin/env bash
# D-023 coupling test (D-024 pre-registered rule): train_v2c = train_v2's settings and seed (2) plus glottal coupling on
# every tube clip (p_coupling 1; per-clip glottis, gap and strength from the sampler). Three S2 Conformer seeds, mixed
# crops 32-400 (as mix_conformer_s0-s2 on train_v2), measured in fp32, closed phase on the GPU. Arg: scratch dir.
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
SA=../voice-analyser-m3/scripts/analysis
F=$A/paired/fp32
ck() { echo runs/small/$1_k35/ckpt_last.pt; }
paired() { $PY -I "$SA/m2_paired_sub.py" "$S/m2cache_sub" $1=$(ck $1) $2=$(ck $2) > $F/$1_vs_$2.txt 2>&1 || fail "paired $1 vs $2"; }
if [ ! -f data/synth/train_v2c/flat/done ] && [ ! -f data/synth/train_v2c/index.json ]; then
  say "v2c: generating data/synth/train_v2c (100k clips, seed 2, p_coupling 1, 32 processes)"
  (cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -c "
from dataclasses import replace
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
write_set('../voice-analyser/data/synth/train_v2c', 100_000, replace(SynthConfig(), p_coupling=1.0), seed=2, workers=32,
          extra_index={'note': 'train_v2 + glottal coupling on every tube clip (D-023 / D-024 test)'})
" > "$S/gen_v2c.log" 2>&1) || fail "generate train_v2c"
fi
say "v2c: data ready"
S2="--model-set channels=8,16,32 --model-set f0_channels=8 --model-set embed=192 --model-set gru_hidden=96 --model-set f0_gru_hidden=64 --model-set temporal=conformer"
for SEED in 0 1 2; do
  N=v2c_conformer_s$SEED; R=runs/small/$N
  mkdir -p $R ${R}_k35
  if [ ! -f ${R}_k35/ckpt_last.pt ]; then
    say "v2c: $R (constant to 2000, decay 1500; crop_frames 32-400)"
    [ -f $R/ckpt_002000.pt ] || $VA train teacher --data data/synth/train_v2c --steps 2000 --out $R $S2 --workers 4 \
      --train-set seed=$SEED --train-set comb_truth_steps=1400 --train-set crop_frames=32,400 >> $R/train.log 2>&1 || fail "$R"
    $VA train teacher --branch-from $R/ckpt_002000.pt --decay-steps 1500 --workers 4 --out ${R}_k35 >> ${R}_k35/train.log 2>&1 || fail "${R}_k35"
    [ -f ${R}_k35/ckpt_last.pt ] || fail "no ckpt_last in ${R}_k35"
  fi
  paired v2_s0 $N
  paired mix_conformer_s$SEED $N
  (cd ../voice-analyser-m3 && VA_DEVICE=cuda PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" $N "../voice-analyser/$(ck $N)" _sub > "$S/add_sub_$N.log" 2>&1) || fail "closed phase $N"
  (cd ../voice-analyser-m3 && VA_DEVICE=cuda PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/add_model.py "$S" $N "../voice-analyser/$(ck $N)" > "$S/add_full_$N.log" 2>&1) || true  # full set, if its frames file exists
done
M="v2_s0 mix_conformer_s0 mix_conformer_s1 mix_conformer_s2 v2c_conformer_s0 v2c_conformer_s1 v2c_conformer_s2"
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u scripts/analysis/invariance_run.py \
  $(for n in $M; do echo -n "$n=small/${n}_k35/ckpt_last.pt "; done) > ../voice-analyser/$A/invariance/small_v2c.txt 2>&1) || fail "probe v2c"
(cd ../voice-analyser-m3 && CP_SUFFIX=_sub ../voice-analyser/.venv/Scripts/python.exe -I scripts/analysis/cp_cmp.py "$S" $(echo $M | tr ' ' ',') \
  v2c_conformer_s0-mix_conformer_s0 v2c_conformer_s1-mix_conformer_s1 v2c_conformer_s2-mix_conformer_s2 > ../voice-analyser/$A/closed_phase/cp_small_v2c.txt 2>&1) || fail "cp_cmp v2c"
say "v2c: done"
