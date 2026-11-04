#!/usr/bin/env bash
# D-079 (registered 2026-11-04): frozen opening probes on D-023's paired small Conformers (mix, v2c2, v2c; three seeds)
# and v3.1 (two seeds). Features (GPU, serial), fits, decision. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d079 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
SA=scripts/analysis
S=runs/small
say "d079: start"
T=""
for s in 0 1 2; do
  T="$T mix_s$s=$S/mix_conformer_s${s}_k35/ckpt_last.pt v2c2_s$s=$S/v2c2_conformer_s${s}_k35/ckpt_last.pt v2c_s$s=$S/v2c_conformer_s${s}_k35/ckpt_last.pt"
done
T="$T v31_s1=runs/d040_5b_s1_b20k/ckpt_last.pt v31_s2=runs/d040_5b_s2_b20k/ckpt_last.pt"
for t in $T; do
  n=${t%%=*}
  [ -f cache/d079/${n}_val.npz ] || { say "d079: features $n"; $PY $SA/d079_probe.py features $t >> runs/d079_probe.log 2>&1 || fail "features $n"; }
done
for t in $T; do
  n=${t%%=*}
  [ -f cache/d079/${n}_fit.json ] || $PY $SA/d079_probe.py fit $n >> runs/d079_probe.log 2>&1 || fail "fit $n"
done
mkdir -p reports/analysis/d079
$PY $SA/decide_d079.py >> runs/d079_probe.log 2>&1 || fail "decide"
cp cache/d079/*_fit.json reports/analysis/d079/
say "d079: $(grep '^DECISION' reports/analysis/d079/decide_d079.txt)"
say "d079: done"
