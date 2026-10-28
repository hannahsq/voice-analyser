#!/usr/bin/env bash
# D-041 (pre-registered 2026-10-28): probe and shift parts through teacher v3's three seeds and D-031's rA / rB / rC, then
# the invariance comparison. Every step skips if its output exists.
set -u
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d041 $*; stopping"; exit 1; }
export VA_DEVICE=cuda OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=./.venv/Scripts/python.exe
C=cache/d041
say "d041: start"
for m in v3_s0:conformer_v2r2_d16k v3_s1:conformer_v2r2_s1_d16k v3_s2:conformer_v2r2_s2_d16k rA:d031_rA_b20k rB:d031_rB_b20k rC:d031_rC_b20k; do
  n=${m%%:*}; r=${m#*:}
  mkdir -p $C/$n
  $PY -u scripts/analysis/d037_cache.py runs/$r/ckpt_last.pt $C/$n probe shift >> $C/$n/cache.log 2>&1 || fail "cache $n"
done
mkdir -p reports/analysis/d041
$PY -u scripts/analysis/d041_invariance.py reports/analysis/d041/d041.txt v3_s0=$C/v3_s0 v3_s1=$C/v3_s1 v3_s2=$C/v3_s2 \
  rA=$C/rA rB=$C/rB rC=$C/rC > reports/analysis/d041/d041.out 2>&1 || fail "scoring"
say "d041: $(grep '^DECISION' reports/analysis/d041/d041.txt)"
