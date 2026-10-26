#!/usr/bin/env bash
# Roadmap 3 (D-028 "windows", default from 2026-10-27): re-measure the reference models' dev numbers windowed, in a new
# scratch dir so whole-file and windowed columns never mix: closed-phase frames without model columns, each reference
# model added windowed (add_model.py), paired whole-vs-windowed reports per model (Hillenbrand, PTDB-TUG RPA), and a
# summary. Every step is skipped if its output exists.
# Args: whole scratch dir (closed-phase frames), new windowed scratch dir.
set -u
S="$1"; SW="$2"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: windowed refs $*"; exit 1; }
export MIOPEN_FIND_MODE=FAST
PY=./.venv/Scripts/python.exe
A=reports/analysis/windowed
MODELS="conformer_v2r2_s1_d16k conformer_v2r2_s2_d16k conformer_v2r2_d16k d031_rA_b20k"
mkdir -p "$SW" $A
say "windowed refs: start"
for f in ptdb_egg_sub arctic_egg_sub; do
  [ -f "$SW/$f.npz" ] || $PY -c "
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
keep = {'speaker', 'sex', 'voice', 'phone', 'utt', 't', 'f0', 'cq', 'ncyc', 'cp', 'cycle', 'praat'}
np.savez(sys.argv[2], **{k: d[k] for k in d.files if k in keep})" "$S/$f.npz" "$SW/$f.npz" || fail "strip $f"
done
for m in $MODELS; do
  say "windowed refs: $m"
  VA_DEVICE=cuda $PY -u scripts/analysis/add_model.py "$SW" $m runs/$m/ckpt_last.pt _sub > "$SW/add_sub_$m.log" 2>&1 || fail "closed phase $m"
  [ -s $A/${m}_whole_vs_windowed.txt ] || VA_TIERS=dev $PY -I scripts/analysis/m2_paired_sub.py "$SW/m2cache_sub" \
    ${m}_whole=runs/$m/ckpt_last.pt@w0 $m=runs/$m/ckpt_last.pt > $A/${m}_whole_vs_windowed.txt 2>&1 || fail "paired $m"
done
$PY -I scripts/analysis/ref_windowed.py "$S" "$SW" $A $MODELS > $A/reference_windowed.txt 2>&1 || fail "summary"
say "windowed refs: done"
