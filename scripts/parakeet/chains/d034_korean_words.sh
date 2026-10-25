#!/usr/bin/env bash
# D-034 student v4, reported check (amended 2026-10-25, before any result): Korean held-out words. Builds
# held_words_zeroth into its own cache (minimal-pairs' cached MFA alignments of Zeroth-Korean val), then scores every v2
# checkpoint and every v4 checkpoint that exists so far with v4's held-out word measure. Runs beside training; the 20k
# branch is scored by d034_h5_sweep.sh. Every step is skipped if its output exists (fid_v4_on.py skips scored checkpoints).
# Run from anywhere: bash d034_korean_words.sh
set -u
cd /c/github/voice-analyser
W=../voice-analyser-parakeet
R=$W/reports/analysis/parakeet
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
export PYTHONPATH=/c/github/voice-analyser VA_DEVICE=cuda MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
PY=$W/.venv/Scripts/python.exe
P=$W/scripts/parakeet
CK=$W/cache/distil_v4_korean
if [ ! -f $CK/held_words_zeroth/done ]; then
  say "d034 v4 (reported): Korean held-out words, targets"
  VA_PART=korean $PY -u $P/distil_cache_v4.py $CK C:/github/minimal-pairs/data/cache/align > $CK.out 2>&1 || { say "FAILED: d034 Korean words targets"; exit 1; }
fi
V2=$(ls $W/runs/student_v2/ckpt_0*.pt)
V4="$(ls $W/runs/student_v4/ckpt_0*.pt) $W/runs/student_v4_b10k/ckpt_010000.pt"
say "d034 v4 (reported): Korean held-out words, scoring v2 and v4 checkpoints"
VA_ONLY_SETS=$CK $PY -u $P/fid_v4_on.py $W/cache/distil_v4 $W/cache/distil_v2 $R/korean_words_v2.jsonl $V2 > $CK.v2.out 2>&1 || { say "FAILED: d034 Korean words, v2"; exit 1; }
VA_ONLY_SETS=$CK $PY -u $P/fid_v4_on.py $W/cache/distil_v4 $W/cache/distil_v4 $R/korean_words_v4.jsonl $V4 > $CK.v4.out 2>&1 || { say "FAILED: d034 Korean words, v4"; exit 1; }
say "d034 v4 (reported): Korean held-out words scored (v2 12k $(grep -o '"step": 12000.*' $R/korean_words_v2.jsonl | grep -o '"r2": [0-9.]*' | head -1); 10k branch $(grep 'b10k' $R/korean_words_v4.jsonl | grep -o '"r2": [0-9.]*' | head -1))"
