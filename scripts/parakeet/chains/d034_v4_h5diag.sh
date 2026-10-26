#!/usr/bin/env bash
# D-034 student v4, reported diagnostic on the 10k branch (queued 2026-10-24, Hannah; no decision): D-033's H5 heads fed
# the student's 80 ms means in Parakeet's place, with the matched-noise control (student_as_parakeet.py), and the
# student's fidelity on the gold audio. Waits for the 10k branch's fidelity breakdown (written by d034_v4.sh).
# Run from anywhere: bash d034_v4_h5diag.sh <scratch dir with m3/>
set -u
SP="$1"
cd /c/github/voice-analyser
W=../voice-analyser-parakeet
R=$W/reports/analysis/parakeet
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
export PYTHONPATH=/c/github/voice-analyser VA_DEVICE=cuda MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
PY=$W/.venv/Scripts/python.exe
P=$W/scripts/parakeet
CK=$W/runs/student_v4_b10k/ckpt_010000.pt
SC=$W/cache/stc_v4_b10k
D=$W/cache/stc_v4_b10k_h5diag
hdir() { [ "$1" = 0 ] && echo "$W/runs/h5" && return; [ "$1" = 2 ] && echo "$W/runs/h5_seed2" && return; echo "$W/runs/h5_s$1"; }

until [ -s $R/fid_v4_b10k.txt ] || grep -q "FAILED: d034 v4" $LOG; do sleep 120; done
[ -s $R/fid_v4_b10k.txt ] || exit 1
if [ ! -f $SC/meta.json ]; then
  say "d034 v4 (reported): H5-on-student diagnostic, 10k branch"
  VA_SETS=synth_train_v2,synth_train_v2s,gold_hillenbrand,gold_ptdb_cp $PY -u $P/student_cache.py "$SP/m3" $CK $SC sv4b10 > $SC.out 2>&1 || { say "FAILED: d034 v4 h5diag cache"; exit 1; }
fi
[ -s $R/d034_v4_b10k_gold_fidelity.txt ] || $PY -I $P/gold_fidelity.py $W/cache/emb_v1 $SC $W/cache/distil_v4 > $R/d034_v4_b10k_gold_fidelity.txt
[ -d $D/student/gold_ptdb_cp ] || $PY -I $P/student_as_parakeet.py $W/cache/emb_v1 $SC $W/cache/distil_v4 $D > $R/d034_v4_b10k_h5diag_fidelity.txt || { say "FAILED: d034 v4 h5diag stand-in"; exit 1; }
H=""; for k in 0 1 2 3 4 5; do H="$H H5#$k=$(hdir $k)/ckpt_last.pt"; done
for kind in student noise; do
  [ -s $R/d034_v4_b10k_h5_on_$kind.txt ] || $PY -u $P/eval_heads.py $D/$kind $R/d034_v4_b10k_h5_on_$kind.txt $H > $D.$kind.out 2>&1 || { say "FAILED: d034 v4 h5diag eval $kind"; exit 1; }
done
say "d034 v4 (reported): H5 on the 10k branch: $(grep '^  H5 (6' $R/d034_v4_b10k_h5_on_student.txt | sed 's/  */ /g' | cut -c1-160)"
