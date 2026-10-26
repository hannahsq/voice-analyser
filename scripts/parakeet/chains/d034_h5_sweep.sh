#!/usr/bin/env bash
# D-034, reported sweep (pre-registered 2026-10-25; no decision): H5 on the student by checkpoint, as the 10k-branch
# diagnostic (d034_v4_h5diag.sh): D-033's six H5 heads fed the student's 80 ms means, the matched-noise control, and gold
# fidelity. v2 at 4k and 8k, v4's constant run at 4k, 8k, 12k and 16k, the 20k branch; then the 20k branch's Korean-word
# score. Runs after the pre-registered 20k-branch steps (called at the top of D-031's chain). Each checkpoint's caches are
# deleted once its reports exist (the 20k branch's scoring cache belongs to the pre-registered heads and is kept).
# Run from anywhere: bash d034_h5_sweep.sh <scratch dir with m3/>
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
hdir() { [ "$1" = 0 ] && echo "$W/runs/h5" && return; [ "$1" = 2 ] && echo "$W/runs/h5_seed2" && return; echo "$W/runs/h5_s$1"; }
H=""; for k in 0 1 2 3 4 5; do H="$H H5#$k=$(hdir $k)/ckpt_last.pt"; done

one() {  # tag, checkpoint, distillation cache (stats), scoring cache (empty: build a temporary one)
  local tag=$1 ck=$2 dc=$3 sc=${4:-} rep=$R/d034_h5sweep_$1
  [ -s ${rep}_on_noise.txt ] && return 0
  local SC=${sc:-$W/cache/stc_sweep_$tag} D=$W/cache/stc_sweep_${tag}_h5diag
  say "d034 (reported): H5 sweep, $tag"
  if [ ! -f $SC/meta.json ]; then
    VA_SETS=synth_train_v2,synth_train_v2s,gold_hillenbrand,gold_ptdb_cp $PY -u $P/student_cache.py "$SP/m3" $ck $SC sw_$tag > $SC.out 2>&1 || { say "FAILED: d034 H5 sweep $tag cache"; return 1; }
  fi
  [ -s ${rep}_gold_fidelity.txt ] || $PY -I $P/gold_fidelity.py $W/cache/emb_v1 $SC $dc > ${rep}_gold_fidelity.txt
  [ -d $D/student/gold_ptdb_cp ] || $PY -I $P/student_as_parakeet.py $W/cache/emb_v1 $SC $dc $D > ${rep}_fidelity.txt || { say "FAILED: d034 H5 sweep $tag stand-in"; return 1; }
  for kind in student noise; do
    [ -s ${rep}_on_$kind.txt ] || $PY -u $P/eval_heads.py $D/$kind ${rep}_on_$kind.txt $H > $D.$kind.out 2>&1 || { say "FAILED: d034 H5 sweep $tag eval $kind"; return 1; }
  done
  rm -rf $D
  [ -z "$sc" ] && rm -rf $SC
  say "d034 (reported): H5 sweep, $tag: $(grep '^  H5 (6' ${rep}_on_student.txt | sed 's/  */ /g' | cut -c1-120)"
}

V2=$W/runs/student_v2 V4=$W/runs/student_v4
one v2_4k $V2/ckpt_004000.pt $W/cache/distil_v2
one v2_8k $V2/ckpt_008000.pt $W/cache/distil_v2
for s in 4 8 12 16; do one v4_${s}k $V4/ckpt_$(printf %06d $((s * 1000))).pt $W/cache/distil_v4; done
B20=${V4}_b20k/ckpt_020000.pt
[ -f $B20 ] && one v4_b20k $B20 $W/cache/distil_v4 $W/cache/stc_v4
[ -f $B20 ] && VA_ONLY_SETS=$W/cache/distil_v4_korean $PY -u $P/fid_v4_on.py $W/cache/distil_v4 $W/cache/distil_v4 $R/korean_words_v4.jsonl $B20 >> $W/cache/distil_v4_korean.v4.out 2>&1
say "d034 (reported): H5 sweep done"
exit 0
