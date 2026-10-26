#!/usr/bin/env bash
# D-034 student v4 (pre-registered 2026-10-24): targets (crops now; word sets once MFA has aligned every corpus, marked
# by <mfa>/ALIGNED_ALL), WSD training (constant to 16k; branches 8k + 2k and 16k + 4k), fidelity by population for both
# branches, heads on the 20k branch (six on 10 ms frames, six on 80 ms means), the rule, the reported diagnostics; then
# D-031's chain resumes. Every step is skipped if its output exists.
# Run from anywhere: bash d034_v4.sh <scratch dir with m3/ and d031_chain_run.sh>
set -u
SP="$1"
cd /c/github/voice-analyser
W=../voice-analyser-parakeet
R=$W/reports/analysis/parakeet
LOG=runs/orchestrator.log
MFA=C:/github/datasets/mfa_en
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
export PYTHONPATH=/c/github/voice-analyser VA_DEVICE=cuda MIOPEN_FIND_MODE=FAST PYTHONIOENCODING=utf-8
PY=$W/.venv/Scripts/python.exe
P=$W/scripts/parakeet
C4=$W/cache/distil_v4
S=$W/runs/student_v4

v4() {
  if [ ! -f $C4/stats.npz ]; then
    say "d034 v4: targets, crops (90,000 at random phases; held-out sets at all eight)"
    VA_PART=crops $PY -u $P/distil_cache_v4.py $C4 $MFA > $W/cache/distil_v4.out 2>&1 || return 1
  fi
  if [ ! -f $C4/done ]; then
    until [ -f $MFA/ALIGNED_ALL ]; do sleep 60; done
    say "d034 v4: targets, isolated words"
    VA_PART=words $PY -u $P/distil_cache_v4.py $C4 $MFA >> $W/cache/distil_v4.out 2>&1 || return 1
  fi
  for f in $W/cache/distil_v2/fid_*.npz; do [ -f $C4/$(basename $f) ] || cp $f $C4/; done

  if [ ! -f $S/ckpt_016000.pt ]; then
    say "d034 v4: training, constant to 16,000"
    VA_STEPS=16000 $PY -u $P/distil_train_v4.py $C4 $S >> $W/runs/student_v4.out 2>&1 || return 1
  fi
  for br in "10 8 2000" "20 16 4000"; do
    set -- $br
    local B=${S}_b$1k
    if [ ! -f $B/ckpt_0$1000.pt ]; then
      say "d034 v4: branch $1k (decay $3 from $2,000)"
      VA_STEPS=$1000 VA_BRANCH_FROM=$S/ckpt_$(printf %06d $(( $2 * 1000 ))).pt VA_DECAY=$3 \
        $PY -u $P/distil_train_v4.py $C4 $B >> $B.out 2>&1 || return 1
    fi
    if [ ! -s $R/fid_v4_b$1k.txt ]; then
      $PY -u $P/fid_breakdown.py $C4 $B/ckpt_0$1000.pt > $W/runs/fid_v4_b$1k.out 2>&1 || { say "FAILED: d034 v4 fid_breakdown $1k"; return 1; }
      grep -v -i warn $W/runs/fid_v4_b$1k.out > $R/fid_v4_b$1k.txt
    fi
  done

  local CK=${S}_b20k/ckpt_020000.pt SC=$W/cache/stc_v4
  if [ ! -f $SC/meta.json ]; then
    say "d034 v4: student scoring cache (20k branch)"
    $PY -u $P/student_cache.py "$SP/m3" $CK $SC sv4 > $SC.out 2>&1 || return 1
  fi
  if [ ! -f $SC/heads/done ]; then
    say "d034 v4: student head cache (40,000 crops)"
    $PY -u $P/student_cache.py "$SP/m3" $CK $SC sv4 --heads 40000 >> $SC.out 2>&1 || return 1
  fi
  [ -f $W/cache/stc_v4_80ms/done ] || $PY -I $P/pool_heads_cache.py $SC/heads $W/cache/heads_l3_6 $C4 $W/cache/stc_v4_80ms || return 1
  [ -d $W/cache/stc_v4_80ms_scoring/student/gold_ptdb_cp ] || $PY -I $P/student_as_parakeet.py $W/cache/emb_v1 $SC $C4 $W/cache/stc_v4_80ms_scoring > $R/d034_v4_80ms_scoring.txt || return 1
  for k in 0 1 2 3 4 5; do
    for kind in 10ms 80ms; do
      local out=$W/runs/head_v4_${kind}_s$k cache=$SC/heads
      [ $kind = 80ms ] && cache=$W/cache/stc_v4_80ms
      [ -f $out/ckpt_last.pt ] && continue
      say "d034 v4: head ($kind), seed $k"
      $PY -u $P/train_cached.py $cache $out 5 6000 adamw $k > $out.out 2>&1 || return 1
    done
  done
  local A10="" A80=""
  for k in 0 1 2 3 4 5; do A10="$A10 S4#$k=$W/runs/head_v4_10ms_s$k/ckpt_last.pt"; A80="$A80 S4x80#$k=$W/runs/head_v4_80ms_s$k/ckpt_last.pt"; done
  if [ ! -s $R/d034_v4_heads_10ms.txt ]; then
    say "d034 v4: evaluation, 10 ms heads"
    $PY -u $P/eval_heads.py $W/cache/emb_v1 $R/d034_v4_heads_10ms.txt $A10 --feat=sv4=$SC > $R/d034_v4_heads_10ms.out 2>&1 || return 1
  fi
  if [ ! -s $R/d034_v4_heads_80ms.txt ]; then
    say "d034 v4: evaluation, 80 ms heads"
    $PY -u $P/eval_heads.py $W/cache/stc_v4_80ms_scoring/student $R/d034_v4_heads_80ms.txt $A80 > $R/d034_v4_heads_80ms.out 2>&1 || return 1
  fi
  local v="does not transfer"
  grep -q "^  S4x80 (6 seeds).*worth a full arm" $R/d034_v4_heads_80ms.txt && v="transfers at 80 ms"
  grep -q "^  S4 (6 seeds).*worth a full arm" $R/d034_v4_heads_10ms.txt && v="transfers"
  grep -q "d034 v4: DECISION" $LOG || say "d034 v4: DECISION $v"
  [ -s $R/d034_v4_gold_fidelity.txt ] || $PY -I $P/gold_fidelity.py $W/cache/emb_v1 $SC $C4 > $R/d034_v4_gold_fidelity.txt
  [ -s $R/d034_v4_free_part.txt ] || $PY -I $P/free_part.py $W/cache/emb_v1 $SC $W/cache/stc_R $W/cache/stc_v4_box8 > $R/d034_v4_free_part.txt 2>&1
}

v4 || say "FAILED: d034 v4; going on to D-031"
say "d034 v4: queue done; D-031 resumes"
bash "$SP/d031_chain_run.sh" "$SP/m3"
