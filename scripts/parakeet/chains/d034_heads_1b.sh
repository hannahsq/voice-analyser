#!/usr/bin/env bash
# D-034 after student v2 (queued 2026-10-23, Hannah): the stage-1 head stage, then stage 1b if the student transfers,
# then D-031's chain in the main repo (resumed from where it was paused) in either case.
#   0. waits for student v2's 7k-12k fidelity breakdown;
#   1. pipeline check (no decision): student v1 step 2000, a 4,000-crop head cache, one head, scored;
#   2. student v2 step 12,000 and control R (its initial weights): scoring caches, 40,000-crop head caches, six heads
#      each (seeds 0-5, train_cached.py's AdamW recipe, 6,000 steps), scored with H5x6 beside; the stage-1 rule on seed
#      means: "transfers" = S2 passes and R fails;
#   3. "transfers" only: stage 1b, student v3 (v2's blend x3, 36,000 steps), its heads, decide_1b.py;
#   4. D-031.
# Head caches (about 33 GB each) are deleted once their heads are trained, except the students that may become stage 2's
# trunk (KEEP: its ensemble distillation of the six F1 heads reads the same features). Every step is skipped if its output exists.
# Run from anywhere: bash d034_heads_1b.sh <scratch dir with m3/ptdb_egg_sub.npz and d031_chain_run.sh>
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
KEEP="v2 v3"  # head caches kept (Hannah freed disk, 2026-10-23)

hdir() {  # D-033's H5 run dirs (seed 0 has no suffix; seed 2 is h5_seed2)
  [ "$1" = 0 ] && echo "$W/runs/h5" && return
  [ "$1" = 2 ] && echo "$W/runs/h5_seed2" && return
  echo "$W/runs/h5_s$1"
}
cache() {  # name, checkpoint, encoder label, head crops
  local C=$W/cache/stc_$1
  if [ ! -f $C/meta.json ]; then
    say "d034: student scoring cache $1"
    $PY -u $P/student_cache.py "$SP/m3" "$2" $C "$3" > $C.out 2>&1 || return 1
  fi
  if [ ! -f $C/heads/done ] && [ ! -f $C/heads_trained ]; then
    say "d034: student head cache $1 ($4 crops)"
    $PY -u $P/student_cache.py "$SP/m3" "$2" $C "$3" --heads "$4" >> $C.out 2>&1 || return 1
  fi
}
heads() {  # name, seeds...
  local C=$W/cache/stc_$1 n=$1
  shift
  for k in "$@"; do
    local out=$W/runs/head_${n}_s$k
    [ -f $out/ckpt_last.pt ] && continue
    say "d034: head on $n, seed $k"
    $PY -u $P/train_cached.py $C/heads $out 5 6000 adamw $k > $out.out 2>&1 || return 1
  done
  case " $KEEP " in *" $n "*) ;; *) rm -f $C/heads/emb.npy ;; esac  # the 33 GB features; labels kept
  touch $C/heads_trained
}
args() {  # label, name -> "label#k=ckpt" for seeds 0-5
  for k in 0 1 2 3 4 5; do echo -n "$1#$k=$W/runs/head_${2}_s$k/ckpt_last.pt "; done
}
h5args() { for k in 0 1 2 3 4 5; do echo -n "H5#$k=$(hdir $k)/ckpt_last.pt "; done; }

stage() {
  until grep -q "d034: student v2 fidelity breakdown done\|FAILED: d034 v2" $LOG; do sleep 60; done
  grep -q "FAILED: d034 v2" $LOG && return 1
  local V1=$W/runs/student_v1/ckpt_002000.pt V2=$W/runs/student_v2/ckpt_012000.pt V0=$W/runs/student_v2/ckpt_init.pt
  [ -f $V2 ] || { say "FAILED: d034 no step-12000 checkpoint"; return 1; }
  [ -s $R/fid_v2_7000_12000.txt ] || cp "$SP/fid_v2_7000_12000.txt" $R/
  [ -s $R/fid_v2_12000.txt ] || sed -n '/^ckpt_012000.pt:/,$p' $R/fid_v2_7000_12000.txt > $R/fid_v2_12000.txt
  [ -s $R/fid_v2_12000.txt ] || { say "FAILED: d034 no step-12000 breakdown"; return 1; }

  # 1. pipeline check
  if [ ! -s $R/d034_pipeline_check.txt ]; then
    cache v1_2000 $V1 sv1 4000 || return 1
    heads v1_2000 0 || return 1
    say "d034: pipeline check (student v1 step 2000, one head)"
    $PY -u $P/eval_heads.py $W/cache/emb_v1 $R/d034_pipeline_check.txt "SV1=$W/runs/head_v1_2000_s0/ckpt_last.pt" \
      --feat=sv1=$W/cache/stc_v1_2000 > $R/d034_pipeline_check.out 2>&1 || return 1
  fi

  # 2. student v2 and control R
  if [ ! -s $R/d034_heads.txt ]; then
    cache v2 $V2 sv2 40000 || return 1
    heads v2 0 1 2 3 4 5 || return 1
    cache R $V0 R 40000 || return 1
    heads R 0 1 2 3 4 5 || return 1
    say "d034: stage-1 evaluation (S2 x6, R x6, H5 x6)"
    $PY -u $P/eval_heads.py $W/cache/emb_v1 $R/d034_heads.txt $(args S2 v2) $(args R R) $(h5args) \
      --feat=sv2=$W/cache/stc_v2 --feat=R=$W/cache/stc_R > $R/d034_heads.out 2>&1 || return 1
  fi
  local s2 r
  s2=$(grep -c "^  S2 (6 seeds).*worth a full arm" $R/d034_heads.txt)
  r=$(grep -c "^  R (6 seeds).*worth a full arm" $R/d034_heads.txt)
  local verdict="does not transfer"
  [ "$s2" = 1 ] && [ "$r" = 0 ] && verdict="transfers"
  [ "$s2" = 1 ] && [ "$r" = 1 ] && verdict="frozen encoder suffices"
  grep -q "d034: DECISION stage 1" $LOG || say "d034: DECISION stage 1: $verdict"
  [ "$verdict" = transfers ] || return 0

  # 3. stage 1b
  local V3=$W/runs/student_v3
  if [ ! -f $W/cache/distil_v3/done ]; then
    local free; free=$(powershell -NoProfile -c '[int]((Get-PSDrive C).Free / 1GB)' | tr -dc 0-9)  # df hangs on the NAS mount
    [ "$free" -ge 100 ] || { say "FAILED: d034 1b needs about 100 GB free, $free GB"; return 1; }
    say "d034: stage 1b targets (270,000 + 2,048 crops)"
    VA_MIX=v3 $PY -u $P/distil_cache.py $W/cache/distil_v3 > $W/cache/distil_v3.out 2>&1 || return 1
  fi
  for f in $W/cache/distil_v2/fid_*.npz; do [ -f $W/cache/distil_v3/$(basename $f) ] || cp $f $W/cache/distil_v3/; done
  if [ ! -f $V3/ckpt_036000.pt ]; then
    say "d034: stage 1b student v3, 36,000 steps"
    VA_STEPS=36000 VA_EVERY=2000 $PY -u $P/distil_train.py $W/cache/distil_v3 $V3 >> $W/runs/student_v3.out 2>&1 || return 1
  fi
  [ -s $R/fid_v3_36000.txt ] || $PY -u $P/fid_breakdown.py $W/cache/distil_v3 $V3/ckpt_036000.pt 2>&1 | grep -v -i warn > $R/fid_v3_36000.txt
  if [ ! -s $R/d034_1b_heads.txt ]; then
    cache v3 $V3/ckpt_036000.pt sv3 40000 || return 1
    heads v3 0 1 2 3 4 5 || return 1
    say "d034: stage 1b evaluation (S3 x6 against S2 x6)"
    $PY -u $P/eval_heads.py $W/cache/emb_v1 $R/d034_1b_heads.txt $(args S2 v2) $(args S3 v3) $(h5args) \
      --feat=sv2=$W/cache/stc_v2 --feat=sv3=$W/cache/stc_v3 > $R/d034_1b_heads.out 2>&1 || return 1
  fi
  $PY -u $P/decide_1b.py $R/fid_v2_12000.txt $R/fid_v3_36000.txt $R/d034_1b_heads.txt \
    $W/runs/student_v2/log.jsonl $V3/log.jsonl > $R/d034_1b_decision.txt 2>&1 || return 1
  say "d034: $(grep '^DECISION' $R/d034_1b_decision.txt) (stage 1b)"
}

stage || say "FAILED: d034 head stage / 1b; going on to D-031"
say "d034: queue done; D-031 resumes"
bash "$SP/d031_chain_run.sh" "$SP/m3"
