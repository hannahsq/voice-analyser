#!/usr/bin/env bash
# v1 decay branch at 6000 (cosine rate there 0.84e-3, decayed 1 - sqrt to 0 at 10k like d06k): the v1 -> v2 comparison
# at matched steps and decay. Runs after the seed replicate measures; the short study waits for it.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
PY=./.venv/Scripts/python.exe
say "v1 branch: waiting for the seed replicate measures"
since() { tail -n +"$(grep -n "v1 branch: waiting for the seed replicate measures" "$LOG" | tail -1 | cut -d: -f1)" "$LOG"; }
until since | grep -q "seed replicate measures done"; do
  since | grep -q "FAILED" && fail "an earlier chain failed; v1 branch not started"; sleep 60; done
say "v1 branch: runs/teacher_v1_d06k (branch at 6000, decay 4000)"
$VA train teacher --branch-from runs/teacher_v1/ckpt_006000.pt --decay-steps 4000 --out runs/teacher_v1_d06k >> runs/teacher_v1_d06k_train.log 2>&1 || fail "teacher_v1_d06k"
[ -f runs/teacher_v1_d06k/ckpt_010000.pt ] || fail "no ckpt_010000 in teacher_v1_d06k"
say "v1 branch trained; measures"
CK=runs/teacher_v1_d06k/ckpt_010000.pt
$PY -I "$S/m2_paired.py" "$S/m2cache" v1d06k=$CK d06k=runs/teacher_v2_d06k/ckpt_010000.pt > runs/m2_paired_v1d06k.txt 2>&1 || fail "paired v1d06k"
$PY -I "$S/m2_paired.py" "$S/m2cache" v1d06k=$CK v2s1=runs/teacher_v2s1_d06k/ckpt_010000.pt > runs/m2_paired_v1d06k_v2s1.txt 2>&1 || fail "paired v1d06k v2s1"
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u "$S/invariance_run.py" d06k=teacher_v2_d06k/ckpt_010000.pt \
   v2s1=teacher_v2s1_d06k/ckpt_010000.pt v1d06k=teacher_v1_d06k/ckpt_010000.pt v1=teacher_v1/ckpt_020000.pt > ../voice-analyser/runs/invariance_v1d06k.txt 2>&1) || fail "probe v1d06k"
say "v1 branch GPU measures done"
# closed phase on the CPU, alongside the short study
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u "$S/add_model.py" "$S" v1d06k ../voice-analyser/$CK > "$S/add_v1d06k.log" 2>&1) || fail "closed phase v1d06k"
say "v1 branch measures done"
