#!/usr/bin/env bash
# Seed replicate of d06k (train_v2, WSD 6000 constant + 4000 decay, training seed 1): the run-to-run noise floor.
set -u
S="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: $*; stopping"; exit 1; }
VA=./.venv/Scripts/va.exe
say "seed replicate: teacher_v2s1 constant phase to 6000 (train_v2, seed 1)"
$VA train teacher --data data/synth/train_v2 --out runs/teacher_v2s1 --steps 6000 --train-set seed=1 >> runs/teacher_v2s1_train.log 2>&1 || fail "teacher_v2s1"
$VA train teacher --branch-from runs/teacher_v2s1/ckpt_006000.pt >> runs/teacher_v2s1_d06k_train.log 2>&1 || fail "branch v2s1"
[ -f runs/teacher_v2s1_d06k/ckpt_010000.pt ] || fail "no ckpt_010000 in teacher_v2s1_d06k"
say "teacher_v2s1_d06k finished; measures"
./.venv/Scripts/python.exe -I "$S/m2_paired.py" "$S/m2cache" d06k=runs/teacher_v2_d06k/ckpt_010000.pt v2s1=runs/teacher_v2s1_d06k/ckpt_010000.pt > runs/m2_paired_v2s1.txt 2>&1 || fail "paired v2s1"
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u "$S/invariance_run.py" d06k=teacher_v2_d06k/ckpt_010000.pt v2s1=teacher_v2s1_d06k/ckpt_010000.pt > ../voice-analyser/runs/invariance_v2s1.txt 2>&1) || fail "probe v2s1"
(cd ../voice-analyser-m3 && PYTHONPATH=. ../voice-analyser/.venv/Scripts/python.exe -u "$S/add_model.py" "$S" v2s1 ../voice-analyser/runs/teacher_v2s1_d06k/ckpt_010000.pt > "$S/add_v2s1.log" 2>&1) || fail "closed phase v2s1"
say "seed replicate measures done"
