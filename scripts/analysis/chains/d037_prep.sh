#!/usr/bin/env bash
# D-037 / D-038 preparation, safe to cut off at any point (a low-power or battery window): feature caches only, which
# show no result, then D-036's remaining head-seed replicate. Order: arm 3's trunk (D-037 decides on it), 1s2's resume
# (D-036 amendment; d036_seeds.sh skips 1s1, whose outputs exist), then teacher v3's, arm 1's and arm 2's trunks. Every
# step resumes or skips: the train cache from its progress file, 1s2 from ckpt_last.pt, finished parts by their "done".
# Args: windowed scratch dir (for 1s2's measures, as d036_seeds.sh), deadline for the seed step as HH:MM [23:59].
set -u
SW="$1"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d037 prep $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
C=cache/d037

cache() {  # name, checkpoint
  say "d037 prep: caching $1"
  $PY -u scripts/analysis/d037_cache.py "$2" $C/$1 >> $C/$1.log 2>&1 || fail "cache $1"
}

mkdir -p $C
say "d037 prep: start"
cache arm3 runs/d036_arm3/ckpt_last.pt
say "d037 prep: d036 1s2"
bash scripts/analysis/chains/d036_seeds.sh "$SW" "${2:-23:59}" || fail "d036 seeds"
cache tv3 runs/conformer_v2r2_s1_d16k/ckpt_last.pt
cache arm1 runs/d036_arm1/ckpt_last.pt
cache arm2 runs/d036_arm2/ckpt_last.pt
say "d037 prep: done"
