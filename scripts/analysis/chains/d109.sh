#!/usr/bin/env bash
# D-109 (registered 2026-11-13): the reconstruction loss on real audio in 5b's decay branch. Both seeds side by side:
# --branch-from runs/d040_5b_sN/ckpt_016000.pt, 4000 decay steps, plus model_recon_loss on the recon stores (VocalSet
# and Zeroth training files), weight and fmax by D-109's rules (args). Stop rule: d109_watch.py at step 17000 (gradient
# norm and formant term against 5b's own branch); on a stop both arms' process trees are killed and nothing is scored.
# Then D-090's scoring and decide_d109.py. Args: windowed scratch dir, recon weight, recon fmax, higher-pole correction (true/false).
# Every step skips if its output exists.
set -u
SW="$1"; W="$2"; FMAX="$3"; HPC="$4"
cd /c/github/voice-analyser
LOG=runs/orchestrator.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
fail() { say "FAILED: d109 $*; stopping"; exit 1; }
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe
VA=./.venv/Scripts/va.exe
SA=scripts/analysis
F=reports/analysis/paired/w400
O=reports/analysis/d109
mkdir -p $F $O

for s in cache/recon/vocalset cache/recon/zeroth; do [ -f $s/index.json ] || fail "recon store $s missing"; done
say "d109: start (recon weight $W, fmax $FMAX, hpc $HPC)"

arm() {  # seed
  local R=runs/d109_s$1
  mkdir -p $R
  [ -f $R/ckpt_020000.pt ] || $VA train teacher --branch-from runs/d040_5b_s$1/ckpt_016000.pt --decay-steps 4000 --workers 8 \
    --out $R --train-set recon_stores=cache/recon/vocalset,cache/recon/zeroth --train-set recon_weight=$W \
    --train-set recon_fmax=$FMAX --train-set recon_hpc=$HPC >> $R/train.log 2>&1 < /dev/null
}
kill_arms() {  # the va.exe trees training d109 runs (loader workers included)
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='va.exe'\" | Where-Object { \$_.CommandLine -match 'd109_s' } | ForEach-Object { taskkill /F /T /PID \$_.ProcessId }" >> $O/watch.txt 2>&1
}
say "d109: training both seeds side by side"
arm 1 & P1=$!
arm 2 & P2=$!
if [ ! -f runs/d109_s1/ckpt_020000.pt ] || [ ! -f runs/d109_s2/ckpt_020000.pt ]; then
  if ! $PY -u $SA/d109_watch.py runs/d109_s1:runs/d040_5b_s1_b20k runs/d109_s2:runs/d040_5b_s2_b20k >> $O/watch.txt 2>&1 < /dev/null; then
    say "d109: stop rule met ($(tail -2 $O/watch.txt | tr '\n' ' '))"
    kill_arms; kill $P1 $P2 2>/dev/null; wait
    fail "stop rule, see $O/watch.txt"
  fi
  say "d109: stop rule passed ($(tail -2 $O/watch.txt | tr '\n' ' '))"
fi
wait $P1 || fail "train s1"
wait $P2 || fail "train s2"
say "d109: training done"

RUNS="d109_s1 d109_s2"
for r in $RUNS; do bash $SA/chains/score_run.sh $r & done
wait
for r in $RUNS; do
  grep -q "\"$r\"\|'$r'" <($PY -c "import numpy as np; print(list(np.load('$SW/ptdb_egg_sub.npz').files))") \
    || $PY -u $SA/add_model.py "$SW" $r runs/$r/ckpt_last.pt _sub > "$SW/add_sub_$r.log" 2>&1 || fail "closed phase $r"
done
paired() {  # ref, run
  [ -s $F/$1_vs_$2.txt ] || VA_TIERS=dev $PY -I $SA/m2_paired_sub.py "$SW/m2cache_sub" $1=runs/$1/ckpt_last.pt $2=runs/$2/ckpt_last.pt \
    > $F/$1_vs_$2.txt 2>&1 || fail "paired $2"
}
for s in 1 2; do paired d040_5b_s${s}_b20k d109_s$s; done
B1=d040_5b_s1_b20k; B2=d040_5b_s2_b20k; C1=d109_s1; C2=d109_s2
[ -s $O/vtl_joint.json ] || VA_SEED_MEANS="recon=$C1+$C2;5b=$B1+$B2" VA_SEED_PAIRS="recon-5b=$C1+$C2:$B1+$B2;s1=$C1:$B1;s2=$C2:$B2" \
  $PY -u $SA/vtl_score_joint.py $O/vtl_joint.txt $B1=runs/$B1/ckpt_last.pt $B2=runs/$B2/ckpt_last.pt \
  $C1=runs/$C1/ckpt_last.pt $C2=runs/$C2/ckpt_last.pt > $O/vtl_joint.out 2>&1 || fail "vtl scoring"
say "d109: scoring done"
$PY -I $SA/decide_d109.py "$SW" > $O/decide_d109.txt 2>&1 || fail "decide"
say "d109: $(grep '^DECISION' $O/decide_d109.txt)"
say "d109: done"
