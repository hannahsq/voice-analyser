#!/usr/bin/env bash
# One run's per-run scoring, safe to run in parallel with other runs' (each writes only its own files): D-053's frames,
# probes 11 (scale), 12 (bw_evidence, after the bwhold / counterfactual / bandwidth probes it reads) and 14 (valley).
# The closed-phase frames (add_model.py: one shared npz) and the paired measures stay serial in the calling chain.
# Arg: run name (runs/<run>/ckpt_last.pt). Every step skips if its output exists.
cd /c/github/voice-analyser
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe; SA=scripts/analysis
r=$1; ck=runs/$r/ckpt_last.pt; L=runs/score_$r.log
ok=0
ls cache/d053/vocalset_$r.npz > /dev/null 2>&1 || $PY -c "import sys; sys.path.insert(0, '$SA'); import d053_iseli_alwan as D
if __name__ == '__main__': D.score('$ck')" >> $L 2>&1 || ok=1
[ -s reports/probes/scale/$r.json ] || $PY -m vocal_analysis.eval.probes.scale score $ck >> $L 2>&1 || ok=1
for p in bwhold counterfactual bandwidth valley; do
  [ -s reports/probes/$p/$r.json ] || $PY -m vocal_analysis.eval.probes.run score $p $ck >> $L 2>&1 || ok=1
done
[ -s reports/probes/bw_evidence/$r.json ] || $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> $L 2>&1 || ok=1
[ $ok = 0 ] || echo "$(date '+%F %T') FAILED: scoring $r" >> runs/orchestrator.log
exit $ok
