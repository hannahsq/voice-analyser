#!/usr/bin/env bash
# D-054: one run's per-run scoring (D-053 frames, probes) in parallel with the chain; closed-phase and paired stay in the chain.
cd /c/github/voice-analyser
export MIOPEN_FIND_MODE=FAST VA_DEVICE=cuda
PY=./.venv/Scripts/python.exe; SA=scripts/analysis
r=$1; ck=runs/$r/ckpt_last.pt; L=runs/d054_helper_$r.log
ls cache/d053/vocalset_$r.npz > /dev/null 2>&1 || $PY -c "import sys; sys.path.insert(0, '$SA'); import d053_iseli_alwan as D
if __name__ == '__main__': D.score('$ck')" >> $L 2>&1
[ -s reports/probes/scale/$r.json ] || $PY -m vocal_analysis.eval.probes.scale score $ck >> $L 2>&1
for p in bwhold counterfactual bandwidth; do
  [ -s reports/probes/$p/$r.json ] || $PY -m vocal_analysis.eval.probes.run score $p $ck >> $L 2>&1
done
[ -s reports/probes/bw_evidence/$r.json ] || $PY -m vocal_analysis.eval.probes.bw_evidence $ck >> $L 2>&1
echo "$(date '+%F %T') d054 helper: $r scored (exit $?)" >> runs/orchestrator.log
