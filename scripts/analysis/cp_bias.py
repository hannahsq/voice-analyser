"""Closed-phase bias per model (PTDB-TUG dev, ptdb_egg_sub.npz): median of model minus closed phase, F1-F3, per sex,
in Hz and cents, on the frames where every listed model and the reference have a value. Lines:
  BIAS <model> <sex> <F1> <F2> <F3>   (Hz)    and    CENTS <model> <sex> <F1> <F2> <F3>
Args: scratch dir, model names (npz keys)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, ms = Path(sys.argv[1]), sys.argv[2:]
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
for sex in ("f", "m"):
    for m in ms:
        hz, ce = [], []
        for j in range(3):
            ok = dev & (d["sex"] == sex) & np.isfinite(d["cp"][:, j]) & np.all([np.isfinite(d[x][:, j]) for x in ms], 0)
            hz.append(np.median(d[m][ok, j] - d["cp"][ok, j]))
            ce.append(np.median(1200 * np.log2(d[m][ok, j] / d["cp"][ok, j])))
        print(f"BIAS {m} {sex} " + " ".join(f"{x:+.1f}" for x in hz))
        print(f"CENTS {m} {sex} " + " ".join(f"{x:+.1f}" for x in ce))
