"""Full-size check of an interactive set (D-026 fail branch, pre-registered 2026-10-15): the d16k limits of
decide_v2m.py on one full-size model instead of the small pair. Exploratory: a pass is evidence that the small-model
failure was a small-model effect and goes back to Hannah; it does not carry the set forward.
  women's closed-phase F1 bias within +-10 Hz, F3 bias >= -10 Hz (PTDB-TUG dev, frames where the model and both
  full-size train_v2 seeds have values); identity resynthesis within +-5 Hz.
Args: scratch dir, model (npz key), identity report, identity key."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, M, IDENT, KEY = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
REF = ["conformer_v2_d16k", "conformer_v2_s1_d16k"]
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])


def bias(m, sex, j):
    ok = dev & (d["sex"] == sex) & np.isfinite(d["cp"][:, j]) & np.all([np.isfinite(d[x][:, j]) for x in [M, *REF]], 0)
    return float(np.median(d[m][ok, j] - d["cp"][ok, j]))


print("Closed phase (PTDB-TUG dev), model minus reference, Hz: F1 / F2 / F3")
for sex in ("f", "m"):
    for m in [*REF, M]:
        print(f"  {sex} {m:26s} " + " / ".join(f"{bias(m, sex, j):+6.1f}" for j in range(3)))
fails = []
b1, b3 = bias(M, "f", 0), bias(M, "f", 2)
print(f"\nwomen's F1 bias {b1:+.1f} (limit +-10), F3 {b3:+.1f} (limit >= -10)")
fails += [] if abs(b1) <= 10 else ["F1 bias"]
fails += [] if b3 >= -10 else ["F3 bias"]
ident = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S+)\s+[+-][\d.]+\s+([+-][\d.]+)", IDENT.read_text(encoding="utf-8"), re.M)}
print(f"identity resynthesis {ident[KEY]:+.1f} (limit +-5)")
fails += [] if abs(ident[KEY]) <= 5 else ["identity"]
print(f"\nfailures: {fails or 'none'}")
print(f"DECISION: {'fail' if fails else 'pass'}")
