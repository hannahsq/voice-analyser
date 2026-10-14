"""Signed Hillenbrand formant error per group (D-026 fail branch): median of model minus Hillenbrand's reference
(Hz and cents), F1-F3, from the m2_paired_sub.py prediction caches (dev tier, as cached). Hillenbrand's references
are whole-vowel LPC measurements, not closed-phase ones: the second reference for women's F3.
Args: cache dir, model names (cache prefixes)."""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
C = Path(sys.argv[1])
print("Hillenbrand, model minus reference, median over tokens: F1 / F2 / F3 (Hz) | (cents)")
for name in sys.argv[2:]:
    f = C / f"{name}__hillenbrand.npz"
    if not f.exists():
        print(f"  {name}: no cache")
        continue
    d = np.load(f, allow_pickle=True)
    for grp in np.unique(d["group"]):
        m = d["group"] == grp
        hz, ce = [], []
        for j in range(3):
            ok = m & np.isfinite(d["ref"][:, j]) & np.isfinite(d["est"][:, j]) & (d["ref"][:, j] > 0)
            hz.append(np.median(d["est"][ok, j] - d["ref"][ok, j]))
            ce.append(np.median(1200 * np.log2(d["est"][ok, j] / d["ref"][ok, j])))
        print(f"  {name:24s} {grp:6s} " + " / ".join(f"{x:+6.1f}" for x in hz) + "  | " + " / ".join(f"{x:+5.1f}" for x in ce))
