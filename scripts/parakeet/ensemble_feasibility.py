"""D-034 ensemble feasibility: per-member signed F1 bias and frame-level error correlations (log frequency).
No ensemble is computed here. Args: dump dir (P/v2/v4 _ptdb.npz and _hillenbrand.npz)."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
TEACH = ["teacher_v3", "tv3_s0", "tv3_s2", "tv2"]
FEATS = ["P", "v2", "v4"]

for tag, groups in (("ptdb", {"women": "f", "men": "m"}), ("hillenbrand", {"all": None})):
    dd = {f: dict(np.load(D / f"{f}_{tag}.npz", allow_pickle=True)) for f in FEATS}
    ref = dd["P"]["ref"]
    for f in FEATS[1:]:
        assert np.array_equal(dd[f]["ref"], ref, equal_nan=True), f"{tag}: {f} frames differ"
    M = {t: dd["P"]["m_" + t] for t in TEACH}
    for f in FEATS:
        seeds = [dd[f][f"m_H5#{k}"] for k in range(6)]
        for k, x in enumerate(seeds):
            M[f"H5/{f}#{k}"] = x
        M[f"H5/{f}"] = np.mean(seeds, 0)
    grp = dd["P"]["group"]
    print(f"\n######## {tag}")
    for gname, gv in groups.items():
        sel = (np.ones(len(ref), bool) if gv is None else grp == gv) & np.isfinite(ref) & (ref > 0)
        for x in M.values():
            sel &= np.isfinite(x)
        lr = np.log2(ref[sel])
        E = {k: v[sel] - lr for k, v in M.items()}
        print(f"\n## {gname}: {sel.sum()} frames")
        print("signed median bias (Hz) / MAE (Hz):")
        for k in TEACH + [f"H5/{f}" for f in FEATS]:
            e = 2 ** M[k][sel] - ref[sel]
            print(f"  {k:12s} {np.median(e):+6.1f} / {np.mean(np.abs(e)):5.1f}")
        r = lambda a, b: float(np.corrcoef(E[a], E[b])[0, 1])  # noqa: E731
        print("error correlations, single members (H5 as six-seed means):")
        keys = TEACH + [f"H5/{f}" for f in FEATS]
        print(" " * 13 + "".join(f"{k:>11s}" for k in keys))
        for a in keys:
            print(f"  {a:11s}" + "".join(f"{r(a, b):11.2f}" for b in keys))
        print("noise floor and groups (mean pairwise r):")
        print(f"  teacher v3 seeds          {np.mean([r(a, b) for a, b in combinations(TEACH[:3], 2)]):.2f}")
        print(f"  teacher v2 vs v3 seeds    {np.mean([r('tv2', b) for b in TEACH[:3]]):.2f}")
        for f in FEATS:
            print(f"  H5 seeds on {f:3s}           {np.mean([r(f'H5/{f}#{a}', f'H5/{f}#{b}') for a, b in combinations(range(6), 2)]):.2f}")
        for f, h in combinations(FEATS, 2):
            print(f"  H5 {f} vs {h} (same seed)   {np.mean([r(f'H5/{f}#{k}', f'H5/{h}#{k}') for k in range(6)]):.2f}")
        tv = [r(t, f"H5/{f}#{k}") for t in TEACH for f in FEATS for k in range(6)]
        print(f"  teacher vs H5 seed        {np.mean(tv):.2f} (range {min(tv):.2f} to {max(tv):.2f})")
