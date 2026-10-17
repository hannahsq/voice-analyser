"""Teacher v3's seeds and their mean, as a labeller would use them (D-025, 2026-10-17, Hannah: for labelling, the
mean of the two best conformer_v2r2_d16k seeds, 1 and 2; one model alone, seed 1). The three-seed mean is shown beside
it. Dev speakers only, from cached predictions (no inference):
Hillenbrand MAE per formant (m2cache_sub) and PTDB-TUG closed-phase F1-F3 median signed bias (median |error|) per sex
(ptdb_egg_sub.npz). The mean is taken per frame in log frequency (the models' axis). A measurement, not a decision.
Args: scratch dir."""
import json
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.eval import tiers as T

sys.stdout.reconfigure(encoding="utf-8")
S = Path(sys.argv[1])
SEEDS = {"s0": "conformer_v2r2_d16k", "s1": "conformer_v2r2_s1_d16k", "s2": "conformer_v2r2_s2_d16k"}
tmap = T.load()
dev = lambda sp: np.array([T.tier_of(str(x), tmap) == "dev" for x in sp])  # noqa: E731
gmean = lambda xs: np.exp2(np.mean(np.log2(np.stack(xs)), 0))  # noqa: E731

h = {k: np.load(S / "m2cache_sub" / f"{m}__hillenbrand.npz", allow_pickle=True) for k, m in SEEDS.items()}
ref, grp, spk = h["s0"]["ref"], h["s0"]["group"], h["s0"]["speaker"]
assert all(np.array_equal(h[k]["speaker"], spk) for k in h)
est = {k: h[k]["est"] for k in h}
est["mean12"] = gmean([est["s1"], est["s2"]])
est["mean3"] = gmean([est[k] for k in SEEDS])
dv = dev(spk)
print(f"Hillenbrand dev ({len(set(spk[dv]))} speakers), MAE Hz")
cols = {"F1 all": (dv, 0), "F1 women": (dv & (grp == "women"), 0), "F2 all": (dv, 1), "F3 all": (dv, 2), "F4 all": (dv, 3)}
print(f"{'':8s}" + "".join(f"{c:>11s}" for c in cols))
for k, e in est.items():
    row = []
    for m, j in cols.values():
        ok = m & np.isfinite(ref[:, j]) & np.isfinite(e[:, j])
        row.append(np.mean(np.abs(e[ok, j] - ref[ok, j])))
    print(f"{k:8s}" + "".join(f"{x:11.1f}" for x in row))

d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
cp = {k: d[m] for k, m in SEEDS.items()}
cp["mean12"] = gmean([cp["s1"], cp["s2"]])
cp["mean3"] = gmean([cp[k] for k in SEEDS])
dvp = dev([s if ":" in str(s) else f"ptdb_tug:{s}" for s in d["speaker"]])
print("\nPTDB-TUG dev closed phase: median signed bias (median |error|), Hz, F1 / F2 / F3")
for sex in ("f", "m"):
    print(f"  {'women' if sex == 'f' else 'men'}")
    for k, e in cp.items():
        row = []
        for j in range(3):
            ok = dvp & (d["sex"] == sex) & np.isfinite(d["cp"][:, j]) & np.all([np.isfinite(cp[x][:, j]) for x in SEEDS], 0)
            err = e[ok, j] - d["cp"][ok, j]
            row.append(f"{np.median(err):+6.1f} ({np.median(np.abs(err)):5.1f})")
        print(f"    {k:6s} " + "   ".join(row))
print(json.dumps({"members": SEEDS}))
