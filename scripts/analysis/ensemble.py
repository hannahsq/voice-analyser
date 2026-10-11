"""Per-frame median ensembles from cached predictions (no inference). Hillenbrand formant MAE from the paired measures'
cache (m2cache_sub; only models whose cache is fp32 are mixed), and closed-phase F1 bias / |error| from the subset
closed-phase frames. Args: scratch dir. Run from the main checkout."""
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S = Path(sys.argv[1])
ENS = {
    "GRU v2, seeds 0-2": ["v2_s0", "v2_s1", "v2_s2"],
    "Conformer mixed, seeds 0-1": ["mix_conformer_s0", "mix_conformer_s1"],
    "all fp32 models (7)": ["v2_s0", "v2_s1", "v2_s2", "conf_v2_s0", "mix_gru_s0", "mix_conformer_s0", "mix_conformer_s1"],
    "all but the fixed-crop Conformer (6)": ["v2_s0", "v2_s1", "v2_s2", "mix_gru_s0", "mix_conformer_s0", "mix_conformer_s1"],
}


def load(name):
    z = np.load(S / "m2cache_sub" / f"{name}__hillenbrand.npz", allow_pickle=True)
    key = json.loads(str(z["__key__"])) if "__key__" in z else {}
    return {k: z[k] for k in z.files}, key


def mae(est, ref, mask, k):
    m = mask & np.isfinite(ref[:, k]) & np.isfinite(est[:, k])
    return np.mean(np.abs(est[m, k] - ref[m, k]))


names = sorted({n for v in ENS.values() for n in v})
data = {n: load(n) for n in names}

ref, grp = data[names[0]][0]["ref"], data[names[0]][0]["group"]
masks = {"F1 all": (np.ones(len(grp), bool), 0), "F1 women": (grp == "women", 0), "F1 children": (np.isin(grp, ["boys", "girls"]), 0),
         "F1 men": (grp == "men", 0), "F2 all": (np.ones(len(grp), bool), 1), "F3 all": (np.ones(len(grp), bool), 2)}
print("\nHillenbrand MAE (Hz): single models, then per-frame median ensembles")
print(f"{'':40s}" + "".join(f"{k:>12s}" for k in masks))
for n in names:
    print(f"{n:40s}" + "".join(f"{mae(data[n][0]['est'], ref, m, k):12.1f}" for m, k in masks.values()))
for label, members in ENS.items():
    est = np.nanmedian(np.stack([data[n][0]["est"] for n in members]), 0)
    best = {key: min(mae(data[n][0]["est"], ref, m, k) for n in members) for key, (m, k) in masks.items()}
    print(f"{'median: ' + label:40s}" + "".join(f"{mae(est, ref, m, k):12.1f}" for m, k in masks.values()))
    print(f"{'  (best single member)':40s}" + "".join(f"{best[key]:12.1f}" for key in masks))

print("\nClosed-phase F1 (PTDB-TUG subset): median signed bias (median |error|), Hz")
d = dict(np.load(S / "ptdb_egg_sub.npz"))
cols = [c for c in ("v2_s0", "v2_s1", "v2_s2", "v3b_s0", "v3b_s1", "v3b_s2", "v1_s0", "v1_s1", "v1_s2", "conf_v2_s0", "mix_gru_s0",
                    "mix_conformer_s0", "mix_conformer_s1") if c in d]
cp_ens = dict(ENS)
cp_ens["v3b, seeds 0-2"] = ["v3b_s0", "v3b_s1", "v3b_s2"]
cp_ens["v1, seeds 0-2"] = ["v1_s0", "v1_s1", "v1_s2"]
cp_ens[f"all {len(cols)} models"] = cols
for sex in ("f", "m"):
    m = (d["sex"] == sex) & np.isfinite(d["cp"][:, 0])
    print(f"  {'women' if sex == 'f' else 'men'}")
    for n in cols:
        e = d[n][m, 0] - d["cp"][m, 0]
        print(f"    {n:38s} {np.nanmedian(e):+6.1f} ({np.nanmedian(np.abs(e)):5.1f})")
    for label, members in cp_ens.items():
        members = [c for c in members if c in d]
        e = np.nanmedian(np.stack([d[c][m, 0] for c in members]), 0) - d["cp"][m, 0]
        print(f"    {'median: ' + label:38s} {np.nanmedian(e):+6.1f} ({np.nanmedian(np.abs(e)):5.1f})")
