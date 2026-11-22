"""D-129: do formant estimates shrink towards the centre at high f0 and under reverberation? From the channel probe's
existing dev outputs (no model is run). Per Hillenbrand group, condition and formant: the OLS slope of the model's
log2 estimate on the reference's log2 value across files (1 = no shrinkage), with speaker-bootstrap 95 % intervals,
for each model given and their mean; and the paired change from clean.

    python scripts/analysis/d129_shrinkage.py <run name> [<run name> ...]    (reports/probes/channel/<run>_raw.npz)
"""

import json
import sys
from pathlib import Path

import numpy as np

CACHE = Path("data/probes/channel_v1")
REPORTS = Path("reports/probes/channel")
CONDS = ("clean", "reverb_0.2", "reverb_0.4", "reverb_0.8")
B = 1000


def references(items):
    """Per Hillenbrand item: median log2 reference F1, F2 over its reference frames (as the probe's estimates)."""
    ref = np.full((len(items), 2), np.nan)
    for i, it in enumerate(items):
        if it["corpus"] != "hillenbrand":
            continue
        r = np.load(CACHE / it["file"])["ref_formants"][:, :2]
        with np.errstate(all="ignore"):
            ref[i] = np.nanmedian(np.where(np.isfinite(r) & (r > 0), np.log2(r), np.nan), 0)
    return ref


def slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def main(runs):
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    items, conds = meta["items"], meta["conditions"]
    ci = [conds.index(c) for c in CONDS]
    ref = references(items)
    est = [np.load(REPORTS / f"{r}_raw.npz")["hf"][:, :, :2] for r in runs]  # (items, conditions, F1-F2) log2
    group = np.array([it.get("group") or "" for it in items])
    spk = np.array([it["speaker"] for it in items])
    rng = np.random.default_rng(0)
    print(f"D-129: slope of log2 estimate on log2 reference (1 = no shrinkage); models {', '.join(runs)}; "
          f"seed mean [95 % speaker bootstrap] (each model)")
    for g in sorted(set(group) - {""}):
        m0 = (group == g) & np.isfinite(ref).all(1)
        sp = np.unique(spk[m0])
        print(f"\n{g}: {m0.sum()} files, {len(sp)} speakers")
        for k, fname in enumerate(("F1", "F2")):
            for c, cn in zip(ci, CONDS):
                ok = m0 & np.all([np.isfinite(e[:, c, k]) & np.isfinite(e[:, 0, k]) for e in est], 0)
                per = [slope(ref[ok, k], e[ok, c, k]) for e in est]
                d_per = [slope(ref[ok, k], e[ok, c, k]) - slope(ref[ok, k], e[ok, 0, k]) for e in est]
                boots, dboots = [], []
                idx = {s: np.flatnonzero(ok & (spk == s)) for s in sp}
                idx = {s: v for s, v in idx.items() if len(v)}
                keys = list(idx)
                for _ in range(B):
                    pick = np.concatenate([idx[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
                    vals = [slope(ref[pick, k], e[pick, c, k]) for e in est]
                    boots.append(np.mean(vals))
                    dboots.append(np.mean([v - slope(ref[pick, k], e[pick, 0, k]) for v, e in zip(vals, est)]))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                line = f"  {fname} {cn:11s} slope {np.mean(per):.3f} [{lo:.3f}, {hi:.3f}] ({', '.join(f'{p:.3f}' for p in per)})"
                if c != 0:
                    dlo, dhi = np.percentile(dboots, [2.5, 97.5])
                    line += f"; change from clean {np.mean(d_per):+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                print(line)


if __name__ == "__main__":
    main(sys.argv[1:] or ["d040_5b_s1_b20k", "d040_5b_s2_b20k"])
