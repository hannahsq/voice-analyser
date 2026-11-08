"""D-089: do the true F6 and F7 help the existing outputs beyond F5? D-088's oracle extended. Frozen v3.1 features
(D-079's cache), D-081 B's heads (d081_oracle_heads.fit), head seeds 1 and 2, train_v2ko's first 3000 clips -> val_v2ko.
Each upper formant enters as log Fk plus its own out-of-band flag (log Fk filled with the training median where
undefined). MAE per output, on all sonorant frames and by tract-length band. Writes reports/analysis/d089/{trunk}.json
and decide_d089.txt (stdout).

    python scripts/analysis/d089_upper_oracle.py v31_s1 v31_s2   (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

O = Path("reports/analysis/d089")
SEEDS = (1, 2)
LOG = lambda x: np.log(np.maximum(x, 1e-6))  # noqa: E731
GROUPS = {
    "formants": (["F1", "F2", "F3", "F4"], lambda z: z["formants"][:, :4]),
    "bandwidths": (["B1", "B2", "B3", "B4"], lambda z: z["bandwidths"][:, :4]),
    "aVTL": (["aVTL"], lambda z: z["length_cm"][:, None]),
    "iVTL": (["iVTL"], lambda z: z["length_inst_cm"][:, None]),
}
# variant -> (formant indices given as values, flags given)
VARIANTS = {"h": ((), ()), "h+flags": ((), (4, 5, 6)), "h+F5": ((4,), (4,)), "h+F6": ((5,), (5,)),
            "h+F5+F6": ((4, 5), (4, 5)), "h+F5+F6+F7": ((4, 5, 6), (4, 5, 6))}
BANDS = {"all": (0, 99), "<14": (0, 14), "14-16.6": (14, 16.6), "16.6-18.5": (16.6, 18.5), ">18.5": (18.5, 99),
         ">=14": (14, 99), ">=16.6": (16.6, 99)}
DECIDING = {"F1", "F4", "iVTL"}


def upper(lab, fills):
    vals, flags = {}, {}
    for k, fill in fills.items():
        f = lab["formants"][:, k]
        bad = ~(np.isfinite(f) & (f > 0))
        vals[k] = np.where(bad, fill, LOG(np.where(bad, 1.0, f))).astype(np.float32)[:, None]
        flags[k] = bad.astype(np.float32)[:, None]
    return vals, flags


def run(trunk, dev):
    from d081_oracle_heads import aligned, fit

    out = O / f"{trunk}.json"
    if out.exists():
        return json.loads(out.read_text())
    (ht, lt), (hv, lv) = aligned(trunk, "train"), aligned(trunk, "val")
    ht, hv = ht.astype(np.float32), hv.astype(np.float32)
    fills = {}
    for k in (4, 5, 6):
        f = lt["formants"][:, k]
        fills[k] = float(np.median(LOG(f[np.isfinite(f) & (f > 0)])))
    (vt, gt), (vv, gv) = upper(lt, fills), upper(lv, fills)
    L = lv["length_cm"]
    res = {}
    for group, (names, target) in GROUPS.items():
        yt, yv = target(lt), target(lv)
        mt = lt["sonorant"].astype(bool) & np.all(np.isfinite(yt) & (yt > 0), 1)
        mv = lv["sonorant"].astype(bool) & np.all(np.isfinite(yv) & (yv > 0), 1)
        res[group] = {}
        for var, (vk, fk) in VARIANTS.items():
            xt = np.column_stack([ht[mt]] + [vt[k][mt] for k in vk] + [gt[k][mt] for k in fk])
            xv = np.column_stack([hv[mv]] + [vv[k][mv] for k in vk] + [gv[k][mv] for k in fk])
            per = {}
            for s in SEEDS:
                err = np.abs(np.exp(fit(xt, LOG(yt[mt]), xv, s, dev)) - yv[mv])
                per[s] = {b: err[(L[mv] >= lo) & (L[mv] < hi)].mean(0).tolist() for b, (lo, hi) in BANDS.items()}
            res[group][var] = per
            print(f"{trunk} {group} {var}: " + ", ".join(f"{n} {v:.4g}" for n, v in
                                                       zip(names, np.mean([per[s]['all'] for s in SEEDS], 0))), flush=True)
    O.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def decide(results):
    def stats(res, group, k, var, band):
        v = [res[group][var][s][band][k] for s in res[group][var]]
        return float(np.mean(v)), abs(v[0] - v[1])

    def adds(group, k, ref, new, band):
        ok, cells = True, []
        for trunk, res in results.items():
            (a, sa), (b, sb) = stats(res, group, k, ref, band), stats(res, group, k, new, band)
            cut = 1 - b / a
            ok &= cut >= 0.03 and a - b > 3 * max(sa, sb)
            cells.append(f"{trunk} {100 * cut:+.1f} %")
        return ok, ", ".join(cells)

    lines, f6, f7 = [], [], []
    for group, (names, _) in GROUPS.items():
        for k, n in enumerate(names):
            for band in BANDS:
                parts = []
                for ref, new in (("h", "h+F5"), ("h", "h+F6"), ("h", "h+flags"), ("h+F5", "h+F5+F6"), ("h+F5+F6", "h+F5+F6+F7")):
                    parts.append(f"{new} vs {ref} {adds(group, k, ref, new, band)[1]}")
                lines.append(f"{n} [{band}]: " + "; ".join(parts))
            if n in DECIDING:
                if adds(group, k, "h+F5", "h+F5+F6", ">=14")[0] and adds(group, k, "h+F5", "h+F5+F6", "all")[0]:
                    f6.append(n)
                if adds(group, k, "h+F5+F6", "h+F5+F6+F7", ">=16.6")[0] and adds(group, k, "h+F5+F6", "h+F5+F6+F7", "all")[0]:
                    f7.append(n)
    print("\n".join(lines))
    print(f"F6 adds beyond F5 (>= 3 %, beyond 3x seed spread, both trunks, >= 14 cm and all frames): {f6 or 'none'}")
    print(f"F7 adds beyond F5 + F6 (same, >= 16.6 cm and all frames): {f7 or 'none'}")
    print(f"DECISION: F6 adds for {', '.join(f6) or 'nothing'}; F7 adds for {', '.join(f7) or 'nothing'}")


if __name__ == "__main__":
    import torch

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    decide({t: run(t, dev) for t in sys.argv[1:]})
