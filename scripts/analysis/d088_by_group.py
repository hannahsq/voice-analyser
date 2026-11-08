"""D-088 follow-up (after its result): the same heads (same seeds, so the all-frame numbers reproduce D-088's), with the
cuts broken down by the clip's tract length (val_v2ko predates the sex-balanced length groups: every clip is from the
uniform coverage draw, so there is no sex label). Bands: < 14 cm (children), 14-16.6 (women-typical, 15.6 +- 1 SD),
16.6-18.5 (men-typical, 17.6 +- 1 SD), > 18.5. Formants, bandwidths and iVTL only. Reporting only.

    python scripts/analysis/d088_by_group.py v31_s1 v31_s2   (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import d088_f5_oracle as M  # noqa: E402

VAL = Path("data/synth/val_v2ko")
KEEP = ("formants", "bandwidths", "iVTL")
BANDS = ("<14 cm", "14-16.6 (women-typical)", "16.6-18.5 (men-typical)", ">18.5 cm")


def clip_groups():
    idx = json.loads((VAL / "index.json").read_text())
    out = {}
    for p in sorted(VAL.glob("shard_*.npz")):
        for m in json.loads(str(np.load(p, allow_pickle=True)["meta"])):
            out[m["index"]] = m["length_group"] or "cover"
    assert len(out) == idx["n_clips"]
    return out


def run(trunk, dev, groups):
    from d081_oracle_heads import aligned, fit

    (ht, lt), (hv, lv) = aligned(trunk, "train"), aligned(trunk, "val")
    ht, hv = ht.astype(np.float32), hv.astype(np.float32)
    f5t = lt["formants"][:, 4]
    fill = float(np.median(M.LOG(f5t[np.isfinite(f5t) & (f5t > 0)])))
    (ft, gt), (fv, gv) = M.f5_inputs(lt, fill), M.f5_inputs(lv, fill)
    grp = np.select([lv["length_cm"] < 14, lv["length_cm"] < 16.6, lv["length_cm"] < 18.5], BANDS[:3], BANDS[3])
    res = {}
    for group in KEEP:
        names, target, logfit = M.GROUPS[group]
        yt, yv = target(lt), target(lv)
        mt = lt["sonorant"].astype(bool) & np.all(np.isfinite(yt) & (yt > 0), 1)
        mv = lv["sonorant"].astype(bool) & np.all(np.isfinite(yv) & (yv > 0), 1)
        inputs = {"h": ([ht], [hv]), "h+flag": ([ht, gt[:, None]], [hv, gv[:, None]]),
                  "h+F5+flag": ([ht, ft[:, None], gt[:, None]], [hv, fv[:, None], gv[:, None]])}
        res[group] = {}
        for var in M.VARIANTS:
            xt = np.column_stack([a[mt] for a in inputs[var][0]])
            xv = np.column_stack([a[mv] for a in inputs[var][1]])
            per = {}
            for s in M.SEEDS:
                err = np.abs(np.exp(fit(xt, M.LOG(yt[mt]), xv, s, dev)) - yv[mv])
                per[s] = {g: err[grp[mv] == g].mean(0).tolist() for g in BANDS} | {"all": err.mean(0).tolist()}
            res[group][var] = per
            print(f"{trunk} {group} {var} done", flush=True)
    return res


if __name__ == "__main__":
    import torch

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    out = M.O / "by_group.json"
    groups = None
    results = json.loads(out.read_text()) if out.exists() else {t: run(t, dev, groups) for t in sys.argv[1:]}
    out.write_text(json.dumps(results, indent=1))
    for group in KEEP:
        for k, n in enumerate(M.GROUPS[group][0]):
            for g in ("all",) + BANDS:
                cells = []
                for trunk, res in results.items():
                    mean = {v: float(np.mean([res[group][v][s][g][k] for s in res[group][v]])) for v in M.VARIANTS}
                    cells.append(f"{trunk} h {mean['h']:.4g}, F5 vs h {100 * (1 - mean['h+F5+flag'] / mean['h']):+.1f} %, "
                                 f"vs flag {100 * (1 - mean['h+F5+flag'] / mean['h+flag']):+.1f} %")
                print(f"{n} [{g}]: " + "; ".join(cells))
