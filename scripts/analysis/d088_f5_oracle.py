"""D-088: would knowing the true F5 help the other outputs? Frozen v3.1 features (D-079's cache), D-081 B's head form
and training (d081_oracle_heads.fit), head seeds 1 and 2, trained on train_v2ko's first 3000 clips, scored on val_v2ko.
Variants: h; h + flag (F5 out of band, i.e. undefined); h + log F5 + flag (log F5 filled with the training median where
undefined). One head per target group; MAE per output. Writes reports/analysis/d088/{trunk}.json and decide_d088.txt.

    python scripts/analysis/d088_f5_oracle.py v31_s1 v31_s2   (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

O = Path("reports/analysis/d088")
SEEDS = (1, 2)
LOG = lambda x: np.log(np.maximum(x, 1e-6))  # noqa: E731
# group -> (outputs' names, raw target, fit in log)
GROUPS = {
    "formants": (["F1", "F2", "F3", "F4"], lambda z: z["formants"][:, :4], True),
    "bandwidths": (["B1", "B2", "B3", "B4"], lambda z: z["bandwidths"][:, :4], True),
    "aVTL": (["aVTL"], lambda z: z["length_cm"][:, None], True),
    "iVTL": (["iVTL"], lambda z: z["length_inst_cm"][:, None], True),
    "glottal": (["glottal_opening"], lambda z: z["glottal_opening"][:, None], False),
    "source": (["Rd", "OQ"], lambda z: np.column_stack([z["rd"], z["open_quotient"]]), False),
}
VARIANTS = ("h", "h+flag", "h+F5+flag")


def f5_inputs(lab, fill):
    f5 = lab["formants"][:, 4]
    flag = ~(np.isfinite(f5) & (f5 > 0))
    lf5 = np.where(flag, fill, LOG(np.where(flag, 1.0, f5)))
    return lf5.astype(np.float32), flag.astype(np.float32)


def run(trunk, dev):
    from d081_oracle_heads import aligned, fit

    out = O / f"{trunk}.json"
    if out.exists():
        return json.loads(out.read_text())
    (ht, lt), (hv, lv) = aligned(trunk, "train"), aligned(trunk, "val")
    ht, hv = ht.astype(np.float32), hv.astype(np.float32)
    f5t = lt["formants"][:, 4]
    fill = float(np.median(LOG(f5t[np.isfinite(f5t) & (f5t > 0)])))
    (ft, gt), (fv, gv) = f5_inputs(lt, fill), f5_inputs(lv, fill)
    subsets = {"all": None, "F5 defined": gv == 0, "short (<14 cm)": lv["length_cm"] < 14}
    res = {"frames_flagged_val": float(gv[lv["sonorant"].astype(bool)].mean())}
    for group, (names, target, logfit) in GROUPS.items():
        yt, yv = target(lt), target(lv)
        mt = lt["sonorant"].astype(bool) & np.all(np.isfinite(yt), 1)
        mv = lv["sonorant"].astype(bool) & np.all(np.isfinite(yv), 1)
        if logfit:
            mt &= np.all(yt > 0, 1)
            mv &= np.all(yv > 0, 1)
        tr = LOG if logfit else (lambda x: x)
        inputs = {"h": ([ht], [hv]), "h+flag": ([ht, gt[:, None]], [hv, gv[:, None]]),
                  "h+F5+flag": ([ht, ft[:, None], gt[:, None]], [hv, fv[:, None], gv[:, None]])}
        res[group] = {}
        for var in VARIANTS:
            xt = np.column_stack([a[mt] for a in inputs[var][0]])
            xv = np.column_stack([a[mv] for a in inputs[var][1]])
            per = {}
            for s in SEEDS:
                p = fit(xt, tr(yt[mt]), xv, s, dev)
                p = np.exp(p) if logfit else p
                err = np.abs(p - yv[mv])
                per[s] = {sub: (err if sel is None else err[sel[mv]]).mean(0).tolist() for sub, sel in subsets.items()}
            res[group][var] = {"by_seed": per, "frames": int(mv.sum())}
            m = np.mean([per[s]["all"] for s in SEEDS], 0)
            print(f"{trunk} {group} {var}: " + ", ".join(f"{n} {v:.4g}" for n, v in zip(names, m)), flush=True)
    O.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def decide(results):
    lines, helped = [], []
    for group, (names, _, _) in GROUPS.items():
        for k, n in enumerate(names):
            ok_all = True
            for sub in ("all", "F5 defined", "short (<14 cm)"):
                cells = []
                for trunk, res in results.items():
                    v = {var: [res[group][var]["by_seed"][s][sub][k] for s in res[group][var]["by_seed"]] for var in VARIANTS}
                    mean = {var: float(np.mean(x)) for var, x in v.items()}
                    spread = max(abs(x[0] - x[1]) for x in v.values())
                    cut_h = 1 - mean["h+F5+flag"] / mean["h"]
                    cut_flag = 1 - mean["h+F5+flag"] / mean["h+flag"]
                    flag_cut = 1 - mean["h+flag"] / mean["h"]
                    ok = (cut_h >= 0.05 and cut_flag >= 0.05 and mean["h"] - mean["h+F5+flag"] > 3 * spread
                          and mean["h+flag"] - mean["h+F5+flag"] > 3 * spread)
                    if sub == "all":
                        ok_all &= ok
                    cells.append(f"{trunk} h {mean['h']:.4g}, flag {100 * flag_cut:+.1f} %, F5 vs h {100 * cut_h:+.1f} %, "
                                 f"vs flag {100 * cut_flag:+.1f} % (3x spread {3 * spread:.3g})")
                lines.append(f"{n} [{sub}]: " + "; ".join(cells))
            if ok_all:
                helped.append(n)
    print("\n".join(lines))
    print(f"F5 helps (all frames, both trunks, >= 5 % against h and h + flag, beyond 3x seed spread): {helped or 'none'}")
    print(f"DECISION: {'F5 helps ' + ', '.join(helped) if helped else 'F5 helps no output'}")


if __name__ == "__main__":
    import torch

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    decide({t: run(t, dev) for t in sys.argv[1:]})
