"""D-083: does the aVTL head gain from a predicted iVTL track (P1) and from its context (P2)? On D-079's cached v3.1
features and D-081's labels (train_v2ko's first 3000 clips, val_v2ko), sonorant frames. Heads as D-081 B; train-set
predictions out of fold (clip parity), val predictions the two folds' mean. Window statistics per clip with scipy's
filters (edges: nearest frame), otherwise as D-081 A3.

    python scripts/analysis/d083_vtl_link.py v31_s1 v31_s2     (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

O = Path("reports/analysis/d083")
SEEDS = (1, 2)
WINDOWS = (0.0, 0.25, 0.5, 1.0, 2.0)


def frames(trunk, split):
    from d081_oracle_heads import aligned

    h, lab = aligned(trunk, split)
    m = lab["sonorant"].astype(bool) & np.isfinite(lab["length_cm"]) & np.isfinite(lab["length_inst_cm"])
    return h[m].astype(np.float32), {k: lab[k][m] for k in ("length_cm", "length_inst_cm", "clip")}


def oof(ht, yt, ct, hv, seed, dev):
    """Out-of-fold train predictions (two folds by clip parity) and the folds' mean on val."""
    from d081_oracle_heads import fit

    pt, pv = np.zeros(len(yt)), np.zeros(len(hv))
    for f in (0, 1):
        tr = ct % 2 == f
        p = fit(ht[tr], yt[tr, None], np.concatenate([ht[~tr], hv]), seed, dev)[:, 0]
        pt[~tr] = p[:(~tr).sum()]
        pv += p[(~tr).sum():] / 2
    return pt, pv


def window_stats(x, clip, half):
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, percentile_filter, uniform_filter1d

    out = np.zeros((len(x), 5), np.float32)
    start = np.flatnonzero(np.r_[True, clip[1:] != clip[:-1]])
    for a, b in zip(start, np.r_[start[1:], len(x)]):
        s = x[a:b]
        if half == 0:
            out[a:b] = s[:, None]
            continue
        w = 2 * half + 1
        out[a:b] = np.column_stack([minimum_filter1d(s, w, mode="nearest"), uniform_filter1d(s, w, mode="nearest"),
                                    maximum_filter1d(s, w, mode="nearest"), percentile_filter(s, 10, w, mode="nearest"),
                                    percentile_filter(s, 90, w, mode="nearest")])
    return out


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


if __name__ == "__main__":
    import torch

    from d081_label_oracle import gb
    from d081_oracle_heads import fit

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    O.mkdir(parents=True, exist_ok=True)
    for trunk in sys.argv[1:]:
        out = O / f"{trunk}.json"
        if out.exists():
            continue
        (ht, lt), (hv, lv) = frames(trunk, "train"), frames(trunk, "val")
        ya_t, ya_v = np.log(lt["length_cm"]), np.log(lv["length_cm"])
        yi_t, yi_v = np.log(lt["length_inst_cm"]), np.log(lv["length_inst_cm"])
        res = {"frames": int(len(yv := lv["length_cm"]))}
        # predicted iVTL and aVTL (out of fold), per head seed
        pi, pa = {}, {}
        for s in SEEDS:
            pi[s] = oof(ht, yi_t, lt["clip"], hv, s, dev)
            pa[s] = oof(ht, ya_t, lt["clip"], hv, s, dev)
        res["iVTL head MAE cm"] = float(np.mean([mae(np.exp(pi[s][1]), lv["length_inst_cm"]) for s in SEEDS]))
        # P1
        p1 = {}
        for name in ("h", "h+iVTL true", "h+iVTL pred"):
            per = {}
            for s in SEEDS:
                extra_t = [] if name == "h" else [yi_t[:, None] if "true" in name else pi[s][0][:, None]]
                extra_v = [] if name == "h" else [yi_v[:, None] if "true" in name else pi[s][1][:, None]]
                p = fit(np.column_stack([ht, *extra_t]), ya_t[:, None], np.column_stack([hv, *extra_v]), s, dev)[:, 0]
                per[s] = mae(np.exp(p), yv)
            p1[name] = {"by_seed": per, "mean": float(np.mean(list(per.values()))), "spread": float(abs(per[1] - per[2]))}
            print(f"{trunk} P1 {name}: {p1[name]['mean']:.4f} cm", flush=True)
        res["P1"] = p1
        # P2 (head seed 1's predictions; trees on CPU)
        p2 = {}
        for w in WINDOWS:
            half = int(round(w * 50))
            ws = lambda x, c: window_stats(x.astype(np.float32), c, half)  # noqa: E731
            parts_t = {"aVTL pred": ws(pa[1][0], lt["clip"]), "iVTL pred": ws(pi[1][0], lt["clip"]), "iVTL true": ws(yi_t, lt["clip"])}
            parts_v = {"aVTL pred": ws(pa[1][1], lv["clip"]), "iVTL pred": ws(pi[1][1], lv["clip"]), "iVTL true": ws(yi_v, lv["clip"])}
            r = {}
            for arm, keys in (("(i) aVTL pred", ["aVTL pred"]), ("(ii) + iVTL pred", ["aVTL pred", "iVTL pred"]),
                              ("(iii) + iVTL true", ["aVTL pred", "iVTL true"])):
                xt = np.column_stack([parts_t[k] for k in keys])
                xv = np.column_stack([parts_v[k] for k in keys])
                r[arm] = mae(np.exp(gb(xt, ya_t).predict(xv)), yv)
            p2[str(w)] = r
            print(f"{trunk} P2 {w} s: " + ", ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
        res["P2"] = p2
        out.write_text(json.dumps(res, indent=1))

    R = {t: json.loads((O / f"{t}.json").read_text()) for t in sys.argv[1:]}
    L = ["D-083: the aVTL head with a predicted iVTL track (v3.1 frozen features; MAE cm on val_v2ko sonorant frames)", ""]
    ok1, ok2 = [], []
    for t, r in R.items():
        p = r["P1"]
        h, tr, pr = p["h"], p["h+iVTL true"], p["h+iVTL pred"]
        rel = pr["mean"] / h["mean"] - 1
        noise = 3 * max(h["spread"], pr["spread"])
        ok1.append(rel <= -0.05 and h["mean"] - pr["mean"] > noise)
        kept = (h["mean"] - pr["mean"]) / (h["mean"] - tr["mean"])
        L.append(f"{t}: iVTL head MAE {r['iVTL head MAE cm']:.3f} cm; P1 h {h['mean']:.4f} -> +true {tr['mean']:.4f} / +pred {pr['mean']:.4f} "
                 f"({100 * rel:+.1f} %, 3x seed spread {noise:.4f}; {100 * kept:.0f} % of the true gain kept)")
        for w, x in r["P2"].items():
            L.append(f"  P2 {float(w):4.2f} s: " + ", ".join(f"{k} {v:.4f}" for k, v in x.items()))
        x2 = r["P2"]["2.0"]
        ok2.append(x2["(ii) + iVTL pred"] <= 0.9 * x2["(i) aVTL pred"])
    L += ["", f"DECISION: P1 {'usable' if all(ok1) else 'not usable'}; P2 iVTL context {'helps' if all(ok2) else 'no clear help'}"]
    (O / "decide_d083.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
