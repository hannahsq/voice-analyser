"""D-081 A: label-only fits (CPU). A1 checks the glottal slope on known predictions; A2 bounds the closed F1 from
perfect acoustic-domain labels; A3 asks how much iVTL context the anatomical length needs. Boosted trees fitted on
train_v2ko's first 3000 clips (`d081_labels.py`), scored on val_v2ko's sonorant frames. The glottal slope (D-078's
addendum, the clip as the item): the label gap's formant-predictable part is fitted on even val clips and the slope
of (acoustic - predicted F1) on the gap's rest is taken on odd clips.

    python scripts/analysis/d081_label_oracle.py
"""

import json
from pathlib import Path

import numpy as np

C, O = Path("cache/d081"), Path("reports/analysis/d081")
N_FIT = 300_000


def load(split):
    z = dict(np.load(C / f"{split}_labels.npz"))
    m = z["sonorant"].astype(bool) & np.isfinite(z["glottal_opening"]) & np.all(np.isfinite(z["formants_acoustic"]), 1)
    return {k: v[m] for k, v in z.items()}


def gb(x, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    i = np.random.default_rng(seed).permutation(len(y))[:N_FIT]
    return HistGradientBoostingRegressor(max_iter=500, random_state=seed).fit(x[i], y[i])


def r2(y, p):
    return float(1 - np.var(y - p) / np.var(y))


class Slope:
    """The glottal slope on val: gap = acoustic - closed F1; its part predictable from the acoustic F1-F4 is fitted on
    even clips; slope of (acoustic - p) on the rest, odd clips. 1 = the closed F1's glottal part fully recovered."""

    def __init__(self, v):
        self.a, c = v["formants_acoustic"][:, 0], v["formants"][:, 0]
        self.g = self.a - c
        even = v["clip"] % 2 == 0
        self.use = ~even
        gp = gb(v["formants_acoustic"][even, :4], self.g[even]).predict(v["formants_acoustic"][self.use, :4])
        self.rest = self.g[self.use] - gp
        self.gp = gp

    def __call__(self, p):
        y = (self.a - p)[self.use]
        m = np.isfinite(y)
        x, y = self.rest[m] - self.rest[m].mean(), y[m] - y[m].mean()
        return float(x @ y / (x @ x))


def window_stats(v, w_frames):
    out = np.zeros((len(v["clip"]), 5), np.float32)
    li = v["length_inst_cm"]
    for c in np.unique(v["clip"]):
        idx = np.flatnonzero(v["clip"] == c)
        x = li[idx]
        for j in range(len(idx)):
            s = x[max(0, j - w_frames): j + w_frames + 1]
            out[idx[j]] = (s.min(), s.mean(), s.max(), np.percentile(s, 10), np.percentile(s, 90))
    return out


if __name__ == "__main__":
    O.mkdir(parents=True, exist_ok=True)
    t, v = load("train"), load("val")
    S = Slope(v)
    res, L = {}, ["D-081 A: label-only fits (train_v2ko 3000 clips -> val_v2ko sonorant frames)", ""]
    # A1
    a1 = {"closed label": S(v["formants"][:, 0]), "acoustic label": S(v["formants_acoustic"][:, 0])}
    p = v["formants_acoustic"][:, 0].copy()
    p[S.use] = S.a[S.use] - S.gp
    a1["acoustic - vowel part"] = S(p)
    ok = abs(a1["closed label"] - 1) <= 0.05 and abs(a1["acoustic label"]) <= 0.05 and abs(a1["acoustic - vowel part"]) <= 0.05
    res["A1"] = {**a1, "valid": ok}
    L.append("A1, the glottal slope on known predictions (expected 1 / 0 / 0): "
             + ", ".join(f"{k} {x:+.3f}" for k, x in a1.items()) + f" -> {'valid' if ok else 'BROKEN'}")
    # A2
    feats = {
        "full": lambda z: np.c_[z["formants_acoustic"][:, :4], np.log(z["bandwidths_acoustic"][:, :4]), z["f0"], z["rd"], z["open_quotient"]],
        "no bandwidths": lambda z: np.c_[z["formants_acoustic"][:, :4], z["f0"], z["rd"], z["open_quotient"]],
    }
    L.append("")
    L.append("A2, the closed F1 from perfect acoustic-domain labels (glottal slope); the clip's gap offset (R², frame / clip mean); "
             "<a> beyond Rd (R²)")
    f_rd = gb(t["rd"][:, None], t["glottal_opening"])
    rd_v = f_rd.predict(v["rd"][:, None])
    res["A2"] = {}
    for name, fx in feats.items():
        xt, xv = fx(t), fx(v)
        ok_t, ok_v = np.all(np.isfinite(xt), 1), np.all(np.isfinite(xv), 1)
        pc = np.full(len(xv), np.nan)
        pc[ok_v] = np.exp(gb(xt[ok_t], np.log(t["formants"][ok_t, 0])).predict(xv[ok_v]))
        po = np.full(len(xv), np.nan)
        po[ok_v] = gb(xt[ok_t], t["gap_offset"][ok_t]).predict(xv[ok_v])
        pa = np.full(len(xv), np.nan)
        pa[ok_v] = gb(xt[ok_t], t["glottal_opening"][ok_t]).predict(xv[ok_v])
        m = ok_v
        cl = np.unique(v["clip"][m])
        cm = lambda q: np.array([q[m][v["clip"][m] == c].mean() for c in cl])  # noqa: E731
        r = {"slope": S(pc), "offset_r2_frame": r2(v["gap_offset"][m], po[m]), "offset_r2_clip": r2(cm(v["gap_offset"]), cm(po)),
             "opening_beyond_rd": r2(v["glottal_opening"][m] - rd_v[m], pa[m] - rd_v[m]), "frames": int(m.sum())}
        res["A2"][name] = r
        L.append(f"  {name:14s} slope {r['slope']:+.3f}; offset R² {r['offset_r2_frame']:.3f} / {r['offset_r2_clip']:.3f}; "
                 f"<a> beyond Rd {r['opening_beyond_rd']:.3f} ({r['frames']} frames)")
    s = res["A2"]["full"]["slope"]
    a2 = "recoverable in principle" if s >= 0.5 else "not recoverable" if s < 0.2 else "partly"
    L.append(f"  -> {a2}")
    res["A2"]["verdict"] = a2
    # A3
    L.append("")
    L.append("A3, the anatomical length from the iVTL track (window = centred, seconds) plus acoustic F1-F4: MAE cm, R²")
    res["A3"] = {}
    for w in (0.0, 0.25, 0.5, 1.0, 2.0):
        wf = int(round(w * 50))  # half-window in 10 ms frames
        xt = np.c_[t["length_inst_cm"], window_stats(t, wf), t["formants_acoustic"][:, :4]]
        xv = np.c_[v["length_inst_cm"], window_stats(v, wf), v["formants_acoustic"][:, :4]]
        pl = gb(xt, t["length_cm"]).predict(xv)
        mae = float(np.mean(np.abs(pl - v["length_cm"])))
        res["A3"][str(w)] = {"mae_cm": mae, "r2": r2(v["length_cm"], pl)}
        L.append(f"  {w:4.2f} s  MAE {mae:.3f} cm, R² {res['A3'][str(w)]['r2']:.3f}")
    m0, m2 = res["A3"]["0.0"]["mae_cm"], res["A3"]["2.0"]["mae_cm"]
    a3 = "context helps" if m2 <= 0.8 * m0 else "no clear help"
    res["A3"]["verdict"] = a3
    L.append(f"  -> {a3} (2 s against frame only: {100 * (m2 / m0 - 1):+.0f} %)")
    (O / "label_oracle.json").write_text(json.dumps(res, indent=1))
    (O / "label_oracle.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
