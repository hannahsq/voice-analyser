"""D-078 addendum (2026-11-04, before registering D-079): does the heads' disagreement follow each frame's own label
gap on val_v2k, or is it a constant? Slope of (acoustic head - closed head) F1 on (acoustic - closed label) F1, per
seed, through the means; 1 = per-frame recovery, 0 = a pure offset. Frame-bootstrap intervals by crop (400 frames).

    python scripts/analysis/d078_per_clip.py
"""

from pathlib import Path

import numpy as np

C = Path("cache/d078")
CROP = 400

if __name__ == "__main__":
    for s in (1, 2):
        zc, za = np.load(C / f"d078_closed_s{s}.npz"), np.load(C / f"d078_acoustic_s{s}.npz")
        pc, pa = zc["v2k:formants|pred"][:, 0], za["v2k:formants|pred"][:, 0]
        lc, la = zc["v2k:formants|ref"][:, 0], zc["v2k:formants_acoustic|ref"][:, 0]
        son = zc["v2k:formants|son"]
        ok = son & np.isfinite(pc) & np.isfinite(pa) & np.isfinite(lc) & np.isfinite(la)
        d, g, crop = (pa - pc)[ok], (la - lc)[ok], (np.arange(len(ok)) // CROP)[ok]

        def fit(idx):
            x, y = g[idx], d[idx]
            xc = x - x.mean()
            return (xc @ (y - y.mean())) / (xc @ xc), np.corrcoef(x, y)[0, 1]

        b, r = fit(np.arange(len(g)))
        rng = np.random.default_rng(0)
        crops = np.unique(crop)
        by = {c: np.flatnonzero(crop == c) for c in crops}
        boot = [fit(np.concatenate([by[c] for c in rng.choice(crops, len(crops))]))[0] for _ in range(500)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # per crop (clip): mean disagreement against mean gap
        mg = np.array([g[by[c]].mean() for c in crops])
        md = np.array([d[by[c]].mean() for c in crops])
        mc = mg - mg.mean()
        bc = (mc @ (md - md.mean())) / (mc @ mc)
        # the shortcut: how much of the gap a model of the visible formant pattern (acoustic label F1-F4) predicts, fitted
        # on half the crops and judged on the other half; then the heads' disagreement against the rest of the gap
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = zc["v2k:formants_acoustic|ref"][ok][:, :4]
        half = np.isin(crop, crops[::2])
        m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X[half], g[half])
        gp = m.predict(X[~half])
        r2 = 1 - np.var(g[~half] - gp) / np.var(g[~half])
        res, dres = g[~half] - gp, d[~half] - d[~half].mean()
        bres = (res - res.mean()) @ dres / ((res - res.mean()) @ (res - res.mean()))
        bpred = (gp - gp.mean()) @ dres / ((gp - gp.mean()) @ (gp - gp.mean()))
        print(f"seed {s}: formant pattern predicts R2 {r2:.2f} of the gap (held-out crops); disagreement slope on the predicted "
              f"part {bpred:.2f}, on the rest {bres:.2f} (r {np.corrcoef(res, dres)[0, 1]:.2f})")
        print(f"seed {s}: frames {len(g)}, gap mean {g.mean():.1f} SD {g.std():.1f} Hz; disagreement mean {d.mean():.1f} SD {d.std():.1f}; "
              f"slope {b:.2f} [{lo:.2f}, {hi:.2f}], r {r:.2f}; per clip: gap SD {mg.std():.1f}, slope {bc:.2f}, "
              f"r {np.corrcoef(mg, md)[0, 1]:.2f}")
