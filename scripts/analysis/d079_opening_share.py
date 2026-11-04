"""D-079 pre-registration check (2026-11-04): how much of the frame's mean glottal opening <a> is set by the open phase
(te, from Rd, which also shapes the source spectrum) and how much by the clip's gap offset (visible only through
coupling's own pole and bandwidth movement)? The same clips (train_v2k's settings) rendered twice, the second with the
gap offset fixed at its range's midpoint (`uniform(c, c)` still takes its draw, so every other draw is the same).

    python scripts/analysis/d079_opening_share.py [n_clips] [workers]
    python scripts/analysis/d079_opening_share.py ceiling   (after d079_render.py: <a> from the frame's Rd label alone,
                                                             fitted on train_v2ko, R² on val_v2ko voiced frames)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 7902  # not a render seed of train_v2k (78) or val_v2k (7801)


def one(args):
    from vocal_analysis.synth.sampler import generate

    i, cfg = args
    c = generate(i, cfg, SEED)
    return c.labels["glottal_opening"], c.labels["sonorant"].astype(bool)


def ceiling():
    from sklearn.ensemble import HistGradientBoostingRegressor

    def load(name, n=None):
        d = Path("data/synth") / name / "flat"
        off = np.load(d / "frame_offsets.npy")
        end = off[n] if n else off[-1]
        a, rd, son = (np.load(d / f"label_{k}.npy", mmap_mode="r")[:end] for k in ("glottal_opening", "rd", "sonorant"))
        m = np.asarray(son).astype(bool) & np.isfinite(a) & np.isfinite(rd)
        return np.asarray(rd)[m][:, None], np.asarray(a)[m]

    xt, yt = load("train_v2ko", 3000)
    xv, yv = load("val_v2ko")
    p = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(xt, yt).predict(xv)
    print(f"ceiling: <a> from the frame's Rd alone, R² {1 - np.var(yv - p) / np.var(yv):.3f} on {len(yv)} val_v2ko voiced frames")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ceiling":
    ceiling()
elif __name__ == "__main__":
    import d078_render

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    full = d078_render.config()
    mid = float(np.mean(full.glottal_gap_offset))
    fixed = replace(full, glottal_gap_offset=(mid, mid))
    with ProcessPoolExecutor(workers) as ex:
        a = list(ex.map(one, [(i, full) for i in range(n)]))
        b = list(ex.map(one, [(i, fixed) for i in range(n)]))
    af, ax, clip = [], [], []
    for k, ((x, s), (y, _)) in enumerate(zip(a, b)):
        m = s & np.isfinite(x) & np.isfinite(y)
        af.append(x[m]), ax.append(y[m]), clip.append(np.full(m.sum(), k))
    af, ax, clip = map(np.concatenate, (af, ax, clip))
    v = af.var()
    print(f"{n} clips, {len(af)} sonorant frames; <a> mean {af.mean():.3f}, SD {af.std():.3f}")
    print(f"variance of <a>: {v:.5f}; with the gap offset fixed: {ax.var():.5f} ({ax.var() / v:.0%}); "
          f"of the difference (the offset's own part): {(af - ax).var():.5f} ({(af - ax).var() / v:.0%})")
    cm = np.array([af[clip == k].mean() for k in np.unique(clip)])
    print(f"between-clip share of <a>'s variance: {cm.var() / v:.0%}")
