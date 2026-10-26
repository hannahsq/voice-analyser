"""D-035 groundwork: glottis-to-lips midline length from 75-Speaker Annot-16's hand-drawn midsagittal contours (rtMRI,
2.4 mm pixels; Shi et al. 2025, derived from Lim et al. 2021). The dataset's terms forbid redistribution, so per-frame
and per-speaker lengths stay outside the repo (out dir); only aggregates are printed.

Per frame: the inner (anterior / inferior) wall is lower lip -> tongue -> epiglottis, the outer wall upper lip, hard
palate, velum, pharyngeal wall and arytenoid (densely resampled). End points: the lips (mean of the two lips' most
anterior points) and the glottis (mean of the epiglottis's and arytenoid's lowest points). Start: midpoints from the
palate and pharyngeal wall to the nearest inner-wall point, smoothed (the outer wall doesn't fold round the tongue tip).
Then grid-line refinement, 30 rounds: each point moves halfway towards the centre between the nearest walls on either
side along its normal, lightly smoothed, resampled at 1 px. Length = the midline's arc length (cm). Frames with the
tongue tip at the palate can still fold; the per-speaker median over its 10 frames absorbs them.
Args: ground-truth json dir, subjects tsv (id, sex, age, height cm), out dir [n overlay plots]."""
import json
import sys
from pathlib import Path

import numpy as np

GT, SUBJ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N_PLOTS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
MM = 2.4  # mm per pixel (200 mm FOV, 84 x 84)
OUT.mkdir(parents=True, exist_ok=True)


def resample(c, step=0.25):
    c = np.asarray(c, float)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    t = np.linspace(0, s[-1], int(np.ceil(s[-1] / step)) + 1)  # keeps both end points
    return np.c_[np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])]


def chain(parts):
    """Join contours end to start, flipping each so it begins nearest the previous end."""
    out = np.asarray(parts[0], float)
    for c in parts[1:]:
        c = np.asarray(c, float)
        if np.hypot(*(c[-1] - out[-1])) < np.hypot(*(c[0] - out[-1])):
            c = c[::-1]
        out = np.r_[out, c]
    return out


def midline(d):
    lips = (d["upper_lip"][np.argmin(np.asarray(d["upper_lip"])[:, 0])], d["lower_lip"][np.argmin(np.asarray(d["lower_lip"])[:, 0])])
    lip = np.mean(lips, 0)
    lo = lambda c: np.asarray(c)[np.argmax(np.asarray(c)[:, 1])]  # noqa: E731  (image y grows downwards)
    glottis = (lo(d["epiglottis"]) + lo(d["arytenoid"])) / 2
    ll = np.asarray(d["lower_lip"], float)
    inner = chain([ll[np.argsort(ll[:, 0])], d["tongue"], d["epiglottis"]])  # starts at the lip front
    inner = resample(inner)
    outer = np.concatenate([resample(d[k]) for k in ("upper_lip", "hard_palate", "velum", "pharyngeal_wall", "arytenoid")])
    # start from the outer wall, which doesn't fold (the inner one turns round the tongue tip): palate then pharynx
    guide = resample(chain([[lip], d["hard_palate"], d["pharyngeal_wall"]])[1:])
    j = np.argmin(((guide[:, None] - inner[None]) ** 2).sum(-1), 1)
    path = resample(np.r_[lip[None], smooth((guide + inner[j]) / 2, 25), glottis[None]], 1.0)
    walls = np.r_[inner, outer]
    for _ in range(30):  # grid-line refinement: centre each point between the nearest walls along its normal
        t = np.gradient(path, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = np.c_[-t[:, 1], t[:, 0]]
        new = path.copy()
        for i in range(1, len(path) - 1):
            r = walls - path[i]
            along, across = r @ t[i], r @ n[i]
            band = np.abs(along) < 0.75
            pos, neg = across[band & (across > 0)], across[band & (across < 0)]
            if len(pos) and len(neg):
                new[i] = path[i] + n[i] * 0.5 * (pos.min() + neg.max()) / 2
        path = resample(np.r_[path[:1], smooth(new[1:-1], 5), path[-1:]], 1.0)
    return path, MM / 10 * np.hypot(*np.diff(path, axis=0).T).sum()


def smooth(c, w):
    k = np.ones(w) / w
    return np.c_[[np.convolve(np.pad(c[:, a], w // 2, "edge"), k, "valid") for a in (0, 1)]].T


subj = {r[0]: (r[1], float(r[2]) if r[2] != "None" else np.nan, float(r[3])) for r in (l.split("\t") for l in SUBJ.read_text().splitlines())}
rows, plots = [], 0
for f in sorted(GT.glob("*.json")):
    d = json.loads(f.read_text())
    path, L = midline(d)
    spk = f.name.split("_")[0]
    rows.append((spk, f.stem, L))
    if plots < N_PLOTS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        for k, c in d.items():
            c = np.asarray(c)
            ax.plot(c[:, 0], c[:, 1], lw=1, label=k)
        ax.plot(path[:, 0], path[:, 1], "k.-", ms=2, lw=1.5)
        ax.set_title(f"{L:.1f} cm")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        fig.savefig(OUT / f"{f.stem}.png", dpi=110)
        plt.close(fig)
        plots += 1
(OUT / "frame_lengths.tsv").write_text("".join(f"{s}\t{n}\t{L:.3f}\n" for s, n, L in rows))
spks = sorted({r[0] for r in rows})
med = {s: np.median([r[2] for r in rows if r[0] == s]) for s in spks}
iqr = {s: np.subtract(*np.percentile([r[2] for r in rows if r[0] == s], [75, 25])) for s in spks}
(OUT / "speaker_lengths.tsv").write_text("".join(f"{s}\t{subj[s][0]}\t{med[s]:.3f}\t{iqr[s]:.3f}\n" for s in spks))
sex = np.array([subj[s][0] for s in spks])
m = np.array([med[s] for s in spks])
h = np.array([subj[s][2] for s in spks])
print(f"{len(rows)} frames, {len(spks)} speakers ({(sex == 'M').sum()} men, {(sex == 'F').sum()} women)")
for g in "MF":
    print(f"  {g}: median of speaker medians {np.median(m[sex == g]):.2f} cm (range {m[sex == g].min():.1f}-{m[sex == g].max():.1f})")
print(f"  women / men ratio {np.median(m[sex == 'F']) / np.median(m[sex == 'M']):.3f}")
print(f"  within-speaker spread across the 10 frames: median IQR {np.median(list(iqr.values())):.2f} cm")
centred = lambda x: np.concatenate([x[sex == g] - x[sex == g].mean() for g in "MF"])  # noqa: E731
print(f"  correlation with height: all {np.corrcoef(m, h)[0, 1]:+.2f}; within sex (pooled, centred) "
      f"{np.corrcoef(centred(m), centred(h))[0, 1]:+.2f}")
