"""D-035 groundwork: glottis-to-lips midline length from 75-Speaker Annot-16's hand-drawn midsagittal contours (rtMRI,
2.4 mm pixels; Shi et al. 2025, derived from Lim et al. 2021). The dataset's terms forbid redistribution, so per-frame
and per-speaker lengths stay outside the repo (out dir); only aggregates are printed.

Per frame, the airway is closed into a polygon so the midline cannot cross a wall:
- lip end: the midpoint of the closest pair of points between the upper and lower lip (the aperture);
- outer wall: upper lip from the aperture -> hard palate -> the velum's oral surface to the uvula tip (its lowest point)
  -> the pharyngeal wall below the tip's height (sealing off the nasopharynx) -> arytenoid to its lowest point;
- inner wall: lower lip from the aperture -> tongue (front to root; the line from the lower lip to the tongue tip seals
  off the space under the tongue) -> epiglottis to its lowest point;
- glottis end (VA_GLOTTIS): "lowest", the midpoint of the epiglottis's and arytenoid's lowest points (A); "level", the
  same with the back wall cut at the epiglottis bottom's height, a level glottal line (B); "curv", the back wall's
  sharpest bend joined to the epiglottis at a set angle (tried, fits worse); "mix" (default, D-035), the frame's length
  is (A + 2 B) / 3: Annot-16's back wall runs below the glottis (its glottal line slopes 30-40 degrees down at the back,
  an annotator's 4-31 degrees up), and 1:2 is the maximum-likelihood mix on 10 annotated frames (pooled w = 0.33-0.35).
Midline: the cheapest 8-connected path on a 4x grid from lip end to glottis end, costing 1 + 1 / (d + 0.25)^2 per pixel
inside the polygon (d: distance to its edge, px), so it keeps to the middle, and 200 per pixel outside (crossed only at
a full closure, such as the tongue tip at the alveolar ridge; reported per frame). Smoothed over about 2 px where that
keeps it inside. Length = its arc length (cm).
Args: ground-truth json dir, subjects tsv (id, sex, age, height cm), out dir [n overlay plots]."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Poly
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

GT, SUBJ, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N_PLOTS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
MM = 2.4  # mm per pixel (200 mm FOV, 84 x 84)
UP = 4  # grid cells per pixel
OUTSIDE = 200.0
GLOTTIS = os.environ.get("VA_GLOTTIS", "mix")  # see the docstring
W_LOWEST = 1 / 3  # "mix": weight of A (lowest) against B (level)
ANGLE = float(os.environ.get("VA_GLOTTAL_ANGLE", "16"))  # "curv": glottal line, degrees, back end higher
PICK = os.environ.get("VA_CURV_PICK", "lowest")  # "curv": which bend of the back wall is the glottis's back end
OUT.mkdir(parents=True, exist_ok=True)


def resample(c, step=0.25):
    c = np.asarray(c, float)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    t = np.linspace(0, s[-1], int(np.ceil(s[-1] / step)) + 1)  # keeps both end points
    return np.c_[np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])]


def smooth(c, w):
    k = np.ones(w) / w
    return np.c_[[np.convolve(np.pad(c[:, a], w // 2, "edge"), k, "valid") for a in (0, 1)]].T


def from_point(c, i, towards):
    """The part of contour c from index i to whichever end is nearer the point `towards`."""
    return c[i:] if np.hypot(*(c[-1] - towards)) < np.hypot(*(c[0] - towards)) else c[i::-1]


def oriented(c, start):
    """Contour c running from the end nearer `start`."""
    return c if np.hypot(*(c[0] - start)) <= np.hypot(*(c[-1] - start)) else c[::-1]


def to_lowest(c):
    return c[: np.argmax(c[:, 1]) + 1]  # image y grows downwards


LAST = {}  # the last airway()'s glottal line ends, for plots


def curv_glottis(ar, ep_full, ep_low):
    """Back of the glottis at the back wall's lowest sharp bend (curvature peak); front where a line from it at
    ANGLE degrees (back end higher) meets the epiglottis, or the straight extension of its last segment.
    Returns the back wall and epiglottis cut at those points."""
    from scipy.signal import find_peaks
    r = smooth(resample(ar), 9)
    th = np.unwrap(np.arctan2(*np.diff(r, axis=0).T[::-1]))
    k = np.abs(np.gradient(th)) / 0.25  # rad per px
    pk, _ = find_peaks(k, prominence=0.15)
    pk = pk[(pk > 4) & (pk < len(r) - 5)]
    if PICK == "bump":  # the sharpest bend below the bump's most forward point (the bump's lower turn)
        pk = pk[pk > np.argmin(r[:, 0])]
        i = pk[np.argmax(k[pk])] if len(pk) else int(np.argmax(r[:, 1]))
    else:  # the lowest bend
        i = pk[np.argmax(r[pk, 1])] if len(pk) else int(np.argmax(r[:, 1]))
    P = r[i]
    u = np.array([-np.cos(np.radians(ANGLE)), np.sin(np.radians(ANGLE))])  # forward and down

    def hit(a, b, seg_only=True):  # P + t u = a + s (b - a)
        M = np.c_[u, a - b]
        if abs(np.linalg.det(M)) < 1e-9:
            return None
        t, s = np.linalg.solve(M, a - P)
        return (t, s) if t > 0 and (0 <= s <= 1 or not seg_only) else None

    e = ep_full
    for j in range(len(e) - 1):
        h = hit(e[j], e[j + 1])
        if h:
            return r[: i + 1], np.r_[e[: j + 1], [P + h[0] * u]]
    h = hit(ep_low[-2], ep_low[-1], seg_only=False)
    if h and h[1] >= 1 and h[0] < 15:  # beyond the epiglottis's end, within 15 px
        return r[: i + 1], np.r_[ep_low, [P + h[0] * u]]
    return r[: i + 1], ep_low


def airway(d):
    c = {k: np.asarray(v, float) for k, v in d.items()}
    ul, ll = c["upper_lip"], c["lower_lip"]
    dd = ((ul[:, None] - ll[None]) ** 2).sum(-1)
    iu, il = np.unravel_index(np.argmin(dd), dd.shape)
    lip = (ul[iu] + ll[il]) / 2
    hp = oriented(c["hard_palate"], lip)
    outer = [from_point(ul, iu, hp[0]), hp]
    vel = to_lowest(oriented(c["velum"], hp[-1]))
    outer.append(vel)
    pw = oriented(c["pharyngeal_wall"], np.array([vel[-1][0], -1e3]))  # top first
    pw = pw[pw[:, 1] >= vel[-1][1]]
    if len(pw):
        outer.append(pw)
    lo = from_point(ll, il, c["tongue"].mean(0))
    tg = oriented(c["tongue"], lo[-1])
    ep = to_lowest(oriented(c["epiglottis"], tg[-1]))
    inner = [lo, tg, ep]
    ar = to_lowest(oriented(c["arytenoid"], outer[-1][-1]))
    if GLOTTIS == "level":  # cut the back wall where it reaches the epiglottis bottom's height: a level glottal line
        k = np.flatnonzero(ar[:, 1] >= ep[-1][1])
        if len(k) and k[0] > 0:
            j = k[0]
            t = (ep[-1][1] - ar[j - 1, 1]) / (ar[j, 1] - ar[j - 1, 1])
            ar = np.r_[ar[:j], [ar[j - 1] + t * (ar[j] - ar[j - 1])]]
    if GLOTTIS == "curv":
        ar, ep = curv_glottis(oriented(c["arytenoid"], outer[-1][-1]), oriented(c["epiglottis"], tg[-1]), ep)
        inner[-1] = ep
    outer.append(ar)
    LAST.update(front=ep[-1], back=ar[-1])
    glottis = (ep[-1] + ar[-1]) / 2
    poly = np.r_[[lip], np.concatenate(outer), [glottis], np.concatenate(inner)[::-1], [lip]]
    return lip, glottis, poly


def midline(d):
    lip, glottis, poly = airway(d)
    lo = np.floor(poly.min(0)) - 2
    hi = np.ceil(poly.max(0)) + 2
    nx, ny = ((hi - lo) * UP).astype(int) + 1
    gx, gy = np.meshgrid(lo[0] + np.arange(nx) / UP, lo[1] + np.arange(ny) / UP)
    pts = np.c_[gx.ravel(), gy.ravel()]
    inside = Poly(poly).contains_points(pts).reshape(ny, nx)
    dist = distance_transform_edt(inside) / UP  # px to the polygon's edge
    cost = np.where(inside, 1 + 1 / (dist + 0.25) ** 2, OUTSIDE)
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = idx[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
        b = idx[max(0, -dy) + dy:ny - max(0, dy) + dy, max(0, -dx) + dx:nx - max(0, dx) + dx]
        step = np.hypot(dx, dy) / UP
        ww = step * (cost.ravel()[a.ravel()] + cost.ravel()[b.ravel()]) / 2
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
        w += [ww, ww]
    g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2).tocsr()
    near = lambda p: int(np.argmin(((pts - p) ** 2).sum(1)))  # noqa: E731
    s, e = near(lip), near(glottis)
    _, pred = dijkstra(g, indices=s, return_predecessors=True)
    node, path = e, [e]
    while node != s:
        node = pred[node]
        path.append(node)
    raw = pts[path[::-1]]
    sm = np.r_[raw[:1], smooth(raw, 2 * UP + 1)[1:-1], raw[-1:]]
    ok = Poly(poly).contains_points(sm)
    raw_in = Poly(poly).contains_points(raw)
    sm[~ok & raw_in] = raw[~ok & raw_in]  # keep the raw point where smoothing would leave the airway
    outside_px = (~raw_in[1:-1]).sum() / UP
    return sm, MM / 10 * np.hypot(*np.diff(sm, axis=0).T).sum(), poly, outside_px


subj = {r[0]: (r[1], float(r[2]) if r[2] != "None" else np.nan, float(r[3])) for r in (l.split("\t") for l in SUBJ.read_text().splitlines())}
rows, plots = [], 0
for f in sorted(GT.glob("*.json")):
    d = json.loads(f.read_text())
    if GLOTTIS == "mix":
        GLOTTIS = "lowest"
        L_a = midline(d)[1]
        GLOTTIS = "level"
        path, L_b, poly, outside = midline(d)
        GLOTTIS = "mix"
        L = W_LOWEST * L_a + (1 - W_LOWEST) * L_b  # the plotted path is B's
    else:
        path, L, poly, outside = midline(d)
    spk = f.name.split("_")[0]
    rows.append((spk, f.stem, L, outside))
    if plots < N_PLOTS:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.fill(poly[:, 0], poly[:, 1], color="0.85", lw=0)
        for k, c in d.items():
            c = np.asarray(c)
            ax.plot(c[:, 0], c[:, 1], lw=1, label=k)
        ax.plot(path[:, 0], path[:, 1], "k-", lw=1.5)
        ax.set_title(f"{L:.1f} cm" + (f" (crosses {outside:.1f} px)" if outside > 0 else ""), fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        fig.savefig(OUT / f"{f.stem}.png", dpi=110)
        plt.close(fig)
        plots += 1
(OUT / "frame_lengths.tsv").write_text("".join(f"{s}\t{n}\t{L:.3f}\t{o:.2f}\n" for s, n, L, o in rows))
spks = sorted({r[0] for r in rows})
med = {s: np.median([r[2] for r in rows if r[0] == s]) for s in spks}
iqr = {s: np.subtract(*np.percentile([r[2] for r in rows if r[0] == s], [75, 25])) for s in spks}
(OUT / "speaker_lengths.tsv").write_text("".join(f"{s}\t{subj[s][0]}\t{med[s]:.3f}\t{iqr[s]:.3f}\n" for s in spks))
sex = np.array([subj[s][0] for s in spks])
m = np.array([med[s] for s in spks])
h = np.array([subj[s][2] for s in spks])
out = np.array([r[3] for r in rows])
print(f"{len(rows)} frames, {len(spks)} speakers ({(sex == 'M').sum()} men, {(sex == 'F').sum()} women); "
      f"{(out > 0).sum()} frames cross a wall (closures; median {np.median(out[out > 0]) if (out > 0).any() else 0:.1f} px)")
for g in "MF":
    print(f"  {g}: median of speaker medians {np.median(m[sex == g]):.2f} cm (range {m[sex == g].min():.1f}-{m[sex == g].max():.1f})")
print(f"  women / men ratio {np.median(m[sex == 'F']) / np.median(m[sex == 'M']):.3f}")
print(f"  within-speaker spread across the 10 frames: median IQR {np.median(list(iqr.values())):.2f} cm")
centred = lambda x: np.concatenate([x[sex == g] - x[sex == g].mean() for g in "MF"])  # noqa: E731
print(f"  correlation with height: all {np.corrcoef(m, h)[0, 1]:+.2f}; within sex (pooled, centred) "
      f"{np.corrcoef(centred(m), centred(h))[0, 1]:+.2f}")
