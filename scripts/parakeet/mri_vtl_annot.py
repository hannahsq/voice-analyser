"""D-035 groundwork: an annotator's glottis marks against mri_vtl.py's end-point rules, on Annot-16 frames upscaled 16x
(png_xy = (mri_xy + 0.5) * 16). Strokes are the pixels that differ from the image handed out: pure red = glottal line
(front to back), pure blue = back wall (replaces Annot-16's "arytenoid"), pure green = epiglottis.
Per frame: the red line's midpoint against each rule's glottis (lowest points; level line), in mm along the midline's
final direction (+ = the mark is deeper) and across it; and the midline length with Annot-16's walls ending at each
rule, and with the annotator's glottis (and their redrawn walls where given).
Args: annotate dir (with annotated/), ground-truth json dir, subjects tsv, out tsv."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

A, GT, SUBJ, OUTF = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
UP, MM = 16, 2.4
src = (Path(__file__).parent / "mri_vtl.py").read_text(encoding="utf-8").split("\nsubj = ")[0]
argv, sys.argv = sys.argv, ["mri_vtl", "x", "x", "x"]


def load_rules():
    out = {}
    for rule in ("lowest", "level"):
        os.environ["VA_GLOTTIS"] = rule
        g = {"__file__": "mri_vtl"}
        exec(compile(src.replace("OUT.mkdir(parents=True, exist_ok=True)", ""), "mri_vtl", "exec"), g)
        out[rule] = g
    return out


R = load_rules()
sys.argv = argv
sex = {l.split("\t")[0]: l.split("\t")[1] for l in SUBJ.read_text().splitlines()}


def strokes(f):
    a = np.asarray(Image.open(f).convert("RGB")).astype(int)
    o = np.asarray(Image.open(A / f.name).convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(a - o).sum(2) > 60)
    px = a[ys, xs]
    sel = {"red": (px[:, 0] > 180) & (px[:, 1] < 90) & (px[:, 2] < 90),
           "blue": (px[:, 2] > 180) & (px[:, 0] < 90) & (px[:, 1] < 120),
           "green": (px[:, 1] > 180) & (px[:, 0] < 90) & (px[:, 2] < 90)}
    return {k: np.c_[xs[m], ys[m]] / UP - 0.5 for k, m in sel.items() if m.sum() > 20}  # MRI px


def order(p, cell=0.25, r=0.8):
    """Points of an open stroke, ordered end to end (geodesic distance from one end)."""
    p = np.unique(np.round(p / cell).astype(int), axis=0) * cell
    pr = np.array(list(cKDTree(p).query_pairs(r)))
    w = np.hypot(*(p[pr[:, 0]] - p[pr[:, 1]]).T)
    G = coo_matrix((np.r_[w, w], (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(len(p),) * 2)
    d0 = dijkstra(G, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da = dijkstra(G, indices=a)
    k = np.isfinite(da)
    return p[k][np.argsort(da[k])]


def posterior_surface(p, cell=0.25):  # a flap drawn as a loop: its back (airway) side, top to bottom
    p = np.unique(np.round(p / cell).astype(int), axis=0) * cell
    return np.array([[p[p[:, 1] == y, 0].max(), y] for y in np.unique(p[:, 1])])


def top_down(c, upto):
    c = c if c[0][1] <= c[-1][1] else c[::-1]
    c = c[c[:, 1] < upto[1]]
    return np.r_[c, [upto]]


rows = []
for f in sorted((A / "annotated").glob("*.png")):
    name = f.name.rsplit("_", 1)[0]
    spk = name.split("_")[0]
    d = json.loads((GT / f"{name}.json").read_text())
    s = strokes(f)
    red = s["red"]
    c = red.mean(0)
    u = np.linalg.svd(red - c)[2][0]
    t = (red - c) @ u
    ends = red[[np.argmin(t), np.argmax(t)]]
    front, back = ends[np.argsort(ends[:, 0])]  # smaller x = anterior
    mark = (front + back) / 2
    res = {"speaker": spk, "sex": sex[spk], "frame": name.rsplit("-", 1)[1], "file": f.name,
           "glottal_angle_deg": float(np.degrees(np.arctan2(back[1] - front[1], back[0] - front[0])))}
    for rule, g in R.items():
        path, L, _, _ = g["midline"](d)
        _, glot, _ = g["airway"](d)
        tan = path[-1] - path[-4]
        tan /= np.linalg.norm(tan)
        off = mark - glot
        res[f"L_{rule}"] = L
        res[f"along_mm_{rule}"] = float(off @ tan * MM)
        res[f"across_mm_{rule}"] = float((off[0] * tan[1] - off[1] * tan[0]) * MM)
    du = {k: np.asarray(v, float) for k, v in d.items()}
    ep = posterior_surface(s["green"]) if "green" in s else du["epiglottis"]
    ar = order(s["blue"]) if "blue" in s else du["arytenoid"]
    du["epiglottis"], du["arytenoid"] = top_down(ep, front), top_down(ar, back)
    path, L, _, _ = R["lowest"]["midline"]({k: v.tolist() for k, v in du.items()})
    res["L_annotator"] = L
    res["redrawn"] = "+".join(k for k in ("blue", "green") if k in s) or "-"
    rows.append(res)

keys = list(rows[0])
Path(OUTF).write_text("\t".join(keys) + "\n" + "".join("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n" for r in rows))
print(f"{len(rows)} annotated frames")
print(f"{'frame':16s} sex  angle  along/across vs lowest (mm)  vs level (mm)   L lowest  level  annotator  redrawn")
for r in rows:
    print(f"{r['speaker'] + ' ' + r['frame']:16s} {r['sex']}  {r['glottal_angle_deg']:+5.0f}   {r['along_mm_lowest']:+5.1f} / {r['across_mm_lowest']:+5.1f}"
          f"        {r['along_mm_level']:+5.1f} / {r['across_mm_level']:+5.1f}    {r['L_lowest']:5.2f}  {r['L_level']:5.2f}  {r['L_annotator']:5.2f}  {r['redrawn']}")
for g in ("M", "F", None):
    m = [r for r in rows if g is None or r["sex"] == g]
    lab = {"M": "men", "F": "women", None: "all"}[g]
    print(f"  {lab:5s} (n={len(m)}): along vs lowest median {np.median([r['along_mm_lowest'] for r in m]):+.1f} mm, vs level "
          f"{np.median([r['along_mm_level'] for r in m]):+.1f} mm; L annotator - lowest {np.median([r['L_annotator'] - r['L_lowest'] for r in m]):+.2f} cm, "
          f"- level {np.median([r['L_annotator'] - r['L_level'] for r in m]):+.2f} cm")
