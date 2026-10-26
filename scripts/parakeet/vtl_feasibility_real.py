"""D-021 item 4, feasibility on real voices before pre-registering the VTL probe: noise floors of the candidate measures
on Hillenbrand dev speakers (each speaker's ~12 h-V-d files), for three per-frame length estimators read at the
reference frames and summarised per file as their median (log2 cm):
- apparent (labeller v3): the uniform-tube length (backend.vtl, equal weights) from the mean of teacher v3 seeds 1 and
  2's F1-F4, calibrated linearly in log2 on synthetic tube crops' true F1-F4 (per-crop median; length not from f0);
- apparent (reference F1-F3): the same from Hillenbrand's own F1-F3 (F4 is mostly missing), calibrated on F1-F3;
- L5 ridge: a per-frame ridge from frozen Parakeet layer 5 to log2 length, fitted on sampled sonorant frames of the
  head cache's labelled crops (tube and cascade, not from f0), the encoder frame nearest each reference frame.
Measures: group medians (cm); women / men ratio of speaker medians; within-speaker SD across files (cents, pooled);
reliability within each adult group (between-speaker variance of speaker means over between + within-file variance);
speaker bootstrap within group (2000) for intervals and paired differences.
Args: head cache dir, Hillenbrand cache dir (cache.py's gold_hillenbrand), out txt."""
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.backend.vtl import apparent_vtl

CACHE, HB, OUTF = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
rng = np.random.default_rng(20261026)

# fit: per-frame ridge on L5, and the apparent-length calibrations, on the head cache
v = np.load(CACHE / "vtl.npz")
lab = np.flatnonzero(np.isfinite(v["length_cm"]) & ~v["from_f0"])
emb = np.load(CACHE / "emb.npy", mmap_mode="r")
son = np.load(CACHE / "sonorant.npy", mmap_mode="r")
fm = np.load(CACHE / "formants.npy", mmap_mode="r")
E = emb.shape[1]
X, Y, a4, a3, yc = [], [], [], [], []
for i in lab:
    s10 = np.asarray(son[i]) > 0.5
    e = np.flatnonzero(s10[(8 * np.arange(E)).clip(max=399)])
    if len(e) < 3:
        continue
    pick = rng.choice(e, min(8, len(e)), replace=False)
    X.append(np.asarray(emb[i][pick, 2], np.float32))
    Y.append(np.full(len(pick), np.log2(v["length_cm"][i])))
    if v["tube"][i]:
        f = np.asarray(fm[i])[s10]
        a4.append(np.nanmedian(apparent_vtl(f)))
        a3.append(np.nanmedian(apparent_vtl(f[:, :3])))
        yc.append(np.log2(v["length_cm"][i]))
X, Y = np.concatenate(X), np.concatenate(Y)
mu, sd = X.mean(0), X.std(0) + 1e-6
Xs = (X - mu) / sd
LAM = 1e3
W = np.linalg.solve(Xs.T @ Xs + LAM * np.eye(1024), Xs.T @ (Y - Y.mean()))
B0 = Y.mean()
cal4 = np.polyfit(np.log2(a4), yc, 1)
cal3 = np.polyfit(np.log2(a3), yc, 1)
p(f"Fitted on {len(lab)} labelled crops: ridge on {len(Y)} sonorant L5 frames (lambda {LAM:g}); calibrations on "
  f"{len(yc)} tube crops: F1-F4 slope {cal4[0]:.3f}, F1-F3 slope {cal3[0]:.3f}")

# Hillenbrand dev: per reference frame, then per file
g, spk = np.load(HB / "group.npy"), np.load(HB / "speaker.npy")
ro, eo, mel = np.load(HB / "ref_off.npy"), np.load(HB / "emb_off.npy"), np.load(HB / "mel.npy")
hemb = np.load(HB / "emb.npy", mmap_mode="r")
lf = (np.load(HB / "m_tv3_s1_log2F.npy") + np.load(HB / "m_tv3_s2_log2F.npy")) / 2
ref = np.load(HB / "ref_formants.npy")
est = {"apparent (labeller v3)": np.polyval(cal4, np.log2(apparent_vtl(2 ** lf))),
       "apparent (reference F1-F3)": np.polyval(cal3, np.log2(apparent_vtl(ref[:, :3])))}
l5 = np.full(len(g), np.nan)
for k in range(len(ro) - 1):
    a, b = ro[k], ro[k + 1]
    n = eo[k + 1] - eo[k]
    e = np.clip(np.round(mel[a:b] / 8).astype(int), 0, n - 1)
    l5[a:b] = ((np.asarray(hemb[eo[k] + e, 5], np.float32) - mu) / sd) @ W + B0
est["L5 ridge"] = l5
fg = g[ro[:-1]]
fs = spk[ro[:-1]]
per_file = {k: np.array([np.nanmedian(x[ro[j]:ro[j + 1]]) for j in range(len(ro) - 1)]) for k, x in est.items()}
spks = {G: sorted(set(fs[fg == G])) for G in "mwbg"}


def measures(e, draw):
    """e: per-file log2 cm; draw: {group: list of speakers (with repeats)}."""
    sm = {G: np.array([np.nanmedian(e[fs == s]) for s in draw[G]]) for G in draw}
    within = np.concatenate([e[fs == s] - np.nanmean(e[fs == s]) for G in draw for s in draw[G]])
    wsd = 1200 * np.sqrt(np.nanmean(within ** 2))
    rel = {}
    for G in "mw":
        means = np.array([np.nanmean(e[fs == s]) for s in draw[G]])
        wv = np.nanmean(np.concatenate([(e[fs == s] - np.nanmean(e[fs == s])) ** 2 for s in draw[G]]))
        rel[G] = np.var(means) / (np.var(means) + wv)
    return {"men": 2 ** np.median(sm["m"]), "women": 2 ** np.median(sm["w"]),
            "children": 2 ** np.median(np.concatenate([sm["b"], sm["g"]])),
            "w/m": 2 ** (np.median(sm["w"]) - np.median(sm["m"])), "within SD": wsd,
            "rel men": rel["m"], "rel women": rel["w"]}


pt = {k: measures(e, spks) for k, e in per_file.items()}
bs = []
for _ in range(2000):
    d = {G: [spks[G][j] for j in rng.integers(0, len(spks[G]), len(spks[G]))] for G in spks}
    bs.append({k: measures(e, d) for k, e in per_file.items()})
p(f"\nHillenbrand dev: {', '.join(f'{G} {len(spks[G])}' for G in 'mwbg')} speakers, {len(fg)} files")
keys = list(pt[next(iter(pt))])
for k in pt:
    p(f"\n{k}")
    for m in keys:
        x = np.array([b[k][m] for b in bs])
        lo, hi = np.percentile(x, [2.5, 97.5])
        p(f"  {m:10s} {pt[k][m]:8.3f} [{lo:8.3f}, {hi:8.3f}]")
p("\nPaired differences (speaker bootstrap): L5 ridge - apparent (labeller v3)")
for m in keys:
    x = np.array([b["L5 ridge"][m] - b["apparent (labeller v3)"][m] for b in bs])
    lo, hi = np.percentile(x, [2.5, 97.5])
    p(f"  {m:10s} {pt['L5 ridge'][m] - pt['apparent (labeller v3)'][m]:+8.3f} [{lo:+8.3f}, {hi:+8.3f}]")
p("\nAnatomy for orientation (Vorperian et al. 2018, ages 18-25): men 17.6, women 15.6 cm, w/m 0.886; children 4-12 "
  "11.4-14.5 cm")
