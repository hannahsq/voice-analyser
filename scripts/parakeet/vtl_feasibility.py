"""D-021 item 4, feasibility before pre-registering the VTL probe (AGENTS: check a hypothesis can hold): is anatomical
tract length decodable from frozen Parakeet layer 5 on the head cache, and how far above a shuffle null?
Per crop: the mean of layer 5 over sonorant encoder frames (frame e at mel 8e) -> ridge -> log2 length, on crops whose
length was not drawn from f0. Split by clip (clip id mod 5 == 0 held out). Also the uniform-tube apparent length from
the crop's *true* F1-F4 (backend.vtl, median over sonorant frames, then a linear calibration in log2): the ceiling of a
formant-derived estimate with perfect formants. Shuffle null: labels permuted across training crops, 5 times.
Args: cache dir, out txt."""
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.backend.vtl import apparent_vtl

CACHE, OUTF = Path(sys.argv[1]), sys.argv[2]
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
v = np.load(CACHE / "vtl.npz")
lab = np.isfinite(v["length_cm"]) & ~v["from_f0"]
idx = np.flatnonzero(lab)
emb = np.load(CACHE / "emb.npy", mmap_mode="r")
son = np.load(CACHE / "sonorant.npy", mmap_mode="r")
fm = np.load(CACHE / "formants.npy", mmap_mode="r")
E = emb.shape[1]
X, Y, Fa, keep = [], [], [], []
for i in idx:
    s = np.asarray(son[i])[(8 * np.arange(E)).clip(max=399)] > 0.5
    f = np.asarray(fm[i])
    fs = np.asarray(son[i]) > 0.5
    if s.sum() < 3:
        continue
    X.append(np.asarray(emb[i, :, 2][s], np.float32).mean(0))
    Fa.append(np.nanmedian(apparent_vtl(f[fs])))
    keep.append(i)
keep = np.array(keep)
X, Fa = np.array(X), np.log2(np.array(Fa))
y = np.log2(v["length_cm"][keep])
tube = v["tube"][keep]
test = v["clip"][keep] % 5 == 0
mu, sd = X[~test].mean(0), X[~test].std(0) + 1e-6
Xs = (X - mu) / sd


def ridge(Xtr, ytr, lam):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    return np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean())), ytr.mean()


def r2(yt, yp):
    return 1 - np.mean((yt - yp) ** 2) / np.var(yt)


p(f"VTL feasibility on {CACHE.name}: {len(keep)} crops (length not from f0; tube {tube.sum()}, cascade {(~tube).sum()}), "
  f"held out {test.sum()} by clip")
inner = (v["clip"][keep] % 5 == 1) & ~test
best = max((r2(y[inner], Xs[inner] @ w + b), lam) for lam in (1e1, 1e2, 1e3, 1e4, 1e5)
           for w, b in [ridge(Xs[~test & ~inner], y[~test & ~inner], lam)])
lam = best[1]
w, b = ridge(Xs[~test], y[~test], lam)
pred = Xs @ w + b
sdc = lambda m: np.std(1200 * (pred[m] - y[m]))  # noqa: E731
p(f"\nParakeet L5 crop mean -> ridge (lambda {lam:g}): held-out R^2 {r2(y[test], pred[test]):.3f} "
  f"(tube {r2(y[test & tube], pred[test & tube]):.3f}, cascade {r2(y[test & ~tube], pred[test & ~tube]):.3f}); "
  f"error SD {sdc(test):.0f} cents (label SD {np.std(1200 * y[test]):.0f})")
rng = np.random.default_rng(0)
nul = []
for _ in range(5):
    ys = y[~test].copy()
    rng.shuffle(ys)
    ws, bs = ridge(Xs[~test], ys, lam)
    nul.append(r2(y[test], Xs[test] @ ws + bs))
p(f"  shuffle null (5 permutations): held-out R^2 {np.mean(nul):+.3f} (max {np.max(nul):+.3f})")
ok = np.isfinite(Fa)
A = np.polyfit(Fa[~test & ok], y[~test & ok], 1)
fp = np.polyval(A, Fa)
p(f"\nTrue F1-F4, uniform-tube apparent length (median over sonorant frames), calibrated: held-out R^2 "
  f"{r2(y[test & ok], fp[test & ok]):.3f} (tube {r2(y[test & ok & tube], fp[test & ok & tube]):.3f}, "
  f"cascade {r2(y[test & ok & ~tube], fp[test & ok & ~tube]):.3f}); slope {A[0]:.3f}")
for name, m in (("tube", tube), ("cascade", ~tube)):
    Am = np.polyfit(Fa[~test & ok & m], y[~test & ok & m], 1)
    p(f"  calibrated on {name} only: held-out R^2 on {name} {r2(y[test & ok & m], np.polyval(Am, Fa[test & ok & m])):.3f}, "
      f"on the other {r2(y[test & ok & ~m], np.polyval(Am, Fa[test & ok & ~m])):.3f}")
