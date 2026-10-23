"""D-034 diagnostic (2026-10-24, reported, no decision): the part of the student's 10 ms output that distillation left
unconstrained. The loss matched only means over fixed eights; a shift-equivariant student can carry, at no cost, content
that averages to zero over every eight: 12.5 Hz and its multiples in the 10 ms trajectory.
  1. Spectrum: mean power over dimensions of each item's 10 ms trajectory (400-frame segments, Hann), peak ratio at
     12.5 / 25 / 37.5 / 50 Hz against neighbouring bins; and the share of energy outside a centred sliding mean of eight.
  2. Probe: ridge regression of log2 F1 (synthetic labels, sonorant frames) from the smooth part (sliding mean of eight),
     the residual (frame minus that mean) and the whole frame; fit on synth_train_v2, scored on synth_train_v2s.
  3. Writes a filtered scoring cache (every set's frames replaced by the sliding mean), for stage 1's heads
     (eval_heads.py --feat=sv2=<out>).
Args: emb_v1 dir (labels), student scoring cache dir, control-R scoring cache dir, out dir for the filtered cache."""
import shutil
import sys
from pathlib import Path

import numpy as np

E1, SC, RC, OUT = map(Path, sys.argv[1:5])


def box8(x: np.ndarray) -> np.ndarray:
    """centred sliding mean of eight (frames t - 4 .. t + 3, the pooling window's alignment), clipped at the ends"""
    c = np.concatenate([np.zeros((1, x.shape[1]), np.float64), np.cumsum(x, 0, dtype=np.float64)])
    i = np.arange(len(x))
    a, b = np.maximum(i - 4, 0), np.minimum(i + 4, len(x))
    return ((c[b] - c[a]) / (b - a)[:, None]).astype(np.float32)


def items(cache, name):
    e, o = np.load(cache / name / "emb.npy", mmap_mode="r"), np.load(cache / name / "emb_off.npy")
    for i in range(len(o) - 1):
        yield np.asarray(e[o[i]:o[i + 1]], np.float32)


def spectrum(cache, name):
    P, n, res, tot = 0.0, 0, 0.0, 0.0
    for x in items(cache, name):
        r = x - box8(x)
        res += (r ** 2).sum()
        tot += ((x - x.mean(0)) ** 2).sum()
        for s in range(0, len(x) - 399, 400):
            seg = x[s:s + 400] - x[s:s + 400].mean(0)
            P = P + (np.abs(np.fft.rfft(seg * np.hanning(400)[:, None], axis=0)) ** 2).mean(1)
            n += 1
    P = P / n
    peaks = []
    for k in (50, 100, 150, 200):  # 0.25 Hz bins: 12.5, 25, 37.5, 50 Hz
        nb = np.r_[k - 10:k - 2, k + 3:min(k + 11, len(P))]
        peaks.append(P[k - 1:k + 2].max() / np.median(P[nb]))
    return peaks, res / tot, n


print("1. spectrum of the 10 ms trajectory: peak / neighbours at 12.5, 25, 37.5, 50 Hz; energy share outside the sliding mean")
for label, cache in (("student v2", SC), ("control R", RC)):
    for name in ("synth_train_v2", "gold_ptdb_cp"):
        pk, share, n = spectrum(cache, name)
        print(f"  {label:10s} {name:16s} " + "  ".join(f"{p:6.2f}" for p in pk) + f"   residual share {share:.3f}  ({n} segments)")


def frames(name):
    X = np.concatenate(list(items(SC, name)))
    f1 = np.load(E1 / name / "formants.npy")[:, 0]
    son = np.load(E1 / name / "sonorant.npy") > 0.5
    ok = son & np.isfinite(f1) & (f1 > 0)
    S = np.concatenate([box8(x) for x in items(SC, name)])
    return X[ok], S[ok], (X - S)[ok], np.log2(f1[ok])


def ridge(A, y, B, z, lam=10.0):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    A, B = (A - mu) / sd, (B - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * len(A) / 1000 * np.eye(A.shape[1]), A.T @ (y - y.mean()))
    p = B @ w + y.mean()
    return 1 - ((z - p) ** 2).sum() / ((z - z.mean()) ** 2).sum()


print("2. ridge probe for log2 F1 (fit synth_train_v2, score synth_train_v2s; R2)")
Xa, Sa, Ra, ya = frames("synth_train_v2")
Xb, Sb, Rb, yb = frames("synth_train_v2s")
for label, A, B in (("whole frame", Xa, Xb), ("smooth part (sliding mean of 8)", Sa, Sb), ("residual (frame - mean)", Ra, Rb)):
    print(f"  {label:34s} {ridge(A, ya, B, yb):.3f}   ({len(ya)} / {len(yb)} frames)")

print("3. filtered scoring cache ->", OUT)
for d in sorted(p for p in SC.iterdir() if (p / "emb_off.npy").exists()):  # the scoring sets, not the head cache
    (OUT / d.name).mkdir(parents=True, exist_ok=True)
    np.save(OUT / d.name / "emb.npy", np.concatenate([box8(x) for x in items(SC, d.name)]).astype(np.float16))
    shutil.copy(d / "emb_off.npy", OUT / d.name / "emb_off.npy")
shutil.copy(SC / "meta.json", OUT / "meta.json")
print("done")
