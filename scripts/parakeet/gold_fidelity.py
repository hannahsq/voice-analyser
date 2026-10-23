"""D-034 reported measure: the student's fidelity to Parakeet layer 5 on the dev gold sets' own audio (whole files), from
the scoring caches: Parakeet's layer 5 from `emb_v1` (encoder frame e at mel 8e), the student's output from
student_cache.py (stored frame k = file frame k + 4, so encoder frame e is the mean of stored frames 8e - 8 .. 8e - 1, as
in distillation). Parakeet standardised with the distillation cache's statistics (the student predicts standardised
targets). Also an alignment check: R^2 with the student shifted by -2..+2 encoder frames (peak expected at 0).
Args: emb_v1 dir, student scoring cache dir, distillation cache dir (stats.npz). CPU only."""
import sys
from pathlib import Path

import numpy as np

E1, SC, DC = map(Path, sys.argv[1:4])
st = np.load(DC / "stats.npz")
MU, SD = st["mean"], st["std"]


def pairs(name, shift=0):
    pe, po = np.load(E1 / name / "emb.npy", mmap_mode="r"), np.load(E1 / name / "emb_off.npy")
    se, so = np.load(SC / name / "emb.npy", mmap_mode="r"), np.load(SC / name / "emb_off.npy")
    P, S = [], []
    for i in range(len(po) - 1):
        p = np.asarray(pe[po[i]:po[i + 1], 5], np.float32)
        s = np.asarray(se[so[i]:so[i + 1]], np.float32)
        for e in range(1, len(p)):
            a = 8 * (e + shift) - 8
            if 0 <= a and a + 8 <= len(s):
                P.append((p[e] - MU) / SD)
                S.append(s[a:a + 8].mean(0))
    return np.array(P), np.array(S)


def score(P, S):
    r2 = 1 - ((S - P) ** 2).sum(0) / ((P - P.mean(0)) ** 2).sum(0)
    cos = (S * P).sum(1) / (np.linalg.norm(S, axis=1) * np.linalg.norm(P, axis=1) + 1e-12)
    return r2.mean(), cos.mean()


for name in ("gold_hillenbrand", "gold_ptdb_cp"):
    P, S = pairs(name)
    r2, cos = score(P, S)
    print(f"{name}: R2 {r2:.3f}, cosine {cos:.3f} ({len(P)} encoder frames)")
    print("  alignment, shift in encoder frames: " + ", ".join(f"{k:+d}: {score(*pairs(name, k))[0]:.3f}" for k in (-2, -1, 0, 1, 2)))
