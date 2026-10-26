"""Paired comparison of two student checkpoints on the same held-out crops: per group, v2 - v1 for R2 (mean per dim),
R2 (variance-weighted) and cosine, with a bootstrap over crops (2000 resamples, the same resample for both models).
Each model is scored in its own standardised space (its cache's stats); R2 and cosine are invariant to per-dimension
affine transforms applied to both prediction and target, so the two are comparable."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fid_breakdown as fb  # noqa: E402  (module-level setup reads argv[1] as the cache dir)
from student import Student  # noqa: E402

C1, CK1, C2, CK2 = map(Path, sys.argv[1:5])
dev = torch.device("cuda")


def preds(cache, ck, audio, target):
    st = np.load(cache / "stats.npz")
    fb.MU, fb.SD = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    m = Student().to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
    m.eval()
    Y, T = fb.predict(m, audio, target)
    return Y.cpu().numpy().astype(np.float64), T.cpu().numpy().astype(np.float64)  # (crops, 50, 1024)


def suff(Y, T):
    """per-crop sufficient statistics: sum T, sum T^2, sum (Y-T)^2 per dim, and the crop's mean frame cosine"""
    cos = (Y * T).sum(2) / (np.linalg.norm(Y, axis=2) * np.linalg.norm(T, axis=2) + 1e-12)
    return T.sum(1), (T ** 2).sum(1), ((Y - T) ** 2).sum(1), cos.mean(1)


def metrics(S, W):
    """metrics for crop-count weights W (resamples, crops); exact equal to pooling the resampled frames"""
    st, stt, se, cos = S
    nf = W.sum(1, keepdims=True) * 50
    ss = W @ stt - (W @ st) ** 2 / nf
    e = W @ se
    return np.stack([np.mean(1 - e / ss, 1), 1 - e.sum(1) / ss.sum(1), (W @ cos) / W.sum(1)], 1)


va = np.load(C1 / "held_vocalset" / "audio.npy", mmap_mode="r")
vt = np.load(C1 / "held_vocalset" / "target.npy", mmap_mode="r")
paths = fb.vocalset_paths()
groups = {"all": np.arange(512)}
for lab, idx in (("task", -3), ("technique", -2)):
    g = np.array([Path(p).parts[idx] for p in paths])
    for v in sorted(set(g)):
        if (g == v).sum() >= 15:
            groups[f"{lab} {v}"] = np.flatnonzero(g == v)
S1 = suff(*preds(C1, CK1, va, vt))
S2 = suff(*preds(C2, CK2, va, vt))
rng = np.random.default_rng(20261023)
print(f"{'VocalSet group':28s} {'n':>3s}  " + "  ".join(f"{m:>27s}" for m in ("R2 per dim v1 -> v2 [95%]", "R2 var-weighted", "cosine")))
for name, ix in groups.items():
    s1, s2 = (tuple(x[ix] for x in S) for S in (S1, S2))
    one = np.ones((1, len(ix)))
    a, b = metrics(s1, one)[0], metrics(s2, one)[0]
    W = np.stack([np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix)) for _ in range(2000)]).astype(np.float64)
    boot = metrics(s2, W) - metrics(s1, W)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    cells = []
    for k in range(3):
        mark = "*" if lo[k] > 0 or hi[k] < 0 else " "
        cells.append(f"{a[k]:5.2f}->{b[k]:5.2f} [{lo[k]:+.2f},{hi[k]:+.2f}]{mark}")
    print(f"{name:28s} {len(ix):3d}  " + "  ".join(f"{c:>27s}" for c in cells), flush=True)
print("* the 95 % interval of v2 - v1 excludes 0")
