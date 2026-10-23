"""D-034 diagnostic (2026-10-24, after the stage-1 result; reported, no decision): do D-033's H5 heads, trained on real
Parakeet layer 5, keep their F1 correction when fed the student's output in Parakeet's place?
The student predicts standardised layer-5 vectors, so its output (destandardised with the distillation statistics and
averaged in eights onto Parakeet's 80 ms frames, as the distillation loss did) is read as layer 5 by the heads.
Control: real layer 5 plus Gaussian noise of the student's per-dimension error variance on each set (same error energy
per dimension, random direction; seeded).
Writes two scoring caches mirroring `emb_v1` for the four sets eval_heads.py reads (layer 5 replaced; other layers zero;
labels and teacher columns copied): <out>/student and <out>/noise. Then: eval_heads.py <out>/student ... H5#k=...
Args: emb_v1 dir, student scoring cache dir (student_cache.py), distillation cache dir (stats.npz), out dir. CPU only."""
import shutil
import sys
from pathlib import Path

import numpy as np

E1, SC, DC, OUT = map(Path, sys.argv[1:5])
st = np.load(DC / "stats.npz")
MU, SD = st["mean"].astype(np.float32), st["std"].astype(np.float32)
SETS = ("synth_train_v2", "synth_train_v2s", "gold_hillenbrand", "gold_ptdb_cp")


def pooled(s: np.ndarray, n_enc: int) -> np.ndarray:
    """student frames (stored k = mel k + 4, standardised) -> (n_enc, 1024) at encoder frames e (mel 8e): the mean of
    stored frames 8e - 8 .. 8e - 1, clipped to the item (frame 0 takes stored 0..3, the last frames the last eight)."""
    out = np.empty((n_enc, s.shape[1]), np.float32)
    for e in range(n_enc):
        a, b = max(8 * e - 8, 0), min(8 * e, len(s))
        if b <= a:
            a, b = (0, min(4, len(s))) if e == 0 else (max(len(s) - 8, 0), len(s))
        out[e] = s[a:b].mean(0)
    return out


rng = np.random.default_rng(20261024)
for name in SETS:
    pe, po = np.load(E1 / name / "emb.npy", mmap_mode="r"), np.load(E1 / name / "emb_off.npy")
    se, so = np.load(SC / name / "emb.npy", mmap_mode="r"), np.load(SC / name / "emb_off.npy")
    assert len(po) == len(so), name
    P = np.asarray(pe[:, 5], np.float32)
    S = np.concatenate([pooled(np.asarray(se[so[i]:so[i + 1]], np.float32), po[i + 1] - po[i]) for i in range(len(po) - 1)])
    Pz = (P - MU) / SD
    err_var = ((S - Pz) ** 2).mean(0)  # the student's per-dimension error, standardised units
    noisy = Pz + rng.standard_normal(Pz.shape).astype(np.float32) * np.sqrt(err_var)
    var = Pz.var(0)
    for kind, Z in (("student", S), ("noise", noisy)):
        d = OUT / kind / name
        d.mkdir(parents=True, exist_ok=True)
        for f in (E1 / name).glob("*.npy"):
            if f.stem != "emb":
                shutil.copy(f, d / f.name)
        emb = np.zeros((len(P), 6, 1024), np.float16)
        emb[:, 5] = Z * SD + MU
        np.save(d / "emb.npy", emb)
        r2 = 1 - ((Z - Pz) ** 2).mean(0) / var
        print(f"{kind:8s} {name:18s} fidelity to layer 5: R2 {r2.mean():.3f} ({len(P)} frames)", flush=True)
