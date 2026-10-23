"""D-034 stage 1c: a head cache of the student's 80 ms means, from its 10 ms head cache (student_cache.py --heads), in
cache_heads.py's format so train_cached.py reads it as Parakeet layer 5: encoder frame e (mel 8e) = the mean of the
student's stored frames 8e - 8 .. 8e - 1 (mels 8e - 4 .. 8e + 3; frame 0 the first four), destandardised with the
distillation statistics. The crops are cache_heads.py's (checked against its labels).
Args: student head cache dir, Parakeet head cache dir (for the frame count and the label check), distillation cache dir
(stats.npz), out dir."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SH, PH, DC, OUT = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
st = np.load(DC / "stats.npz")
MU, SD = st["mean"].astype(np.float32), st["std"].astype(np.float32)
src = np.load(SH / "emb.npy", mmap_mode="r")
N, T, _ = src.shape
E = np.load(PH / "emb.npy", mmap_mode="r").shape[1]
for k in ("formants", "f0"):  # the same crops
    a, b = np.load(SH / f"{k}.npy", mmap_mode="r"), np.load(PH / f"{k}.npy", mmap_mode="r")
    assert np.array_equal(np.asarray(a[:200]), np.asarray(b[:200]), equal_nan=True), k
win = [(0, 4)] + [(8 * e - 8, 8 * e) for e in range(1, E)]
assert win[-1][1] <= T
dst = np.lib.format.open_memmap(OUT / "emb.npy", "w+", np.float16, (N, E, 1, 1024))
for i in range(0, N, 500):
    x = np.asarray(src[i:i + 500], np.float32)
    dst[i:i + len(x), :, 0] = np.stack([x[:, a:b].mean(1) for a, b in win], 1) * SD + MU
dst.flush()
for f in SH.glob("*.npy"):
    if f.stem != "emb":
        shutil.copy(f, OUT / f.name)
(OUT / "done").write_text(json.dumps({"n": N, "layers": [5], "seed": 999, "crop_frames": T, "pooled_from": str(SH)}))
print(f"{N} crops, {E} frames of 80 ms")
