"""D-021 item 4 (VTL probe): per-crop tract-length labels for cache_heads.py's cache, from the set's clip labels
(scripts/analysis/vtl_labels.py). The cache keeps no clip ids, so they are recovered from the seeded draws
(MixedCrops' part choice, SynthCrops' clip choice; fixed=True) and checked against the cached formant labels: every
train_v2 clip (2-4 s) is shorter than the 400-frame crop, so a crop starts at the clip's first frame.
Resynthesised crops get NaN (their tract length is unknown).
Writes <cache>/vtl.npz: length_cm (N,) float32, from_f0, tube (bool), clip (int, -1 for resynthesised crops).
Run from the main checkout: python scripts/parakeet/vtl_cache_labels.py <cache dir>"""
import json
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.train.data import split_ids

CACHE = Path(sys.argv[1])
done = json.loads((CACHE / "done").read_text())
N, SEED = done["n"], done["seed"]
cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))["train"]
flat = Path(cfg3["data"]) / "flat"
train_ids, _ = split_ids(json.loads((Path(cfg3["data"]) / "index.json").read_text())["n_clips"])
sizes = [json.loads((Path(d) / "flat" / "done").read_text())["n_clips"] for d in cfg3["resynth"]]
frac = cfg3["resynth_frac"]
p = np.array([1 - frac] + [frac * n / sum(sizes) for n in sizes])
p = p / p.sum()

clip = np.full(N, -1, np.int64)
for i in range(N):  # MixedCrops.__getitem__ then SynthCrops.__getitem__ (fixed, epoch_size N != len(clip_ids))
    rng = np.random.default_rng([SEED, i, 17])
    if int(rng.choice(len(p), p=p)) == 0:
        j = int(rng.integers(N))
        clip[i] = train_ids[np.random.default_rng([SEED, j]).integers(len(train_ids))]
syn = clip >= 0

fo = np.load(flat / "frame_offsets.npy")
lf = np.load(flat / "label_formants.npy", mmap_mode="r")
cf = np.load(CACHE / "formants.npy", mmap_mode="r")
bad = 0
for i in np.flatnonzero(syn)[::20]:
    a, b = fo[clip[i]], fo[clip[i] + 1]
    if not np.array_equal(np.asarray(lf[a:b, :4], np.float32), np.asarray(cf[i, : b - a]), equal_nan=True):
        bad += 1
print(f"{syn.sum()} synthetic crops of {N}; checked {len(np.flatnonzero(syn)[::20])}, mismatched {bad}")
assert bad == 0

L = np.load(flat / "clip_length_cm.npy")
F = np.load(flat / "clip_length_from_f0.npy")
T = np.load(flat / "clip_tube.npy")
ix = np.where(syn, clip, 0)
length = np.where(syn, L[ix], np.nan).astype(np.float32)
from_f0, tube = syn & F[ix], syn & T[ix]
np.savez(CACHE / "vtl.npz", length_cm=length, from_f0=from_f0, tube=tube, clip=clip)
print(f"labelled (synthetic, not from f0): tube {(tube & ~from_f0).sum()}, cascade {(syn & ~tube & ~from_f0).sum()}; "
      f"distinct clips {len(np.unique(clip[syn]))}")
