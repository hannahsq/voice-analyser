"""Does resynthesis itself move F1 (D-026)? VocalSet val recordings re-filtered through their own smoothed teacher_v2_d16k
tracks with no perturbation (identity: alpha 1, no offsets, bandwidths x1, no nasal pair), with bandwidths only
scaled (x0.6, x1.6), and at alpha 0.9 / 1.1 for reference. Each model reads the original and each variant; per
recording the median F1 over sonorant voiced frames minus the label (the re-filter's F1), and for the original minus
the smoothed track. Args: n recordings, name=ckpt ... ($VA_DEVICE=cuda)."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.data.pools import vocalset
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.resynth import corpus
from vocal_analysis.resynth.render import ResynthConfig, refilter, source_from_audio
from vocal_analysis.resynth.tracks import PerturbConfig, perturb

sys.stdout.reconfigure(encoding="utf-8")
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
n_rec = int(sys.argv[1])
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
sha = corpus.file_sha256("runs/teacher_v2_d16k/ckpt_last.pt")
adir = Path("data/resynth/analysis/vocalset") / sha
root = Path("data/vocalset")
rows = [r for r in read_manifest("data/manifests/vocalset.csv") if r.split == "val" and vocalset.resynth_ok(r)]
rows = [rows[int(i)] for i in np.unique(np.linspace(0, len(rows) - 1, n_rec).round())]
cfg = ResynthConfig()
VARIANTS = {"identity": dict(alpha=(1.0, 1.0), offset_sd=(0.0, 0.0), bw_scale=(1.0, 1.0), upper_sd=0.0, p_nasal=0.0),
            "bw x0.6": dict(alpha=(1.0, 1.0), offset_sd=(0.0, 0.0), bw_scale=(0.6, 0.6), upper_sd=0.0, p_nasal=0.0),
            "bw x1.6": dict(alpha=(1.0, 1.0), offset_sd=(0.0, 0.0), bw_scale=(1.6, 1.6), upper_sd=0.0, p_nasal=0.0),
            "alpha 0.9": dict(alpha=(0.9, 0.9), offset_sd=(0.0, 0.0), bw_scale=(1.0, 1.0), upper_sd=0.0, p_nasal=0.0),
            "alpha 1.1": dict(alpha=(1.1, 1.1), offset_sd=(0.0, 0.0), bw_scale=(1.0, 1.0), upper_sd=0.0, p_nasal=0.0)}
res = {m: {v: [] for v in ["original", *VARIANTS]} for m in models}
f0s = []
for r in rows:
    path = adir / (r.path.replace("/", "__").rsplit(".", 1)[0] + ".npz")
    if not path.exists():
        continue
    a = dict(np.load(path))
    audio = corpus.load_audio(root / r.path)
    src = source_from_audio(audio, a, cfg)
    son = src["voiced"] & src["sonorant"]
    if son.sum() < 20:
        continue
    f0s.append(np.nanmedian(src["f0"][son]))
    ys = {"original": (audio, src["tracks"][:, 0])}
    for v, kw in VARIANTS.items():
        p = perturb(src["tracks"], src["f0"], src["voiced"], np.random.default_rng(0), PerturbConfig(**kw))
        y = refilter(src["source"], p["formants"], p["bandwidths"], p["extras"])
        ys[v] = (y, p["formants"][:, 0])
    for v, (y, lab) in ys.items():
        y = (0.5 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
        for name, m in models.items():
            pr = analyse(m, y, dev)["formants"][:, 0]
            k = min(len(pr), len(lab), len(son))
            ok = son[:k] & np.isfinite(pr[:k]) & np.isfinite(lab[:k])
            res[name][v].append(np.median(pr[:k][ok] - lab[:k][ok]))
f0s = np.array(f0s)
print(f"{len(f0s)} VocalSet val recordings (tracker teacher_v2_d16k). Median over recordings of the median F1 error (Hz)")
print("'original': the model on the untouched recording minus the smoothed track; others: on the resynthesis minus its label")
print(f"{'model':22s} " + "".join(f"{v:>11s}" for v in ["original", *VARIANTS]) + "   | identity, f0 < 300 / >= 300")
for name in models:
    e = {v: np.array(x) for v, x in res[name].items()}
    lo = f0s < 300
    print(f"{name:22s} " + "".join(f"{np.median(x):+11.1f}" for x in e.values())
          + f"   | {np.median(e['identity'][lo]):+.1f} / {np.median(e['identity'][~lo]):+.1f}  (n {lo.sum()} / {(~lo).sum()})")
