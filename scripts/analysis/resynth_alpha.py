"""Is the F1 overread on resynthesised clips residue of the original voice? Labels = tracked formants / alpha, so residue
pulls the error toward the original: error ~ label * (alpha - 1). Per clip: median signed F1 error on sonorant frames;
regress on log(alpha). The intercept (alpha = 1, labels at the original positions) is the overread without that pull.
Args: resynth set dir, n clips, name=ckpt ..."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
root, n_clips = Path(sys.argv[1]), int(sys.argv[2])
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {n: load_teacher(c, dev)[0] for n, c in (a.split("=", 1) for a in sys.argv[3:])}
clips = []
for sh in sorted(root.glob("shard_*.npz")):
    d = np.load(sh, allow_pickle=True)
    meta = d["meta"].item() if d["meta"].shape == () else d["meta"]
    meta = json.loads(meta) if isinstance(meta, str) else meta
    ao, fo = d["audio_offsets"], d["frame_offsets"]
    for i, m in enumerate(meta):
        clips.append((d["audio"][ao[i]:ao[i + 1]].astype(np.float32), d["label_formants"][fo[i]:fo[i + 1], 0],
                      d["label_sonorant"][fo[i]:fo[i + 1]].astype(bool), d["label_f0"][fo[i]:fo[i + 1]], m["perturb"]["alpha"]))
rng = np.random.default_rng(0)
clips = [clips[i] for i in sorted(rng.choice(len(clips), min(n_clips, len(clips)), replace=False))]
la = np.log([c[4] for c in clips])
print(f"{root}: {len(clips)} clips, alpha {np.exp(la.min()):.2f}-{np.exp(la.max()):.2f}")
print(f"{'model':12s} {'median err':>10s} | median err by alpha: <0.9 / 0.9-0.96 / 0.96-1.04 / 1.04-1.1 / >1.1 | alpha 0.96-1.04 at f0 150-300")
for name, m in models.items():
    e, f0m = [], []
    for audio, f1, son, f0, _ in clips:
        p = analyse(m, audio, dev)["formants"][:, 0]
        k = min(len(p), len(f1))
        ok = son[:k] & np.isfinite(f1[:k]) & np.isfinite(p[:k])
        e.append(np.median(p[:k][ok] - f1[:k][ok]) if ok.sum() > 10 else np.nan)
        f0m.append(np.nanmedian(np.where(ok, f0[:k], np.nan)) if ok.sum() > 10 else np.nan)
    e, f0m = np.array(e), np.array(f0m)
    g = np.isfinite(e)
    al = np.exp(la)
    bins = [(0, 0.9), (0.9, 0.96), (0.96, 1.04), (1.04, 1.1), (1.1, 9)]
    cells = " / ".join(f"{np.median(e[g & (al >= lo) & (al < hi)]):+6.1f}" for lo, hi in bins)
    mid = g & (al >= 0.96) & (al < 1.04) & (f0m >= 150) & (f0m < 300)
    print(f"{name:12s} {np.median(e[g]):+10.1f} | {cells} | {np.median(e[mid]):+6.1f} (n {mid.sum()})", flush=True)
