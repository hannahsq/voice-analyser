"""alpha 0.96-1.04 clips of a resynth set: per-clip median F1 error against the clip's QC (tract_left: fraction of the
tract left in the source; ripple dB), source H1-H2 and f0. Args: set dir, name=ckpt ..."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from vocal_analysis.models.infer import analyse, load_teacher
sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1]); dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {n: load_teacher(c, dev)[0] for n, c in (a.split("=", 1) for a in sys.argv[2:])}
clips = []
for sh in sorted(root.glob("shard_*.npz")):
    d = np.load(sh, allow_pickle=True); meta = d["meta"].item() if d["meta"].shape == () else d["meta"]
    meta = json.loads(meta) if isinstance(meta, str) else meta
    ao, fo = d["audio_offsets"], d["frame_offsets"]
    for i, m in enumerate(meta):
        if 0.96 <= m["perturb"]["alpha"] < 1.04:
            s = slice(fo[i], fo[i + 1]); h = d["label_source_h_db"][s]
            clips.append((d["audio"][ao[i]:ao[i + 1]].astype(np.float32), d["label_formants"][s, 0], d["label_sonorant"][s].astype(bool),
                          np.nanmedian(d["label_f0"][s]), m["qc"]["tract_left_median"], m["qc"]["ripple_median_db"],
                          np.nanmedian(h[:, 0] - h[:, 1]), np.nanmedian(d["label_bandwidths"][s, 0]), m["speaker_id"]))
print(f"{len(clips)} clips with alpha 0.96-1.04")
X = {"tract_left": [c[4] for c in clips], "ripple dB": [c[5] for c in clips], "source H1-H2": [c[6] for c in clips],
     "f0": [c[3] for c in clips], "label B1": [c[7] for c in clips]}
for name, m in models.items():
    e = []
    for audio, f1, son, *_ in clips:
        p = analyse(m, audio, dev)["formants"][:, 0]; k = min(len(p), len(f1))
        ok = son[:k] & np.isfinite(f1[:k]) & np.isfinite(p[:k])
        e.append(np.median(p[:k][ok] - f1[:k][ok]) if ok.sum() > 10 else np.nan)
    e = np.array(e); g = np.isfinite(e)
    tl = np.array(X["tract_left"])
    print(f"\n{name}: median {np.median(e[g]):+.1f} Hz; by tract_left <0 / 0-0.1 / >0.1: " +
          " / ".join(f"{np.median(e[g & (tl >= a) & (tl < b)]):+.1f} (n {(g & (tl >= a) & (tl < b)).sum()})" for a, b in [(-9, 0), (0, 0.1), (0.1, 9)]))
    for k_, v in X.items():
        v = np.array(v, float); ok = g & np.isfinite(v)
        print(f"  rho(err, {k_:13s}) {spearmanr(v[ok], e[ok]).statistic:+.2f}")
    spk = np.array([c[8] for c in clips])
    print("  per singer: " + "  ".join(f"{s.split(':')[1]} {np.median(e[g & (spk == s)]):+.0f}" for s in np.unique(spk)))
