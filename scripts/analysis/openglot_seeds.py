"""Seed contrasts on OPENGLOT Repository III (evaluation only; D-026): LF excitation through a loudspeaker into 3D-printed
tract replicas, so a real tract and channel but a synthetic source and no glottis. If a seed's uniform F1 offset on real
voices (mix_conformer_s1 +13 Hz over s0 / s2 for women) comes from the real tract or channel, it should show here.
Per file: each model's median F1 over the middle of the 200 ms file; contrasts are paired per file (b minus the mean of
the others), median over files with f0 <= 300 Hz, by sex, with files where any contrasted model is more than 150 Hz from
fR1 (assignment failures) left out. Args: resonances json, out file, contrast spec 'b:a1,a2;b:a1 ...', name=ckpt ..."""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
FR1 = {k: v[0] for k, v in json.load(open(sys.argv[1])).items()}
contrasts = [(c.split(":")[0], c.split(":")[1].split(",")) for c in sys.argv[3].split(";")]
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[4:]}
rows = []
for f in sorted(Path("data/openglot/RepositoryIII").glob("*/*Hz.wav")):
    m = re.match(r"(Male|Female)_([A-Za-z]+)_(\d+)Hz$", f.stem)
    replica, f0 = f"{m.group(1)}_{m.group(2)}", int(m.group(3))
    if f0 > 300:
        continue
    x, sr = sf.read(f)
    x16 = resample_poly(x[:, 1], 16000, sr).astype(np.float32)  # second channel: radiated pressure (openglot_rep3.py)
    x16 = 0.5 * x16 / (np.abs(x16).max() + 1e-12)
    n = len(x16) // 160 + 1
    mid = slice(int(0.15 * n), int(0.85 * n))
    rows.append((m.group(1), replica, f0, {k: float(np.nanmedian(analyse(mod, x16, dev)["formants"][mid, 0])) - FR1[replica]
                                          for k, mod in models.items()}))
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    p = lambda *a: (print(*a, file=fh, flush=True), print(*a, flush=True))  # noqa: E731
    p(f"OPENGLOT III, f0 100-300 Hz: {len(rows)} files. F1 minus the replica's fR1 (Hz)")
    for sex in ("Male", "Female"):
        sel = [r for r in rows if r[0] == sex]
        p(f"\n{sex} ({len(sel)} files): median error, all files / without failures (|err| > 150 Hz for that model)")
        for k in models:
            e = np.array([r[3][k] for r in sel])
            p(f"  {k:22s} {np.nanmedian(e):+6.1f} / {np.nanmedian(e[np.abs(e) <= 150]):+6.1f}   failures {np.sum(np.abs(e) > 150)}")
        p("  paired contrasts (median over files, [25th, 75th percentile], files used):")
        for b, others in contrasts:
            d = []
            for r in sel:
                v = [r[3][b], *[r[3][a] for a in others]]
                if all(np.isfinite(v)) and max(abs(x) for x in v) <= 150:
                    d.append(v[0] - np.mean(v[1:]))
            d = np.array(d)
            p(f"    {b} - mean({', '.join(others)}): {np.median(d):+6.1f} [{np.percentile(d, 25):+.0f}, {np.percentile(d, 75):+.0f}]  n {len(d)}"
              if len(d) else f"    {b}: no usable files")
        p("  per replica, median error:")
        for rep in sorted({r[1] for r in sel}):
            rs = [r for r in sel if r[1] == rep]
            p(f"    {rep:10s} fR1 {FR1[rep]:4d}: " + "  ".join(f"{k} {np.nanmedian([r[3][k] for r in rs]):+.0f}" for k in models))
