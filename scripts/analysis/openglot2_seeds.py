"""Seed contrasts on OPENGLOT Repository II (evaluation only; D-026): Story's physically modelled vowels with
source-tract interaction (simulated, not recorded), 4 vowels x 4 f0 x 3 adductions per sex. Per file: each model's
median F1 over the middle of the file minus the calculated fR1; contrasts paired per file (b minus the mean of the
others), files where a contrasted model is more than 150 Hz off left out; by sex and adduction. Also each model's
median error, to compare with its real closed-phase bias. Args: out file, contrast spec 'b:a1,a2;...', name=ckpt ..."""
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
FR1 = {"Male": {"a": 752, "i": 340, "u": 367, "ae": 693}, "Female": {"a": 848, "i": 379, "u": 420, "ae": 795}}  # Table 2
contrasts = [(c.split(":")[0], c.split(":")[1].split(",")) for c in sys.argv[2].split(";")]
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[3:]}
rows = []
for f in sorted(Path("data/openglot").glob("RepositoryII_*/**/*.wav")):
    m = re.match(r"(Male|Female)_([A-Za-z]+)_(\d+)Hz_(small|medium|large)$", f.stem)
    if not m:
        continue
    sex, vowel = m.group(1), m.group(2).lower()
    x, sr = sf.read(f)
    x = x[:, 0] if x.ndim == 2 else x
    x16 = resample_poly(x, 16000, sr).astype(np.float32)
    x16 = 0.5 * x16 / (np.abs(x16).max() + 1e-12)
    n = len(x16) // 160 + 1
    mid = slice(int(0.3 * n), int(0.8 * n))
    rows.append((sex, m.group(4), {k: float(np.nanmedian(analyse(mod, x16, dev)["formants"][mid, 0])) - FR1[sex][vowel]
                                   for k, mod in models.items()}))
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    p = lambda *a: (print(*a, file=fh, flush=True), print(*a, flush=True))  # noqa: E731
    p(f"OPENGLOT II: {len(rows)} files. F1 minus calculated fR1 (Hz)")
    for sex in ("Male", "Female"):
        sel = [r for r in rows if r[0] == sex]
        p(f"\n{sex}: median error (files within 150 Hz) by adduction small / medium / large")
        for k in models:
            p(f"  {k:22s} " + " / ".join(f"{np.median([r[2][k] for r in sel if r[1] == a and abs(r[2][k]) <= 150]):+6.1f}"
                                         for a in ("small", "medium", "large")))
        p("  paired contrasts, all adductions (median [25th, 75th], n), then small / medium / large:")
        for b, others in contrasts:
            def con(rs):
                d = [r[2][b] - np.mean([r[2][a] for a in others]) for r in rs
                     if max(abs(r[2][x]) for x in (b, *others)) <= 150]
                return np.array(d)
            d = con(sel)
            p(f"    {b} - mean({', '.join(others)}): {np.median(d):+6.1f} [{np.percentile(d, 25):+.0f}, {np.percentile(d, 75):+.0f}] n {len(d)};  "
              + " / ".join(f"{np.median(con([r for r in sel if r[1] == a])):+.1f}" for a in ("small", "medium", "large")))
