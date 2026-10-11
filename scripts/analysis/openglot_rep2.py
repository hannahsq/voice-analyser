"""OPENGLOT Repository II (Alku et al. 2019; free for research use with citation; evaluation only): physically modelled vowels with
source-tract interaction, male and female, 4 vowels x 4 f0 x 3 degrees of vocal-fold adduction (separation 0.03-0.09 cm;
wider = breathier, a larger gap). Each reader's median F1 (and F2) over the middle of the file against the calculated
resonances (paper Table 2). Args: out file, name=ckpt ... Run from the m3 worktree; data in ../voice-analyser/data/openglot."""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher

FR = {  # Table 2: fR1-fR4 (Hz)
    "Male": {"a": (752, 1095, 2616, 3169), "i": (340, 2237, 2439, 3668), "u": (367, 1180, 2395, 3945), "ae": (693, 1521, 2435, 3252)},
    "Female": {"a": (848, 1210, 2923, 3637), "i": (379, 2634, 4256, 5395), "u": (420, 1264, 2714, 4532), "ae": (795, 1700, 2692, 3740)}}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
root = Path("../voice-analyser/data/openglot")
rows = []
for f in sorted(root.glob("RepositoryII_*/**/*.wav")):
    m = re.match(r"(Male|Female)_([A-Za-z]+)_(\d+)Hz_(small|medium|large)$", f.stem)
    if not m:
        continue
    sex, vowel, f0 = m.group(1), m.group(2).lower(), int(m.group(3))
    add = {"Male": {"small": "0.03", "medium": "0.06", "large": "0.09"},  # vocal-process separation (cm): larger = wider gap
           "Female": {"small": "0.05", "medium": "0.07", "large": "0.09"}}[sex][m.group(4)]
    x, sr = sf.read(f)
    x = x[:, 0] if x.ndim == 2 else x
    x16 = resample_poly(x, 16000, sr).astype(np.float32)
    x16 = 0.5 * x16 / (np.abs(x16).max() + 1e-12)
    n = len(x16) // 160 + 1
    mid = slice(int(0.3 * n), int(0.8 * n))
    t = np.arange(n)[mid] * 0.01
    est = {"praat": np.nanmedian(praat_default(x16, 16000, t, {})[:, :2], axis=0)}
    for name, mod in models.items():
        est[name] = np.nanmedian(analyse(mod, x16, dev)["formants"][mid, :2], axis=0)
    fr = FR[sex][vowel]
    rows.append((sex, vowel, f0, add, {k: (v[0] - fr[0], v[1] - fr[1]) for k, v in est.items()}))
readers = ["praat", *models]
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    p = lambda *a: (print(*a, file=fh, flush=True), print(*a, flush=True))  # noqa: E731
    p(f"OPENGLOT Repository II: {len(rows)} files. F1 minus calculated fR1 (Hz): median over vowels and f0, by sex and adduction")
    for sex in ("Male", "Female"):
        for add in sorted({r[3] for r in rows if r[0] == sex}):
            sel = [r for r in rows if r[0] == sex and r[3] == add]
            p(f"  {sex:6s} adduction {add}: " + "  ".join(f"{k} F1 {np.median([r[4][k][0] for r in sel]):+5.0f} (F2 {np.median([r[4][k][1] for r in sel]):+5.0f})" for k in readers)
              + f"   [{len(sel)} files]")
    p("\nPer file (F1 error):")
    for r in rows:
        p(f"  {r[0]:6s} {r[1]:3s} f0 {r[2]:3d} add {r[3]}: " + "  ".join(f"{k} {r[4][k][0]:+5.0f}" for k in readers))
