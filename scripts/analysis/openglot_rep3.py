"""OPENGLOT Repository III (Alku et al. 2019; free for research use with citation; evaluation only): LF excitation
played through a loudspeaker into 3D-printed vocal tract replicas (no glottis, so no source-tract coupling), f0 100-500
Hz in 10 Hz steps. Each reader's median F1 over the 200 ms segment against the replica's first resonance, digitised from
its measured sweep (openglot_sweeps.py). By sex and f0 band. Args: resonances json, out file, name=ckpt ...
Run from the m3 worktree."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher

FR1 = {k: v[0] for k, v in json.load(open(sys.argv[1])).items()}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[3:]}
readers = ["praat", *models]
rows = []
for f in sorted(Path("../voice-analyser/data/openglot/RepositoryIII").glob("*/*Hz.wav")):
    m = re.match(r"(Male|Female)_([A-Za-z]+)_(\d+)Hz$", f.stem)
    replica, f0 = f"{m.group(1)}_{m.group(2)}", int(m.group(3))
    x, sr = sf.read(f)
    x = x[:, 1]  # the second channel is the radiated pressure here (the first is the LF flow), against the readme
    x16 = resample_poly(x, 16000, sr).astype(np.float32)
    x16 = 0.5 * x16 / (np.abs(x16).max() + 1e-12)
    n = len(x16) // 160 + 1
    mid = slice(int(0.15 * n), int(0.85 * n))  # the files are 200 ms: skip the edges
    t = np.arange(n)[mid] * 0.01
    est = {"praat": np.nanmedian(praat_default(x16, 16000, t, {})[:, 0])}
    for name, mod in models.items():
        est[name] = np.nanmedian(analyse(mod, x16, dev)["formants"][mid, 0])
    rows.append((m.group(1), replica, f0, {k: v - FR1[replica] for k, v in est.items()}))
bands = ((100, 150), (160, 250), (260, 350), (360, 500))
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    p = lambda *a: (print(*a, file=fh, flush=True), print(*a, flush=True))  # noqa: E731
    p(f"OPENGLOT Repository III (loudspeaker + 3D-printed tracts, no coupling): {len(rows)} files. "
      f"F1 minus the replica's measured fR1 (Hz), median over replicas and f0 in the band")
    p("fR1 (Hz): " + ", ".join(f"{k} {v}" for k, v in FR1.items()))
    for sex in ("Male", "Female"):
        for lo, hi in bands:
            sel = [r for r in rows if r[0] == sex and lo <= r[2] <= hi]
            if sel:
                p(f"  {sex:6s} f0 {lo}-{hi}: " + "  ".join(f"{k} {np.nanmedian([r[3][k] for r in sel]):+5.0f}" for k in readers) + f"   [{len(sel)} files]")
    p("\nPer replica, f0 100-300 Hz:")
    for rep in FR1:
        sel = [r for r in rows if r[1] == rep and r[2] <= 300]
        p(f"  {rep:10s} fR1 {FR1[rep]:4d}: " + "  ".join(f"{k} {np.nanmedian([r[3][k] for r in sel]):+5.0f}" for k in readers))
