"""Birkholz et al. (2019) supplementary audio: the teacher and Praat on vowels made with real (physical) and simulated
source-filter coupling, at five lung pressures (peak glottal area 11-60 mm^2). Reads F1-F3 per pressure; reports the
change from the lowest pressure and, for the simulations, the offset from their closed-glottis LF renders.
Args: data dir, out file, name=ckpt. Run from the m3 worktree. Evaluation only (no licence stated)."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
D, OUT = Path(sys.argv[1]), open(sys.argv[2], "w", encoding="utf-8")
name, ck = sys.argv[3].split("=", 1)
torch.set_num_threads(6)
dev = torch.device("cpu")
model = load_teacher(ck, dev)[0]
P_KPA = (0.8, 1.0, 1.2, 1.4, 1.6)
AREA = [11.27 + (p * 1000 - 800) * 0.0606 for p in P_KPA]  # mm^2, the paper's fit
VOWELS = {"i": "i", "y": "y", "u": "u", "e": "e", "oe": "ø", "o": "o", "ae": "ɛ", "schwa": "ə", "a": "a", "cylinder": "cyl"}
CLOSE = {"i", "y", "u", "e", "oe", "o"}  # the paper's strongly affected group (fR1 < 350 Hz)
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731


def read(path):
    y, sr = sf.read(path)
    y = resample_poly(y, 160, 441) if sr == 44100 else resample_poly(y, 320, 441)
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


def measure(y):
    """Median F1-F3 over the middle 60 % of y: (teacher, Praat)."""
    t = np.arange(0.2, 0.8, 0.01) * len(y) / 16000
    te = analyse(model, y, dev)["formants"][np.round(t / 0.01).astype(int), :3]
    pr = praat_default(y, 16000, t, {})[:, :3]
    return np.nanmedian(te, 0), np.nanmedian(pr, 0)


res = {}
for kind in ("physical-models", "simulations"):
    for v in VOWELS:
        f = D / f"sounds-of-{kind}" / f"{'cylinder' if v == 'cylinder' else 'vowel-' + v}-sequence.wav"
        y = read(f)
        seg = len(y) // 5
        res[(kind, v)] = [measure(y[i * seg:(i + 1) * seg]) for i in range(5)]
        if kind == "simulations":
            lf = D / "sounds-of-simulations" / f"LF-{'cylinder' if v == 'cylinder' else 'vowel-' + ('ue' if v == 'y' else v)}.wav"
            res[(kind, v, "LF")] = measure(read(lf))

p(f"{name} and Praat (Burg defaults) on Birkholz et al. 2019 supplementary audio; 16 kHz; peak glottal area by pressure: "
  + ", ".join(f"{a:.0f}" for a in AREA) + " mm^2")
for kind in ("physical-models", "simulations"):
    p(f"\n{kind}: F1 per pressure (Hz){'; LF = closed-glottis render' if kind == 'simulations' else ''}")
    p(f"{'vowel':6s}" + (f"{'LF':>12s}" if kind == "simulations" else "") + "".join(f"{f'{a:.0f} mm2':>14s}" for a in AREA)
      + f"{'rise 11->60 (teacher / Praat)':>32s}")
    for v, sym in VOWELS.items():
        r = res[(kind, v)]
        lf = f"{res[(kind, v, 'LF')][0][0]:5.0f}/{res[(kind, v, 'LF')][1][0]:<5.0f} " if kind == "simulations" else ""
        cells = "".join(f"{f'{t[0]:.0f}/{q[0]:.0f}':>14s}" for t, q in r)
        p(f"{sym:6s}{lf:>12s}{cells}{f'{r[-1][0][0] - r[0][0][0]:+.0f} / {r[-1][1][0] - r[0][1][0]:+.0f}':>32s}")
    for grp, keys in (("close (fR1 < 350)", [v for v in VOWELS if v in CLOSE]), ("open, cylinder", [v for v in VOWELS if v not in CLOSE])):
        for k, lab in ((0, "F1"), (1, "F2"), (2, "F3")):
            dt = [res[(kind, v)][-1][0][k] - res[(kind, v)][0][0][k] for v in keys]
            dq = [res[(kind, v)][-1][1][k] - res[(kind, v)][0][1][k] for v in keys]
            p(f"  {grp:18s} {lab} rise 11->60 mm2, median: {name} {np.median(dt):+5.0f}  Praat {np.median(dq):+5.0f} Hz")
    if kind == "simulations":
        for k, lab in ((0, "F1"), (1, "F2"), (2, "F3")):
            for i, a in ((0, AREA[0]), (4, AREA[4])):
                dt = [res[(kind, v)][i][0][k] - res[(kind, v, "LF")][0][k] for v in VOWELS]
                dq = [res[(kind, v)][i][1][k] - res[(kind, v, "LF")][1][k] for v in VOWELS]
                p(f"  {lab} vs own closed-glottis LF render at {a:.0f} mm2, median: {name} {np.median(dt):+5.0f}  Praat {np.median(dq):+5.0f} Hz")
