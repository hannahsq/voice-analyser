"""Does a channel or source-tilt difference make seeds diverge? The sensitivity.py vowels (8 Peterson & Barney vowels,
men f0 120, women f0 210, Rd 1.0) after a whole-signal filter: spectral tilt (dB / octave about 1 kHz), high-pass,
telephone band, and Rd. Per condition and model: median over vowels of F1 minus the true F1, and the change from the
unfiltered render. Seeds that differ by a uniform offset on real voices (D-025) but not on the plain renders should
diverge under the condition that real recordings share. Args: name=ckpt ... ($VA_DEVICE=cuda for speed)."""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv
from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {n: load_teacher(c, dev)[0] for n, c in (a.split("=", 1) for a in sys.argv[1:])}
FS = sv.FS


def tilt(y, db_per_oct):
    Y = np.fft.rfft(y)
    f = np.maximum(np.fft.rfftfreq(len(y), 1 / FS), 50.0)
    return np.fft.irfft(Y * (f / 1000.0) ** (db_per_oct / 6.0206), len(y))


def band(y, lo=None, hi=None):
    if lo and hi:
        sos = butter(4, [lo, hi], "bandpass", fs=FS, output="sos")
    elif lo:
        sos = butter(2, lo, "highpass", fs=FS, output="sos")
    else:
        sos = butter(4, hi, "lowpass", fs=FS, output="sos")
    return sosfiltfilt(sos, y)


C = [("none", {}, None), ("tilt -6 dB/oct", {}, lambda y: tilt(y, -6)), ("tilt -3 dB/oct", {}, lambda y: tilt(y, -3)),
     ("tilt +3 dB/oct", {}, lambda y: tilt(y, 3)), ("tilt +6 dB/oct", {}, lambda y: tilt(y, 6)),
     ("high-pass 150 Hz", {}, lambda y: band(y, lo=150)), ("high-pass 300 Hz", {}, lambda y: band(y, lo=300)),
     ("telephone 300-3400 Hz", {}, lambda y: band(y, 300, 3400)), ("low-pass 4 kHz", {}, lambda y: band(y, hi=4000)),
     ("Rd 0.6 (pressed)", {"rd": 0.6}, None), ("Rd 1.6", {"rd": 1.6}, None), ("Rd 2.4 (breathy)", {"rd": 2.4}, None)]
print("F1 error vs true F1 (Hz), median over 8 PB vowels; change from 'none' in brackets")
for sex in ("men", "women"):
    print(f"\n{sex} f0 {sv.F0[sex]:.0f}\n{'condition':24s}" + "".join(f"{n[:21]:>22s}" for n in models))
    base = {}
    for cname, p, filt in C:
        ys = []
        for v, fv in enumerate(sv.PB[sex]):
            y = sv.render(fv, sex, sv.BASE | p, np.random.default_rng([v, 7]))
            y = filt(y) if filt else y
            ys.append((0.5 * y / np.abs(y).max()).astype(np.float32))
        cells = []
        for n, m in models.items():
            e = float(np.median([np.nanmedian(analyse(m, y, dev)["formants"][30:71, 0]) - fv[0] for y, fv in zip(ys, sv.PB[sex])]))
            base.setdefault(n, e)
            cells.append(f"{e:+5.0f} ({e - base[n]:+4.0f})")
        print(f"{cname:24s}" + "".join(f"{c:>22s}" for c in cells), flush=True)
