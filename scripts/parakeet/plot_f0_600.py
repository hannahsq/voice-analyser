"""Excess power (student minus Parakeet, dB) of the 10 ms and 2 ms output trajectories on the synthetic held-out crops:
all crops, and the crops with median voiced f0 500-700 Hz, with vertical lines where 600 Hz and its harmonics fold to."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("C:/github/voice-analyser-parakeet/reports/analysis/parakeet")
OUT = Path(sys.argv[1])


def load(f):
    d = defaultdict(list)
    for r in csv.DictReader(open(R / f, encoding="utf-8")):
        d[(r["model"], r["set"])].append((float(r["hz"]), float(r["power_db"])))
    return {k: np.array(v) for k, v in d.items()}


def images(f0, fs, kmax=6):
    """where k * f0 folds to at sampling rate fs"""
    out = {}
    for k in range(1, kmax + 1):
        a = (k * f0) % fs
        out.setdefault(round(min(a, fs - a), 3), []).append(k)
    return out


def smooth(y, n):
    return np.convolve(y, np.ones(n) / n, mode="same") if n > 1 else y


SETS = {"held_synth_v2": ("-", "synthetic v2"), "held_synth_v2s": ("--", "synthetic v2s")}
COL = {"v2": "#c96442", "v4": "#3b6fb6"}
rows = [("all crops (10 ms: 512 per set; 2 ms: first 128)", "v2_12k_v4_10k"), ("median voiced f0 500-700 Hz (45 crops per set)", "v2_12k_v4_10k_f0_600")]
cols = [("10 ms output (100 Hz sampling)", "spectrum_vs_parakeet_{}_spectrum.csv", 100, 1), ("2 ms output (500 Hz sampling)", "subhop_2ms_{}_spectrum.csv", 500, 5)]
fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
for i, (rlab, tag) in enumerate(rows):
    for j, (clab, pat, fs, sm) in enumerate(cols):
        a = ax[i, j]
        d = load(pat.format(tag))
        for st in ("v2", "v4"):
            for s, (ls, slab) in SETS.items():
                p, q = d[(st, s)], d[("parakeet", s)]
                a.plot(p[1:, 0], smooth(p[1:, 1] - q[1:, 1], sm), ls, color=COL[st], lw=1, label=f"student {st}, {slab}")
        for hz, ks in images(600, fs).items():
            a.axvline(hz, color="0.35", ls=":", lw=1.2)
            a.text(hz, 1.0, " " + ", ".join(f"{k * 600}" for k in ks) + " Hz", rotation=90, va="top", ha="left" if hz < fs / 2 * 0.9 else "right",
                   fontsize=7, color="0.3", transform=a.get_xaxis_transform())
        a.axhline(0, color="0.6", lw=0.8)
        a.set_xlim(0, fs / 2)
        a.set_title(f"{clab}\n{rlab}", fontsize=9)
        a.set_xlabel("Hz")
        if j == 0:
            a.set_ylabel("student − Parakeet power, dB")
ax[0, 0].legend(fontsize=7, loc="lower left")
fig.suptitle("Excess power of the students' output trajectories against Parakeet (v2 step 12,000, v4 10k branch); dotted: where 600 Hz and its harmonics alias to"
             + "\n2 ms spectra smoothed over 5 bins (1.3 Hz)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
