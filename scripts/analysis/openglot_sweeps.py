"""Resonances of the OPENGLOT Repository III vocal-tract replicas, digitised from the vector sweep plots
(`*_sweep.pdf`: average amplitude response, 80-7350 Hz, log frequency axis). The curve's end points give the
frequency axis (checked against the decade grid lines); the dB axis comes from the horizontal grid lines (0 to -50 dB
in 10 dB steps). Resonances: peaks with at least 3 dB prominence below 5 kHz. Needs pypdf. Writes a JSON
{replica: [peak Hz, ...]} for openglot_rep3.py. Args: out json. Run from the m3 worktree."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from scipy.signal import find_peaks

ROOT = Path("../voice-analyser/data/openglot/RepositoryIII")


def paths(data):
    out, cur = [], []
    for tok in re.finditer(r"([-\d.]+)\s+([-\d.]+)\s+(m|l)\b|(\bS\b|\bf\b|\bn\b)", data):
        if tok.group(3):
            if tok.group(3) == "m" and cur:
                out.append(cur)
                cur = []
            cur.append((float(tok.group(1)), float(tok.group(2))))
        elif cur:
            out.append(cur)
            cur = []
    return out


res = {}
for pdf in sorted(ROOT.glob("*/*_sweep.pdf")):
    ps = paths(PdfReader(pdf).pages[0].get_contents().get_data().decode("latin-1"))
    curve = np.array(max(ps, key=len))
    x, y = curve[:, 0], curve[:, 1]
    lf = np.log10(80.0) + (x - x[0]) / (x[-1] - x[0]) * (np.log10(7350.0) - np.log10(80.0))
    vert = sorted({round(p[0][0], 2) for p in ps if len(p) == 2 and abs(p[0][0] - p[1][0]) < 1e-3 and abs(p[0][1] - p[1][1]) > 20})
    horiz = sorted({round(p[0][1], 2) for p in ps if len(p) == 2 and abs(p[0][1] - p[1][1]) < 1e-3 and abs(p[0][0] - p[1][0]) > 20})
    to_hz = lambda xx: 10 ** (np.log10(80.0) + (xx - x[0]) / (x[-1] - x[0]) * (np.log10(7350.0) - np.log10(80.0)))  # noqa: E731
    decades = [round(float(to_hz(v))) for v in vert]
    # dB axis: horizontal grid lines at 0, -10, ..., -50 dB; y grows downward on the page here
    if len(horiz) >= 2:
        step = np.median(np.diff(horiz))
        db = -(y - horiz[0]) / step * 10.0
    else:
        db = -(y - y.min()) / (y.max() - y.min()) * 50.0
    f = 10 ** lf
    grid = np.logspace(np.log10(80), np.log10(7350), 4000)
    dbg = np.interp(np.log10(grid), lf, db)
    pk, props = find_peaks(dbg, prominence=3.0)
    peaks = [round(float(grid[i])) for i in pk if grid[i] < 5000]
    name = pdf.parent.name
    res[name] = peaks
    print(f"{name:10s} grid lines at {decades} Hz; peaks (>= 3 dB prominence, < 5 kHz): {peaks}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
