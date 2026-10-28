"""D-046: QCP-FB as a formant reference without an EGG. Stage 1 on D-043's renders.

Per render (D-043's interactive tube renderer, render D), the same five frame times (0.3-0.7 s), median over frames,
minus the closed-glottis poles (Hz), for:
  cp       the closed-phase reference as D-043 (10 kHz, order 12, ideal instants from the area);
  fb@R     QCP-FB at rate R (8, 10, 16 kHz), GCIs from SEDREAMS (f0_mean = the render's f0);
  fbI@R    QCP-FB at rate R with ideal GCIs (the area's closures);
  q@8      forward-only QCP at 8 kHz, SEDREAMS GCIs.
Also per render: SEDREAMS' identification rate (true closures, shifted by the glottis-to-lips lag that `speech_lag`
measures, with an estimate within +-0.5 ms) and the share of frames each method returns. "Ideal GCIs" are those
lag-shifted closures.

    python scripts/analysis/d046_qcp.py stage1 <out dir> [--smoke | --high]
--smoke: development renders off the grid (women, f0 160 / 240, open quotient 0.50 / 0.65, gaps 0.03 / 0.08, every
third vowel); --high: the 400-800 Hz grid (women's size).
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

FRAME_TIMES = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
RATES = (8000.0, 10000.0, 16000.0)
METHODS = ("cp", *[f"fb@{int(r / 1000)}" for r in RATES], *[f"fbI@{int(r / 1000)}" for r in RATES], "q@8")


def one(job):
    from scipy.signal import resample_poly

    from cp_ref_grid import FS, render_one
    from vocal_analysis.eval import closed_phase as cp
    from vocal_analysis.eval import qcp

    size, f0, oq, gap, vowel = job
    y, area, F, _ = render_one(size, f0, oq, gap, vowel, "D")
    n = len(y)
    out = {}
    x10, e10 = resample_poly(y, 10000, FS), resample_poly(-area, 10000, FS)
    # true closures: the area pulse's open phase spans cycle fraction [0, oq); it closes at oq. Shifted by the
    # acoustic lag from glottis to lips, measured as the reference measures it (`speech_lag`)
    lag = cp.speech_lag(x10, cp.egg_instants(e10, 10000.0)["gci"], 10000.0) / 10000.0
    k = np.arange(int(n * f0 / FS) + 2)
    gci_t = (k + oq) / f0 + lag
    gci_t = gci_t[gci_t * FS < n]
    r = cp.analyse(x10, e10, 10000.0, FRAME_TIMES, order=12)["cp"]
    out["cp"] = r[:, :3]
    sed_hit = np.nan
    for rate in RATES:
        tag = int(rate / 1000)
        a = qcp.analyse(y, FS, FRAME_TIMES, f0_mean=f0, fs=rate)
        out[f"fb@{tag}"] = a["F"][:, :3]
        out[f"fbI@{tag}"] = qcp.analyse(y, FS, FRAME_TIMES, fs=rate, gci=np.round(gci_t * rate).astype(int))["F"][:, :3]
        if tag == 8:
            est = a["gci"] / rate
            sel = (gci_t > 0.05) & (gci_t < n / FS - 0.05)
            sed_hit = float(np.mean([np.min(np.abs(est - g)) <= 0.0005 for g in gci_t[sel]])) if len(est) else 0.0
            out["q@8"] = qcp.analyse(y, FS, FRAME_TIMES, f0_mean=f0, fs=rate, forward_only=True)["F"][:, :3]
    row = [*job, float(F[0]), float(F[1]), float(F[2]), sed_hit]
    for m in METHODS:
        e = out[m] - F[:3]
        ok = np.isfinite(e[:, 0])
        row += [float(ok.mean()), *(np.nanmedian(e, 0) if ok.any() else [np.nan] * 3)]
    return tuple(row)


def columns():
    cols = "size f0 oq gap vowel F1 F2 F3 sedreams_hit".split()
    for m in METHODS:
        cols += [f"{m}_cov", f"{m}_dF1", f"{m}_dF2", f"{m}_dF3"]
    return cols


def stage1(out: Path, mode: str):
    from cp_ref_grid import F0S, GAPS, HIGH_F0S, OQS, SIZES
    from vocal_analysis.synth import tube

    vowels = list(tube.STORY_VOWELS)
    if mode == "smoke":
        jobs = list(itertools.product(["women"], (160.0, 240.0), (0.50, 0.65), (0.03, 0.08), vowels[::3]))
    elif mode == "high":
        jobs = list(itertools.product(["women"], HIGH_F0S, OQS, GAPS, vowels))
    else:
        jobs = list(itertools.product(SIZES, F0S, OQS, GAPS, vowels))
    from multiprocessing import Pool

    t0 = time.time()
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=2)
    print(f"{len(rows)} renders in {time.time() - t0:.0f} s", flush=True)
    name = {"smoke": "d046_stage1_smoke", "high": "d046_stage1_high"}.get(mode, "d046_stage1")
    (out / f"{name}.tsv").write_text("\t".join(columns()) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    summary(rows, out / f"{name}.txt")


def summary(rows, path):
    cols = columns()
    c = {k: i for i, k in enumerate(cols)}
    lines = [f"D-046 stage 1 ({len(rows)} renders): median over renders of the per-render median error (Hz); coverage = mean share of frames",
             f"SEDREAMS identification rate (+-0.5 ms): median {np.nanmedian([r[c['sedreams_hit']] for r in rows]):.3f}, "
             f"min {np.nanmin([r[c['sedreams_hit']] for r in rows]):.3f}", ""]
    for m in METHODS:
        e = np.array([[r[c[f"{m}_dF{k}"]] for k in (1, 2, 3)] for r in rows], float)
        cov = np.mean([r[c[f"{m}_cov"]] for r in rows])
        med = np.nanmedian(e, 0)
        mad = np.nanmedian(np.abs(e), 0)
        lines.append(f"{m:7s} cov {cov:.2f}  F1 {med[0]:+6.1f} (|.| {mad[0]:5.1f})  F2 {med[1]:+6.1f} (|.| {mad[1]:5.1f})  F3 {med[2]:+6.1f} (|.| {mad[2]:5.1f})")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "stage1":
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if "--smoke" in sys.argv else "high" if "--high" in sys.argv else "grid"
    stage1(out, mode)


if __name__ == "__main__":
    main()
