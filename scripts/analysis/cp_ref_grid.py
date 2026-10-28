"""D-043: the closed-phase reference's own bias on audio with known resonances, over the conditions of real voices.

Extends D-026's render check (two operating points: men f0 110, women f0 210, open quotient 0.5, complete closure)
to a grid: two tract sizes (D-026's men 17.5 cm and women 15 cm shapes) x f0 x open quotient x closed-phase gap,
Story's 10 vowels each, 1 s static. Render: the interactive tube renderer with the load's feedback (D-026's "D"),
Rosenberg area pulse; at gap 0 also without feedback ("C"), which separates the feedback's effect. EGG = the negated
glottal area (contact falls as the area rises), as in D-026, so instants are ideal; the reference method itself
(`eval.closed_phase.analyse`, 10 kHz, order 12, as ptdb_egg.py) is what is tested.

Per render: median over 0.3-0.7 s of the closed-phase F1-F3 minus the closed-glottis poles (Hz), and the share of
frames the method returns (it needs two whole cycles and 3 x order closed-phase samples). Output: one row per
render (tsv) and a summary by condition (median over vowels, vowel bootstrap 95 % interval).

    python scripts/analysis/cp_ref_grid.py <out dir> [--smoke | --high]

--high (D-043 addendum, exploratory): women's size only, f0 330-800 Hz, the f0 limit of the cycle detection raised
to 1,000 Hz, and the method at order 12 (as used) and order 8 (a shorter look-back, 0.8 ms at 10 kHz).
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

FS, FS_CP, ORDER, DUR = 16000, 10000, 12, 1.0
SIZES = {"men": (17.5, 1.0, 1.0), "women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3)}  # L, area scale, pharynx
F0S = (100.0, 140.0, 180.0, 220.0, 270.0, 330.0)
OQS = (0.45, 0.57, 0.70)  # PTDB-TUG dev women's EGG open quotient, 10th / 50th / 90th percentile
GAPS = (0.0, 0.05, 0.10)  # closed-phase area as a fraction of the peak
TIMES = np.arange(0.3, 0.71, 0.01)
HIGH_F0S = (330.0, 400.0, 500.0, 600.0, 700.0, 800.0)
HIGH_F0_RANGE = (50.0, 1000.0)


def render_one(size, f0, oq, gap, vowel, render):
    """One static render at FS: (pressure scaled to peak 0.5, glottal area in cm^2, closed-glottis poles F, B)."""
    from vocal_analysis.synth import interactive as it
    from vocal_analysis.synth import tract, tube
    from vocal_analysis.synth.coupling import Glottis

    L, asc, phar = SIZES[size]
    s = L / 17.5
    g = Glottis(0.12 * s * s, -0.1, 1.4 * s, 0.3 * s, 8000.0, 1 / s)  # as D-026's render check
    n = int(DUR * FS)
    sh = tube.build_shape(*tube.STORY_VOWELS[vowel], L, pharynx_ratio=phar, area_scale=asc)
    F, B = tube.poles(sh, tract.F_MAX_HZ)
    area = g.peak_area_cm2 * it.area_pulse(np.arange(n) * f0 / FS, oq, 1.3, gap)
    nb = n // 16 + 1
    y = it.render(area, [it.tract_responses(sh, FS, sub_scale=1 / s)], np.zeros(nb, int), np.zeros(nb), g,
                  coupled=render == "D")["pressure"]
    return 0.5 * y / np.abs(y).max(), area, F, B


def one(job):
    from scipy.signal import resample_poly

    from vocal_analysis.eval import closed_phase as cp

    size, f0, oq, gap, vowel, render, order = job
    y, area, F, _ = render_one(size, f0, oq, gap, vowel, render)
    kw = {} if f0 <= 500 and order == ORDER else {"f0_range": HIGH_F0_RANGE}
    r = cp.analyse(resample_poly(y, FS_CP, FS), resample_poly(-area, FS_CP, FS), FS_CP, TIMES, order=order, **kw)
    ok = np.isfinite(r["cp"][:, 0])
    err = np.nanmedian(r["cp"][:, :3], 0) - F[:3] if ok.any() else np.full(3, np.nan)
    return (*job, float(ok.mean()), *map(float, err))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from vocal_analysis.synth import tube

    vowels = list(tube.STORY_VOWELS)
    if "--smoke" in sys.argv:
        jobs = [("women", 220.0, 0.57, 0.05, vowels[0], "D", ORDER), ("women", 330.0, 0.70, 0.10, vowels[1], "D", ORDER)]
    elif "--high" in sys.argv:
        jobs = [("women", f0, oq, gp, v, "D", o) for o, f0, oq, gp, v in itertools.product((ORDER, 8), HIGH_F0S, OQS, GAPS, vowels)]
    else:
        jobs = [(sz, f0, oq, gp, v, "D", ORDER) for sz, f0, oq, gp, v in itertools.product(SIZES, F0S, OQS, GAPS, vowels)]
        jobs += [(sz, f0, oq, 0.0, v, "C", ORDER) for sz, f0, oq, v in itertools.product(SIZES, F0S, OQS, vowels)]
    from multiprocessing import Pool

    t0 = time.time()
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    print(f"{len(rows)} renders in {time.time() - t0:.0f} s", flush=True)
    cols = "size f0 oq gap vowel render order coverage dF1 dF2 dF3".split()
    name = "cp_ref_grid_smoke" if "--smoke" in sys.argv else "cp_ref_grid_high" if "--high" in sys.argv else "cp_ref_grid"
    tsv = out / f"{name}.tsv"
    tsv.write_text("\t".join(cols) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    if "--smoke" in sys.argv:
        for r in rows:
            print(r)
        return
    summarise(rows, out / f"{name}.txt")


def summarise(rows, path):
    rng = np.random.default_rng(0)
    lines = ["D-043: closed-phase reference minus closed-glottis poles (Hz), median over 10 vowels [vowel bootstrap 95 %];",
             "coverage = share of frames returned (mean over vowels)", ""]
    keys = sorted({r[:4] + r[5:7] for r in rows}, key=lambda k: (k[5], k[4], k[0], k[3], k[2], k[1]))
    for size, f0, oq, gap, render, order in keys:
        sel = [r for r in rows if r[:4] == (size, f0, oq, gap) and r[5:7] == (render, order)]
        e = np.array([r[8:11] for r in sel], float)
        cov = np.mean([r[7] for r in sel])
        cells = []
        for k in range(3):
            x = e[np.isfinite(e[:, k]), k]
            if len(x) < 3:
                cells.append("   n/a")
                continue
            bs = np.median(x[rng.integers(0, len(x), (2000, len(x)))], 1)
            cells.append(f"{np.median(x):+6.1f} [{np.percentile(bs, 2.5):+5.1f}, {np.percentile(bs, 97.5):+5.1f}]")
        lines.append(f"{render} order {order:2d} {size:5s} f0 {f0:5.0f} OQ {oq:.2f} gap {gap:.2f}  cov {cov:.2f}  F1 {cells[0]}  F2 {cells[1]}  F3 {cells[2]}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
