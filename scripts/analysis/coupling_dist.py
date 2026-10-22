"""How strong is train_v2c's coupling per clip? Samples tube targets with a glottis as the sampler does (train_v2c's
glottal settings), and reports the quasi-static open-glottis F1 shift and B1 ratio at the peak opening, the same after
`coupling_scale` (uniform 0-1), and the cycle-mean opening a(t) at the clip's Rd (closed-phase gap included).
Effective whole-cycle shift ~ df1 * scale * mean(a). Grouped by tract length (<15.5 cm ~ women / children, >=15.5 men)
and by closed-glottis F1 (<350 Hz close vowels)."""
import sys

import numpy as np

from vocal_analysis.synth import coupling, tract, tube
from vocal_analysis.synth.glottal import open_phase

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
rng = np.random.default_rng(7)
rows = []
for i in range(N):
    length = rng.uniform(13.0, 18.5)
    spk = tube.sample_speaker(rng, length)
    g = coupling.sample_glottis(rng, length)
    scale = rng.uniform(0.0, 1.0)
    rd = rng.uniform(0.3, 2.7)
    f, b, info = tube.sample_target(rng, spk, tract.N_POLES, tract.F_MAX_HZ, glottis=g)
    te = open_phase(round(rd, 2))
    gap = float(coupling.gap_from_open_quotient(te, g.gap_offset))
    frac = np.linspace(0, 1, 2000, endpoint=False)
    a = gap + (1 - gap) * np.where(frac < te, np.sin(np.pi * frac / te) ** 2, 0.0)
    rows.append((length, f[0], b[0], info["open_df"][0], info["open_bm"][0], scale, a.mean(), gap, g.peak_area_cm2, rd))
r = np.array(rows)
L, F1, B1, df, bm, sc, am, gap, area, rd = r.T
eff = df * sc * am * F1  # Hz, rough whole-cycle F1 shift
effb = (bm - 1) * sc * am * B1  # Hz, rough whole-cycle B1 increase
eff1 = df * am * F1  # same at scale 1
effb1 = (bm - 1) * am * B1


def q(x):
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:7.1f}" for v in np.percentile(x, [10, 25, 50, 75, 90]))


print(f"{N} tube targets with a glottis (train_v2c settings). Percentiles 10 / 25 / 50 / 75 / 90")
for name, k in [("short tract <15.5 cm", L < 15.5), ("long tract >=15.5 cm", L >= 15.5), ("F1 < 350 Hz", F1 < 350),
                ("F1 >= 350 Hz", F1 >= 350), ("all", np.ones_like(L, bool))]:
    print(f"\n{name} (n={k.sum()}): peak area {np.median(area[k]):.3f} cm2, gap {np.median(gap[k]):.2f}, mean a(t) {np.median(am[k]):.2f}")
    print(f"  F1 shift at peak opening, %        {q(100 * df[k])}")
    print(f"  B1 ratio at peak opening           {q(bm[k])}")
    print(f"  cycle-mean F1 shift, Hz, scale 1   {q(eff1[k])}")
    print(f"  cycle-mean F1 shift, Hz, as v2c    {q(eff[k])}")
    print(f"  cycle-mean B1 increase, Hz, scale 1 {q(effb1[k])}")
    print(f"  cycle-mean B1 increase, Hz, as v2c  {q(effb[k])}")
