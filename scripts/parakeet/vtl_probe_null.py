"""D-035, reported after the result (decides nothing): the f0-invariance guard's noise floor for each head. Per clip,
the head's estimate as vtl_score.py; the median |slope| at f0 <= 250 Hz with f0 permuted within each (vowel, length)
group (1000 permutations): what per-clip noise alone gives with no f0 dependence. Also each seed alone.
Args: vtl cache dir, out txt."""
import sys
from pathlib import Path

import numpy as np

sys.argv, ARGS = sys.argv[:1] + [sys.argv[1], "nul.txt"], sys.argv[1:]
import vtl_score as S  # noqa: E402  (runs its scoring; only its helpers are used here)

OUT = open(ARGS[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
rng = np.random.default_rng(1)
low = S.low


def med_abs_slope(e, f0):
    return np.median([abs(1200 * np.polyfit(f0[(S.kid == k) & low], e[(S.kid == k) & low], 1)[0]) for k in range(len(S.keys))])


p("Probe guard noise floor (f0 <= 250 Hz; median |slope|, cents per octave)")
for arm in S.ARMS:
    e = S.probe_est(arm)
    obs = med_abs_slope(e, S.f0)
    nul = []
    for _ in range(1000):
        f = S.f0.copy()
        for k in range(len(S.keys)):
            m = (S.kid == k) & low
            f[m] = rng.permutation(f[m])
        nul.append(med_abs_slope(e, f))
    seeds = [med_abs_slope(S.probe_est(arm, s), S.f0) for s in range(3)]
    resid = np.concatenate([e[(S.kid == k) & low] - np.polyval(np.polyfit(S.f0[(S.kid == k) & low], e[(S.kid == k) & low], 1), S.f0[(S.kid == k) & low]) for k in range(len(S.keys))])
    p(f"  {arm:3s} observed {obs:5.1f}; shuffle null median {np.median(nul):5.1f} [95 % {np.percentile(nul, 95):5.1f}]; "
      f"P(null >= observed) {np.mean(np.array(nul) >= obs):.3f}; seeds {' '.join(f'{x:.1f}' for x in seeds)}; per-clip residual SD {1200 * resid.std():.0f} c")
