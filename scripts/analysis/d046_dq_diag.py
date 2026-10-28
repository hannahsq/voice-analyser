import itertools, sys
import numpy as np
sys.path.insert(0, "scripts/analysis")

def run(job):
    from scipy.signal import resample_poly
    from cp_ref_grid import FS, render_one
    from vocal_analysis.eval import closed_phase as cp, qcp
    f0, oq, gap, vowel, render, dq = job
    y, area, F, _ = render_one("women", f0, oq, gap, vowel, render)
    x10, e10 = resample_poly(y, 10000, FS), resample_poly(-area, 10000, FS)
    lag = cp.speech_lag(x10, cp.egg_instants(e10, 10000.0)["gci"], 10000.0) / 10000.0
    k = np.arange(int(len(y) * f0 / FS) + 2); g = (k + oq) / f0 + lag; g = g[g * FS < len(y)]
    old = qcp.DQ
    try:
        a = qcp.analyse(y, FS, np.array([0.3, 0.4, 0.5, 0.6, 0.7]), fs=8000.0, gci=np.round(g * 8000).astype(int)) if dq is None else None
        if dq is not None:
            w = qcp.qcp_weights  # weights with a different duration quotient
            x = resample_poly(y, 1, 2)
            gg = np.round(g * 8000).astype(int)
            ww = w(len(x), gg, 8000.0, dq=dq)
            xp = np.r_[x[0], x[1:] - 0.97 * x[:-1]]
            Fs = []
            for t in (0.3, 0.4, 0.5, 0.6, 0.7):
                c = int(t * 8000); s = slice(c - 120, c + 120)
                f, _ = qcp.lpc_formants(qcp.fb_wlp(xp[s], ww[s], 13), 8000.0); Fs.append(f[:1] if len(f) else [np.nan])
            return (render, dq, float(np.nanmedian(Fs) - F[0]))
    finally:
        qcp.DQ = old
    return (render, dq, float(np.nanmedian(a["F"][:, 0]) - F[0]))

if __name__ == "__main__":
    from multiprocessing import Pool
    from vocal_analysis.synth import tube
    v = list(tube.STORY_VOWELS)[::3]
    jobs = [(f0, oq, gp, vw, r, dq) for f0, oq, gp, vw, r, dq in itertools.product((160.0, 240.0), (0.5, 0.65), (0.0,), v, ("C", "D"), (None, 0.3))]
    with Pool(12) as p:
        rows = p.map(run, jobs)
    for r in ("C", "D"):
        for dq in (None, 0.3):
            print(r, "DQ", 0.7 if dq is None else dq, "F1 median err %+.1f" % np.median([q[2] for q in rows if q[0] == r and q[1] == dq]))
