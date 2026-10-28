"""Where do R's order-4 remainder roots fall? Women's region renders, at R's chosen F1 / B1."""
import itertools
import sys

import numpy as np

sys.path.insert(0, "scripts/analysis")


def run(job):
    import cp_ref_grid as g
    import d044_inverse as d
    from scipy.signal import lfilter, resample_poly
    from vocal_analysis.eval import closed_phase as cp
    size, f0, oq, gap, vowel = job
    y, area, F, B = g.render_one(size, f0, oq, gap, vowel, "D")
    x = resample_poly(y, d.FS_CP, g.FS); eg = resample_poly(-area, d.FS_CP, g.FS)
    x_pre = np.r_[x[0], np.diff(x)]
    inst = cp.egg_instants(eg, d.FS_CP); lag = cp.speech_lag(x, inst["gci"], d.FS_CP)
    ref = cp.analyse(x, eg, d.FS_CP, np.array([0.5]), order=12, lag=lag)["cp"][0, 0]
    cyc = d.frame_cycles(inst, lag, 0.5, len(x))
    f1s = ref * 2.0 ** (d.CENTS / 1200)
    sc = np.array([[d.criteria(x_pre, x_pre, cyc, f, b, F[1:4], B[1:4])[0] for b in d.B1S] for f in f1s])
    i, j = np.unravel_index(np.argmin(sc), sc.shape)
    a = d.inverse_poly([f1s[i], *F[1:4]], [d.B1S[j], *B[1:4]])
    segs = [(gg + d.GUARD, o - d.GUARD) for gg, o, _ in cyc if o - d.GUARD > gg + d.GUARD]
    c = cp.covariance_lpc(lfilter(a, [1.0], x_pre), segs, d.NUISANCE)
    roots = np.roots(np.r_[1.0, c])
    rf = np.abs(np.angle(roots)) * d.FS_CP / (2 * np.pi)
    rb = -np.log(np.abs(roots)) * d.FS_CP / np.pi
    cplx = np.imag(roots) > 1e-9
    near = [(f, b) for f, b in zip(rf[cplx], rb[cplx]) if 0.6 * F[0] < f < 1.6 * F[0]]
    return gap, oq, F[0], f1s[i] - F[0], ref - F[0], near, sorted(np.round(rf[cplx]))


if __name__ == "__main__":
    from multiprocessing import Pool
    from vocal_analysis.synth import tube
    jobs = list(itertools.product(["women"], (180.0, 220.0, 270.0), (0.45, 0.57, 0.70), (0.0, 0.05, 0.10), list(tube.STORY_VOWELS)))
    with Pool(10) as p:
        rows = p.map(run, jobs)
    for key, sel in (("gap 0", lambda r: r[0] == 0.0), ("gap 0.05", lambda r: r[0] == 0.05), ("gap 0.10", lambda r: r[0] == 0.10), ("OQ 0.70", lambda r: r[1] == 0.70)):
        s = [r for r in rows if sel(r)]
        has = [r for r in s if r[5]]
        print(f"{key:9s} n={len(s)}  remainder pole within 0.6-1.6 x F1: {len(has)/len(s):.0%};  "
              f"median R err {np.median([r[3] for r in s]):+.1f}, ref err {np.median([r[4] for r in s]):+.1f};  "
              f"with a near pole: R err {np.median([r[3] for r in has]) if has else float('nan'):+.1f}, ref {np.median([r[4] for r in has]) if has else float('nan'):+.1f};  "
              f"without: R {np.median([r[3] for r in s if not r[5]]) if len(has) < len(s) else float('nan'):+.1f}, ref {np.median([r[4] for r in s if not r[5]]) if len(has) < len(s) else float('nan'):+.1f}")
    nb = [b for r in rows if r[0] == 0.10 for _, b in r[5]]
    print("gap 0.10 near-F1 remainder poles: median bandwidth %.0f Hz (n=%d)" % (np.median(nb) if nb else np.nan, len(nb)))
    allf = np.concatenate([r[6] for r in rows]) if rows else []
    print("all remainder pole frequencies, quartiles:", np.percentile(allf, [10, 25, 50, 75, 90]).round(0))
