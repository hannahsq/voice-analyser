"""D-077: KL's whole-cycle acoustic response against the cycle-mean pole (D-074's CM) and teacher v3.1. Per D-060 file:
the coupled render, the same glottal area uncoupled (no acoustic load), and the uncoupled flow through a rigid glottis.
F_eff: F1 of S(U_g uncoupled, out coupled) / S(U_g uncoupled); F_rig: the same fit on the rigid render (the estimator
check); dE = F_eff - F_rig.

    python scripts/analysis/d077_kl_response.py [--serial]
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("reports/analysis/d077")
CACHE = Path("cache/d077")
SEED = 5077


def fit_f1(x, ug, f1):
    """F1 and B1 of the resonance fitted to log |S(ug, x) / S(ug)| / f, and the bins used; NaN if under 15 or at a bound"""
    from scipy.optimize import least_squares
    from scipy.signal import coherence, csd, welch

    from vocal_analysis.synth import kl

    f, sxy = csd(ug, x, kl.FS, nperseg=4096)
    _, sxx = welch(ug, kl.FS, nperseg=4096)
    _, coh = coherence(ug, x, kl.FS, nperseg=4096)
    H = np.abs(sxy / sxx) / np.maximum(f, 1)
    k = (f > 0.55 * f1) & (f < 1.6 * f1) & (coh > 0.95) & (sxx > 1e-6 * sxx.max())
    if k.sum() < 15:
        return np.nan, np.nan, int(k.sum())
    ff, hh = f[k], np.log(H[k])

    def res(p):
        F, B, g, s = p
        return -0.5 * np.log((ff**2 - F**2) ** 2 + (ff * B) ** 2) + g + s * np.log(ff / f1) - hh

    lo, hi = [0.6 * f1, 10, -np.inf, -5], [1.5 * f1, 600, np.inf, 5]
    p = least_squares(res, [f1, 80, hh.max() + np.log(f1 * 80), 0], bounds=(lo, hi)).x
    at_bound = any(abs(p[i] - b[i]) < 1e-3 * abs(b[i]) for i in range(2) for b in (lo, hi))
    return (np.nan, np.nan, int(k.sum())) if at_bound else (float(p[0]), float(p[1]), int(k.sum()))


def one(job):
    sex, v, f0, gap = job
    from vocal_analysis.synth import branches as br
    from vocal_analysis.synth import kl, tube

    import d060_kl_gap as D

    tag = "".join(c + "^" if c.isupper() else c for c in v)
    path = CACHE / f"{sex}_{tag}_{f0}_{gap:.1f}.npy"
    if path.exists():
        return job, tuple(np.load(path))
    L, peak = D.SIZE[sex]
    rel = L / tube.STORY_LENGTH_CM
    s = tube.build_shape(*tube.STORY_VOWELS[v], L, area_scale=rel**2)
    (F, _, _), = kl.frame_labels([s])
    n = kl.FS
    frames = n // kl.HOP + 1
    t = np.arange(frames) * kl.HOP / kl.FS
    f0_t = f0 * 2.0 ** (D.VIB_ST * np.sin(2 * np.pi * D.VIB_HZ * t) / 12)
    ag = kl.glottal_area(f0_t, n, kl.FS, peak, oq=0.6, sq=1.3, gap=gap)
    sub = br.subglottal(L, kl.section_length())
    kw = dict(subglottal=sub)
    c = kl.render([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel, **kw)
    u = kl.render([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel, coupled=False, **kw)
    ugu = u["ug"].astype(float)
    r = kl.render([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel, flow=ugu, **kw)
    fe, be, ne = fit_f1(c["audio"].astype(float), ugu, F[0])
    fr, brr, nr = fit_f1(r["audio"].astype(float), ugu, F[0])
    res = (F[0], fe, be, ne, fr, brr, nr)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(res, float))
    return job, res


def boot(x, rng, n=2000):
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.median(x)), *np.percentile(b, [2.5, 97.5])


def main():
    import d060_kl_gap as D
    import d074_gap_candidates as C

    jobs = [(sex, v, f0, gap) for sex in D.SIZE for v in D.MAIN["vowels"] for f0 in D.MAIN["f0"][sex] for gap in D.GAPS]
    if "--serial" in sys.argv:
        got = [one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(6) as ex:
            got = list(ex.map(one, jobs, chunksize=4))
    R = dict(got)
    rows = [ln.split("\t") for ln in C.TSV.read_text().splitlines()[1:]]
    err = {(r[0], float(r[1]), r[2], int(r[3])): (float(r[4]) + float(r[5])) / 2 for r in rows}
    cand, _ = C.candidates()
    dE = {j: R[j][1] - R[j][4] for j in jobs}
    bad = [j for j in jobs if not np.isfinite(dE[j])]
    rng = np.random.default_rng(SEED)
    L = ["D-077: KL's whole-cycle response (uncoupled flow -> coupled output) against the cycle-mean pole and v3.1", ""]
    L.append(f"files left out (under 15 bins or a fit at its bound): {len(bad)} of {len(jobs)}"
             + (": " + ", ".join(f"{s[0]}{v}{f}/{g}" for s, v, f, g in bad[:20]) if bad else ""))
    L.append("estimator check, F_rig - label, median by sex: "
             + ", ".join(f"{sx} {np.nanmedian([R[j][4] - R[j][0] for j in jobs if j[0] == sx]):+.1f} Hz" for sx in D.SIZE))
    L.append("")
    L.append("per vowel, median dE / CM / CL / teacher error, Hz, gaps 0 / 0.3")
    for sx in D.SIZE:
        for v in D.MAIN["vowels"]:
            cells = []
            for g in (0.0, 0.3):
                de = np.nanmedian([dE[sx, v, f, g] for f in D.MAIN["f0"][sx]])
                te = np.median([err[sx, g, v, f] for f in D.MAIN["f0"][sx]])
                cells.append(f"{de:+6.1f} / {cand[sx, v, g][1]:+5.1f} / {cand[sx, v, g][0]:+5.1f} / {te:+6.1f}")
            L.append(f"  {sx:6s} {v:3s} {cells[0]}   |   {cells[1]}")
    L.append("")
    verdict = {}

    def growth(sx, f):
        pairs = [(v, f0) for v in D.MAIN["vowels"] for f0 in D.MAIN["f0"][sx]]
        x = np.array([f(sx, v, f0, 0.3) - f(sx, v, f0, 0.0) for v, f0 in pairs])
        return x[np.isfinite(x)]

    def level(sx, f):
        x = np.array([f(sx, v, f0, g) for v in D.MAIN["vowels"] for f0 in D.MAIN["f0"][sx] for g in D.GAPS])
        return x[np.isfinite(x)]

    tests = {
        "(1) dE - CM": lambda sx, v, f0, g: dE[sx, v, f0, g] - cand[sx, v, g][1],
        "    dE - CL (reported)": lambda sx, v, f0, g: dE[sx, v, f0, g] - cand[sx, v, g][0],
        "(2) teacher - dE": lambda sx, v, f0, g: err[sx, g, v, f0] - dE[sx, v, f0, g],
    }
    for name, f in tests.items():
        L.append(name)
        for sx in D.SIZE:
            m, lo, hi = boot(growth(sx, f), rng)
            lv = float(np.median(level(sx, f)))
            g_ok = lo <= 0 <= hi and abs(m) <= 5
            verdict[name, sx] = (g_ok, abs(lv) <= 10)
            byg = " / ".join(f"{np.median([f(sx, v, f0, g) for v in D.MAIN['vowels'] for f0 in D.MAIN['f0'][sx] if np.isfinite(f(sx, v, f0, g))]):+.1f}"
                             for g in D.GAPS)
            L.append(f"  {sx:6s} growth residual {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] -> {'within' if g_ok else 'outside'}; "
                     f"level median {lv:+6.1f} (by gap {byg})")
        L.append("")
    L.append("reported: fitted B1 of H_eff, median by gap: " + "; ".join(
        f"{sx} " + " / ".join(f"{np.nanmedian([R[j][2] for j in jobs if j[0] == sx and j[3] == g]):.0f}" for g in D.GAPS)
        for sx in D.SIZE) + " Hz (rigid: " + "; ".join(f"{sx} {np.nanmedian([R[j][5] for j in jobs if j[0] == sx]):.0f}" for sx in D.SIZE) + ")")
    rk = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    for sx in D.SIZE:
        tg = [np.mean([err[sx, 0.3, v, f] - err[sx, 0.0, v, f] for f in D.MAIN["f0"][sx]]) for v in D.MAIN["vowels"]]
        dg = [np.nanmean([dE[sx, v, f, 0.3] - dE[sx, v, f, 0.0] for f in D.MAIN["f0"][sx]]) for v in D.MAIN["vowels"]]
        cg = [cand[sx, v, 0.3][1] - cand[sx, v, 0.0][1] for v in D.MAIN["vowels"]]
        L.append(f"reported: Spearman over vowels, {sx}: teacher growth vs dE growth {np.corrcoef(rk(tg), rk(dg))[0, 1]:+.2f}; "
                 f"dE growth vs CM growth {np.corrcoef(rk(dg), rk(cg))[0, 1]:+.2f}")
    L.append("")
    c1 = all(verdict["(1) dE - CM", sx][0] and verdict["(1) dE - CM", sx][1] for sx in D.SIZE)
    c2 = all(verdict["(2) teacher - dE", sx][0] for sx in D.SIZE)
    sign = np.median(np.concatenate([growth(sx, tests["(2) teacher - dE"]) for sx in D.SIZE]))
    L.append(f"DECISION: (1) CM {'predicts' if c1 else 'does not predict'} the response; "
             f"(2) v3.1 {'follows the response' if c2 else 'does not follow the response (' + ('over' if sign > 0 else 'under') + ')'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d077.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
