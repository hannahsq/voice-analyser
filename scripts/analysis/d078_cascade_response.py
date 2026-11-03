"""Pre-registration check for D-078: on coupled cascade clips, does D-077's response estimator land on the cycle-mean
pole F1 (1 + df <a>)? Steady tube targets drawn as D-075, coupling at scale 1, LF source with D-060's vibrato; the source
is the same with and without coupling, so H_eff = S(src, coupled out) / S(src); rigid = the same poles held still."""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SEED = 5078


def fit(x, src, f1):
    from scipy.optimize import least_squares
    from scipy.signal import coherence, csd, welch

    f, sxy = csd(src, x, 16000, nperseg=4096)
    _, sxx = welch(src, 16000, nperseg=4096)
    _, coh = coherence(src, x, 16000, nperseg=4096)
    k = (f > 0.55 * f1) & (f < 1.6 * f1) & (coh > 0.95) & (sxx > 1e-6 * sxx.max())
    if k.sum() < 15:
        return np.nan
    ff, hh = f[k], np.log(np.abs(sxy / sxx)[k])

    def res(p):
        F, B, g, s = p
        return -0.5 * np.log((ff**2 - F**2) ** 2 + (ff * B) ** 2) + g + s * np.log(ff / f1) - hh

    lo, hi = [0.6 * f1, 10, -np.inf, -5], [1.5 * f1, 600, np.inf, 5]
    p = least_squares(res, [f1, 80, hh.max() + np.log(f1 * 80), 0], bounds=(lo, hi)).x
    return np.nan if any(abs(p[i] - b[i]) < 1e-3 * abs(b[i]) for i in range(2) for b in (lo, hi)) else float(p[0])


def one(i):
    from vocal_analysis.synth import coupling, filters, tract, tube, voice
    from vocal_analysis.synth.glottal import open_phase, source_coefficients

    rng = np.random.default_rng([SEED, i])
    length = rng.uniform(13.0, 18.5)
    spk = tube.sample_speaker(rng, length)
    g = coupling.sample_glottis(rng, length)
    rd = float(np.round(rng.uniform(0.3, 2.7), 2))
    F, B, info = tube.sample_target(rng, spk, tract.N_POLES, tract.F_MAX_HZ, glottis=g)
    df, bm = np.array(info["open_df"]), np.array(info["open_bm"])
    f0 = 220.0 if length < 15.5 else 120.0
    n = voice.FS
    nc = n // voice.CTRL + 1
    t = np.arange(nc) * voice.CTRL / voice.FS
    f0_c = f0 * 2.0 ** (np.sin(2 * np.pi * 5 * t) / 12)
    cycles, _ = voice._glottal_phase(voice._per_sample(f0_c), np.ones(4096), float(voice.FS))
    kmax_ = int(np.ceil(voice.NYQ_TAPER_HZ[1] / f0_c.min()))
    amp = source_coefficients(np.full(nc, rd), kmax_)
    kk = np.minimum(np.ceil(voice.NYQ_TAPER_HZ[1] / f0_c).astype(np.int64), kmax_)
    src = voice._additive(cycles, np.abs(amp), np.unwrap(np.angle(amp), axis=0), kk, voice.CTRL)
    te = open_phase(rd)
    gap = float(coupling.gap_from_open_quotient(te, g.gap_offset))
    a = voice.glottal_opening(cycles, np.full(len(cycles), te), gap)
    P = int(np.sum(np.isfinite(F)))
    pf = np.repeat(F[:P, None], len(src), 1)
    pb = np.repeat(B[:P, None], len(src), 1)
    m = min(coupling.N_COUPLED, P)
    cf, cb = pf.copy(), pb.copy()
    cf[:m] *= 1.0 + df[:m, None] * a
    cb[:m] *= 1.0 + (bm[:m, None] - 1.0) * a
    out_c = filters.filter_tv(src, cf, cb, voice.FS)
    out_r = filters.filter_tv(src, pf, pb, voice.FS)
    fe, fr = fit(out_c, src, F[0]), fit(out_r, src, F[0])
    return length, F[0], df[0], gap, a.mean(), fe - fr, F[0] * df[0] * a.mean(), F[0] * df[0] * gap, fr - F[0]


if __name__ == "__main__":
    sys.path.insert(0, ".")
    with ProcessPoolExecutor(6) as ex:
        r = np.array(list(ex.map(one, range(int(sys.argv[1]) if len(sys.argv) > 1 else 60))))
    L, F1, df, gap, am, dE, cm, cl, rig = r.T
    ok = np.isfinite(dE)
    print(f"{ok.sum()} of {len(r)} fitted; rigid check (F_rig - label) median {np.nanmedian(rig):+.1f} Hz")
    for name, x in (("dE - CM", dE - cm), ("dE - CL", dE - cl)):
        print(f"  {name}: median {np.nanmedian(x):+.1f}, IQR {np.nanpercentile(x, 25):+.1f} / {np.nanpercentile(x, 75):+.1f} Hz")
    print(f"  dE median {np.nanmedian(dE):+.1f}, CM median {np.median(cm[ok]):+.1f}, CL median {np.median(cl[ok]):+.1f}")
    print(f"  slope of dE on CM (through 0): {np.nansum(dE * cm) / np.sum(cm[ok] ** 2):.2f}; corr {np.corrcoef(dE[ok], cm[ok])[0, 1]:+.2f}")
    print(f"  |closed - CM| median {np.median(np.abs(cm)):.1f} Hz (the label gap D-078 has to resolve)")
