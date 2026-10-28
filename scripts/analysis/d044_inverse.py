"""D-044: label-free F1 checks by inverse filtering. R, the closed-phase ripple (a robustness check of the closed-phase
reference, which minimises the same quantity); J, a joint source-filter fit (an LF pulse fitted to the inverse-filtered
flow derivative over the whole cycle).

For each frame (the cycles wholly within +-20 ms), the voice is inverse-filtered with candidate F1 / B1 and F2-F4 held
fixed: F1 on a grid of +-300 cents around the reference's F1 in 10-cent steps, B1 in B1S. Each criterion's F1 is its
optimum over both. At 10 kHz, as the reference.
  R: e = pre-emphasised speech through the inverse filter (monic, so candidates are comparable); the closed phases'
     (GCI + guard to GOI - guard) energy, summed over the frame: the reference's own least-squares criterion.
  J: e = speech (no pre-emphasis: the pressure is already the derivative of the lip flow) through the inverse filter,
     high-passed at 40 Hz: the flow derivative. Per cycle, LF flow derivatives at RDS, their main excitation at the
     GCI +- 0.3 ms, amplitude by least squares; the best fit's residual energy, summed over cycles (absolute, as
     ARX-LF's least squares). A ratio to the frame's energy (as first registered) is minimised by inflating the
     denominator; see D-044's amendment.

Stage 1 (`stage1 <out dir> [--smoke]`): D-043's renders (render D, both sizes, f0 100-330 Hz, every open quotient and
gap, 10 vowels), true F2-F4 and bandwidths, instants from the glottal area; also F2-F4 moved by +3 % and -3 %. Five
frames per render (0.3-0.7 s), median over frames. Writes d044_stage1.tsv and d044_stage1.txt (the registered gate).
Stage 2 (real voices) is added only if stage 1 allows it.
"""

from __future__ import annotations

import itertools
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

FS_CP = 10000
GUARD = 3  # samples at 10 kHz (0.3 ms), as the reference
CENTS = np.arange(-300, 301, 10)
B1S = (40.0, 60.0, 90.0, 130.0, 200.0, 300.0)
RDS = np.linspace(0.3, 2.7, 25)
SHIFTS = np.arange(-3, 4)  # samples: +-0.3 ms in 0.1 ms steps
HALF_S = 0.02
FRAME_TIMES = (0.3, 0.4, 0.5, 0.6, 0.7)
PERTURB = (0.0, 0.03, -0.03)
NUISANCE = 4  # order of the free remainder filter (D-044 amendment)
REF_LEAKY = {"gap 0.10": 8.2, "OQ 0.70": 12.3}  # D-043: the reference's women's-region F1 error there (registered)


def inverse_poly(freqs, bws, fs=FS_CP) -> np.ndarray:
    """A(z) of an all-pole filter with these resonances: the FIR that undoes it."""
    a = np.array([1.0])
    for f, b in zip(freqs, bws):
        r = np.exp(-np.pi * b / fs)
        a = np.convolve(a, [1.0, -2 * r * np.cos(2 * np.pi * f / fs), r * r])
    return a


@lru_cache(maxsize=None)
def lf_bank(n: int) -> np.ndarray:
    """(len(RDS) * len(SHIFTS), n): LF flow derivatives one period long, rolled so the main excitation (te) falls at
    sample 0 + shift (a cycle runs GCI to GCI), each normalised to unit energy."""
    from vocal_analysis.synth.glottal import lf_params, lf_waveform

    rows = []
    for rd in RDS:
        w = lf_waveform(float(rd), n)
        te = int(round(lf_params(float(rd)).te * n))
        for s in SHIFTS:
            v = np.roll(w, -te + s)
            rows.append(v / np.linalg.norm(v))
    return np.array(rows)


def criteria(x_pre, x_raw, cycles, f1, b1, rest_f, rest_b):
    """(R, J) for one candidate. cycles: (gci, goi, next_gci) sample triples at 10 kHz.

    A nuisance filter C(z), monic of order NUISANCE, is fitted by covariance least squares to the closed phases of the
    candidate-filtered pre-emphasised speech: it absorbs the source's tilt and the resonances above F4 (which the
    reference's order 12 has room for), so F1 is not pulled to stand in for them. Both criteria use A(z) C(z)."""
    from scipy.signal import lfilter

    from vocal_analysis.eval.closed_phase import covariance_lpc

    a = inverse_poly([f1, *rest_f], [b1, *rest_b])
    segs = [(g + GUARD, o - GUARD) for g, o, _ in cycles if o - GUARD > g + GUARD]
    e = []
    for x in (x_pre, x_raw):  # each signal gets its own remainder (pre-emphasis changes the tilt it must absorb)
        c = covariance_lpc(lfilter(a, [1.0], x), segs, NUISANCE)
        e.append(lfilter(np.convolve(a, np.r_[1.0, c]) if c is not None else a, [1.0], x))
    e_r, e_j = e
    closed = resid = 0.0
    for g, o, nx in cycles:
        if o - GUARD > g + GUARD:
            seg = e_r[g + GUARD:o - GUARD]
            closed += seg @ seg
        v = e_j[g:nx]
        resid += v @ v - np.max((lf_bank(nx - g) @ v) ** 2)  # unit-energy templates: what the best one explains
    return closed, resid


def frame_cycles(inst, lag, t, n):
    g, o, nx = inst["gci"] + lag, inst["goi"] + lag, inst["next_gci"] + lag
    c, half = t * FS_CP, HALF_S * FS_CP
    sel = (g >= c - half) & (nx <= c + half) & (g >= 16) & (nx <= n)
    return list(zip(g[sel], o[sel], nx[sel]))


def best_f1(x_pre, x_raw, cycles, centre, rest_f, rest_b):
    """Each criterion's F1: the minimum over the F1 x B1 grid."""
    f1s = centre * 2.0 ** (CENTS / 1200)
    scores = np.array([[criteria(x_pre, x_raw, cycles, f, b, rest_f, rest_b) for b in B1S] for f in f1s])  # (F, B, 2)
    out = []
    for k in range(2):
        i, _ = np.unravel_index(np.argmin(scores[..., k]), scores.shape[:2])
        out.append(f1s[i])
    return out


def stage1_one(job):
    from scipy.signal import butter, resample_poly, sosfiltfilt

    from cp_ref_grid import FS, render_one
    from vocal_analysis.eval import closed_phase as cp

    size, f0, oq, gap, vowel, perturb = job
    y, area, F, B = render_one(size, f0, oq, gap, vowel, "D")
    x = resample_poly(y, FS_CP, FS)
    eg = resample_poly(-area, FS_CP, FS)
    x_pre = np.r_[x[0], np.diff(x)]
    x_raw = sosfiltfilt(butter(2, 40.0, "highpass", fs=FS_CP, output="sos"), x)
    inst = cp.egg_instants(eg, FS_CP)
    lag = cp.speech_lag(x, inst["gci"], FS_CP)
    ref = cp.analyse(x, eg, FS_CP, np.array(FRAME_TIMES), order=12, lag=lag)["cp"][:, 0]
    rest_f, rest_b = F[1:4] * (1 + perturb), B[1:4]
    rows = []
    for t, r in zip(FRAME_TIMES, ref):
        cyc = frame_cycles(inst, lag, t, len(x))
        if not np.isfinite(r) or len(cyc) < 2:
            continue
        rows.append((r, *best_f1(x_pre, x_raw, cyc, r, rest_f, rest_b)))
    if not rows:
        return (*job, np.nan, np.nan, np.nan, np.nan)
    med = np.median(np.array(rows), 0) - F[0]
    return (*job, float(F[0]), *map(float, med))


def stage1(out: Path, smoke: bool):
    from cp_ref_grid import F0S, GAPS, OQS, SIZES
    from vocal_analysis.synth import tube

    vowels = list(tube.STORY_VOWELS)
    if smoke:  # development renders, off the gate's grid (D-044 amendment)
        jobs = list(itertools.product(["women"], (160.0, 240.0), (0.50, 0.65), (0.03, 0.08), vowels[::3], [0.0]))
    else:
        jobs = list(itertools.product(SIZES, F0S, OQS, GAPS, vowels, PERTURB))
    from multiprocessing import Pool

    t0 = time.time()
    with Pool(12) as pool:
        rows = pool.map(stage1_one, jobs, chunksize=2)
    print(f"{len(rows)} renders in {time.time() - t0:.0f} s", flush=True)
    cols = "size f0 oq gap vowel perturb F1_true dRef dR dJ".split()
    name = "d044_stage1_smoke" if smoke else "d044_stage1"
    (out / f"{name}.tsv").write_text("\t".join(cols) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    if smoke:
        for r in rows:
            print(r)
        return
    gate(rows, out / f"{name}.txt")


def pooled(rows, k, rng):
    """Median of column k over rows, with a vowel bootstrap (vowels resampled, all their rows kept)."""
    vow = sorted({r[4] for r in rows})
    by = {v: np.array([r[k] for r in rows if r[4] == v], float) for v in vow}
    x = np.concatenate(list(by.values()))
    x = x[np.isfinite(x)]
    bs = []
    for _ in range(2000):
        z = np.concatenate([by[v] for v in rng.choice(vow, len(vow))])
        bs.append(np.nanmedian(z))
    return float(np.median(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def gate(rows, path):
    rng = np.random.default_rng(0)
    COL = {"reference": 7, "R": 8, "J": 9}
    lines = ["D-044 stage 1: F1 error (Hz) against the closed-glottis pole, median over 5 frames per render;",
             "pooled medians with a vowel bootstrap (95 %). Render D, instants from the area, true F2-F4 unless noted.", ""]
    fmt = lambda m: f"{m[0]:+6.1f} [{m[1]:+5.1f}, {m[2]:+5.1f}] n={m[3]}"  # noqa: E731
    base = [r for r in rows if r[5] == 0.0]
    W = [r for r in base if r[0] == "women" and r[1] in (180.0, 220.0, 270.0)]
    res = {k: pooled(W, c, rng) for k, c in COL.items()}
    for k in COL:
        lines.append(f"women's region      {k:9s} {fmt(res[k])}")
    leaky = {}
    for label, sel in (("gap 0.10", [r for r in W if r[3] == 0.10]), ("OQ 0.70", [r for r in W if r[2] == 0.70])):
        leaky[label] = {k: pooled(sel, c, rng) for k, c in COL.items()}
        for k in COL:
            lines.append(f"{label:19s} {k:9s} {fmt(leaky[label][k])}")
    for p in PERTURB[1:]:
        sel = [r for r in rows if r[5] == p and r[0] == "women" and r[1] in (180.0, 220.0, 270.0)]
        for k in ("R", "J"):
            lines.append(f"F2-F4 x{1 + p:.2f}         {k:9s} {fmt(pooled(sel, COL[k], rng))}")
    M = [r for r in base if r[0] == "men" and r[1] in (100.0, 140.0)]
    for k in COL:
        lines.append(f"men's region        {k:9s} {fmt(pooled(M, COL[k], rng))}")
    cells = sorted({r[:4] for r in base})
    cm = lambda c, k: np.nanmedian([r[k] for r in base if r[:4] == c])  # noqa: E731
    ref_c = np.array([cm(c, 7) for c in cells])
    for k in ("R", "J"):
        v = np.array([cm(c, COL[k]) for c in cells])
        ok = np.isfinite(v) & np.isfinite(ref_c)
        lines.append(f"correlation over {ok.sum()} cells of {k}'s error with the reference's: {np.corrcoef(v[ok], ref_c[ok])[0, 1]:+.2f}")
    lines.append("")
    r_pass = abs(res["R"][0]) <= 4
    j_unbiased = abs(res["J"][0]) <= 4
    j_better = all(abs(leaky[l]["J"][0]) <= REF_LEAKY[l] - 4 for l in leaky)
    lines.append(f"GATE R: {'passes' if r_pass else 'fails'} (women's region |median| <= 4 Hz)")
    lines.append("GATE J: " + ("passes as an arbiter" if j_unbiased and j_better else
                                "unbiased but not independent of the reference where it matters (stage 2 reported only)"
                                if j_unbiased else "fails (women's region |median| > 4 Hz): stops, reported"))
    lines.append("(J is compared with D-043's registered values, +8.2 at gap 0.10 and +12.3 at OQ 0.70; the reference"
                 " column here is recomputed on 5 frames per render, reported beside them.)")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "stage1":
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stage1(out, "--smoke" in sys.argv)


if __name__ == "__main__":
    main()
