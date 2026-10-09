"""Closed-phase formant analysis with an EGG reference (D-019).

While the glottis is closed, the vocal tract is cut off from the trachea and receives no excitation, so the
speech signal is the free decay of the tract's resonances: an all-pole recursion that covariance linear
prediction fits exactly. Formants from the closed phases alone are therefore free of the source spectrum and of
the subglottal coupling that raises and broadens F1 while the glottis is open. The EGG (electroglottograph)
marks the phases: contact rises sharply at glottal closure (a peak of the differentiated EGG, dEGG) and falls at
opening (a smaller dEGG trough). Closed-phase covariance LPC is the classic reference method of glottal inverse
filtering (Wong, Markel & Gray 1979; Alku 2011, review).

Functions:
    egg_instants(egg, fs)                    glottal closure and opening instants from the EGG
    speech_lag(speech, gci, fs)              acoustic delay of the speech behind the EGG
    covariance_lpc(x, segments, order)       one predictor fitted jointly over several segments
    lpc_formants(a, fs)                      resonance frequencies and bandwidths from a predictor
    analyse(speech, egg, fs, times)          closed-phase and whole-cycle formants at given times

All sample indices refer to the arrays passed in; `fs` is their common sampling rate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, find_peaks, lfilter, sosfiltfilt

F0_RANGE = (50.0, 500.0)


def egg_instants(egg: np.ndarray, fs: float, f0_range: tuple[float, float] = F0_RANGE,
                 min_open_strength: float = 0.15) -> dict:
    """Glottal closure (GCI) and opening (GOI) instants from the EGG, cycle by cycle.

    dEGG is the derivative of the EGG band-limited to 20 Hz–3 kHz. Its sign is set so that closures are
    positive peaks: the polarity whose largest excursions are sharper (higher 99.9th percentile). GCIs are dEGG
    peaks above 15 % of that percentile, at least one shortest period apart. A cycle runs from one GCI to the
    next and is kept if its period lies in `f0_range` and differs from its neighbour's by less than 30 %. Its GOI
    is the deepest dEGG trough between 10 % and 95 % of the period, kept if at least `min_open_strength` times
    the closing peak (otherwise the cycle has no reliable opening and is dropped).

    Returns gci (int array of cycle starts), goi (int, same length), next_gci (int, cycle ends), polarity (+1 if
    the raw EGG rises at closure).
    """
    sos = butter(2, [20.0, min(3000.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
    d = np.gradient(sosfiltfilt(sos, np.asarray(egg, float)))
    hi, lo = np.percentile(d, 99.9), -np.percentile(d, 0.1)
    polarity = 1 if hi >= lo else -1
    d = d * polarity
    scale = max(hi, lo)
    peaks, _ = find_peaks(d, height=0.15 * scale, distance=max(1, int(fs / f0_range[1])))
    tmin, tmax = fs / f0_range[1], fs / f0_range[0]
    gci, goi, nxt = [], [], []
    for k in range(len(peaks) - 1):
        a, b = peaks[k], peaks[k + 1]
        T = b - a
        if not tmin <= T <= tmax:
            continue
        neighbours = [peaks[j + 1] - peaks[j] for j in (k - 1, k + 1) if 0 <= j < len(peaks) - 1]
        if not any(abs(Tn / T - 1) < 0.3 for Tn in neighbours):
            continue
        lo_i, hi_i = a + int(0.1 * T), a + int(0.95 * T)
        if hi_i <= lo_i:
            continue
        o = lo_i + int(np.argmin(d[lo_i:hi_i]))
        if -d[o] < min_open_strength * d[a]:
            continue
        gci.append(a)
        goi.append(o)
        nxt.append(b)
    return {"gci": np.array(gci, int), "goi": np.array(goi, int), "next_gci": np.array(nxt, int), "polarity": polarity}


def _lpc_residual(x: np.ndarray, fs: float, order: int, frame_s: float = 0.025) -> np.ndarray:
    """Autocorrelation-LPC residual, frame by frame (Hann windows, half overlap), overlap-added."""
    n = int(frame_s * fs)
    hop = n // 2
    win = np.hanning(n)
    out = np.zeros(len(x) + n)
    for s in range(0, max(1, len(x) - n), hop):
        seg = x[s : s + n] * win
        r = np.correlate(seg, seg, "full")[n - 1 : n + order]
        if r[0] <= 0:
            continue
        col = r[:-1].copy()
        col[0] *= 1 + 1e-9  # white-noise correction
        a = solve_toeplitz(col, -r[1:])
        e = lfilter(np.r_[1.0, a], [1.0], x[s : s + n])
        out[s : s + n] += e * win
    return out[: len(x)]


def speech_lag(speech: np.ndarray, gci: np.ndarray, fs: float, max_ms: tuple[float, float] = (-1.0, 3.0),
               order: int | None = None) -> int:
    """Delay (samples) of the speech behind the EGG: the shift that puts the most LPC-residual energy at the GCIs.

    The excitation at each closure is a sharp peak of the speech's LPC residual, reaching the microphone after
    the sound travels from the glottis (about 0.5–1 ms for a close microphone). Residual magnitude is summed over
    GCIs for each shift in `max_ms`, smoothed over ±0.1 ms, and the best shift returned (positive: speech late).
    """
    if len(gci) == 0:
        return 0
    order = order or int(fs / 1000) + 2
    e = np.abs(_lpc_residual(np.asarray(speech, float), fs, order))
    shifts = np.arange(int(max_ms[0] * fs / 1000), int(max_ms[1] * fs / 1000) + 1)
    ok = lambda i: (i >= 0) & (i < len(e))  # noqa: E731
    score = np.array([e[gci[ok(gci + s)] + s].sum() for s in shifts])
    w = max(1, int(0.0001 * fs))
    score = np.convolve(score, np.ones(2 * w + 1) / (2 * w + 1), "same")
    return int(shifts[int(np.argmax(score))])


def covariance_lpc(x: np.ndarray, segments: list[tuple[int, int]], order: int) -> np.ndarray | None:
    """Predictor a (x[n] ≈ -Σ a_k x[n-k]) minimising the squared error summed over the samples of all segments
    [start, end) (covariance method; samples before a segment's start may be used as predictors). None if the
    segments hold fewer than 2·order samples."""
    rows = []
    for a, b in segments:
        a = max(a, order)
        if b > a:
            idx = np.arange(a, b)
            rows.append(x[idx[:, None] - np.arange(order + 1)[None, :]])
    if not rows:
        return None
    X = np.concatenate(rows)
    if len(X) < 2 * order:
        return None
    R = X.T @ X
    try:
        return np.linalg.solve(R[1:, 1:] + 1e-12 * np.trace(R) * np.eye(order), -R[1:, 0])
    except np.linalg.LinAlgError:
        return None


def lpc_formants(a: np.ndarray, fs: float, f_min: float = 90.0, max_bw: float = 600.0) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, bandwidths) in Hz of the predictor's complex pole pairs, ascending, keeping poles above
    f_min and below fs/2 - 50 Hz with bandwidth under max_bw (wider ones shape the spectrum's tilt, not formants)."""
    r = np.roots(np.r_[1.0, a])
    r = r[np.imag(r) > 0]
    f = np.angle(r) * fs / (2 * np.pi)
    bw = -np.log(np.maximum(np.abs(r), 1e-12)) * fs / np.pi
    keep = (f > f_min) & (f < fs / 2 - 50) & (bw < max_bw)
    o = np.argsort(f[keep])
    return f[keep][o], bw[keep][o]


def analyse(speech: np.ndarray, egg: np.ndarray, fs: float, times: np.ndarray, order: int = 12,
            window_s: float = 0.04, guard_s: float = 0.0003, lag: int | None = None, n_formants: int = 4) -> dict:
    """Closed-phase and whole-cycle covariance-LPC formants at `times` (seconds).

    For each time, the cycles lying wholly within ±window_s/2 are used: their closed phases, from GCI + guard to
    GOI - guard and shifted by the speech lag, form the closed-phase fit; the whole cycles, GCI to next GCI, the
    control fit with the same estimator. Speech is pre-emphasised by a first difference (lip radiation), as usual
    for LPC formant analysis. A frame needs at least two cycles and 3·order closed-phase samples; otherwise NaN.

    Returns arrays (T, n_formants) "cp" and "cycle" (Hz) with their bandwidths "cp_bw" and "cycle_bw", plus f0 (from
    the cycles' periods), closed quotient "cq" and "n_cycles" per time, and the lag used.
    """
    x = np.asarray(speech, float)
    x = np.r_[x[0], np.diff(x)]
    inst = egg_instants(egg, fs)
    lag = speech_lag(speech, inst["gci"], fs) if lag is None else lag
    g, o, nx = inst["gci"] + lag, inst["goi"] + lag, inst["next_gci"] + lag
    guard = int(round(guard_s * fs))
    T = len(times)
    res = {k: np.full((T, n_formants), np.nan) for k in ("cp", "cp_bw", "cycle", "cycle_bw")}
    res.update(f0=np.full(T, np.nan), cq=np.full(T, np.nan), n_cycles=np.zeros(T, int), lag=lag)
    half = window_s / 2 * fs
    for i, t in enumerate(times):
        c = t * fs
        sel = (g >= c - half) & (nx <= c + half) & (g >= order)
        if sel.sum() < 2:
            continue
        cp_seg = [(a + guard, b - guard) for a, b in zip(g[sel], o[sel]) if b - guard > a + guard]
        if sum(b - a for a, b in cp_seg) < 3 * order:
            continue
        res["n_cycles"][i] = sel.sum()
        periods = nx[sel] - g[sel]
        res["f0"][i] = fs / np.mean(periods)
        res["cq"][i] = np.mean((o[sel] - g[sel]) / periods)
        for key, segs in (("cp", cp_seg), ("cycle", list(zip(g[sel], nx[sel])))):
            a = covariance_lpc(x, segs, order)
            if a is None:
                continue
            f, bw = lpc_formants(a, fs)
            m = min(n_formants, len(f))
            res[key][i, :m], res[key + "_bw"][i, :m] = f[:m], bw[:m]
    return res
