"""Quasi-closed-phase forward-backward linear prediction (QCP-FB) formants, with no EGG (D-046).

Weighted linear prediction whose weight emphasises each glottal cycle's closed phase and attenuates the main
excitation (QCP; Airaksinen, Raitio, Story & Alku 2014), fitted with forward and backward prediction sharing one set of
coefficients over the covariance range (Gowda, Airaksinen & Alku 2017, JASA 142(3), eqs. 14-18):

    F = sum_n w_n (x_n + sum_k a_k x_{n-k})^2 + sum_n w_n (x_n + sum_k a_k x_{n+k})^2.

Glottal closure instants come from the speech itself (SEDREAMS; Drugman & Dutoit 2009, Drugman et al. 2012),
implemented here from the published method. The QCP weight follows the authors' `qcp_wt` (github.com/njaygowda/ftrack,
Apache-2.0): per cycle, 1 from GCI + PQ T0 for DQ T0 samples, linear ramps at each end, a small floor elsewhere.

Settings are in time units so the sample rate can change (default 8 kHz, as published): order = fs/1000 + 5 (13 at
8 kHz), 30 ms rectangular frames, pre-emphasis 1 - 0.97 z^-1, ramp 0.875 ms (7 samples at 8 kHz).

Functions:
    sedreams(x, fs, f0_mean)                   GCIs (samples) from speech
    qcp_weights(n, gci, fs, ...)               the QCP weight per sample
    fb_wlp(x, w, order)                        weighted forward-backward covariance LP
    analyse(x, fs_in, times, ...)              QCP-FB (or QCP) formants at given times
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .closed_phase import lpc_formants

DQ, PQ, RAMP_S, W_MIN = 0.7, 0.05, 0.000875, 1e-5
FRAME_S, PREEMPH = 0.030, 0.97


def default_order(fs: float) -> int:
    return int(round(fs / 1000)) + 5


def _lp_residual(x: np.ndarray, fs: float, order: int) -> np.ndarray:
    """Overlap-added LP residual: 25 ms Hann-windowed autocorrelation LP every 5 ms, each frame's residual scaled to
    the frame's energy."""
    from scipy.linalg import solve_toeplitz
    from scipy.signal import lfilter

    n_win, hop = int(round(0.025 * fs)), int(round(0.005 * fs))
    win = np.hanning(n_win)
    out = np.zeros(len(x))
    for s in range(0, len(x) - n_win, hop):
        seg = x[s:s + n_win] * win
        r = np.correlate(seg, seg, "full")[n_win - 1:n_win + order]
        if r[0] <= 0:
            continue
        try:
            a = np.r_[1.0, solve_toeplitz(r[:order], -r[1:order + 1])]
        except np.linalg.LinAlgError:
            continue
        e = lfilter(a, [1.0], seg)
        ee = e @ e
        if ee > 0:
            out[s:s + n_win] += e * np.sqrt((seg @ seg) / ee)
    m = np.max(np.abs(out))
    return out / m if m > 0 else out


def sedreams(x: np.ndarray, fs: float, f0_mean: float, window_periods: float = 1.75) -> np.ndarray:
    """Glottal closure instants (sample indices) by SEDREAMS.

    1. The mean-based signal: x averaged under a Blackman window `window_periods` mean periods long, high-passed
       (here a 4th-order Butterworth at 40 Hz, zero phase), so it oscillates once per cycle.
    2. Each cycle's interval runs from a minimum of the mean-based signal to the next maximum; GCIs fall at a fairly
       constant relative position inside it.
    3. That position is estimated as the median relative position of the strong peaks (|r| > 0.4) of the LP
       residual r; each GCI is the residual's largest value within (position - 0.25, position + 0.35) of its
       interval.
    The residual's polarity is set so excitations are positive (the sign of its skewness)."""
    x = np.asarray(x, float)
    res = _lp_residual(x, fs, int(round(fs / 1000)) + 2)
    if np.mean(res ** 3) < 0:
        res = -res
    half = int(round(window_periods * fs / f0_mean / 2))
    win = np.blackman(2 * half + 1)
    mbs = np.convolve(x, win / len(win), "same")
    mbs = sosfiltfilt(butter(4, 40.0, "highpass", fs=fs, output="sos"), mbs)
    d = np.diff(mbs)
    maxima = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    minima = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    pairs = []
    for m in minima:
        k = np.searchsorted(maxima, m, side="right")
        if k < len(maxima):
            pairs.append((m, maxima[k]))
    if not pairs:
        return np.array([], int)
    lo = np.array([p[0] for p in pairs])
    hi = np.array([p[1] for p in pairs])
    d_res = np.diff(np.abs(res))
    peaks = np.flatnonzero((d_res[:-1] > 0) & (d_res[1:] <= 0)) + 1
    peaks = peaks[np.abs(res[peaks]) > 0.4]
    rel = []
    for p in peaks:
        j = int(np.argmin(np.abs(lo - p)))
        if hi[j] > lo[j]:
            rel.append((p - lo[j]) / (hi[j] - lo[j]))
    ratio = float(np.median(rel)) if rel else 0.5
    gci = []
    for a, b in zip(lo, hi):
        span = b - a
        s0 = max(0, a + int(round((ratio - 0.25) * span)))
        s1 = min(len(res), a + int(round((ratio + 0.35) * span)) + 1)
        if s1 > s0:
            gci.append(s0 + int(np.argmax(res[s0:s1])))
    return np.unique(np.array(gci, int))


def qcp_weights(n: int, gci: np.ndarray, fs: float, dq: float = DQ, pq: float = PQ, ramp_s: float = RAMP_S,
                w_min: float = W_MIN) -> np.ndarray:
    """The QCP weight for n samples: per cycle (one GCI to the next, period T), 1 over [GCI + round(pq T),
    GCI + round(pq T) + round(dq T)), with linear ramps of round(ramp_s fs) samples inside each end (excluding the
    end values, as the authors' `qcp_wt`), w_min elsewhere. The last cycle reuses the previous period."""
    w = np.full(n, w_min)
    nr = int(round(ramp_s * fs))
    up = np.linspace(w_min, 1.0, nr + 2)[1:-1] if nr > 0 else np.array([])
    gci = np.sort(np.asarray(gci, int))
    if len(gci) < 2:
        return w
    periods = np.diff(gci)
    for i, g in enumerate(gci):
        T = periods[min(i, len(periods) - 1)]
        t2 = int(round(pq * T))
        t1 = min(int(round(dq * T)), T - t2)
        a, b = g + t2, min(g + t2 + t1, n)
        if a >= n or b <= a:
            continue
        w[a:b] = 1.0
        if nr > 0 and b - a >= 2 * nr:
            w[a:a + nr] = up
            w[b - nr:b] = up[::-1]
    return w


def fb_wlp(x: np.ndarray, w: np.ndarray, order: int, forward_only: bool = False) -> np.ndarray | None:
    """Predictor a (x[n] ~ -sum_k a_k x[n-k]) minimising the weighted forward (and backward) squared error over the
    covariance range of x: forward errors at n = order..N-1, backward errors at n = 0..N-1-order, each weighted by
    w[n]. None if the weighted system is degenerate."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    N = len(x)
    if N <= 2 * order:
        return None
    k = np.arange(order + 1)
    nf = np.arange(order, N)
    Xf = x[nf[:, None] - k[None, :]]
    D = (Xf * w[nf, None]).T @ Xf
    if not forward_only:
        nb = np.arange(0, N - order)
        Xb = x[nb[:, None] + k[None, :]]
        D = D + (Xb * w[nb, None]).T @ Xb
    tr = np.trace(D)
    if not np.isfinite(tr) or tr <= 0:
        return None
    try:
        return np.linalg.solve(D[1:, 1:] + 1e-12 * tr * np.eye(order), -D[1:, 0])
    except np.linalg.LinAlgError:
        return None


def analyse(x: np.ndarray, fs_in: float, times: np.ndarray, f0_mean: float | None = None, fs: float = 8000.0,
            order: int | None = None, gci: np.ndarray | None = None, forward_only: bool = False,
            n_formants: int = 4, dq: float = DQ) -> dict:
    """QCP-FB formants at `times` (s). x at fs_in is resampled to fs. GCIs from SEDREAMS (needs f0_mean, Hz) unless
    `gci` is given (sample indices at fs, e.g. ideal ones on synthetic audio). Per time, a FRAME_S rectangular frame
    centred on it, pre-emphasised; formants from the predictor's roots (`closed_phase.lpc_formants`). `dq` is the
    weight's duration quotient (0.7 as published; D-046's arm B uses 0.3, inside the closed phase).

    Returns "F", "bw" (T, n_formants) Hz with NaN where no fit, "gci" (samples at fs), "fs", "order"."""
    x = np.asarray(x, float)
    if fs != fs_in:
        from fractions import Fraction

        fr = Fraction(int(round(fs)), int(round(fs_in)))
        x = resample_poly(x, fr.numerator, fr.denominator)
    order = default_order(fs) if order is None else order
    if gci is None:
        if f0_mean is None:
            raise ValueError("f0_mean is needed to detect GCIs")
        gci = sedreams(x, fs, f0_mean)
    w = qcp_weights(len(x), gci, fs, dq=dq)
    xp = np.r_[x[0], x[1:] - PREEMPH * x[:-1]]
    half = int(round(FRAME_S * fs / 2))
    T = len(times)
    F = np.full((T, n_formants), np.nan)
    B = np.full((T, n_formants), np.nan)
    for i, t in enumerate(times):
        c = int(round(t * fs))
        a0, a1 = c - half, c + half
        if a0 < 0 or a1 > len(x):
            continue
        a = fb_wlp(xp[a0:a1], w[a0:a1], order, forward_only)
        if a is None:
            continue
        f, b = lpc_formants(a, fs)
        m = min(n_formants, len(f))
        F[i, :m], B[i, :m] = f[:m], b[:m]
    return {"F": F, "bw": B, "gci": np.asarray(gci, int), "fs": fs, "order": order}
