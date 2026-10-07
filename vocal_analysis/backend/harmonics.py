"""Harmonic amplitudes by least squares at k f0 (plan §8.1)."""

from __future__ import annotations

import numpy as np

N_PERIODS = 3.0
MIN_WINDOW_S = 0.006  # enough samples for the fit at very high f0


def fit_frame(x: np.ndarray, sr: int, centre: int, f0: float, k_max: int, causal: bool = False,
              n_periods: float = N_PERIODS, fit_all: bool = True) -> np.ndarray:
    """Complex amplitudes of harmonics 1..k_max (NaN above Nyquist) for one frame.

    Hann-weighted least squares of DC + cosine/sine pairs over ~n_periods periods, centred
    on `centre` (or ending at it when causal). With fit_all, every harmonic below Nyquist
    is in the model (only the first k_max are returned), so unmodelled harmonics can't leak
    into the estimates. Amplitude convention: A cos(2 pi k f0 t + phi) returns A exp(i phi),
    with t = 0 at the window centre.
    """
    n = int(max(round(n_periods * sr / f0), MIN_WINDOW_S * sr))
    start = centre - n + 1 if causal else centre - n // 2
    lo, hi = max(start, 0), min(start + n, len(x))
    if hi - lo < n // 2:
        return np.full(k_max, np.nan + 0j)
    seg = x[lo:hi]
    t = (np.arange(lo, hi) - (start + n / 2)) / sr
    k_fit = max(k_max, int(0.98 * sr / 2 / f0)) if fit_all else k_max
    k = np.arange(1, k_fit + 1)
    k = k[k * f0 < sr / 2]
    ok = np.arange(1, k_max + 1) * f0 < sr / 2
    arg = 2 * np.pi * f0 * t[:, None] * k[None, :]
    basis = np.concatenate([np.ones((len(t), 1)), np.cos(arg), -np.sin(arg)], 1)
    w = np.sqrt(np.hanning(len(t) + 2)[1:-1])
    coef, *_ = np.linalg.lstsq(basis * w[:, None], seg * w, rcond=None)
    out = np.full(k_max, np.nan + 0j)
    amps = coef[1 : 1 + len(k)] + 1j * coef[1 + len(k) :]
    out[ok] = amps[: ok.sum()]
    return out


def harmonic_amplitudes(x: np.ndarray, sr: int, times: np.ndarray, f0: np.ndarray, k_max: int = 40,
                        causal: bool = False) -> np.ndarray:
    """|amplitude| in dB, shape (frames, k_max). NaN where f0 is missing or the harmonic is above Nyquist."""
    out = np.full((len(times), k_max), np.nan)
    for i, (t, f) in enumerate(zip(times, f0)):
        if np.isfinite(f) and f > 0:
            a = fit_frame(x, sr, int(round(t * sr)), float(f), k_max, causal)
            with np.errstate(divide="ignore"):
                out[i] = 20 * np.log10(np.abs(a))
    return out


def amplitude_near(h_db: np.ndarray, f0: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """An: amplitude of the strongest harmonic within +-max(f0/2, 15 % of Fn) of Fn (plan §8.1)."""
    k = np.arange(1, h_db.shape[1] + 1)
    fk = f0[:, None] * k[None, :]
    half = np.maximum(f0 / 2, 0.15 * fn)[:, None]
    m = np.abs(fk - fn[:, None]) <= half
    vals = np.where(m & np.isfinite(h_db), h_db, -np.inf)
    best = vals.max(axis=1)
    return np.where(np.isfinite(best), best, np.nan)
