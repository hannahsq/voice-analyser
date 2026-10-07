"""Digital second-order resonators and anti-resonators, normalised to unity gain at DC.

One definition serves both synthesis paths: `response` gives the exact complex
transfer function (harmonic path, back-end corrections), and `filter_tv` runs the
same sections in the time domain with per-sample coefficients (noise path). The
magnitude of a resonator is exactly the Iseli-Alwan G_i(f) of plan §8.2.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def coefficients(freq: np.ndarray, bw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, b1, b2) with y[n] = a x[n] + b1 y[n-1] + b2 y[n-2]; a = 1 - b1 - b2 gives unity DC gain."""
    r = np.exp(-np.pi * np.asarray(bw, float) / fs)
    b1 = 2.0 * r * np.cos(2.0 * np.pi * np.asarray(freq, float) / fs)
    b2 = -r * r
    return 1.0 - b1 - b2, b1, b2


def response(f: np.ndarray, poles_f: np.ndarray, poles_b: np.ndarray, fs: float,
             zeros_f: np.ndarray | None = None, zeros_b: np.ndarray | None = None) -> np.ndarray:
    """Complex response of a cascade at frequencies f.

    poles_f/poles_b have shape (..., P) and broadcast against f[..., None]; NaN poles are
    skipped (treated as absent). Zeros are anti-resonators (inverse resonators).
    """
    z1 = np.exp(-2j * np.pi * np.asarray(f, float) / fs)[..., None]

    def sections(freqs, bws):
        a, b1, b2 = coefficients(freqs, bws, fs)
        with np.errstate(invalid="ignore"):
            h = a / (1.0 - b1 * z1 - b2 * z1 * z1)
        return np.where(np.isfinite(h), h, 1.0)

    h = np.prod(sections(poles_f, poles_b), axis=-1)
    if zeros_f is not None:
        h = h / np.prod(sections(zeros_f, zeros_b), axis=-1)
    return h


def resonator_gain_db(f: np.ndarray, freq: np.ndarray, bw: np.ndarray, fs: float) -> np.ndarray:
    """Iseli-Alwan G(f) in dB for one resonator per entry (plan §8.2), = 20 log10 |response|."""
    r = np.exp(-np.pi * bw / fs)
    th = 2.0 * np.pi * freq / fs
    w = 2.0 * np.pi * f / fs
    num = 1.0 - 2.0 * r * np.cos(th) + r * r
    den = np.sqrt((1.0 - 2.0 * r * np.cos(th - w) + r * r) * (1.0 - 2.0 * r * np.cos(th + w) + r * r))
    return 20.0 * np.log10(num / den)


@njit(cache=True)
def _filter_tv(x, a, b1, b2, is_zero):
    """Cascade of sections with per-sample coefficients; shape (S, N) each. Zeros invert the section."""
    n_sec, n = a.shape
    y = x.copy()
    for s in range(n_sec):
        u1 = 0.0  # previous input (zeros) or output (poles)
        u2 = 0.0
        if is_zero[s]:
            for i in range(n):
                xi = y[i]
                v = (xi - b1[s, i] * u1 - b2[s, i] * u2) / a[s, i]
                u2 = u1
                u1 = xi
                y[i] = v
        else:
            for i in range(n):
                v = a[s, i] * y[i] + b1[s, i] * u1 + b2[s, i] * u2
                u2 = u1
                u1 = v
                y[i] = v
    return y


def filter_tv(x: np.ndarray, freqs: np.ndarray, bws: np.ndarray, fs: float, is_zero: np.ndarray | None = None) -> np.ndarray:
    """Filter x through sections whose frequencies/bandwidths vary per sample.

    freqs, bws: (S, N). Sections with NaN frequency at a sample pass the signal unchanged
    there (a = 1, b = 0).
    """
    a, b1, b2 = coefficients(freqs, bws, fs)
    off = ~np.isfinite(a)
    a[off], b1[off], b2[off] = 1.0, 0.0, 0.0
    flags = np.zeros(len(freqs), np.bool_) if is_zero is None else np.asarray(is_zero, np.bool_)
    return _filter_tv(np.asarray(x, np.float64), a, b1, b2, flags)


def hawks_miller_bandwidth(f: np.ndarray, f0: np.ndarray | float) -> np.ndarray:
    """Hawks & Miller (1995) bandwidth (Hz) for a formant at f, scaled for f0.

    Coefficients as in VoiceSauce's func_getBWHawksMiller.m. VERIFY against the paper before
    relying on the back-end correction; the generator only uses it as a sampling prior.
    """
    f = np.asarray(f, float)
    s = 1.0 + 0.25 * (np.asarray(f0, float) - 132.0) / 88.0
    c1 = (165.327516, -6.73636734e-1, 1.80874446e-3, -4.52201682e-6, 7.49514000e-9, -4.70219241e-12)
    c2 = (15.8146139, 8.10159009e-2, -9.79728215e-5, 5.28725064e-8, -1.07099364e-11, 7.91528509e-16)
    lo = sum(c * f**i for i, c in enumerate(c1))
    hi = sum(c * f**i for i, c in enumerate(c2))
    return s * np.where(f < 500.0, lo, hi)
