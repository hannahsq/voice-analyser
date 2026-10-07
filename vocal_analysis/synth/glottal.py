"""LF glottal flow derivative (Fant, Liljencrants & Lin 1985) parameterised by Rd (Fant 1995).

Time is normalised to one period (T0 = 1) and the excitation amplitude Ee = 1, so the
waveform shape, and hence every Fourier coefficient, depends on Rd alone. The
synthesiser uses the flow *derivative* as the excitation, which already includes the
lip-radiation characteristic, so no separate radiation filter is applied.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

RD_MIN, RD_MAX = 0.3, 2.7
N_HARMONICS = 256  # enough for f0 = 30 Hz up to 7.7 kHz


@dataclass(frozen=True)
class LFParams:
    tp: float  # peak flow (zero crossing of the derivative)
    te: float  # main excitation (most negative derivative)
    ta: float  # return-phase time constant
    alpha: float
    epsilon: float
    e0: float


def lf_params(rd: float) -> LFParams:
    """Fant (1995) Rd regression -> (Ra, Rk, Rg), then the LF constants for Ee = 1, T0 = 1."""
    if not RD_MIN - 1e-9 <= rd <= RD_MAX + 1e-9:
        raise ValueError(f"Rd {rd} outside [{RD_MIN}, {RD_MAX}]")
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    rg = rk * (0.5 + 1.2 * rk) / (0.44 * rd - 4.0 * ra * (0.5 + 1.2 * rk))
    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    ta = ra
    tb = 1.0 - te
    if not (0 < tp < te < 1 and 0 < ta < tb):
        raise ValueError(f"inconsistent LF timing for Rd {rd}: tp={tp}, te={te}, ta={ta}")

    # Return phase: epsilon * ta = 1 - exp(-epsilon * (1 - te)).
    eps = brentq(lambda e: e * ta - 1.0 + np.exp(-e * tb), 1e-6 / ta, 1e3 / ta)
    ret_area = -(1.0 / (eps * ta)) * ((1.0 - np.exp(-eps * tb)) / eps - tb * np.exp(-eps * tb))
    wg = np.pi / tp

    def net_area(a: float) -> float:
        e0 = -1.0 / (np.exp(a * te) * np.sin(wg * te))
        open_area = e0 * (np.exp(a * te) * (a * np.sin(wg * te) - wg * np.cos(wg * te)) + wg) / (a * a + wg * wg)
        return open_area + ret_area

    alpha = brentq(net_area, -50.0, 200.0)  # flow returns to zero at the end of the period
    e0 = -1.0 / (np.exp(alpha * te) * np.sin(wg * te))
    return LFParams(tp, te, ta, alpha, eps, e0)


def lf_waveform(rd: float, n: int) -> np.ndarray:
    """One period of the flow derivative, n samples at t = i / n, Ee = 1."""
    p = lf_params(rd)
    t = np.arange(n) / n
    out = np.empty(n)
    op = t <= p.te
    out[op] = p.e0 * np.exp(p.alpha * t[op]) * np.sin(np.pi / p.tp * t[op])
    tr = t[~op]
    out[~op] = -(1.0 / (p.epsilon * p.ta)) * (np.exp(-p.epsilon * (tr - p.te)) - np.exp(-p.epsilon * (1.0 - p.te)))
    return out


def open_phase(rd: float) -> float:
    """Fraction of the period during which the glottis is open (te; the return phase is short)."""
    return lf_params(rd).te


RD_GRID = np.round(np.arange(RD_MIN, RD_MAX + 1e-9, 0.01), 2)


@functools.lru_cache(maxsize=1)
def harmonic_table() -> tuple[np.ndarray, np.ndarray]:
    """(log magnitude, unwrapped phase) of Fourier coefficients c_k, k = 1..N_HARMONICS, on RD_GRID.

    c_k = integral over one period of E(t) exp(-2 pi i k t) dt. The derivative waveform is
    continuous, so its spectrum falls off fast and an 8192-point FFT is accurate to ~1e-6
    relative at k = 256.
    """
    n = 8192
    coefs = np.array([np.fft.rfft(lf_waveform(rd, n))[1 : N_HARMONICS + 1] / n for rd in RD_GRID])
    logmag = np.log(np.abs(coefs))
    phase = np.unwrap(np.unwrap(np.angle(coefs), axis=0), axis=1)
    return logmag, phase


def source_coefficients(rd: np.ndarray, k_max: int = N_HARMONICS) -> np.ndarray:
    """Complex c_k for k = 1..k_max at each Rd (shape rd.shape + (k_max,)), interpolated on RD_GRID.

    Log magnitude and phase are interpolated linearly in Rd; grid spacing is 0.01.
    """
    logmag, phase = harmonic_table()
    rd = np.clip(np.asarray(rd, float), RD_MIN, RD_MAX)
    pos = (rd - RD_MIN) / 0.01
    i0 = np.clip(np.floor(pos).astype(int), 0, len(RD_GRID) - 2)
    w = (pos - i0)[..., None]
    lm = (1 - w) * logmag[i0, :k_max] + w * logmag[i0 + 1, :k_max]
    ph = (1 - w) * phase[i0, :k_max] + w * phase[i0 + 1, :k_max]
    return np.exp(lm + 1j * ph)


def source_h_db(rd: np.ndarray, k_max: int = 8) -> np.ndarray:
    """Source harmonic amplitudes 20 log10 |c_k| (dB re Ee = 1), k = 1..k_max."""
    return 20.0 * np.log10(np.abs(source_coefficients(rd, k_max)))
