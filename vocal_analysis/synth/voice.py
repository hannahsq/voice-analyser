"""Signal synthesis from control-rate trajectories (plan §5.1).

Voiced component: additive synthesis. Harmonic k at control point c has complex amplitude
c_k(Rd_c) * H_c(k f0_c), where c_k is the LF source spectrum and H_c the tract cascade
response, so the output is band-limited at any f0 and its harmonic amplitudes equal
source x |H(k f0)| exactly on stationary, unperturbed settings. Jitter and shimmer are
applied per glottal period.

With glottal coupling (D-023, `synthesize(..., coupling=...)`), the voiced component is instead the same
band-limited LF source rendered in the time domain and filtered by `filters.filter_tv`, with every coupled pole's
frequency and bandwidth moving within each cycle towards its open-glottis value as the glottis opens (the harmonic
path can't vary the tract within a period). Labels stay the closed-glottis poles.

Noise components go through the same cascade in the time domain (`filters.filter_tv`):
aspiration (modulated by glottal opening, level set by HNR), whisper (unmodulated, through
the tract) and frication (high-passed, not through the tract). Noise sources are flow, so
they get the radiation first difference; the LF derivative already includes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt

from . import filters
from .glottal import source_coefficients

FS = 16000
CTRL = 16  # samples per control point (1 ms)
NYQ_TAPER_HZ = (7400.0, 7800.0)  # harmonics fade out across this band


@dataclass
class Trajectories:
    """Control-rate trajectories, C points at CTRL-sample spacing (point c at sample c * CTRL)."""

    f0: np.ndarray  # (C,) Hz, defined everywhere (used for phase even when unvoiced)
    rd: np.ndarray  # (C,)
    voice_amp: np.ndarray  # (C,) linear amplitude of the voiced source (0 = unvoiced)
    whisper_amp: np.ndarray  # (C,) unmodulated noise through the tract
    fric_amp: np.ndarray  # (C,) high-passed noise, not through the tract
    poles_f: np.ndarray  # (C, P) all poles (oral first), NaN = absent
    poles_b: np.ndarray
    zeros_f: np.ndarray  # (C, Z)
    zeros_b: np.ndarray

    @property
    def n_samples(self) -> int:
        return (len(self.f0) - 1) * CTRL


@njit(cache=True)
def _glottal_phase(f0, jitter, fs):
    """Phase in cycles per sample and period index; period m runs at f0 * jitter[m]."""
    n = len(f0)
    ph = np.empty(n)
    idx = np.empty(n, np.int64)
    acc = 0.0
    m = 0
    for i in range(n):
        ph[i] = acc
        idx[i] = m
        acc += f0[i] * jitter[min(m, len(jitter) - 1)] / fs
        while acc >= m + 1:
            m += 1
    return ph, idx


@njit(cache=True)
def _additive(cycles, mag, phs, kmax, ctrl):
    n = len(cycles)
    y = np.zeros(n)
    two_pi = 2.0 * np.pi
    for i in range(n):
        c0 = i // ctrl
        w = (i - c0 * ctrl) / ctrl
        c1 = c0 + 1
        kk = max(kmax[c0], kmax[c1])
        base = two_pi * cycles[i]
        s = 0.0
        for k in range(kk):
            m = (1.0 - w) * mag[c0, k] + w * mag[c1, k]
            if m != 0.0:
                s += m * np.cos((k + 1) * base + (1.0 - w) * phs[c0, k] + w * phs[c1, k])
        y[i] = s
    return y


def harmonic_amplitudes(traj: Trajectories, k_max: int, fs: float = FS) -> np.ndarray:
    """Complex amplitude of harmonics 1..k_max at each control point: c_k(Rd) H(k f0) * taper."""
    k = np.arange(1, k_max + 1)
    fk = traj.f0[:, None] * k[None, :]
    h = filters.response(fk, traj.poles_f[:, None, :], traj.poles_b[:, None, :], fs,
                         traj.zeros_f[:, None, :], traj.zeros_b[:, None, :])
    taper = np.clip((NYQ_TAPER_HZ[1] - fk) / (NYQ_TAPER_HZ[1] - NYQ_TAPER_HZ[0]), 0.0, 1.0)
    return source_coefficients(traj.rd, k_max) * h * taper


def _per_sample(ctrl_values: np.ndarray) -> np.ndarray:
    """Linear interpolation from control points to samples (last control point excluded)."""
    c = len(ctrl_values) - 1
    pos = np.arange(c * CTRL) / CTRL
    return np.interp(pos, np.arange(c + 1), ctrl_values)


def _sections_per_sample(freqs: np.ndarray, bws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, N) per-sample section frequencies/bandwidths; sections absent at either end stay absent."""
    if freqs.shape[1] == 0:
        return np.zeros((0, (len(freqs) - 1) * CTRL)), np.zeros((0, (len(freqs) - 1) * CTRL))
    f = np.stack([_per_sample(np.nan_to_num(freqs[:, s], nan=-1.0)) for s in range(freqs.shape[1])])
    b = np.stack([_per_sample(np.nan_to_num(bws[:, s], nan=1.0)) for s in range(freqs.shape[1])])
    absent = np.stack([_per_sample(np.isnan(freqs[:, s]).astype(float)) for s in range(freqs.shape[1])]) > 0
    f[absent] = np.nan
    return f, b


def tract_filter(x: np.ndarray, traj: Trajectories, fs: float = FS) -> np.ndarray:
    pf, pb = _sections_per_sample(traj.poles_f, traj.poles_b)
    zf, zb = _sections_per_sample(traj.zeros_f, traj.zeros_b)
    freqs = np.concatenate([pf, zf])
    bws = np.concatenate([pb, zb])
    is_zero = np.r_[np.zeros(len(pf), bool), np.ones(len(zf), bool)]
    return filters.filter_tv(x, freqs, bws, fs, is_zero)


def glottal_opening(cycles: np.ndarray, te_s: np.ndarray, gap: float = 0.0) -> np.ndarray:
    """Opening per sample as a fraction of the peak: sin^2 over the open phase (as the aspiration modulation), and
    the residual gap during the closed phase."""
    frac = cycles % 1.0
    return gap + (1.0 - gap) * np.where(frac < te_s, np.sin(np.pi * frac / te_s) ** 2, 0.0)


def _open_phase_te(traj: Trajectories) -> np.ndarray:
    from .glottal import open_phase

    return _per_sample(np.array([open_phase(r) for r in np.round(traj.rd, 2)]))


def _voiced_coupled(traj: Trajectories, cycles: np.ndarray, coupling: dict, k_max: int, fs: float) -> np.ndarray:
    """The LF source alone (band-limited, as the harmonic path) through the tract in the time domain, poles moving
    with the glottal opening: F(t) = F (1 + df a(t)), B(t) = B (1 + (bm - 1) a(t)). coupling: "df", "bm" (C, P) at
    control points, P = the oral poles they cover (extra poles and zeros don't move); "gap" (fraction of the peak)."""
    k = np.arange(1, k_max + 1)
    fk = traj.f0[:, None] * k[None, :]
    taper = np.clip((NYQ_TAPER_HZ[1] - fk) / (NYQ_TAPER_HZ[1] - NYQ_TAPER_HZ[0]), 0.0, 1.0)
    amp = source_coefficients(traj.rd, k_max) * taper
    kmax = np.minimum(np.ceil(NYQ_TAPER_HZ[1] / traj.f0).astype(np.int64), k_max)
    src = _additive(cycles, np.abs(amp) * traj.voice_amp[:, None], np.unwrap(np.angle(amp), axis=0), kmax, CTRL)
    a = glottal_opening(cycles, _open_phase_te(traj), coupling.get("gap", 0.0))
    pf, pb = _sections_per_sample(traj.poles_f, traj.poles_b)
    n_c = coupling["df"].shape[1]
    df = np.stack([_per_sample(coupling["df"][:, j]) for j in range(n_c)])
    bm = np.stack([_per_sample(coupling["bm"][:, j]) for j in range(n_c)])
    pf[:n_c] = pf[:n_c] * (1.0 + df * a)
    pb[:n_c] = pb[:n_c] * (1.0 + (bm - 1.0) * a)
    zf, zb = _sections_per_sample(traj.zeros_f, traj.zeros_b)
    is_zero = np.r_[np.zeros(len(pf), bool), np.ones(len(zf), bool)]
    return filters.filter_tv(src, np.concatenate([pf, zf]), np.concatenate([pb, zb]), fs, is_zero)


def synthesize(traj: Trajectories, rng: np.random.Generator, jitter: float = 0.0, shimmer_db: float = 0.0,
               hnr_db: float | None = None, fs: float = FS, coupling: dict | None = None) -> dict[str, np.ndarray]:
    """Render the components. Returns dict of float64 arrays (same length) and 'cycles' (glottal phase).

    jitter: SD of the per-period relative period perturbation; shimmer_db: SD of per-period amplitude (dB).
    hnr_db: voiced-to-aspiration energy ratio over voiced samples; None = no aspiration.
    coupling: glottal coupling within the cycle (`_voiced_coupled`); None = the harmonic path.
    """
    n = traj.n_samples
    f0_s = _per_sample(traj.f0)
    n_periods = int(np.sum(f0_s) / fs * 1.1) + 16
    jit = np.exp(rng.normal(0.0, jitter, n_periods)) if jitter > 0 else np.ones(n_periods)
    shim = 10 ** (rng.normal(0.0, shimmer_db, n_periods) / 20) if shimmer_db > 0 else np.ones(n_periods)
    cycles, period = _glottal_phase(f0_s, jit, float(fs))

    k_max = int(min(256, np.ceil(NYQ_TAPER_HZ[1] / max(traj.f0.min(), 1.0))))
    if coupling is not None:
        voiced = _voiced_coupled(traj, cycles, coupling, k_max, fs) * shim[period]
    else:
        amp = harmonic_amplitudes(traj, k_max, fs)
        kmax = np.minimum(np.ceil(NYQ_TAPER_HZ[1] / traj.f0).astype(np.int64), k_max)
        mag = np.abs(amp) * traj.voice_amp[:, None]
        phs = np.unwrap(np.angle(amp), axis=0)
        voiced = _additive(cycles, mag, phs, kmax, CTRL) * shim[period]

    out = {"voiced": voiced, "cycles": cycles}
    va = _per_sample(traj.voice_amp)
    if hnr_db is not None:
        from .glottal import open_phase

        te = np.array([open_phase(r) for r in np.round(traj.rd, 2)])
        frac = cycles % 1.0
        te_s = _per_sample(te)
        mod = np.where(frac < te_s, 0.25 + 0.75 * np.sin(np.pi * frac / te_s) ** 2, 0.25)
        src = np.diff(rng.standard_normal(n + 1)) * mod * va
        asp = tract_filter(src, traj, fs)
        on = va > 0.5 * va.max() if va.max() > 0 else np.zeros(n, bool)
        if on.any():
            ev, ea = np.mean(voiced[on] ** 2), np.mean(asp[on] ** 2)
            if ea > 0:
                asp *= np.sqrt(ev / ea / 10 ** (hnr_db / 10))
        out["aspiration"] = asp
    wa = _per_sample(traj.whisper_amp)
    if wa.any():
        out["whisper"] = tract_filter(np.diff(rng.standard_normal(n + 1)) * wa, traj, fs) * 0.05
    fa = _per_sample(traj.fric_amp)
    if fa.any():
        sos = butter(4, rng.uniform(2000, 5000), "highpass", fs=fs, output="sos")
        out["frication"] = sosfilt(sos, rng.standard_normal(n)) * fa * 0.05
    out["total"] = sum(v for k, v in out.items() if k not in ("cycles",))
    return out
