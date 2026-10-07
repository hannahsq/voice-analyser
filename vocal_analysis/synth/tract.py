"""Vocal-tract resonance patterns for the resonator-cascade synthesiser (plan §5.1, fallback tract).

A pattern is F1..F8 (Hz, ascending, NaN above the usable band) with bandwidths, plus
optional extra pole-zero pairs (nasal or subglottal coupling). F1-F3 come from a
VTL-scaled vowel space or from broad log-uniform coverage; F4 and up follow the uniform
tube (2n - 1) c / 4L. The poles used for synthesis are the labels, by construction.

The vowel space is Peterson & Barney (1952) adult-male means, scaled by 17.5 cm / L. It
is a population prior from a different study, not data from any gold set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import hawks_miller_bandwidth

C_CM_S = 35000.0
N_POLES = 8
F_MAX_HZ = 7600.0  # poles above this are dropped (NaN); fs = 16 kHz
MIN_SPACING_HZ = 150.0
REF_L_CM = 17.5

# Peterson & Barney (1952), adult male means: F1, F2, F3 (Hz).
PB_MALE = {
    "i": (270, 2290, 3010), "I": (390, 1990, 2550), "E": (530, 1840, 2480), "ae": (660, 1720, 2410),
    "A": (730, 1090, 2440), "O": (570, 840, 2410), "U": (440, 1020, 2240), "u": (300, 870, 2240),
    "V": (640, 1190, 2390), "3r": (490, 1350, 1690),
}
_PB = np.array(list(PB_MALE.values()), float)


@dataclass
class TractPattern:
    freqs: np.ndarray  # (N_POLES,) oral resonances, ascending, NaN if absent
    bws: np.ndarray
    extra_pole_f: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra_pole_b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra_zero_f: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra_zero_b: np.ndarray = field(default_factory=lambda: np.zeros(0))


def tube_resonances(length_cm: float, n: int = N_POLES) -> np.ndarray:
    """Closed-open uniform tube: F_n = (2n - 1) c / 4L."""
    return (2 * np.arange(1, n + 1) - 1) * C_CM_S / (4.0 * length_cm)


def _enforce_order(f: np.ndarray) -> np.ndarray:
    out = f.copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + MIN_SPACING_HZ)
    out[out > F_MAX_HZ] = np.nan
    return out


def sample_formants(rng: np.random.Generator, length_cm: float, p_uniform: float = 0.3, p_cluster: float = 0.1) -> np.ndarray:
    """F1..F8 for one articulatory target."""
    s = REF_L_CM / length_cm
    tube = tube_resonances(length_cm)
    if rng.random() < p_uniform:
        f1 = np.exp(rng.uniform(np.log(200), np.log(1100))) * s ** 0.5
        f2 = np.exp(rng.uniform(np.log(max(f1 + 200, 600 * s)), np.log(2900 * s)))
        f3 = np.exp(rng.uniform(np.log(max(f2 + 200, 1600 * s)), np.log(max(f2 + 400, 3600 * s))))
        f13 = np.array([f1, f2, f3])
    else:
        i, j = rng.choice(len(_PB), 2, replace=False)
        w = rng.random()
        f13 = np.exp((1 - w) * np.log(_PB[i]) + w * np.log(_PB[j])) * s * np.exp(rng.normal(0, 0.08, 3))
    upper = tube[3:] * np.exp(rng.normal(0, 0.05, N_POLES - 3))
    f = np.concatenate([np.sort(f13), upper])
    if rng.random() < p_cluster:  # singer's formant cluster: F3-F5 bunched near 2.5-3.2 kHz (scaled)
        centre = rng.uniform(2500, 3200) * s ** 0.5
        gap = rng.uniform(250, 500)
        f[2:5] = centre + gap * np.array([-1.0, 0.0, 1.0])
    return _enforce_order(f)


def sample_bandwidths(rng: np.random.Generator, freqs: np.ndarray, f0: float) -> np.ndarray:
    """Hawks & Miller x log-uniform(0.6, 1.6) (plan §5.1). Above 5 kHz the H&M polynomial is out of
    its fitted range, so bandwidth grows linearly from its 5 kHz value."""
    f = np.where(np.isfinite(freqs), freqs, 1000.0)
    hm = np.where(f <= 5000, hawks_miller_bandwidth(np.minimum(f, 5000), f0), hawks_miller_bandwidth(5000.0, f0) * f / 5000)
    b = np.maximum(hm, 20.0) * np.exp(rng.uniform(np.log(0.6), np.log(1.6), len(f)))
    return np.where(np.isfinite(freqs), b, np.nan)


def sample_extras(rng: np.random.Generator, length_cm: float, p_nasal: float = 0.15, p_subglottal: float = 0.2) -> dict:
    """Extra pole-zero pairs. Nasal: pole 200-1200 Hz with a zero 100-400 Hz away. Subglottal:
    weak pairs near 600/1400/2200 Hz (adult), scaled by size. Not formant labels; kept in meta."""
    pf, pb, zf, zb = [], [], [], []
    if rng.random() < p_nasal:
        p = rng.uniform(200, 1200)
        z = p + rng.choice([-1, 1]) * rng.uniform(100, 400)
        pf.append(p), pb.append(rng.uniform(50, 200)), zf.append(max(z, 150.0)), zb.append(rng.uniform(50, 200))
    if rng.random() < p_subglottal:
        s = REF_L_CM / length_cm
        for base in (600.0, 1400.0, 2200.0):
            p = base * s * np.exp(rng.normal(0, 0.05))
            pf.append(p), pb.append(rng.uniform(100, 300))
            zf.append(p + rng.uniform(30, 80)), zb.append(rng.uniform(100, 300))
    a = lambda v: np.array(v, float)  # noqa: E731
    return {"extra_pole_f": a(pf), "extra_pole_b": a(pb), "extra_zero_f": a(zf), "extra_zero_b": a(zb)}
