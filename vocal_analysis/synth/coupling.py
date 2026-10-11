"""Glottal coupling within the cycle (D-023): how far each tube resonance moves while the glottis is open.

The closed-glottis tube (`tube`) drives the tract with an ideal volume-velocity source. With the glottis open, the
source has a finite impedance: in Norton form, a volume-velocity source in parallel with Zs = Zg + Zsub at the
glottis, so the tube's denominator (C Zr + D) becomes (C Zr + D) + (A Zr + B) / Zs.
- Zg, the glottal slit (Flanagan 1972; Ishizaka & Flanagan 1972): kinetic resistance linearised about the mean flow,
  rho U / Ag^2 with U = Ag sqrt(2 dP / rho), viscous resistance 12 mu d lg^2 / Ag^3, and inertance rho d / Ag.
- Zsub, the subglottal system: parallel resonances at the input-impedance maxima Ishizaka, Matsudaira & Kaneko (1976)
  measured (640, 1400, 2100 Hz; magnitudes within their 2.5-5 MPa s m^-3), scaled by speaker size.
Each closed-glottis pole is continued to the open-glottis one as the coupling is ramped in, so it stays the same
resonance. Values are quasi-static (the opening held fixed); `voice.synthesize` interpolates between the closed and
open values with the glottal opening through each cycle. Labels stay the closed-glottis resonances.

Check (D-023): a uniform 2.89 cm^2 duct with a static 0.51 cm^2 opening and no flow gives F1 +12 %; Barney, De Stefano
& Henrich (2007) measured +13 %.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tube import MU, RHO, TubeShape, _radiation, chain_matrix, poles

# Ishizaka et al. (1976) subglottal impedance maxima for adult men: Hz, peak |Z| (cgs acoustic ohm), bandwidth Hz.
SUBGLOTTAL = ((640.0, 45.0, 160.0), (1400.0, 35.0, 200.0), (2100.0, 25.0, 300.0))
REF_L_CM = 17.5
N_COUPLED = 5  # poles moved (F1-F5); higher ones are left as they are
MAX_REL_SHIFT = 0.6  # a continuation that moves a pole further than this has switched branch: no shift for that pole


# Residual opening during the 'closed' phase (fraction of the peak) against the open quotient, from Holmberg, Hillman &
# Perkell (1988, Tables AI-AVI; minimum / peak flow, 25 men and 20 women in loud, normal and soft voice; men 0.22 /
# 0.32 / 0.45 at OQ 0.57 / 0.60 / 0.73, women 0.33 / 0.41 / 0.52 at 0.71 / 0.76 / 0.82): within each sex it rises 1.4x
# as fast as the open quotient; at equal OQ, men's line is 0.28 at OQ 0.60 and women's 0.10 lower (their higher OQ
# gives them similar or larger gaps). Flow minima include leakage that high-speed video doesn't show as area (Patel et
# al. 2026: minimum / maximum area 0.01 men, 0.07 women), so each clip's offset spans from about the video values to
# the flow values.
GAP_AT_REF, OQ_REF, GAP_PER_OQ, GAP_MAX = 0.28, 0.60, 1.4, 0.6


def gap_from_open_quotient(oq: np.ndarray, offset: float) -> np.ndarray:
    """Residual opening per sample from the rendered open quotient (the LF te, set by Rd) and the clip's offset."""
    return np.clip(GAP_AT_REF + offset + GAP_PER_OQ * (np.asarray(oq) - OQ_REF), 0.0, GAP_MAX)


@dataclass
class Glottis:
    """Per-clip glottis and subglottal system. peak_area_cm2: largest opening in a cycle; gap_offset: shifts the
    residual opening during the 'closed' phase (`gap_from_open_quotient`)."""

    peak_area_cm2: float
    gap_offset: float
    length_cm: float  # glottal (membranous) length
    depth_cm: float  # vocal-fold thickness along the flow
    pressure: float  # transglottal pressure, dyn/cm^2
    sub_scale: float  # subglottal resonance frequencies multiplied by this


def sample_glottis(rng: np.random.Generator, tract_length_cm: float, peak_area_cm2: tuple[float, float] = (0.05, 0.22),
                   gap_offset: tuple[float, float] = (-0.3, 0.1)) -> Glottis:
    """Peak area log-uniform at 17.5 cm, scaled by (length / 17.5)^2: Holmberg et al.'s (1988) peak flows through
    Bernoulli at each group's pressure give men 0.06-0.24 cm^2 (mean 0.12) and women 0.04-0.12 (0.07; 0.05-0.15 at
    17.5 cm), about the same in soft, normal and loud voice, so the opening is not tied to Rd; only the closed-phase gap
    is. Transglottal pressure 3.6-12 cm H2O (theirs: 3.2-16, normal 4.3-9.7). Glottal length and depth scale with
    the tract."""
    s = tract_length_cm / REF_L_CM
    return Glottis(
        peak_area_cm2=float(np.exp(rng.uniform(*np.log(peak_area_cm2))) * s * s),
        gap_offset=float(rng.uniform(*gap_offset)),
        length_cm=float(1.4 * s * np.exp(rng.normal(0, 0.1))),
        depth_cm=float(0.3 * s * np.exp(rng.normal(0, 0.1))),
        pressure=float(rng.uniform(3500.0, 12000.0)),
        sub_scale=float(np.exp(rng.normal(0, 0.07)) / s),
    )


def _chain(s: np.ndarray, shape: TubeShape) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full glottis-to-lips chain matrix at complex s, with every loss."""
    return chain_matrix(s, shape)


def source_impedance(s: np.ndarray, area_cm2: float, g: Glottis, flow: bool = True, subglottal: bool = True) -> np.ndarray:
    z = 12.0 * MU * g.depth_cm * g.length_cm**2 / area_cm2**3 + s * RHO * g.depth_cm / area_cm2
    if flow:
        z = z + RHO * np.sqrt(2.0 * g.pressure / RHO) / area_cm2
    if subglottal:
        for f, r, b in SUBGLOTTAL:
            w0, bw = 2 * np.pi * f * g.sub_scale, 2 * np.pi * b * g.sub_scale
            z = z + r * bw * s / (s * s + bw * s + w0 * w0)
    return z


def _denominator(s, shape, area, g, alpha, **kw):
    A, B, C, D = _chain(s, shape)
    zr = _radiation(s, shape.areas[-1])
    return (C * zr + D) + alpha * (A * zr + B) / source_impedance(s, area, g, **kw)


def open_poles(shape: TubeShape, area_cm2: float, g: Glottis, freqs: np.ndarray | None = None, bws: np.ndarray | None = None,
               n: int = N_COUPLED, steps: int = 40, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies and bandwidths (Hz) of the first n resonances with the glottis open at area_cm2, continued from the
    closed-glottis poles (computed here unless given). Poles whose continuation fails are NaN."""
    if freqs is None or bws is None:
        freqs, bws = poles(shape)
    n = min(n, int(np.sum(np.isfinite(freqs))))
    s = -np.pi * bws[:n] + 2j * np.pi * freqs[:n]
    ok = np.ones(n, bool)
    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        for _ in range(50):
            h = 1e-5 * np.abs(s)
            d = _denominator(np.concatenate([s, s + h, s - h]), shape, area_cm2, g, alpha, **kw)
            with np.errstate(invalid="ignore", divide="ignore"):
                step = d[:n] / ((d[n:2 * n] - d[2 * n:]) / (2 * h))
            step = np.where(np.isfinite(step), step, 0.0)
            s = s - step
            if np.all(np.abs(step) < 2 * np.pi * 0.01):
                break
        else:
            ok &= np.abs(step) < 2 * np.pi * 1.0
    F, B = s.imag / (2 * np.pi), -s.real / np.pi
    ok &= (B > 0) & (np.abs(F / freqs[:n] - 1) <= MAX_REL_SHIFT)
    return np.where(ok, F, np.nan), np.where(ok, B, np.nan)


def open_shifts(shape: TubeShape, freqs: np.ndarray, bws: np.ndarray, g: Glottis) -> tuple[np.ndarray, np.ndarray]:
    """Per pole (len(freqs)): relative frequency shift and bandwidth ratio at the peak opening. Poles not coupled
    (above N_COUPLED, absent, or whose continuation failed) get 0 and 1."""
    df, bm = np.zeros(len(freqs)), np.ones(len(freqs))
    Fo, Bo = open_poles(shape, g.peak_area_cm2, g, freqs, bws)
    k = len(Fo)
    good = np.isfinite(Fo)
    df[:k] = np.where(good, Fo / freqs[:k] - 1.0, 0.0)
    bm[:k] = np.where(good, Bo / bws[:k], 1.0)
    return df, bm
