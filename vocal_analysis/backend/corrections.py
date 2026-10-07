"""Iseli-Alwan formant corrections (plan §8.2).

H*(f) = H(f) - sum_i G_i(f), with G_i the DC-normalised second-order resonator gain. The
formula is `synth.filters.resonator_gain_db`, the same one the synthesiser's resonators
obey, so correcting with the true poles undoes the tract exactly.
"""

from __future__ import annotations

import numpy as np

from ..synth.filters import hawks_miller_bandwidth, resonator_gain_db
from .vtl import C_CM_S, apparent_vtl

N_UPPER = 4  # poles F5..F8 added from the apparent VTL when upper_from_vtl


def formant_gain_db(f: np.ndarray, formants: np.ndarray, bandwidths: np.ndarray | None, f0: np.ndarray,
                    sr: float) -> np.ndarray:
    """Summed G_i(f) in dB. f: (frames, K); formants/bandwidths: (frames, N). Missing bandwidths
    (None or NaN) fall back to Hawks & Miller; missing formants contribute nothing."""
    formants = np.asarray(formants, float)
    bw = np.full_like(formants, np.nan) if bandwidths is None else np.asarray(bandwidths, float)
    hm = hawks_miller_bandwidth(np.nan_to_num(formants, nan=1000.0), np.asarray(f0, float)[:, None])
    bw = np.where(np.isfinite(bw), bw, hm)
    g = resonator_gain_db(np.asarray(f)[..., None], formants[:, None, :], bw[:, None, :], sr)
    return np.nansum(np.where(np.isfinite(formants)[:, None, :], g, 0.0), axis=-1)


def upper_poles(formants: np.ndarray, sr: float) -> np.ndarray:
    """Uniform-tube resonances n = N+1..N+N_UPPER at the apparent VTL of the given formants
    (NaN above 0.95 x Nyquist). DC-normalised resonators still boost frequencies below their
    peak, so leaving these out biases H2* relative to H1* at high f0 (~0.4 dB per pole at 1 kHz)."""
    formants = np.asarray(formants, float)
    length = apparent_vtl(formants)
    n = np.arange(formants.shape[1] + 1, formants.shape[1] + N_UPPER + 1)
    f = (2 * n[None, :] - 1) * C_CM_S / (4 * length[:, None])
    return np.where(f < 0.95 * sr / 2, f, np.nan)


def correct(h_db: np.ndarray, f0: np.ndarray, formants: np.ndarray, bandwidths: np.ndarray | None, sr: float,
            upper_from_vtl: bool = False) -> np.ndarray:
    """Corrected harmonic amplitudes H_k* (dB), shape like h_db (frames, K).

    upper_from_vtl adds poles above the supplied formants at their uniform-tube positions
    (bandwidths from Hawks & Miller); use it when only F1-F4 are known.
    """
    formants = np.asarray(formants, float)
    if upper_from_vtl:
        up = upper_poles(formants, sr)
        bw = np.full(formants.shape, np.nan) if bandwidths is None else np.asarray(bandwidths, float)
        formants, bandwidths = np.concatenate([formants, up], 1), np.concatenate([bw, np.full(up.shape, np.nan)], 1)
    k = np.arange(1, h_db.shape[1] + 1)
    fk = np.asarray(f0, float)[:, None] * k[None, :]
    return h_db - formant_gain_db(fk, formants, bandwidths, f0, sr)


def monte_carlo(h_db: np.ndarray, f0: np.ndarray, formant_samples: np.ndarray, bandwidth_samples: np.ndarray | None,
                sr: float, measure, q=(2.5, 50.0, 97.5)) -> np.ndarray:
    """Percentiles of measure(H*) over S samples of (F, B) (plan §8.2 uncertainty).

    formant_samples: (S, frames, N). measure maps corrected (frames, K) -> (frames,).
    Returns (len(q), frames).
    """
    vals = np.stack([measure(correct(h_db, f0, formant_samples[s], None if bandwidth_samples is None else bandwidth_samples[s], sr))
                     for s in range(len(formant_samples))])
    return np.nanpercentile(vals, q, axis=0)
