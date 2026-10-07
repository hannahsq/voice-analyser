"""Spectral slope measures (plan §8.3)."""

from __future__ import annotations

import numpy as np


def nearest_harmonic(f0: np.ndarray, target_hz: float, k_max: int) -> np.ndarray:
    """Index (0-based) of the harmonic nearest target_hz, clipped to the available range."""
    k = np.rint(target_hz / np.asarray(f0, float)).astype(int)
    return np.clip(k, 1, k_max) - 1


def _pick(h: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return h[np.arange(len(h)), idx]


def kreiman_garellek(h_db: np.ndarray, h_corr_db: np.ndarray, f0: np.ndarray) -> dict[str, np.ndarray]:
    """H1*-H2*, H2*-H4*, H4*-H2kHz*, H2kHz*-H5kHz (H5kHz uncorrected), plus raw H1-H2."""
    k_max = h_db.shape[1]
    i2k = nearest_harmonic(f0, 2000.0, k_max)
    i5k = nearest_harmonic(f0, 5000.0, k_max)
    return {
        "H1H2": h_db[:, 0] - h_db[:, 1],
        "H1H2c": h_corr_db[:, 0] - h_corr_db[:, 1],
        "H2H4c": h_corr_db[:, 1] - h_corr_db[:, 3],
        "H4H2kc": h_corr_db[:, 3] - _pick(h_corr_db, i2k),
        "H2kH5k": _pick(h_corr_db, i2k) - _pick(h_db, i5k),
    }


def regression_slope(h_db: np.ndarray, f0: np.ndarray, band_hz: tuple[float, float] = (0.0, 5000.0)) -> np.ndarray:
    """Least-squares slope (dB/octave) of harmonic amplitude against log2 frequency within band_hz."""
    k = np.arange(1, h_db.shape[1] + 1)
    fk = np.asarray(f0, float)[:, None] * k[None, :]
    out = np.full(len(h_db), np.nan)
    for i in range(len(h_db)):
        m = (fk[i] > band_hz[0]) & (fk[i] <= band_hz[1]) & np.isfinite(h_db[i])
        if m.sum() >= 3:
            out[i] = np.polyfit(np.log2(fk[i, m]), h_db[i, m], 1)[0]
    return out
