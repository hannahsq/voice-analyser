"""Vocal tract length from formant spacing (plan §8.4)."""

from __future__ import annotations

import numpy as np

C_CM_S = 35000.0


def apparent_vtl(formants: np.ndarray, sigma_hz: np.ndarray | None = None) -> np.ndarray:
    """Per-frame L (cm) from the uniform-tube fit through the origin, F_n = (2n - 1)/2 * dF.

    dF = sum(w F x) / sum(w x^2) with x_n = (2n - 1)/2 and w_n = 1 / sigma_n^2 (equal weights if
    sigma is None); L = c / (2 dF). Missing formants are skipped.
    """
    f = np.asarray(formants, float)
    x = (2 * np.arange(1, f.shape[1] + 1) - 1) / 2.0
    w = np.ones_like(f) if sigma_hz is None else 1.0 / np.asarray(sigma_hz, float) ** 2
    w = np.where(np.isfinite(f) & np.isfinite(w), w, 0.0)
    num = np.sum(w * np.nan_to_num(f) * x, axis=1)
    den = np.sum(w * x * x, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = num / den
        return np.where(den > 0, C_CM_S / (2 * df), np.nan)


def anatomical_vtl(apparent_cm: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Weighted median and 2.5-97.5 % weighted range over frames: (median, low, high).

    A placeholder aggregate until the MRI calibration (plan §8.4) exists.
    """
    v = np.asarray(apparent_cm, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    m = np.isfinite(v) & (w > 0)
    if not m.any():
        return (np.nan, np.nan, np.nan)
    order = np.argsort(v[m])
    vs, ws = v[m][order], w[m][order]
    cw = np.cumsum(ws) / ws.sum()
    q = lambda p: float(vs[np.searchsorted(cw, p)])  # noqa: E731
    return q(0.5), q(0.025), q(0.975)
