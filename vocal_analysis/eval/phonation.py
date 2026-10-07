"""Phonation categories from a laryngograph (EGG) signal, for hard-mode f0 evaluation (decisions D-009).

For each voiced reference frame with period T = 1/f0_ref, correlate the high-passed EGG
with itself shifted by T (r1) and by T/2 (rh), over a 2T window centred on the frame:

- modal:        repeats at T (r1 high) and not at T/2 (rh strongly negative)
- ref_error_2x: repeats at T/2 too, i.e. the folds cycle at 2 x f0_ref: the reference halved
- alternating:  repeats at T but only partly at T/2: alternate cycles differ (period doubling),
                so f0_ref and 2 x f0_ref are both defensible answers
- aperiodic:    repeats at neither (creak, irregular phonation)
- weak_egg:     too little EGG signal to say anything (and the reference, derived from it, is unreliable)

Thresholds are round numbers chosen from the EGG distributions on PTDB-TUG (modal frames have
median r1 0.99 and rh -0.81), not from any estimator's errors, so they can't favour one.
"""

from __future__ import annotations

import numpy as np

SR = 16000
HIGHPASS_HZ = 40.0  # removes slow larynx movement from the EGG
R1_PERIODIC = 0.75
RH_NOT_PERIODIC = -0.5
RH_PERIODIC = 0.75
WEAK_REL_RMS = 0.1  # x the file's median EGG RMS over voiced frames

CATEGORIES = ("modal", "ref_error_2x", "alternating", "aperiodic", "weak_egg")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def egg_lag_correlations(egg: np.ndarray, sr: int, times: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, ...]:
    """(r1, rh, rms) per frame; NaN for unvoiced frames and frames too close to the edges."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    x = np.asarray(egg, float)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    x = sosfiltfilt(butter(4, HIGHPASS_HZ, "highpass", fs=SR, output="sos"), x)
    r1, rh, rms = (np.full(len(f0), np.nan) for _ in range(3))
    for i in np.flatnonzero(np.asarray(f0) > 0):
        period = SR / f0[i]
        c, half_w = int(round(times[i] * SR)), int(round(period))
        lags = int(round(period)), int(round(period / 2))
        if c - half_w - lags[0] // 2 < 0 or c + half_w + lags[0] > len(x):
            continue
        seg = x[c - half_w : c + half_w]
        rms[i] = np.sqrt(np.mean(seg * seg))
        r1[i], rh[i] = (_pearson(x[c - half_w - L // 2 : c + half_w - L // 2], x[c - half_w - L // 2 + L : c + half_w - L // 2 + L]) for L in lags)
    return r1, rh, rms


def classify(r1: np.ndarray, rh: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """Category per frame ('' where r1 is undefined: unvoiced or at the file edge). `rms` is per file."""
    r1, rh, rms = (np.asarray(a, float) for a in (r1, rh, rms))
    ok = np.isfinite(r1) & np.isfinite(rh)
    rel = rms / np.median(rms[ok]) if ok.any() else rms
    out = np.full(len(r1), "", dtype="<U12")
    with np.errstate(invalid="ignore"):
        out[ok & (r1 < R1_PERIODIC) & (rh < RH_PERIODIC)] = "aperiodic"
        out[ok & (r1 >= R1_PERIODIC) & (rh > RH_NOT_PERIODIC)] = "alternating"
        out[ok & (r1 >= R1_PERIODIC) & (rh <= RH_NOT_PERIODIC)] = "modal"
        out[ok & (rh >= RH_PERIODIC)] = "ref_error_2x"
        out[ok & (rel < WEAK_REL_RMS)] = "weak_egg"
    return out
