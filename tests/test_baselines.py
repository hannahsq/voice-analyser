"""Baselines against exact answers on synthetic signals (plan §0)."""

import numpy as np
import pytest

SR = 16000


def harmonic_tone(f0_of_t, seconds=1.0, n_harm=10):
    t = np.arange(int(seconds * SR)) / SR
    f0 = f0_of_t(t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, n_harm + 1) if k * f0.max() < SR / 2)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def resonator_vowel(f0, formants, bws, seconds=0.5):
    """Impulse train through a cascade of 2nd-order resonators (exact poles)."""
    from scipy.signal import lfilter

    n = int(seconds * SR)
    src = np.zeros(n)
    src[:: int(SR / f0)] = 1.0
    y = src
    for f, b in zip(formants, bws):
        r = np.exp(-np.pi * b / SR)
        a = [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r]
        y = lfilter([sum(a)], a, y)
    return (y / np.abs(y).max() * 0.5).astype(np.float32)


@pytest.mark.parametrize("name", ["pesto", "crepe", "praat_ac"])
def test_f0_baseline_step_timing(name):
    pytest.importorskip("parselmouth" if name.startswith("praat") else "torch")
    from vocal_analysis.baselines.f0 import F0_BASELINES

    # A fifth, not an octave: after an octave step the old period still fits, which delays
    # autocorrelation trackers for reasons unrelated to timestamps.
    x = harmonic_tone(lambda t: np.where(t < 0.5, 220.0, 330.0))
    t, f0, _ = F0_BASELINES[name].predict(x, SR)
    cents = lambda est, ref: 1200 * np.log2(est / ref)  # noqa: E731
    steady_lo = (t > 0.1) & (t < 0.4)
    steady_hi = (t > 0.6) & (t < 0.9)
    assert np.median(np.abs(cents(f0[steady_lo], 220.0))) < 20
    assert np.median(np.abs(cents(f0[steady_hi], 330.0))) < 20
    # The step must appear at t = 0.5 s (within one hop) if timestamps are frame centres.
    mid = (t > 0.3) & (t < 0.7)
    step_t = t[mid][np.argmax(f0[mid] > 269.0)]  # 269 Hz = geometric midpoint
    assert abs(step_t - 0.5) <= 0.011, step_t


@pytest.mark.parametrize("name", ["praat_default", "praat_sweep", "praat_fasttrack", "tvlp"])
def test_formant_baseline_on_resonator_vowel(name):
    from vocal_analysis.baselines.formants import FORMANT_BASELINES

    true = np.array([700.0, 1200.0, 2600.0, 3500.0])
    x = resonator_vowel(120.0, true, [80, 90, 120, 150])
    times = np.array([0.15, 0.25, 0.35])
    est = FORMANT_BASELINES[name].predict(x, SR, times, {"start_s": 0.05, "end_s": 0.45, "group": "m"})
    rel = np.abs(est - true) / true
    assert np.nanmedian(rel[:, :3]) < 0.05, est


def test_align_nearest_within_tolerance():
    from vocal_analysis.eval.report import _align

    est_t = np.arange(0, 1.0, 0.01)
    (out,) = _align(np.array([-0.1, 0.0, 0.014, 0.016, 0.994, 1.2]), est_t, est_t * 100, max_dt=0.005)
    np.testing.assert_allclose(out, [np.nan, 0.0, 1.0, 2.0, 99.0, np.nan])
