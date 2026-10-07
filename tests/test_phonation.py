"""Phonation categories on synthetic laryngograph signals with known answers (decisions D-009)."""

import numpy as np

from vocal_analysis.eval import phonation as P

SR = P.SR


def egg(cycle_hz, amps, seconds=1.0):
    """Glottal-cycle waveform at cycle_hz; cycle k has amplitude amps[k % len(amps)]."""
    period = int(round(SR / cycle_hz))
    n = np.arange(period)
    shape = np.sin(2 * np.pi * n / period) + 0.3 * np.sin(4 * np.pi * n / period)
    k = int(seconds * SR) // period
    return np.concatenate([amps[i % len(amps)] * shape for i in range(k)])


def labels(x, ref_hz):
    times = np.arange(0.05, len(x) / SR - 0.05, 0.01)
    f0 = np.full(len(times), float(ref_hz))
    return P.classify(*P.egg_lag_correlations(x, SR, times, f0))


def test_modal():
    assert set(labels(egg(120, [1.0]), 120)) == {"modal"}


def test_reference_halved():
    # Folds cycle at 240 Hz, reference says 120 Hz.
    assert set(labels(egg(240, [1.0]), 120)) == {"ref_error_2x"}


def test_alternating_cycles():
    # Cycles at 240 Hz alternate strong/weak, so the waveform repeats at 120 Hz.
    assert set(labels(egg(240, [1.0, 0.2]), 120)) == {"alternating"}


def test_aperiodic():
    x = np.random.default_rng(0).standard_normal(SR)
    assert set(labels(x, 120)) == {"aperiodic"}


def test_weak_egg_relative_to_file():
    x = egg(120, [1.0])
    x[SR // 2 :] *= 0.02
    lab = labels(x, 120)
    assert lab[0] == "modal" and lab[-1] == "weak_egg"


def test_unvoiced_frames_unlabelled():
    x = egg(120, [1.0])
    times = np.array([0.1, 0.2, 0.3])
    lab = P.classify(*P.egg_lag_correlations(x, SR, times, np.array([120.0, 0.0, 120.0])))
    assert lab.tolist() == ["modal", "", "modal"]
