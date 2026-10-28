"""QCP-FB (D-046): the weight function, the weighted forward-backward fit and SEDREAMS, on synthetic signals."""

import numpy as np
import pytest
from scipy.signal import lfilter

from vocal_analysis.eval import qcp

FS = 8000
POLES = ((500.0, 60.0), (1500.0, 90.0), (2500.0, 120.0), (3500.0, 150.0))


def _a(poles=POLES, fs=FS):
    a = np.array([1.0])
    for f, b in poles:
        r = np.exp(-np.pi * b / fs)
        a = np.convolve(a, [1.0, -2 * r * np.cos(2 * np.pi * f / fs), r * r])
    return a


def _voice(f0=150.0, seconds=0.5, fs=FS, seed=0, poles=POLES, open_noise=0.3):
    """An all-pole tract excited by a negative impulse at each closure and noise through the open phase (the last
    40 % of each cycle); returns (speech, true GCIs)."""
    rng = np.random.default_rng(seed)
    n, T = int(seconds * fs), fs / f0
    gci = np.round(np.arange(T, n - 2 * T, T)).astype(int)
    e = np.zeros(n)
    for g in gci:
        o = g + int(0.6 * T)
        nxt = g + int(T)
        e[g] = -8.0
        e[o:nxt] += open_noise * rng.normal(size=nxt - o)
    return lfilter([1.0], _a(poles, fs), e), gci


def test_weights_match_the_authors_construction():
    # One cycle of 100 samples at 8 kHz: 1 from 5 to 75 (PQ 0.05, DQ 0.7), 7-sample ramps inside each end.
    w = qcp.qcp_weights(250, np.array([0, 100, 200]), 8000.0)
    up = np.linspace(qcp.W_MIN, 1.0, 9)[1:-1]
    assert np.allclose(w[5:12], up)
    assert np.allclose(w[12:68], 1.0)
    assert np.allclose(w[68:75], up[::-1])
    assert np.allclose(w[75:105], qcp.W_MIN)
    assert np.allclose(w[105:112], up)


def test_order_scales_with_rate():
    assert [qcp.default_order(f) for f in (8000, 10000, 16000)] == [13, 15, 21]


@pytest.mark.parametrize("forward_only", [False, True])
def test_fb_wlp_recovers_known_poles_on_free_decay(forward_only):
    # Between two excitations the signal is the tract's free decay, which covariance LP fits exactly. Frequencies
    # only: forward-backward prediction mirrors poles across the unit circle, biasing bandwidths on a decay.
    x, gci = _voice(open_noise=0.0)
    seg = x[gci[4] + 1:gci[5] - 1]
    a = qcp.fb_wlp(seg, np.ones(len(seg)), 8, forward_only)
    f, _ = qcp.lpc_formants(a, FS)
    assert np.allclose(f[:4], [p[0] for p in POLES], rtol=0.01)


def test_qcp_weighting_gives_the_four_formants_despite_open_phase_noise():
    # Noise through the open phase: with the QCP weight (ideal GCIs) the fit returns the four poles and nothing else.
    x, gci = _voice(f0=220.0, open_noise=1.0, seed=3)
    xp = np.r_[x[0], x[1:] - qcp.PREEMPH * x[:-1]]
    seg = slice(1500, 1740)
    w = qcp.qcp_weights(len(x), gci, FS)
    f = qcp.lpc_formants(qcp.fb_wlp(xp[seg], w[seg], 13), FS)[0]
    assert len(f) == 4
    assert np.allclose(f, [p[0] for p in POLES], rtol=0.02)


@pytest.mark.parametrize("f0", [100.0, 200.0, 300.0])
def test_sedreams_finds_the_closures(f0):
    x, gci = _voice(f0=f0, seconds=1.0, seed=1)
    est = qcp.sedreams(x, FS, f0)
    tol = int(round(0.0005 * FS)) + 1
    hit = np.mean([np.min(np.abs(est - g)) <= tol for g in gci[2:-2]])
    assert hit >= 0.95


def test_analyse_with_ideal_gcis_reads_the_poles():
    x, gci = _voice(f0=200.0, seconds=0.6, open_noise=0.1)
    r = qcp.analyse(x, FS, np.array([0.2, 0.3, 0.4]), gci=gci)
    assert np.all(np.abs(np.nanmedian(r["F"][:, :2], 0) - [500.0, 1500.0]) < [15.0, 30.0])
    assert r["order"] == 13
