"""Back end against exact answers on synthetic data (plan §8, M1)."""

import numpy as np
import pytest

from vocal_analysis.backend import corrections, harmonics, slope, vtl
from vocal_analysis.synth import filters as F
from vocal_analysis.synth import glottal as G
from vocal_analysis.synth.sampler import SynthConfig, generate
from vocal_analysis.synth.voice import Trajectories, synthesize

FS = 16000
PF = np.array([550.0, 1650.0, 2600.0, 3500.0, 4400.0, 5300.0])
PB = np.array([70.0, 100.0, 130.0, 180.0, 250.0, 300.0])


def stationary(f0, rd=1.2, c=601):
    tr = Trajectories(f0=np.full(c, f0), rd=np.full(c, rd), voice_amp=np.ones(c), whisper_amp=np.zeros(c),
                      fric_amp=np.zeros(c), poles_f=np.tile(PF, (c, 1)), poles_b=np.tile(PB, (c, 1)),
                      zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
    return synthesize(tr, np.random.default_rng(0))["voiced"]


@pytest.mark.parametrize("f0", [87.3, 140.0, 221.7, 600.0, 1180.0])
def test_harmonic_fit_exact_on_clean_frames(f0):
    """Plan M1: harmonic fit < 0.1 dB on clean frames (any f0, non-integer periods)."""
    x = stationary(f0)
    k = 8
    got = harmonics.harmonic_amplitudes(x, FS, np.array([0.3]), np.array([f0]), k_max=k)[0]
    kk = np.arange(1, k + 1)
    exact = 20 * np.log10(np.abs(G.source_coefficients(np.array(1.2), k) * F.response(kk * f0, PF, PB, FS)))
    ok = kk * f0 < 7400
    assert np.max(np.abs(got[ok] - exact[ok])) < 0.1


def test_causal_fit_matches_centred_on_stationary_signal():
    x = stationary(200.0)
    a = harmonics.harmonic_amplitudes(x, FS, np.array([0.3]), np.array([200.0]), 6)
    b = harmonics.harmonic_amplitudes(x, FS, np.array([0.3]), np.array([200.0]), 6, causal=True)
    assert np.allclose(a, b, atol=1e-6)  # every harmonic is modelled, so the window position doesn't matter


@pytest.mark.parametrize("rd", [0.4, 1.0, 1.8, 2.6])
@pytest.mark.parametrize("f0", [110.0, 250.0, 520.0])
def test_corrected_h1h2_equals_source(rd, f0):
    """Plan M1: corrected H1-H2 matches source H1-H2 when the true F/B are supplied (< 0.5 dB)."""
    x = stationary(f0, rd)
    h = harmonics.harmonic_amplitudes(x, FS, np.array([0.3]), np.array([f0]), k_max=4)
    hc = corrections.correct(h, np.array([f0]), PF[None], PB[None], FS)
    src = G.source_h_db(np.array(rd), 2)
    assert abs((hc[0, 0] - hc[0, 1]) - (src[0] - src[1])) < 0.5
    # With F1-F4 only (as the model outputs), poles above F4 still boost H2 relative to H1 at high
    # f0; adding them back from the apparent VTL removes most of that bias.
    err4 = lambda **kw: abs(np.diff(corrections.correct(h, np.array([f0]), PF[None, :4], PB[None, :4], FS, **kw)[0, 1::-1])[0]  # noqa: E731
                            - (src[0] - src[1]))
    assert err4(upper_from_vtl=True) < 0.5
    assert err4(upper_from_vtl=True) <= err4() + 1e-9


def test_correction_on_generated_clip_with_labels():
    cfg = SynthConfig(duration_s=(1.0, 1.0), contour_probs_low={"flat": 1}, contour_probs_high={"flat": 1}, p_register_step=0,
                      rd_wander_sd=(0, 0), loudness_sd_db=(0, 0), jitter=(0, 0), shimmer_db=(0, 0), hnr_db=None,
                      voiced_seg_s=(5, 5), target_every_s=(5, 5), p_nasal=0, p_subglottal=0, p_high_f0=0)
    for idx in range(3):
        clip = generate(idx, cfg, seed=11)
        lab = clip.labels
        seg = next(s for s in clip.meta["segments"] if s["kind"] == "voiced")
        fi = int((seg["start_s"] + 0.3) / 0.01)
        t, f0 = lab["t"][fi : fi + 1], lab["f0"][fi : fi + 1]
        h = harmonics.harmonic_amplitudes(clip.audio.astype(float), FS, t, f0, k_max=4)
        hc = corrections.correct(h, f0, lab["formants"][fi : fi + 1], lab["bandwidths"][fi : fi + 1], FS)
        src = lab["source_h_db"][fi]
        assert abs((hc[0, 0] - hc[0, 1]) - (src[0] - src[1])) < 0.5


def test_hawks_miller_fallback_when_bandwidths_missing():
    f0 = np.array([150.0])
    g_none = corrections.formant_gain_db(np.array([[300.0]]), PF[None, :3], None, f0, FS)
    g_hm = corrections.formant_gain_db(np.array([[300.0]]), PF[None, :3],
                                       F.hawks_miller_bandwidth(PF[None, :3], 150.0), f0, FS)
    assert np.allclose(g_none, g_hm)


def test_vtl_exact_for_uniform_tube():
    for length in (11.0, 14.5, 17.5, 19.0):
        fn = (2 * np.arange(1, 5) - 1) * 35000 / (4 * length)
        assert vtl.apparent_vtl(fn[None])[0] == pytest.approx(length)
        sig = np.array([[200.0, 100.0, 50.0, 50.0]])
        assert vtl.apparent_vtl(fn[None], sig)[0] == pytest.approx(length)
    assert np.isnan(vtl.apparent_vtl(np.full((1, 4), np.nan))[0])


def test_regression_slope_recovers_known_tilt():
    f0 = np.array([120.0])
    k = np.arange(1, 41)
    h = (-9.0 * np.log2(k * 120.0))[None]
    assert slope.regression_slope(h, f0)[0] == pytest.approx(-9.0)


def test_kreiman_garellek_picks_nearest_harmonics():
    f0 = np.array([200.0])
    h = np.arange(40, dtype=float)[None]  # H_k = k - 1
    m = slope.kreiman_garellek(h, h, f0)
    assert m["H1H2c"][0] == -1 and m["H2H4c"][0] == -2
    assert m["H4H2kc"][0] == 3 - 9  # 2 kHz = harmonic 10
    assert m["H2kH5k"][0] == 9 - 24  # 5 kHz = harmonic 25


def test_amplitude_near_formant():
    h = np.array([[0.0, 5.0, 9.0, 4.0, -3.0]])
    assert harmonics.amplitude_near(h, np.array([200.0]), np.array([590.0]))[0] == 9.0
