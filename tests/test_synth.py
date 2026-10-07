"""Synthetic generator against exact answers (plan §0, §5.1)."""

import numpy as np
import pytest
from scipy.signal import freqz, lfilter

from vocal_analysis.synth import filters as F
from vocal_analysis.synth import glottal as G
from vocal_analysis.synth import tract
from vocal_analysis.synth.channel import augment, synth_rir
from vocal_analysis.synth.sampler import SynthConfig, generate
from vocal_analysis.synth.voice import Trajectories, synthesize

FS = 16000


# --------------------------------------------------------------------------- glottal source


@pytest.mark.parametrize("rd", [0.3, 0.8, 1.5, 2.7])
def test_lf_closes_and_peaks_at_te(rd):
    p = G.lf_params(rd)
    w = G.lf_waveform(rd, 16384)
    assert abs(w.mean()) < 1e-5  # flow returns to zero over the period
    assert w[int(round(p.te * 16384))] == pytest.approx(-1.0, abs=2e-3)  # Ee = 1 at te
    assert abs(w[0]) < 1e-9 and abs(w[-1]) < 1e-3  # continuous at the period boundary


def test_source_h1_h2_rises_with_rd():
    rd = np.linspace(0.3, 2.7, 25)
    h = G.source_h_db(rd, 2)
    assert np.all(np.diff(h[:, 0] - h[:, 1]) > 0)
    assert -6 < h[0, 0] - h[0, 1] < 0 and 15 < h[-1, 0] - h[-1, 1] < 25


def test_source_coefficients_interpolate_exactly_on_grid():
    direct = np.fft.rfft(G.lf_waveform(1.23, 8192))[1:9] / 8192
    assert np.allclose(G.source_coefficients(np.array(1.23), 8), direct, rtol=1e-9)


# --------------------------------------------------------------------------- filters


def _cascade_ba(pf, pb):
    b, a = np.array([1.0]), np.array([1.0])
    for f, bw in zip(pf, pb):
        A, B1, B2 = F.coefficients(f, bw, FS)
        b, a = np.convolve(b, [A]), np.convolve(a, [1, -B1, -B2])
    return b, a


def test_response_matches_freqz_and_iseli_alwan():
    pf, pb = np.array([500.0, 1500, 2500]), np.array([60.0, 90, 120])
    f = np.linspace(0, 8000, 101)
    h = F.response(f, pf, pb, FS)
    _, hz = freqz(*_cascade_ba(pf, pb), worN=f, fs=FS)
    assert np.allclose(h, hz, rtol=1e-10)
    assert np.allclose(F.resonator_gain_db(f[:, None], pf, pb, FS).sum(-1), 20 * np.log10(np.abs(h)), atol=1e-9)
    assert abs(h[0]) == pytest.approx(1.0)  # unity DC gain


def test_time_varying_filter_matches_lfilter_and_zero_inverts_pole():
    pf, pb = np.array([500.0, 1500, 2500]), np.array([60.0, 90, 120])
    x = np.random.default_rng(0).standard_normal(3000)
    fr, bw = np.repeat(pf[:, None], len(x), 1), np.repeat(pb[:, None], len(x), 1)
    y = F.filter_tv(x, fr, bw, FS)
    assert np.allclose(y, lfilter(*_cascade_ba(pf, pb), x), atol=1e-9)
    assert np.allclose(F.filter_tv(y, fr, bw, FS, is_zero=[1, 1, 1]), x, atol=1e-9)


def test_uniform_tube_resonances():
    assert np.allclose(tract.tube_resonances(17.5, 3), [500.0, 1500.0, 2500.0])


# --------------------------------------------------------------------------- additive synthesis


@pytest.mark.parametrize("f0", [125.0, 500.0, 1000.0])
def test_harmonics_equal_source_times_tract(f0):
    """Plan §5.1: on clean, stationary settings, harmonic amplitudes = source x |H(k f0)|."""
    c = 801
    pf, pb = np.array([500.0, 1500, 2500, 3500]), np.array([60.0, 90, 120, 150])
    zf, zb = np.array([[900.0]] * c), np.array([[80.0]] * c)
    tr = Trajectories(f0=np.full(c, f0), rd=np.full(c, 1.1), voice_amp=np.ones(c), whisper_amp=np.zeros(c),
                      fric_amp=np.zeros(c), poles_f=np.tile(pf, (c, 1)), poles_b=np.tile(pb, (c, 1)), zeros_f=zf, zeros_b=zb)
    y = synthesize(tr, np.random.default_rng(0))["voiced"]
    n = 1024  # whole number of periods for all three f0
    spec = np.fft.rfft(y[4000 : 4000 + n]) * 2 / n
    k = np.arange(1, 8)
    k = k[k * f0 < 7400]
    meas = np.abs(spec[(k * f0 * n / FS).astype(int)])
    exact = np.abs(G.source_coefficients(np.array(1.1), len(k)) * F.response(k * f0, pf, pb, FS, np.array([900.0]), np.array([80.0])))
    assert np.allclose(20 * np.log10(meas), 20 * np.log10(exact), atol=1e-6)


STEADY = SynthConfig(
    duration_s=(1.0, 1.0), contour_probs_low={"flat": 1}, contour_probs_high={"flat": 1}, p_register_step=0,
    rd_wander_sd=(0, 0), loudness_sd_db=(0, 0), jitter=(0, 0), shimmer_db=(0, 0), hnr_db=None,
    voiced_seg_s=(5, 5), target_every_s=(5, 5), p_nasal=0, p_subglottal=0, low_f0_hz=(100, 200), p_high_f0=0,
)


@pytest.mark.parametrize("index", range(4))
def test_emitted_labels_describe_the_audio(index):
    """Harmonic amplitudes measured from generated audio = gain x source(Rd label) x |H(k f0; F, B labels)|."""
    clip = generate(index, STEADY, seed=7)
    lab = clip.labels
    seg = next(s for s in clip.meta["segments"] if s["kind"] == "voiced")
    f0 = lab["f0"][0]
    period = FS / f0
    n = int(round(8 * period))
    start = int((seg["start_s"] + 0.2) * FS)
    x = clip.audio[start : start + n].astype(float)
    t = np.arange(n) / FS
    k = np.arange(1, 6)
    basis = np.concatenate([np.cos(2 * np.pi * f0 * k[None] * t[:, None]), np.sin(2 * np.pi * f0 * k[None] * t[:, None])], 1)
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    meas = np.hypot(coef[:5], coef[5:])
    fi = int((start + n / 2) / FS / 0.01)
    pf, pb = lab["formants"][fi], lab["bandwidths"][fi]  # every oral pole is labelled (NaN = absent)
    h = F.response(k * f0, pf, pb, FS)
    expect = clip.meta["gain"] * np.abs(G.source_coefficients(np.array(lab["rd"][fi]), 5) * h)
    assert np.allclose(20 * np.log10(meas), 20 * np.log10(expect), atol=0.05)


def test_generate_is_deterministic():
    a, b = generate(11, seed=3), generate(11, seed=3)
    assert np.array_equal(a.audio, b.audio)
    assert all(np.array_equal(a.labels[k], b.labels[k], equal_nan=True) for k in a.labels)
    assert not np.array_equal(a.audio, generate(12, seed=3).audio)


@pytest.mark.parametrize("how", ["f1_near_h1", "f1_near_h2", "f2_f3_close"])
def test_constraints_keep_formants_ordered(how):
    clip = generate(5, SynthConfig(formant_constraint=how, p_register_step=0))
    f = clip.labels["formants"][:, :4]
    assert np.all(np.diff(f, axis=1) > 0)


# --------------------------------------------------------------------------- channel


def test_channel_keeps_timing():
    rng = np.random.default_rng(0)
    rir = synth_rir(rng, 0.5)
    assert np.argmax(np.abs(rir)) == 0
    x = np.zeros(FS)
    x[8000] = 1.0
    for i in range(20):
        y, params = augment(x, np.random.default_rng(i))
        assert np.isfinite(y).all() and len(y) == len(x)
        if not params.get("noise") and not params.get("phone"):
            assert abs(int(np.argmax(np.abs(y))) - 8000) <= 2, params
