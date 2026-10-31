import numpy as np
import pytest

from vocal_analysis.synth import kl, tube


def _lossless_ir(n: int, m: int = 1 << 14) -> np.ndarray:
    """Impulse response of a lossless uniform tube (3 cm^2), rigid glottis, ideal open lips (reflection -1)."""
    areas, loss, lip = np.full((1, n), 3.0), np.ones((1, n)), np.array([[-1.0, 0.0, 0.0]])
    u, _ = kl._run(areas, loss, lip, m, np.zeros(m), 0.0, 0.0, np.zeros(4), 1.0 / (kl.FS * kl.OVERSAMPLE),
                   kl.section_length(), True)
    return u


def test_section_length_gives_quarter_wave_resonances():
    """One section per sample of delay: the original's 2 L fs / c sections made the tube twice as long."""
    n = 64
    u = _lossless_ir(n, 1 << 16)
    f = np.fft.rfftfreq(len(u), 1.0 / (kl.FS * kl.OVERSAMPLE))
    db = 20 * np.log10(np.abs(np.fft.rfft(u)) + 1e-30)
    from scipy.signal import find_peaks

    peaks = f[find_peaks(db[f < 4000], prominence=3)[0]]
    want = [(2 * k - 1) * tube.C_CM_S / (4 * n * kl.section_length()) for k in range(1, 5)]
    assert np.allclose(peaks[:4], want, atol=2.0)


def test_closed_glottis_reflects_fully():
    """GL-3 corrected (p+ = rho c Ug / A0 + p-): with the glottis closed and no losses, nothing decays. The original's
    halved boundary lost half the wave at every return."""
    u = _lossless_ir(32)
    q = len(u) // 4
    first, last = np.sqrt(np.mean(u[:q] ** 2)), np.sqrt(np.mean(u[-q:] ** 2))
    assert last / first > 0.999


def test_lip_filter_is_an_open_end_at_dc():
    b0, b1, a1 = kl._lip_filter(4.0, kl.FS * kl.OVERSAMPLE)
    assert abs((b0 + b1) / (1 + a1) + 1.0) < 1e-12


def test_uniform_areas_keep_volume_and_length():
    s = tube.build_shape(*tube.STORY_VOWELS["A"], 16.0, pg=0.4, pm=0.6)  # unequal tubelets (Tier III)
    a = kl.uniform_areas(s)
    dx = kl.section_length()
    assert abs(len(a) * dx - s.length_cm) <= dx / 2 + 1e-9
    assert abs(a.sum() * (s.length_cm / len(a)) - np.sum(s.areas * s.lengths)) < 1e-6 * np.sum(s.areas * s.lengths)


@pytest.mark.parametrize("vowel", ["i", "A", "u", "E"])
def test_frame_poles_match_the_continuous_tube(vowel):
    """The waveguide's resonances, measured by vector fitting, against tube.poles of the same discretised tube: F1-F4
    within 2 % (the discretisation and the frequency-independent viscous loss differ; bandwidths are not compared).
    /i/'s close F2 / F3 pair must come out as two poles."""
    s = tube.build_shape(*tube.STORY_VOWELS[vowel], 17.5)
    a = kl.uniform_areas(s)
    ft, _ = tube.poles(tube.TubeShape(a, np.full(len(a), kl.section_length())), 5000.0)
    (fk, bk), = kl.frame_poles([s])
    assert len(fk) >= 4 and np.all(bk[:4] > 0)
    assert np.all(np.abs(fk[:4] / ft[:4] - 1) < 0.02)


def test_render_is_finite_and_the_load_lowers_the_flow():
    n = 4000
    frames = n // kl.HOP + 1
    shapes = [tube.build_shape(*tube.STORY_VOWELS["A"], 17.5)] * frames
    ag = kl.glottal_area(np.full(frames, 120.0), n, kl.FS, 0.12)
    on = kl.render(shapes, ag, 8000.0, 0.3, 1.0)
    off = kl.render(shapes, ag, 8000.0, 0.3, 1.0, coupled=False)
    assert np.isfinite(on["audio"]).all() and len(on["audio"]) == n
    assert 0 < on["ug"].max() < off["ug"].max()
