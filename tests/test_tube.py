"""Lossy-tube tract (Story 2005 area function) against exact and published answers (plan §0, §5.1)."""

import numpy as np
import pytest

from vocal_analysis.synth import filters as F
from vocal_analysis.synth import tube as T
from vocal_analysis.synth.sampler import SynthConfig, generate

C = T.C_CM_S


def _uniform(length=17.5, area=3.0, n=44):
    return T.TubeShape(np.full(n, area), np.full(n, length / n))


def _lossless_poles(shape):
    """Roots of the lossless, open-ended tube (loss = 0), refined by bisection on the real axis."""
    f = np.arange(1.0, 6000.0, 0.5)
    d = T._denominator(2j * np.pi * f, shape, 0.0).real
    out = []
    for i in np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
        lo, hi = f[i], f[i + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.sign(T._denominator(2j * np.pi * np.array([mid]), shape, 0.0).real[0]) == np.sign(d[i]):
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return np.array(out)


def test_lossless_uniform_tube_is_quarter_wave():
    f = _lossless_poles(_uniform())
    assert np.allclose(f[:5], (2 * np.arange(1, 6) - 1) * C / (4 * 17.5), atol=1e-6)


@pytest.mark.parametrize("a1,a2,l1,l2", [(1.0, 7.0, 8.0, 9.0), (7.0, 1.0, 9.0, 8.0)])
def test_lossless_two_tube_matches_analytic(a1, a2, l1, l2):
    """Closed-open two-tube resonator: cos(kl1) cos(kl2) = (A1 / A2) sin(kl1) sin(kl2)."""
    shape = T.TubeShape(np.r_[np.full(20, a1), np.full(20, a2)], np.r_[np.full(20, l1 / 20), np.full(20, l2 / 20)])
    f = _lossless_poles(shape)
    k = 2 * np.pi * f / C
    assert len(f) >= 4
    assert np.allclose(np.cos(k * l1) * np.cos(k * l2) - a1 / a2 * np.sin(k * l1) * np.sin(k * l2), 0.0, atol=1e-9)


def test_poles_are_roots_and_describe_the_response():
    """Exact roots of the denominator; near each pole |H| peaks at ~F with a -3 dB width of ~B."""
    shape = T.build_shape(-2.3, 1.4, 17.0)
    f, b = T.poles(shape)
    s = -np.pi * b + 2j * np.pi * f
    assert np.all(np.abs(T._denominator(s, shape)) < 1e-9)
    assert np.all(np.diff(f) > 0) and np.all(b > 0)
    for fk, bk in zip(f[:3], b[:3]):
        grid = np.linspace(fk - 3 * bk, fk + 3 * bk, 4001)
        mag = np.abs(T.transfer(grid, shape))
        peak = grid[np.argmax(mag)]
        above = grid[mag >= mag.max() / np.sqrt(2)]
        assert abs(peak - fk) < 0.1 * bk
        assert above[-1] - above[0] == pytest.approx(bk, rel=0.15)


def test_dc_gain_is_one():
    assert abs(T.transfer(np.array([0.0]), T.build_shape(1.0, -1.0, 15.0))[0]) == pytest.approx(1.0, abs=1e-6)


def test_losses_raise_f1_and_give_plausible_bandwidths():
    """Yielding walls raise F1 of a uniform tube above c / 4L; bandwidths are tens of Hz (Fant 1972)."""
    f, b = T.poles(_uniform())
    assert 500.0 < f[0] < 560.0
    assert np.all((b[:4] > 15) & (b[:4] < 150))


def test_story_vowels_erratum():
    """Story's corrected Table IV vowels (2009 erratum) through this tube: the vowel space's shape.
    Relations, not norms, since the speaker is Story's: front vowels have high F2 and F1 rising with
    openness, back rounded vowels low F2 and small lips, /A/ the highest F1."""
    f = {v: T.poles(T.build_shape(*q, T.STORY_LENGTH_CM))[0][:2] for v, q in T.STORY_VOWELS.items()}
    lips = {v: T.vowel_areas(*q)[-1] for v, q in T.STORY_VOWELS.items()}
    assert f["i"][0] < f["I"][0] < f["E"][0] < f["ae"][0]
    assert f["i"][1] > f["I"][1] > f["E"][1] > 1700
    assert all(f[v][1] < 1350 for v in ("V", "A", "O", "U", "o", "u"))
    assert max(f, key=lambda v: f[v][0]) == "A"
    assert f["i"][0] < 350 and f["u"][0] < 400 and f["i"][1] > 2200
    assert all(lips[v] < 0.5 for v in ("U", "o", "u")) and lips["i"] > 1.0


# Story (2005) Fig. 10: neutral point and the ends of the [IU] and [Au] trajectories, read off the figure.
FIG10 = [((0.0, 0.0), (610, 1470)), ((-2.3, 1.4), (500, 2000)), ((1.8, -2.0), (500, 1000)),
         ((3.8, 0.6), (750, 1200)), ((-3.2, -2.15), (300, 1000))]


@pytest.mark.parametrize("q,f12", FIG10)
def test_story_fig10_points(q, f12):
    f, _ = T.poles(T.build_shape(*q, T.STORY_LENGTH_CM))
    assert f[0] == pytest.approx(f12[0], rel=0.17)
    assert f[1] == pytest.approx(f12[1], rel=0.10)


def test_story_fig10_mesh_extent():
    """The q-plane box covers the F1-F2 region of Fig. 10(b): F1 0.2-0.85 kHz, F2 0.55-2.55 kHz."""
    f12 = np.array([T.poles(T.build_shape(a, b, T.STORY_LENGTH_CM))[0][:2]
                    for a in np.linspace(*T.Q1_RANGE, 10) for b in np.linspace(*T.Q2_RANGE, 7)])
    assert 0.18e3 < f12[:, 0].min() < 0.3e3 and 0.75e3 < f12[:, 0].max() < 0.9e3
    assert 0.5e3 < f12[:, 1].min() < 0.7e3 and 2.3e3 < f12[:, 1].max() < 2.7e3


def test_tier_three_adds_exact_length():
    shape = T.build_shape(0.5, -0.5, 16.0, pharynx_ratio=0.8, pg=0.7, pm=1.2)
    assert shape.length_cm == pytest.approx(16.0 + 0.7 + 1.2, abs=1e-9)


def test_constriction_hits_target_area():
    a = T.vowel_areas(0.0, 0.0)
    c = T.constriction(44, a[23], 24, 0.3, 8)
    assert (a * c)[23] == pytest.approx(0.3)
    assert c[:5] == pytest.approx(1.0, abs=1e-6) and c[-3:] == pytest.approx(1.0, abs=1e-3)


def test_longer_tract_lowers_formants():
    f15, _ = T.poles(T.build_shape(0.0, 0.0, 15.0))
    f18, _ = T.poles(T.build_shape(0.0, 0.0, 18.0))
    assert np.all(f18[:4] < f15[:4])


TUBE = SynthConfig(p_tube=1.0, p_register_step=0)


@pytest.mark.parametrize("index", range(3))
def test_tube_clip_labels_are_its_poles(index):
    """A tube clip's frame labels are what was synthesised: ascending, below F_MAX, and its target poles."""
    clip = generate(index, TUBE, seed=5)
    assert clip.meta["tract"] == "tube"
    f = clip.labels["formants"]
    ok = np.isfinite(f[:, :4]).all(1)
    assert ok.all() and np.all(np.diff(f[:, :4], axis=1) > 0)
    first = clip.meta["tube"]["targets"][0]
    spk = clip.meta["tube"]["speaker"]
    shape = T.build_shape(first["q1"], first["q2"], spk["length_cm"], pharynx_ratio=spk["pharynx_ratio"],
                          area_scale=spk["area_scale"], omega_warp=np.array(spk["omega_warp"]),
                          constrict=first.get("constrict"), pg=first["pg_cm"], pm=first["pm_cm"],
                          perimeter_factor=spk["perimeter_factor"])
    pf, pb = T.poles(shape)
    k = np.isfinite(f[0])
    assert np.allclose(f[0][k], pf[: k.sum()], rtol=1e-9)
    assert np.allclose(clip.labels["bandwidths"][0][k], pb[: k.sum()] * spk["bw_scale"], rtol=1e-9)


def test_tube_cascade_response_tracks_tube_below_4khz():
    """The cascade built from the tube's poles (what is synthesised) stays close to the tube's own response
    for an open vowel; the residual is the frequency dependence of the losses."""
    shape = T.build_shape(1.0, 0.5, 17.0)
    f, b = T.poles(shape)
    grid = np.arange(60.0, 4000.0, 5.0)
    keep = f < 7600
    d = 20 * np.log10(np.abs(T.transfer(grid, shape)) / np.abs(F.response(grid, f[keep], b[keep], 16000)))
    assert np.max(np.abs(d)) < 4.0


# A shape found in sampling where a heavily damped root passes a sharp one in frequency as the losses ramp up
# (the old "frequencies stay ascending" check rejected it as a lost root).
CROSSING = dict(q1=1.689, q2=1.35, length_cm=19.837, pharynx_ratio=1.101, area_scale=1.037, pg=0.456, pm=0.0,
                perimeter_factor=1.076, constrict={"lc": 29.22, "rc": 7.494, "sc": 1.736, "ac": 0.159},
                omega_warp=np.array([
                    1.169, 1.164, 1.154, 1.139, 1.121, 1.101, 1.078, 1.054, 1.03, 1.007, 0.986, 0.966, 0.95, 0.937, 0.927,
                    0.92, 0.918, 0.918, 0.923, 0.93, 0.94, 0.952, 0.965, 0.979, 0.994, 1.008, 1.02, 1.03, 1.038, 1.042,
                    1.043, 1.04, 1.035, 1.026, 1.015, 1.003, 0.989, 0.975, 0.961, 0.949, 0.938, 0.93, 0.924, 0.921]))


def test_poles_survive_a_frequency_crossing():
    k = dict(CROSSING)
    shape = T.build_shape(k.pop("q1"), k.pop("q2"), k.pop("length_cm"), **k)
    f, b = T.poles(shape, f_max_hz=20000.0)  # keep every root, so none is cut at the band edge
    s = -np.pi * b + 2j * np.pi * f
    assert np.all(np.abs(T._denominator(s, shape)) < 1e-9)
    assert np.all(np.diff(f) >= 0) and np.all(b > 0)
    grid = np.arange(T.GRID_HZ, 20000.0 + 400.0, T.GRID_HZ)  # the seed grid poles() searches for this f_max
    d0 = T._denominator(2j * np.pi * grid, shape, 0.0).real
    assert len(f) == np.count_nonzero(np.sign(d0[:-1]) * np.sign(d0[1:]) < 0)  # one root per lossless root
    assert b.max() > 200  # the damped root is there
