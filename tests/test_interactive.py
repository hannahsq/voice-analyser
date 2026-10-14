"""Interactive glottal source (synth.interactive, D-026)."""

import numpy as np
import pytest

from vocal_analysis.synth import interactive as it
from vocal_analysis.synth import tube
from vocal_analysis.synth.coupling import Glottis

FS = 16000


@pytest.fixture(scope="module")
def shape():
    return tube.build_shape(*tube.STORY_VOWELS["A"], 17.5)


@pytest.fixture(scope="module")
def resp(shape):
    return it.tract_responses(shape, FS)


def _area(n, f0=120.0, oq=0.6, gap=0.0, peak=0.12):
    return peak * it.area_pulse(np.arange(n) * f0 / FS, oq, 1.3, gap)


def _render(area, resp, g, coupled=True):
    nb = len(area) // 16 + 1
    return it.render(area, [resp], np.zeros(nb, int), np.zeros(nb), g, coupled=coupled)


def test_impulse_responses_match_the_tube(shape, resp):
    f = np.fft.rfftfreq(4096, 1 / FS)
    zf, hf = it.tract_frequency_responses(shape, f)
    band = (f > 50) & (f < 7000)
    ht = np.fft.rfft(resp.h, 4096)
    assert np.max(np.abs(20 * np.log10(np.abs(ht[band]) / np.abs(hf[band])))) < 0.1
    zt = np.fft.rfft(resp.z, 4096)
    assert np.max(np.abs(zt[band] - zf[band])) < 0.01 * np.abs(zf[band]).max()
    # the transfer is the one the labels come from: 1 at DC, peaks at the tube's poles
    F, _ = tube.poles(shape)
    assert abs(np.abs(hf[1]) - 1.0) < 0.05
    assert abs(f[band][np.argmax(np.abs(hf[band]) * (f[band] < 1000))] - F[0]) < 2 * (f[1] - f[0])


def test_uncoupled_flow_is_bernoulli_through_the_area(resp):
    g = Glottis(0.12, 0.0, 1.4, 1e-4, 8000.0, 1.0)  # a thin glottis: inertance and viscosity negligible
    area = _area(FS // 4)
    u = _render(area, resp, g, coupled=False)["flow"]
    assert np.allclose(u, area * np.sqrt(2 * 8000.0 / it.RHO), rtol=0.02, atol=1.0)


def test_interaction_lowers_and_skews_the_flow(resp):
    g = Glottis(0.12, 0.0, 1.4, 0.3, 8000.0, 1.0)
    area = _area(FS)
    per = FS // 120
    out = {c: _render(area, resp, g, coupled=c)["flow"][FS // 2:] for c in (False, True)}
    a = area[FS // 2:]
    lag = {c: np.median([np.argmax(u[k * per:(k + 1) * per]) - np.argmax(a[k * per:(k + 1) * per]) for k in range(20)])
           for c, u in out.items()}
    assert out[True].max() < 0.9 * out[False].max()  # the tract's inertive load limits the flow
    assert lag[True] > lag[False] + 5  # and delays its peak (skewing; about 0.8 ms here)
    # formant ripple: the coupled flow's open phase carries energy near F1 that the uncoupled one lacks
    spec = {c: np.abs(np.fft.rfft(np.diff(u) * np.hanning(len(u) - 1))) for c, u in out.items()}
    f = np.fft.rfftfreq(len(out[True]) - 1, 1 / FS)
    band = (f > 600) & (f < 1000)
    assert spec[True][band].sum() / spec[True].sum() > spec[False][band].sum() / spec[False].sum()


def test_closed_glottis_is_silent_and_rendering_is_finite(resp):
    g = Glottis(0.12, 0.0, 1.4, 0.3, 8000.0, 1.0)
    assert not _render(np.zeros(4000), resp, g)["pressure"].any()
    o = _render(_area(FS // 2, f0=300.0, oq=0.9, gap=0.6, peak=0.3), resp, g)
    assert np.isfinite(o["pressure"]).all() and np.isfinite(o["flow"]).all()


def test_render_follows_targets(shape, resp):
    """Two targets: block weights move the output from one tract's response to the other's."""
    other = it.tract_responses(tube.build_shape(*tube.STORY_VOWELS["i"], 17.5), FS)
    g = Glottis(0.12, 0.0, 1.4, 0.3, 8000.0, 1.0)
    area = _area(FS)
    nb = FS // 16 + 1
    a = it.render(area, [resp, other], np.zeros(nb, int), np.zeros(nb), g)["pressure"]
    b = it.render(area, [resp, other], np.zeros(nb, int), np.ones(nb), g)["pressure"]
    c = it.render(area, [other], np.zeros(nb, int), np.zeros(nb), g)["pressure"]
    assert not np.allclose(a, b) and np.allclose(b, c)


@pytest.mark.parametrize("gap", [0.0, 0.25, 0.43])
def test_area_pulse_source_is_lf_like_with_a_gap(resp, gap):
    """No spectral nulls and an LF-like roll-off at any gap (a smooth sin^2 pulse had both problems), and H1-H2
    rises with the gap."""
    g = Glottis(0.073, 0.0, 1.2, 0.26, 8000.0, 1.0)
    f0, per = 210.0, FS / 210.0

    def h(gap):
        u = _render(_area(FS, f0=f0, oq=0.65, gap=gap, peak=0.073), resp, g, coupled=False)["flow"]
        seg = np.diff(u)[FS // 2:FS // 2 + int(per * 20)]
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1 / FS)
        return np.array([20 * np.log10(X[np.argmin(np.abs(f - k * f0))]) for k in range(1, 11)])
    x = h(gap) - h(gap)[0]
    assert x[1:].min() > -35 and -14 < (x[9] - x[1]) / np.log2(5) < -5
    if gap > 0:
        assert -x[1] > -(h(0.0) - h(0.0)[0])[1] + 1.5


def test_sampler_interactive_clips():
    from vocal_analysis.synth.sampler import SynthConfig, generate

    cfg = SynthConfig(p_tube=1.0, p_interactive=1.0, p_subglottal=0.0, duration_s=(1.0, 1.0))
    a, b = generate(3, cfg, 5), generate(3, cfg, 5)
    assert np.array_equal(a.audio, b.audio) and np.isfinite(a.audio).all()
    m = a.meta["tube"]
    assert m["interactive"] and m["glottis"] is not None and "open_df" not in m["targets"][0]
    # at the start (before any transition) the labels are the first target's closed-glottis tube poles,
    # with the tube's own bandwidths
    spk = m["speaker"]
    t0 = m["targets"][0]
    shape = tube.build_shape(t0["q1"], t0["q2"], spk["length_cm"], pharynx_ratio=spk["pharynx_ratio"],
                             area_scale=spk["area_scale"], omega_warp=np.array(spk["omega_warp"]), constrict=t0.get("constrict"),
                             pg=t0["pg_cm"], pm=t0["pm_cm"], perimeter_factor=spk["perimeter_factor"])
    F, B = tube.poles(shape)
    assert np.allclose(a.labels["formants"][0, :3], F[:3], rtol=1e-6)
    assert np.allclose(a.labels["bandwidths"][0, :3], B[:3], rtol=1e-6)
    assert np.isnan(a.labels["source_h_db"]).all()
    # the control arm: same draws and labels, no feedback, different audio
    c = generate(3, SynthConfig(p_tube=1.0, p_interactive=1.0, p_subglottal=0.0, duration_s=(1.0, 1.0),
                                interactive_coupled=False), 5)
    assert np.array_equal(c.labels["formants"], a.labels["formants"], equal_nan=True) and not np.allclose(c.audio, a.audio)
    # off by default: no interactive clips and no extra draws
    assert not generate(3, SynthConfig(p_tube=1.0, duration_s=(1.0, 1.0)), 5).meta["tube"]["interactive"]


def test_lf_area_pulse_gives_lf_flow_without_feedback():
    from vocal_analysis.synth.coupling import Glottis

    fs, f0 = 16000, 125.0
    g = Glottis(0.12, -0.1, 1.4, 0.3, 8000.0, 1.0)
    shape = tube.build_shape(*tube.STORY_VOWELS[list(tube.STORY_VOWELS)[0]], 17.5)
    resp = it.tract_responses(shape, fs, sub_scale=1.0)
    cyc = np.arange(fs // 2) * f0 / fs
    nb = len(cyc) // 16 + 1
    grid, tab = it.lf_flow_table()
    for rd in (0.6, 1.2, 2.2):
        area = g.peak_area_cm2 * it.lf_area_pulse(cyc, rd, g)
        assert 0.0 <= area.min() and area.max() <= g.peak_area_cm2 + 1e-12
        u = it.render(area, [resp], np.zeros(nb, int), np.zeros(nb), g, coupled=False)["flow"][fs // 4:]
        target = np.interp((np.arange(len(u)) + fs // 4) * f0 / fs % 1.0 * 2048, np.arange(2048), tab[np.argmin(abs(grid - rd))])
        assert np.corrcoef(u, target)[0, 1] > 0.99


def test_sampler_lf_pulse_option_crosses_pulse_and_feedback():
    from dataclasses import replace

    from vocal_analysis.synth.sampler import SynthConfig, generate

    cfg = replace(SynthConfig(), p_interactive=1.0, p_interactive_coupled=0.5, p_lf_pulse=0.5)
    seen = set()
    for i in range(40):
        t = generate(i, cfg, 3).meta["tube"]
        if t and t["interactive"]:
            seen.add((t["pulse"], t["feedback"]))
    assert seen == {("lf", True), ("lf", False), ("rosenberg", True), ("rosenberg", False)}


def test_closure_sharpness():
    cyc = np.linspace(0.0, 3.0, 3000, endpoint=False)
    fr = cyc % 1.0
    te, sq = 0.6, 1.3
    tp = te * sq / (1 + sq)
    tn = te - tp
    rosenberg = np.where(fr < tp, 0.5 * (1 - np.cos(np.pi * fr / tp)), np.where(fr < te, np.cos(np.pi * (fr - tp) / (2 * tn)), 0.0))
    assert np.allclose(it.area_pulse(cyc, te, sq, 0.0, 1.0), rosenberg)
    sharp = it.area_pulse(cyc, te, sq, 0.0, 3.0)
    assert np.all((sharp >= 0) & (sharp <= 1)) and np.allclose(sharp[fr < tp], rosenberg[fr < tp])
    assert np.all(sharp[fr >= te] == 0.0)

    def hf_share(a):  # the area derivative's harmonics 15-60 against 1-5 (3 cycles: harmonic k at bin 3k)
        s = np.abs(np.fft.rfft(np.diff(a))) ** 2
        return s[45:181:3].sum() / s[3:16:3].sum()

    assert hf_share(sharp) > 2.0 * hf_share(rosenberg)


def test_sampler_closure_option():
    from dataclasses import replace

    from vocal_analysis.synth.sampler import SynthConfig, generate

    base = replace(SynthConfig(), p_interactive=1.0, p_interactive_coupled=0.5)
    a = generate(4, base, 2)
    assert a.meta["tube"] is None or not a.meta["tube"]["interactive"] or a.meta["tube"]["closure"] == 1.0
    vals = [generate(i, replace(base, interactive_closure=(1.0, 3.0)), 2).meta["tube"] for i in range(12)]
    vals = [t["closure"] for t in vals if t and t["interactive"]]
    assert vals and all(1.0 <= v <= 3.0 for v in vals) and len(set(vals)) == len(vals)
