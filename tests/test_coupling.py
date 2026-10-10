import numpy as np

from vocal_analysis.synth import coupling, tube
from vocal_analysis.synth.sampler import SynthConfig, generate
from vocal_analysis.synth.voice import CTRL, FS, Trajectories, synthesize


def test_static_opening_matches_barney_et_al():
    """Uniform 17 x 17 mm duct, static 0.51 cm^2 opening, no flow: Barney, De Stefano & Henrich (2007) measured F1 +13 %."""
    duct = tube.TubeShape(np.full(44, 2.89), np.full(44, 17.5 / 44))
    F, B = tube.poles(duct)
    Fo, Bo = coupling.open_poles(duct, 0.51, coupling.Glottis(0.51, 0.0, 1.4, 0.3, 8000.0, 1.0), F, B, flow=False)
    assert 0.10 < Fo[0] / F[0] - 1 < 0.15
    assert Bo[0] > B[0]


def test_open_glottis_raises_f1_and_b1_most_for_close_vowels():
    g = coupling.Glottis(0.1, 0.0, 1.4, 0.3, 8000.0, 1.0)
    shifts = {}
    for v in ("i", "u", "A"):
        shape = tube.build_shape(*tube.STORY_VOWELS[v], 17.5)
        F, B = tube.poles(shape)
        df, bm = coupling.open_shifts(shape, F, B, g)
        assert df[0] > 0 and bm[0] > 1.3
        shifts[v] = df[0]
    assert min(shifts["i"], shifts["u"]) > shifts["A"]  # Birkholz et al. (2019): fR1 < 350 Hz most affected


def test_coupled_path_without_coupling_matches_harmonic_path():
    c = 501
    pf, pb = np.full((c, 8), np.nan), np.full((c, 8), np.nan)
    pf[:, :4], pb[:, :4] = [600, 1300, 2500, 3500], [60, 80, 120, 150]
    tr = Trajectories(f0=np.full(c, 150.0), rd=np.full(c, 1.2), voice_amp=np.ones(c), whisper_amp=np.zeros(c), fric_amp=np.zeros(c),
                      poles_f=pf, poles_b=pb, zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
    h = synthesize(tr, np.random.default_rng(0))["voiced"]
    z = synthesize(tr, np.random.default_rng(0), coupling={"df": np.zeros((c, 4)), "bm": np.ones((c, 4))})["voiced"]
    seg = slice(2000, (c - 1) * CTRL)  # past the time-domain filter's start-up
    assert np.allclose(z[seg], h[seg], atol=1e-3 * np.abs(h[seg]).max())


def test_coupled_clips_are_deterministic_and_keep_closed_glottis_labels():
    cfg = SynthConfig(p_tube=1.0, p_coupling=1.0, duration_s=(1.0, 1.0))
    a, b = generate(2, cfg, 7), generate(2, cfg, 7)
    assert np.array_equal(a.audio, b.audio)
    assert a.meta["tube"]["glottis"] is not None
    t0 = a.meta["tube"]["targets"][0]
    assert len(t0["open_df"]) == coupling.N_COUPLED
    assert 0.0 <= a.meta["tube"]["coupling_scale"] <= 1.0
    # labels are the targets' closed-glottis poles: at the start, before any transition, F1 is the first target's pole
    shape = tube.build_shape(t0["q1"], t0["q2"], a.meta["tube"]["speaker"]["length_cm"],
                             pharynx_ratio=a.meta["tube"]["speaker"]["pharynx_ratio"], area_scale=a.meta["tube"]["speaker"]["area_scale"],
                             omega_warp=np.array(a.meta["tube"]["speaker"]["omega_warp"]), constrict=t0.get("constrict"),
                             pg=t0["pg_cm"], pm=t0["pm_cm"], perimeter_factor=a.meta["tube"]["speaker"]["perimeter_factor"])
    assert np.isclose(a.labels["formants"][0, 0], tube.poles(shape)[0][0], rtol=1e-6)
    assert generate(2, SynthConfig(p_tube=1.0, duration_s=(1.0, 1.0)), 7).meta["tube"]["glottis"] is None
    assert FS == 16000
