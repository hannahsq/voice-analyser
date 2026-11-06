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


def test_closed_phase_gap_follows_open_quotient_like_holmberg_et_al():
    # men's loud / normal / soft (Holmberg, Hillman & Perkell 1988): minimum / peak flow 0.22 / 0.32 / 0.45
    gap = coupling.gap_from_open_quotient(np.array([0.57, 0.60, 0.73]), 0.0)
    assert np.allclose(gap, [0.22, 0.32, 0.45], atol=0.06)
    assert np.all(np.diff(gap) > 0)
    assert coupling.gap_from_open_quotient(0.3, -0.3) == 0.0  # never negative


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
    # full-strength F1 changes the audio but not the draws or the labels
    c = generate(2, SynthConfig(p_tube=1.0, p_coupling=1.0, duration_s=(1.0, 1.0), coupling_full_poles=1), 7)
    assert c.meta["tube"] == a.meta["tube"] and np.array_equal(c.labels["formants"], a.labels["formants"], equal_nan=True)
    assert not np.array_equal(c.audio, a.audio)
    assert FS == 16000


def test_acoustic_labels_are_the_cycle_mean_and_change_nothing_else():
    """D-078: acoustic_labels adds "formants_acoustic" = F (1 + df <a>) on coupled poles; audio, draws and the closed-glottis
    labels stay as without it."""
    cfg = SynthConfig(p_tube=1.0, p_coupling=1.0, coupling_scale=(1.0, 1.0), duration_s=(1.0, 1.0))
    a = generate(3, cfg, 7)
    b = generate(3, SynthConfig(**{**cfg.__dict__, "acoustic_labels": True}), 7)
    assert "formants_acoustic" not in a.labels
    assert np.array_equal(a.audio, b.audio) and a.meta["tube"] == b.meta["tube"]
    assert np.array_equal(a.labels["formants"], b.labels["formants"], equal_nan=True)
    f, fa = b.labels["formants"], b.labels["formants_acoustic"]
    rel = fa[:, :coupling.N_COUPLED] / f[:, :coupling.N_COUPLED] - 1
    df1 = np.array([tg["open_df"][0] for tg in b.meta["tube"]["targets"]])
    # F1 moves by df times a mean opening between 0 and 1 (df interpolated between the targets'), in df's direction
    assert np.all(np.abs(rel[:, 0]) <= np.abs(df1).max() + 1e-9) and np.median(rel[:, 0]) * np.median(df1) > 0
    assert np.array_equal(f[:, coupling.N_COUPLED:], fa[:, coupling.N_COUPLED:], equal_nan=True)
    # an uncoupled clip's acoustic label is its closed-glottis label
    u = generate(3, SynthConfig(p_tube=1.0, duration_s=(1.0, 1.0), acoustic_labels=True), 7)
    assert np.array_equal(u.labels["formants_acoustic"], u.labels["formants"], equal_nan=True)
    # D-079: the opening label is the <a> the acoustic label used, in [0, 1]; NaN on an uncoupled clip
    op = b.labels["glottal_opening"]
    assert op.shape == (len(f),) and np.all((op >= 0) & (op <= 1)) and op.std() > 0
    ok = op > 1e-3  # F1's shift over the opening is df, interpolated between the targets'
    k = rel[ok, 0] / op[ok]
    assert ok.mean() > 0.9 and np.all((k >= df1.min() - 1e-9) & (k <= df1.max() + 1e-9))
    assert np.all(np.isnan(u.labels["glottal_opening"]))
    # D-081: the cycle-mean bandwidths use the same <a>, with bm in place of df; the clip's coupling as drawn
    bw, ba, bm = b.labels["bandwidths"], b.labels["bandwidths_acoustic"], b.labels["coupling_bm"]
    np.testing.assert_allclose(ba[:, 0], bw[:, 0] * (1 + (bm[:, 0] - 1) * op), rtol=1e-9)
    np.testing.assert_allclose(fa[:, 0], f[:, 0] * (1 + b.labels["coupling_df"][:, 0] * op), rtol=1e-9)
    assert np.all(b.labels["gap_offset"] == b.labels["gap_offset"][0]) and -0.3 <= b.labels["gap_offset"][0] <= 0.1
    oq = b.labels["open_quotient"]
    assert oq.shape == (len(f),) and np.all((oq > 0) & (oq < 1))


def test_coupled_aspiration_follows_the_coupled_tract_and_changes_nothing_else():
    """D-085: coupled_aspiration sends aspiration through the coupled tract. With no pole movement (df 0, bm 1) that is
    the closed tract (same aspiration); with movement, only the aspiration changes; without coupling the flag does
    nothing. The same draws throughout."""
    c = 301
    pf, pb = np.full((c, 8), np.nan), np.full((c, 8), np.nan)
    pf[:, :4], pb[:, :4] = [600, 1300, 2500, 3500], [60, 80, 120, 150]
    tr = Trajectories(f0=np.full(c, 150.0), rd=np.full(c, 1.2), voice_amp=np.ones(c), whisper_amp=np.zeros(c), fric_amp=np.zeros(c),
                      poles_f=pf, poles_b=pb, zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
    still = {"df": np.zeros((c, 4)), "bm": np.ones((c, 4))}
    moving = {"df": np.full((c, 4), 0.1), "bm": np.full((c, 4), 2.0), "gap": 0.2}
    run = lambda cp, flag: synthesize(tr, np.random.default_rng(0), hnr_db=15.0, coupling=cp, coupled_aspiration=flag)  # noqa: E731
    seg = slice(2000, (c - 1) * CTRL)
    a, b = run(still, False), run(still, True)
    assert np.allclose(a["aspiration"][seg], b["aspiration"][seg], atol=1e-3 * np.abs(a["aspiration"][seg]).max())
    a, b = run(moving, False), run(moving, True)
    assert np.array_equal(a["voiced"], b["voiced"])
    assert not np.allclose(a["aspiration"][seg], b["aspiration"][seg], atol=1e-2 * np.abs(a["aspiration"][seg]).max())
    for k in a:
        assert a[k].shape == b[k].shape
    u0 = synthesize(tr, np.random.default_rng(0), hnr_db=15.0)
    u1 = synthesize(tr, np.random.default_rng(0), hnr_db=15.0, coupled_aspiration=True)
    assert all(np.array_equal(u0[k], u1[k]) for k in u0)
    cfg = SynthConfig(p_tube=1.0, p_coupling=1.0, coupling_scale=(1.0, 1.0))
    assert np.array_equal(generate(3, cfg, 5).audio, generate(3, cfg, 5).audio)
