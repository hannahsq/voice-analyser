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


def test_flat_store_regenerates_from_index_and_seed(tmp_path):
    """a stored clip is generate(index, the set's config, its seed) stored as int16: analyses regenerate clips this way
    (D-034's jitter arms) instead of keeping the shards' metadata"""
    import json

    from vocal_analysis.synth.shards import INT16_SCALE, write_set
    from vocal_analysis.train.data import LABEL_KEYS, consolidate

    cfg = SynthConfig(duration_s=(1.0, 1.0))
    flat = consolidate(write_set(tmp_path / "s", 3, cfg, seed=5, shard_size=2, workers=1, log=lambda m: None), log=lambda m: None)
    ix = json.loads((flat.parent / "index.json").read_text())
    cfg2 = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ix["synth_config"].items()})
    audio, a_off, f_off = (np.load(flat / f) for f in ("audio.npy", "audio_offsets.npy", "frame_offsets.npy"))
    for i in range(3):
        c = generate(i, cfg2, ix["seed"])
        stored = np.clip(np.round(c.audio * INT16_SCALE), -32768, 32767).astype(np.int16)
        assert np.array_equal(stored, audio[a_off[i]:a_off[i + 1]])
        for k in LABEL_KEYS:
            assert np.array_equal(np.load(flat / f"label_{k}.npy")[f_off[i]:f_off[i + 1]], c.labels[k], equal_nan=True)


@pytest.mark.parametrize("settings", [{}, {"p_length_from_f0": 0.0, "length_dist": "population"}, {"p_length_from_f0": 1.0}])
def test_clip_tract_matches_generate(settings):
    """clip_tract makes only the draws generate makes before the tract, so it recovers a stored clip's tract length,
    whether it came from the f0, and tube or cascade (D-021's VTL labels for train_v2, whose shards are gone)"""
    from vocal_analysis.synth.sampler import clip_tract

    cfg = SynthConfig(duration_s=(1.0, 1.5), **settings)
    kinds = set()
    for i in range(8):
        c = generate(i, cfg, seed=11)
        length, from_f0, use_tube = clip_tract(i, cfg, seed=11)
        assert length == c.meta["length_cm"] and np.all(c.labels["length_cm"] == length)
        assert from_f0 == c.meta["length_from_f0"]
        assert use_tube == (c.meta["tract"] == "tube")
        kinds.add(use_tube)
        if "p_length_from_f0" in settings:
            assert from_f0 == (settings["p_length_from_f0"] == 1.0)
    assert kinds == {True, False}  # both kinds checked


def test_fixed_jitter_keeps_the_random_stream():
    """fixing jitter to the value a clip drew reproduces it exactly, so a fixed-jitter arm changes only the period
    perturbations (D-034); a different value changes the audio but not the labels"""
    from dataclasses import replace

    cfg = SynthConfig(duration_s=(1.0, 1.0))
    i = next(i for i in range(20) if generate(i, cfg, seed=2).labels["voiced"].mean() > 0.5)  # whispered clips have no periods
    a = generate(i, cfg, seed=2)
    j = a.meta["jitter"]
    b = generate(i, replace(cfg, jitter=(j, j)), seed=2)
    assert np.array_equal(a.audio, b.audio)
    c = generate(i, replace(cfg, jitter=(0.02, 0.02)), seed=2)
    assert len(c.audio) == len(a.audio) and not np.array_equal(c.audio, a.audio)
    assert all(np.array_equal(a.labels[k], c.labels[k], equal_nan=True) for k in a.labels)


def test_aspiration_tilt_changes_only_the_aspiration():
    """D-048: a tilted aspiration keeps every other component and label (same random draws); 0 dB/oct is white
    before the tract, and a steeper negative tilt puts relatively more of the noise below 1 kHz"""
    from dataclasses import replace

    cfg = SynthConfig(duration_s=(1.0, 1.0), hnr_db=(10.0, 10.0))
    i = next(i for i in range(20) if generate(i, cfg, seed=2).labels["voiced"].mean() > 0.5)
    a = generate(i, cfg, seed=2, keep_components=True)
    b = generate(i, replace(cfg, asp_tilt_db_oct=-6.0), seed=2, keep_components=True)
    assert np.array_equal(a.components["voiced"], b.components["voiced"])
    assert all(np.array_equal(a.labels[k], b.labels[k], equal_nan=True) for k in a.labels)
    assert not np.array_equal(a.components["aspiration"], b.components["aspiration"])
    assert np.isclose(np.mean(a.components["aspiration"] ** 2), np.mean(b.components["aspiration"] ** 2), rtol=0.05)

    def low_share(x):
        p = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 16000)
        return p[f < 1000].sum() / p.sum()

    assert low_share(b.components["aspiration"]) > low_share(a.components["aspiration"])


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


# --------------------------------------------------------------------------- data sets


def test_original_suite_cases_are_pinned():
    """suite_v1's first ten cases keep the settings they were written with (no tube, no codec)."""
    from vocal_analysis.synth.shards import suite_cases

    cases = suite_cases()
    assert list(cases)[:10] == ["reference", "high_f0", "f1_near_h1", "f1_near_h2", "f2_f3_close", "nasal", "breathy",
                                "wide_vibrato", "low_snr", "phone"]
    from vocal_analysis.synth.sampler import V1_SETTINGS

    for name in list(cases)[:10]:
        synth, ch = cases[name]
        assert all(getattr(synth, k) == v for k, v in V1_SETTINGS.items()) and ch.p_codec == 0.0, name
    assert cases["tube"][0].p_tube == 1.0 and cases["codec"][1].p_codec == 1.0
    assert cases["whisper"][0].p_whisper_clip == 1.0 and cases["whisper"][0].sonorant_includes_whisper


def test_write_set_refuses_different_settings(tmp_path):
    from dataclasses import replace

    from vocal_analysis.synth.shards import write_set

    cfg = replace(STEADY, duration_s=(0.3, 0.3))
    write_set(tmp_path, 2, cfg, seed=0, shard_size=2, workers=1, log=lambda m: None)
    write_set(tmp_path, 2, cfg, seed=0, shard_size=2, workers=1, log=lambda m: None)  # same settings: resumes
    with pytest.raises(ValueError, match="synth_config"):
        write_set(tmp_path, 2, replace(cfg, p_tube=0.5), seed=0, shard_size=2, workers=1, log=lambda m: None)


def test_write_set_reads_old_indexes_with_legacy_defaults(tmp_path):
    """An index written before the post-v1 fields existed means V1_SETTINGS: same settings under them,
    different settings otherwise."""
    import json
    from dataclasses import replace

    from vocal_analysis.synth.sampler import V1_SETTINGS
    from vocal_analysis.synth.shards import write_set

    cfg = replace(STEADY, duration_s=(0.3, 0.3), **V1_SETTINGS)
    write_set(tmp_path, 2, cfg, seed=0, shard_size=2, workers=1, log=lambda m: None)
    idx = json.loads((tmp_path / "index.json").read_text())
    for k in V1_SETTINGS:
        idx["synth_config"].pop(k)
    (tmp_path / "index.json").write_text(json.dumps(idx))
    write_set(tmp_path, 2, cfg, seed=0, shard_size=2, workers=1, log=lambda m: None)  # same settings: no error
    assert "p_tube" not in json.loads((tmp_path / "index.json").read_text())["synth_config"]  # and left as written
    with pytest.raises(ValueError):
        write_set(tmp_path, 2, replace(cfg, p_tube=0.6), seed=0, shard_size=2, workers=1, log=lambda m: None)


# --------------------------------------------------------------------------- f0-length coupling, F1 band, whisper (D-012)


def test_v1_coupled_length_is_the_old_formula():
    from dataclasses import replace

    from vocal_analysis.synth.sampler import V1_SETTINGS, _coupled_length

    cfg = replace(SynthConfig(), **V1_SETTINGS)
    for centre in (60.0, 110.0, 230.0, 900.0):
        a = _coupled_length(np.random.default_rng(1), cfg, centre)
        b = float(np.clip(17.5 - 4.0 * np.log2(centre / 110.0) + np.random.default_rng(1).normal(0, 1.0), 10.0, 20.0))
        assert a == b


def test_coupled_length_redraws_instead_of_piling_up():
    from vocal_analysis.synth.sampler import _coupled_length

    cfg = SynthConfig()
    rng = np.random.default_rng(0)
    hi = np.array([_coupled_length(rng, cfg, 1200.0) for _ in range(2000)])
    assert hi.min() >= 10.0 and hi.max() <= 20.0
    assert np.mean(hi < 10.2) < 0.03  # v1 put ~45 % of these at exactly 10 cm
    assert 12.5 < np.median(hi) < 14.0  # capped at 350 Hz: adult-singer-sized tracts stay common
    women = np.array([_coupled_length(rng, cfg, 220.0) for _ in range(2000)])
    assert 14.5 < np.median(women) < 15.5


def test_f1_band_constraint():
    from vocal_analysis.synth.sampler import _constrain

    rng = np.random.default_rng(0)
    for _ in range(200):
        f0 = rng.uniform(150, 350)
        f = _constrain(rng, tract.sample_formants(rng, 15.0), f0, "f1_band", (1.5, 4.5))
        assert 1.5 * f0 <= f[0] <= 4.5 * f0
        assert np.all(np.diff(f[np.isfinite(f)]) > 0)


def test_whispered_spectrum_is_the_labelled_tract():
    """Noise-only excitation: the time-averaged spectrum equals |H(f)|^2 x the radiation first difference
    (white noise source), so the pole labels describe whispered audio as well as voiced."""
    from scipy.signal import welch

    c = 4001
    pf, pb = np.array([600.0, 1700, 2600, 3500, 4500]), np.array([80.0, 100, 140, 180, 250])
    tr = Trajectories(f0=np.full(c, 120.0), rd=np.full(c, 1.0), voice_amp=np.zeros(c), whisper_amp=np.ones(c),
                      fric_amp=np.zeros(c), poles_f=np.tile(pf, (c, 1)), poles_b=np.tile(pb, (c, 1)),
                      zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
    y = synthesize(tr, np.random.default_rng(0))["total"]
    f, pxx = welch(y, FS, nperseg=1024)
    band = (f > 150) & (f < 6000)
    model = np.abs(F.response(f, pf, pb, FS)) ** 2 * np.abs(1 - np.exp(-2j * np.pi * f / FS)) ** 2
    d = 10 * np.log10(pxx[band] / model[band])
    assert np.max(np.abs(d - np.median(d))) < 1.5


def test_whisper_clip_and_h_onsets_label_formants():
    from dataclasses import replace

    w = generate(3, replace(SynthConfig(), p_whisper_clip=1.0, p_register_step=0), seed=2)
    lab = w.labels
    assert not lab["voiced"].any() and lab["whisper"].any()
    assert np.array_equal(lab["sonorant"], lab["whisper"])
    assert np.isfinite(lab["formants"][lab["sonorant"], :4]).all()
    assert all(s["kind"] != "voiced" for s in w.meta["segments"])

    h = generate(4, replace(SynthConfig(), p_h_onset=1.0, p_whisper_clip=0.0, p_register_step=0), seed=2)
    onsets = [s for s in h.meta["segments"] if s["kind"] == "h_onset"]
    assert onsets
    t = h.labels["t"]
    for s in onsets:
        m = (t >= s["start_s"] + 0.02) & (t < s["end_s"] - 0.005)  # after the 20 ms rise
        if m.any():
            assert h.labels["sonorant"][m].all() and not h.labels["voiced"][m].any()

    v1 = generate(4, replace(SynthConfig(), sonorant_includes_whisper=False))
    assert np.array_equal(v1.labels["sonorant"], v1.labels["voiced"])


def test_write_set_keeps_provenance_when_nothing_is_written(tmp_path):
    import json
    from dataclasses import replace

    from vocal_analysis.synth.shards import write_set

    cfg = replace(STEADY, duration_s=(0.3, 0.3))
    write_set(tmp_path, 2, cfg, seed=0, shard_size=2, workers=1, log=lambda m: None)
    idx = json.loads((tmp_path / "index.json").read_text())
    assert "complete" not in idx
    (tmp_path / "index.json").write_text(json.dumps(idx | {"git_commit": "original"}))
    write_set(tmp_path, 2, cfg, seed=0, shard_size=2, workers=1, log=lambda m: None)
    assert json.loads((tmp_path / "index.json").read_text())["git_commit"] == "original"


def test_length_inst_replay_matches_generate():
    from dataclasses import replace

    from vocal_analysis.synth.sampler import SynthConfig, clip_length_inst, generate

    base = SynthConfig(duration_s=(1.0, 1.5), p_tube=0.7)
    for cfg in (base, replace(base, articulation="sustained", p_tube=1.0, duration_s=(3.0, 4.0))):
        for i in range(6):
            c = generate(i, cfg, seed=3)
            np.testing.assert_allclose(clip_length_inst(i, cfg, seed=3), c.labels["length_inst_cm"], rtol=0, atol=1e-9)


def test_sustained_articulation_moves_length_and_keeps_anatomy():
    from vocal_analysis.synth.sampler import SynthConfig, generate

    cfg = SynthConfig(duration_s=(8.0, 8.0), p_tube=1.0, articulation="sustained")
    c = generate(1, cfg, seed=0)
    inst, anat = c.labels["length_inst_cm"], c.labels["length_cm"]
    assert np.ptp(anat) == 0 and np.ptp(inst) > 0.2  # the tube's length moves while the anatomical label stays
    assert np.all(np.abs(inst - anat) <= 1.5 * anat[0] / 17.5 + 1e-6)  # within PG + PM range at this size


def test_default_config_draws_unchanged_by_articulation_option():
    from vocal_analysis.synth.sampler import SynthConfig, generate

    cfg = SynthConfig(duration_s=(1.0, 1.5), p_tube=1.0)
    c = generate(4, cfg, seed=1)
    assert all(t.get("pg_cm") is not None for t in c.meta["tube"]["targets"])


def test_fixed_articulation_pairs_clips():
    from dataclasses import replace

    from vocal_analysis.synth.sampler import SynthConfig, generate

    cfg = SynthConfig(duration_s=(1.0, 1.5), p_tube=1.0, articulation="fixed")
    a, b = generate(2, cfg, seed=5), generate(2, replace(cfg, artic_pg_cm=0.4), seed=5)
    np.testing.assert_array_equal(a.labels["f0"], b.labels["f0"])  # same source draws
    np.testing.assert_array_equal(a.labels["length_cm"], b.labels["length_cm"])  # same anatomy
    assert np.all(b.labels["length_inst_cm"] > a.labels["length_inst_cm"])  # a lowered larynx lengthens the tract
    assert np.nanmedian(b.labels["formants"][:, 0]) < np.nanmedian(a.labels["formants"][:, 0])
