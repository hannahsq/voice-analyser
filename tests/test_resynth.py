"""Analysis-resynthesis labeller (plan §5.2), checked on synthetic voices whose source and tract are known."""

import json
from dataclasses import replace

import numpy as np
import pytest

from vocal_analysis.backend.harmonics import harmonic_amplitudes
from vocal_analysis.data.manifest import ManifestRow
from vocal_analysis.resynth import corpus, inverse, render, tracks
from vocal_analysis.synth import filters
from vocal_analysis.synth.glottal import source_h_db
from vocal_analysis.synth.tract import bandwidth_prior, tube_resonances
from vocal_analysis.synth.voice import CTRL, Trajectories, synthesize

FS = 16000


def _vowel(f0, formants, rd=1.0, dur=0.6, tract=True):
    """Stationary vowel: F1-F3 given, F4 and up from a 17.5 cm uniform tube. Returns (audio, poles, bandwidths)."""
    c = int(dur * FS / CTRL) + 1
    f = np.r_[formants, tube_resonances(17.5)[len(formants):]]
    f[f > 7600] = np.nan
    b = bandwidth_prior(f, f0)
    pf, pb = (np.tile(f, (c, 1)), np.tile(b, (c, 1))) if tract else (np.zeros((c, 0)), np.zeros((c, 0)))
    traj = Trajectories(np.full(c, float(f0)), np.full(c, rd), np.ones(c), np.zeros(c), np.zeros(c), pf, pb,
                        np.zeros((c, 0)), np.zeros((c, 0)))
    return synthesize(traj, np.random.default_rng(0))["voiced"], f, b


def _harmonics(x, f0, k=8):
    n = len(x) // 160 + 1
    t = np.arange(n) * 0.01
    return harmonic_amplitudes(x, FS, t[10:-10], np.full(n - 20, float(f0)), k)


def test_lpc_recovers_ar_coefficients():
    rng = np.random.default_rng(0)
    a = np.array([1.0, -1.6, 0.9])
    from scipy.signal import lfilter

    x = lfilter([1.0], a, rng.standard_normal(200000))
    assert np.allclose(inverse.lpc(x, 2), a, atol=0.01)


def test_track_inverse_with_true_poles_returns_the_source():
    """Anti-resonators are the exact inverse of the synthesiser's resonators."""
    rd, f0 = 1.2, 140.0
    x, f, b = _vowel(f0, np.array([650.0, 1100.0, 2500.0]), rd)
    n = len(x) // 160 + 1
    s = inverse.track_inverse(x, np.tile(f, (n, 1)), np.tile(b, (n, 1)), upper=False)
    h = np.median(_harmonics(s, f0), axis=0)
    assert np.allclose(h, source_h_db(np.array([rd]), 8)[0], atol=0.05)


def _steady(n):
    return np.r_[np.zeros(5, bool), np.ones(n - 10, bool), np.zeros(5, bool)]


@pytest.mark.parametrize("f0", [110.0, 220.0])
def test_iaif_recovers_a_clean_source_at_low_f0(f0):
    rd = 1.0
    x, f, b = _vowel(f0, np.array([700.0, 1220.0, 2600.0]), rd)
    s, polys = inverse.iaif(x)
    n = len(x) // 160 + 1
    assert polys.shape == (n, 19)
    h = np.median(_harmonics(s, f0), axis=0)
    true = source_h_db(np.array([rd]), 2)[0]
    assert abs((h[0] - h[1]) - (true[0] - true[1])) < 1.0
    tr = np.tile(f[:4], (n, 1))
    q_src = render.source_qc(s, np.full(n, f0), _steady(n), tr)
    q_speech = render.source_qc(x, np.full(n, f0), _steady(n), tr)
    assert abs(np.nanmedian(q_src["tract_left"])) < 0.2  # IAIF over-removes a little
    assert abs(np.nanmedian(q_speech["tract_left"]) - 1.0) < 0.05  # nothing removed: all of the tract is left


def test_qc_ignores_noise_but_not_a_badly_placed_inverse_filter():
    f0 = 150.0
    x, f, b = _vowel(f0, np.array([500.0, 1500.0, 2500.0]))
    n = len(x) // 160 + 1
    tr, bw, v = np.tile(f[:4], (n, 1)), np.tile(b[:4], (n, 1)), _steady(n)
    good = inverse.track_inverse(x, tr, bw)
    bad = inverse.track_inverse(x, tr * [1.12, 0.9, 1.08, 1.0], bw)
    noisy = good + np.random.default_rng(0).standard_normal(len(good)) * np.std(good) * 0.3
    q = {k: render.source_qc(s, np.full(n, f0), v, tr) for k, s in [("good", good), ("bad", bad), ("noisy", noisy)]}
    med = {k: (np.nanmedian(d["tract_left"]), np.nanmedian(d["ripple_db"])) for k, d in q.items()}
    assert abs(med["good"][0]) < 0.05 and med["good"][1] < 1.0
    assert med["bad"][1] > 3.0  # misplaced anti-resonators leave peak-dip pairs the fit can't explain
    assert abs(med["noisy"][0]) < 0.1 and med["noisy"][1] > med["good"][1] + 1.0  # noise raises the RMS, not beta


def test_refilter_imposes_exactly_the_label_poles():
    """Plan §5.1 test, for the re-filter: output harmonics = source harmonics x |H(k f0)|."""
    f0, rd = 180.0, 0.8
    src, _, _ = _vowel(f0, np.array([]), rd, tract=False)
    n = len(src) // 160 + 1
    f = np.r_[820.0, 1300.0, 2700.0, 3600.0, 4500.0, np.nan, np.nan, np.nan]
    b = bandwidth_prior(f, f0)
    y = render.refilter(src, np.tile(f, (n, 1)), np.tile(b, (n, 1)))
    k = np.arange(1, 9)
    gain = 20 * np.log10(np.abs(filters.response(k * f0, f[None, :], b[None, :], FS)))
    assert np.allclose(np.median(_harmonics(y, f0), 0) - np.median(_harmonics(src, f0), 0), gain, atol=0.05)


def test_smooth_tracks_fills_gaps_and_keeps_steady_tracks():
    t = 60
    f = np.tile([500.0, 1500.0, 2500.0, 3500.0], (t, 1))
    f[20:30] = [3000.0, 200.0, 7000.0, 100.0]  # tracker garbage in a non-sonorant gap
    f[40:] *= 1.2
    son = np.ones(t, bool)
    son[20:30] = False
    son[34:40] = False
    out = tracks.smooth_tracks(f, son)
    assert np.allclose(out[:20], [500, 1500, 2500, 3500]) and np.allclose(out[20:30], out[19])
    assert np.allclose(out[42:], np.array([500, 1500, 2500, 3500]) * 1.2)
    assert np.all(np.diff(np.log(out[33:41, 0])) > 0)  # log-linear across the second gap


def test_perturb_scales_orders_and_is_deterministic():
    t = 100
    base = np.tile([600.0, 1400.0, 2500.0, 3400.0], (t, 1))
    cfg = tracks.PerturbConfig(alpha=(1.25, 1.25), offset_sd=(0.0, 0.0), p_nasal=0.0)
    p = tracks.perturb(base, np.full(t, 200.0), np.ones(t, bool), np.random.default_rng(0), cfg)
    assert p["formants"].shape == (t, tracks.N_LABEL_FORMANTS)
    assert np.allclose(p["formants"][:, :4], base / 1.25)
    f = p["formants"]
    ok = np.isfinite(f)
    assert np.all(np.diff(np.where(ok, f, np.inf), axis=1)[ok[:, 1:]] >= 150)
    assert np.all(np.isnan(f[f > 7600])) and np.all(np.isfinite(p["bandwidths"]) == ok)
    q = tracks.perturb(base, np.full(t, 200.0), np.ones(t, bool), np.random.default_rng(0))
    r = tracks.perturb(base, np.full(t, 200.0), np.ones(t, bool), np.random.default_rng(0))
    assert np.array_equal(q["formants"], r["formants"], equal_nan=True) and q["info"] == r["info"]
    ratio = q["formants"][:, :4] / base * q["info"]["alpha"]
    assert 0.8 < ratio.min() and ratio.max() < 1.25 and ratio.std() > 0.005  # slow offsets present


def test_cut_points():
    x = np.random.default_rng(0).standard_normal(FS * 14)
    n = len(x) // 160 + 1
    cuts = render.cut_points(x, n, (3.0, 6.0))
    lengths = np.diff(cuts) * 0.01
    assert cuts[0] == 0 and cuts[-1] == n and np.all(lengths >= 3.0 - 1e-9) and np.all(lengths[:-1] <= 6.0)
    assert render.cut_points(x[: FS * 5], 501, (3.0, 6.0)) == [0, 501]


def _fake_corpus(tmp_path, n_files=3):
    """Synthetic recordings at 44.1 kHz with 'analysis' files holding their true labels."""
    import soundfile as sf
    from scipy.signal import resample_poly

    from vocal_analysis.synth.sampler import SynthConfig, generate

    cfg = SynthConfig(p_tube=0.0, p_whisper_clip=0.0, p_nasal=0.0, p_subglottal=0.0, duration_s=(4.0, 8.0),
                      p_high_f0=0.0, low_f0_hz=(90.0, 200.0))
    root, adir = tmp_path / "corpus", tmp_path / "out" / "analysis"
    adir.mkdir(parents=True)
    rows = []
    for i in range(n_files):
        c = generate(i, cfg, seed=5)
        rel = f"s{i}/take.wav"
        (root / f"s{i}").mkdir(parents=True)
        sf.write(root / rel, resample_poly(c.audio, 441, 160) * 0.5, 44100, subtype="FLOAT")
        need = len(corpus.load_audio(root / rel)) // 160 + 1
        lab = c.labels
        pad = lambda v: np.concatenate([v, np.repeat(v[-1:], need - len(v), 0)]) if len(v) < need else v[:need]  # noqa: E731
        np.savez(adir / corpus._analysis_name(rel), f0=pad(np.where(lab["voiced"], lab["f0"], 100.0)),
                 voiced_p=pad(lab["voiced"].astype(float)), sonorant_p=pad(lab["sonorant"].astype(float)),
                 formants=pad(lab["formants"][:, :4]), bandwidths=pad(lab["bandwidths"][:, :4]),
                 formant_sigma_cents=pad(np.zeros((len(lab["t"]), 4))))
        rows.append(ManifestRow(rel, "fake", f"fake:s{i}", "train", 44100, len(c.audio) / FS, "CC0"))
    return root, adir, rows


def test_render_corpus_end_to_end(tmp_path):
    from vocal_analysis.synth.shards import ShardReader
    from vocal_analysis.train.data import LABEL_KEYS, SynthCrops, consolidate

    root, adir, rows = _fake_corpus(tmp_path)
    out = tmp_path / "out" / "set"
    cfg = render.ResynthConfig(variants=2)
    corpus.render(rows, root, adir, out, cfg, seed=3, files_per_shard=2, workers=1, log=lambda *a: None)
    index = json.loads((out / "index.json").read_text())
    assert index["n_clips"] == sum(index["shard_clips"]) > 0 and index["kind"] == "resynth"
    assert index["qc"]["n_rejected"] <= index["n_clips"] // 2  # true tracks: the sources are clean
    reader = ShardReader(out)
    for i in range(len(reader)):
        audio, lab, meta = reader[i]
        assert set(LABEL_KEYS) <= set(lab) and (len(lab["t"]) - 1) * 160 <= len(audio) <= len(lab["t"]) * 160
        son = lab["sonorant"]
        assert np.all(np.isfinite(lab["formants"][son, :4])) and np.all(np.isfinite(lab["bandwidths"][son, :4]))
        assert meta["labels"]["formants"] == "exact" and meta["source_path"] in [r.path for r in rows]
        assert np.all(lab["sonorant"] <= np.isfinite(lab["formants"][:, 0]))
    with pytest.raises(IndexError):
        reader[len(reader)]
    # resuming writes nothing and keeps the index; different settings are refused
    before = {p.name: p.stat().st_mtime_ns for p in out.glob("shard_*.npz")}
    corpus.render(rows, root, adir, out, cfg, seed=3, files_per_shard=2, workers=1, log=lambda *a: None)
    assert before == {p.name: p.stat().st_mtime_ns for p in out.glob("shard_*.npz")}
    with pytest.raises(ValueError, match="different settings"):
        corpus.render(rows, root, adir, out, replace(cfg, variants=1), seed=3, files_per_shard=2, workers=1)
    flat = consolidate(out, log=lambda *a: None)
    seg, lab = SynthCrops(flat, np.arange(len(reader)), channel=None, fixed=True)[0]
    assert lab["formants"].shape == (200, 4) and np.isfinite(seg).all()


def test_render_variants_differ_and_regenerate(tmp_path):
    root, adir, rows = _fake_corpus(tmp_path, n_files=1)
    cfg = render.ResynthConfig()
    with np.load(adir / corpus._analysis_name(rows[0].path)) as z:
        analysis = {k: z[k] for k in z.files}
    audio = corpus.load_audio(root / rows[0].path)
    src = render.source_from_audio(audio, analysis, cfg)
    a, _ = render.render_variant(audio, src, np.random.default_rng([1, 0, 0]), cfg)
    b, _ = render.render_variant(audio, src, np.random.default_rng([1, 0, 0]), cfg)
    c, _ = render.render_variant(audio, src, np.random.default_rng([1, 0, 1]), cfg)
    assert np.array_equal(a[0][0], b[0][0]) and not np.allclose(a[0][1]["formants"][:, 0], c[0][1]["formants"][:, 0])
