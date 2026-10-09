"""Closed-phase EGG analysis (D-019) and the CMU ARCTIC loader, on synthetic signals and tiny fixtures."""

import tarfile

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter

from vocal_analysis.data.download import extract
from vocal_analysis.eval import closed_phase as cp
from vocal_analysis.eval.datasets import cmu_arctic

FS = 16000
POLES = ((500.0, 60.0), (1500.0, 90.0), (2500.0, 120.0), (3500.0, 150.0))


def _a(poles=POLES, fs=FS):
    a = np.array([1.0])
    for f, b in poles:
        r = np.exp(-np.pi * b / fs)
        a = np.convolve(a, [1.0, -2 * r * np.cos(2 * np.pi * f / fs), r * r])
    return a


def _voice(f0=125.0, cq=0.45, seconds=0.5, lag=0, seed=0, poles=POLES):
    """Speech from an all-pole tract excited at each closure (an impulse, the main excitation of real voices)
    and through the open phase (noise), silent while closed; and an EGG whose contact rises fast at closure
    (0.4 ms) and falls slower at opening (1.2 ms), both raised-cosine, so their steepest points are the instants."""
    rng = np.random.default_rng(seed)
    n, T = int(seconds * FS), int(round(FS / f0))
    gci = np.arange(T, n - 2 * T, T)
    goi = gci + int(cq * T)
    e = np.zeros(n)
    egg = np.zeros(n)
    ramp = lambda m: 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / (m - 1))  # noqa: E731
    for g, o in zip(gci, goi):
        nxt = g + T
        e[g] = -8.0
        e[o:nxt] = rng.normal(size=nxt - o) * np.hanning(nxt - o)
        rise, fall = int(0.0004 * FS) + 1, int(0.0012 * FS) + 1
        egg[g - rise // 2 : g + rise - rise // 2] = ramp(rise)
        egg[g + rise - rise // 2 : o - fall // 2] = 1.0
        egg[o - fall // 2 : o + fall - fall // 2] = 1.0 - ramp(fall)
    x = lfilter([1.0], _a(poles), e)
    x = np.r_[np.zeros(lag), x[: n - lag]] if lag > 0 else x
    return x, egg + 0.001 * rng.normal(size=n), gci, goi


def test_covariance_lpc_is_exact_on_closed_phases():
    x, _, gci, goi = _voice()
    segs = [(g + 1, o) for g, o in zip(gci, goi)]  # closed phase: from closure to opening
    a = cp.covariance_lpc(x, segs, order=8)
    np.testing.assert_allclose(np.r_[1.0, a], _a(), atol=1e-3)  # poles near |z| = 1: ill-conditioned coefficients
    f, bw = cp.lpc_formants(a, FS)
    np.testing.assert_allclose(f, [p[0] for p in POLES], atol=0.5)
    np.testing.assert_allclose(bw, [p[1] for p in POLES], atol=0.5)


def test_whole_cycles_are_not_exact():
    x, _, gci, _ = _voice()
    a = cp.covariance_lpc(x, [(g, g + 128) for g in gci], order=8)
    f, _ = cp.lpc_formants(a, FS, max_bw=5000)
    assert np.max(np.abs(np.r_[f, np.zeros(4)][:4] - [p[0] for p in POLES])) > 1.0


@pytest.mark.parametrize("sign", [1, -1])
def test_egg_instants(sign):
    _, egg, gci, goi = _voice()
    got = cp.egg_instants(sign * egg, FS)
    assert got["polarity"] == sign
    match = lambda ref, est: np.array([np.min(np.abs(est - r)) for r in ref[1:-1]])  # noqa: E731
    assert np.max(match(gci, got["gci"])) <= 2
    assert np.max(match(goi, got["goi"])) <= 2
    assert len(got["gci"]) >= len(gci) - 3


def test_analyse_recovers_formants_and_lag():
    x, egg, gci, _ = _voice(lag=12)
    res = cp.analyse(x, egg, FS, np.array([0.15, 0.25, 0.35]), order=8, guard_s=0.0002)
    assert abs(res["lag"] - 12) <= 1  # within the ±0.1 ms smoothing of the lag search
    np.testing.assert_allclose(res["cp"][:, 0], 500.0, atol=2.0)
    np.testing.assert_allclose(res["cp"][:, 1], 1500.0, atol=3.0)
    np.testing.assert_allclose(res["f0"], 125.0, rtol=0.01)
    assert np.all((res["cq"] > 0.35) & (res["cq"] < 0.55))


def test_cmu_arctic_loader(tmp_path):
    d = tmp_path / "cmu_us_slt_arctic"
    (d / "orig").mkdir(parents=True)
    (d / "lab").mkdir()
    sf.write(d / "orig" / "arctic_a0001.wav", np.zeros((3200, 2)), 32000)
    sf.write(d / "orig" / "arctic_a0002.wav", np.zeros((3200, 2)), 32000)  # no labels -> not in manifest
    (d / "lab" / "arctic_a0001.lab").write_text("#\n0.10 125 pau\n0.20 125 iy\n0.24 125 t\n0.30 125 ah\n0.40 125 ay\n")
    other = tmp_path / "cmu_us_clb_arctic" / "orig"  # a voice without EGG is ignored
    other.mkdir(parents=True)
    sf.write(other / "arctic_a0001.wav", np.zeros((3200, 2)), 32000)
    rows = cmu_arctic.scan(tmp_path)
    assert [r.path for r in rows] == ["cmu_us_slt_arctic/orig/arctic_a0001.wav"]
    assert rows[0].speaker_id == "cmu_arctic:slt" and rows[0].split == "gold"
    item = cmu_arctic.load(tmp_path, rows[0])
    # iy (0.10-0.20 s) qualifies; t is no vowel; ah lasts 60 ms (< 70); ay is a diphthong
    np.testing.assert_allclose(item.times, [0.135, 0.15, 0.165])
    assert item.meta["phones"] == ["iy"] * 3 and item.meta["channel"] == 0 and item.meta["sex"] == "f"


def test_extract_include_filter_on_tar_bz2(tmp_path):
    src = tmp_path / "src" / "v"
    (src / "orig").mkdir(parents=True)
    (src / "festival").mkdir()
    (src / "orig" / "a.wav").write_bytes(b"x")
    (src / "festival" / "big.bin").write_bytes(b"y")
    arc = tmp_path / "v.tar.bz2"
    with tarfile.open(arc, "w:bz2") as t:
        t.add(src, arcname="v")
    extract(arc, tmp_path / "out", include=("v/orig/*.wav",))
    assert (tmp_path / "out" / "v" / "orig" / "a.wav").exists()
    assert not (tmp_path / "out" / "v" / "festival").exists()
