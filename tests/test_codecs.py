"""Codec augmentation keeps timing and leaves other channel draws unchanged (plan §5.1)."""

from dataclasses import replace

import numpy as np
import pytest

from vocal_analysis.synth import codecs
from vocal_analysis.synth.channel import ChannelConfig, augment

pytestmark = pytest.mark.skipif(not codecs.available(), reason="ffmpeg not installed")
FS = 16000


def _vowel_burst(n: int, start: int, f0: float = 140.0) -> np.ndarray:
    """A short harmonic burst in silence."""
    t = np.arange(n) / FS
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
    env = np.zeros(n)
    env[start : start + 4000] = np.hanning(4000)
    return (0.4 * x * env / np.abs(x * env).max()).astype(np.float32)


def _noise_burst(n: int, start: int) -> np.ndarray:
    x = np.zeros(n, np.float32)
    x[start : start + 4000] = np.random.default_rng(0).standard_normal(4000) * np.hanning(4000) * 0.3
    return x


def _lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[np.argmax([np.dot(a[2000:-2000], b[2000 - k : len(b) - 2000 - k]) for k in lags])])


@pytest.mark.parametrize("codec", list(codecs.CODECS))
def test_codec_output_is_aligned_and_same_length(codec):
    """Waveform lag on noise (unambiguous) is within a sample; on a harmonic burst, whose phase low-rate
    speech codecs don't keep, the energy envelope is within 1 ms. Lowest bitrate: the most distortion."""
    kbps = codecs.CODECS[codec][2][0]
    noise = _noise_burst(23_457, 9_001)
    y = codecs.apply(noise, codec, kbps)
    assert len(y) == len(noise) and np.isfinite(y).all()
    assert abs(_lag(y.astype(float), noise.astype(float), 60)) <= 1
    x = _vowel_burst(23_457, 9_001)
    env = lambda s: np.convolve(s.astype(float) ** 2, np.ones(160), "same")  # noqa: E731
    assert abs(_lag(env(codecs.apply(x, codec, kbps)), env(x), 400)) <= 16


def test_codec_stage_leaves_other_draws_unchanged():
    """Draws without a codec are bit-identical to p_codec = 0: the codec stage draws last."""
    x = _vowel_burst(16000, 4000)
    with_codec, without = ChannelConfig(), replace(ChannelConfig(), p_codec=0.0)
    n_coded = 0
    for i in range(40):
        y1, p1 = augment(x, np.random.default_rng(i), with_codec)
        y0, p0 = augment(x, np.random.default_rng(i), without)
        if "codec" in p1:
            n_coded += 1
            assert {k: v for k, v in p1.items() if not k.startswith("codec")} == p0
        else:
            assert np.array_equal(y1, y0) and p1 == p0
    assert 0 < n_coded < 40


def test_augment_always_codes_when_asked():
    cfg = replace(ChannelConfig(), p_clean=0.0, p_codec=1.0, codecs=("opus",))
    y, p = augment(_vowel_burst(8000, 2000), np.random.default_rng(0), cfg)
    assert p["codec"] == "opus" and p["codec_kbps"] in codecs.CODECS["opus"][2] and len(y) == 8000



@pytest.mark.parametrize("codec", list(codecs.IN_PROCESS))
def test_in_process_codecs_match_the_executable(codec):
    """PyAV and the ffmpeg executable code the same way, once each path's own delay is removed: AAC and MP3
    bit-identical; Opus voip differs between the two builds' libopus versions, so its distortion is compared."""
    pytest.importorskip("av")
    x = codecs._probe()
    kbps = codecs.CODECS[codec][2][1]

    def aligned(in_process):
        y = codecs._roundtrip(x, codec, kbps, in_process=in_process, stream=False)
        d = int(np.argmax([np.dot(y[k : k + len(x) - 2000], x[: len(x) - 2000]) for k in range(1200)]))
        return y[d : d + len(x) - 2000]

    a, b, ref = aligned(True), aligned(False), x[: len(x) - 2000]
    if codec == "opus":
        snr = lambda y: 10 * np.log10(np.sum(ref**2) / np.sum((y - ref) ** 2))  # noqa: E731
        assert abs(snr(a) - snr(b)) < 3.0
    else:
        assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("codec", ["amr_nb", "amr_wb"])
def test_amr_stream_matches_one_shot_and_stays_aligned(codec):
    """A stream's first clip is bit-identical to a process per call, up to the last frames (a call flushes the encoder
    at the end of its input; a stream codes on into the silence after the clip); later clips, coded after others,
    stay aligned (noise: within a sample)."""
    kbps = codecs.CODECS[codec][2][0]
    x = codecs._probe()
    st = codecs.AmrStream(codec, kbps)
    try:
        n = codecs.LEAD + len(x) - 2000
        assert np.array_equal(st.roundtrip(x)[:n], codecs._roundtrip(x, codec, kbps, stream=False)[:n])
        d = codecs.delay(codec, kbps)
        for i, n in enumerate((23_457, 8_000, 40_001)):
            noise = _noise_burst(n, n // 2 - 2000)
            st.roundtrip(_vowel_burst(5_000 + 777 * i, 1000))  # something else in between: another frame phase
            y = st.roundtrip(np.pad(noise, (0, 2048)))[d : d + n]
            assert len(y) == n and abs(_lag(y.astype(float), noise.astype(float), 60)) <= 1
    finally:
        st.close()

