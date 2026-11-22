"""M5: the streaming VQT equals the batch VQT, whatever the chunk sizes."""

import numpy as np
import pytest

from vocal_analysis.features import vqt
from vocal_analysis.rt.streaming import HALF, StreamingVQT, log_power


def _stream(x, sizes):
    s, out, i, k = StreamingVQT(), [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out.append(s.push(x[i:i + n]))
        i, k = i + n, k + 1
    out.append(s.flush())
    return np.concatenate(out)


@pytest.mark.parametrize("sizes", [[1], [37], [160], [1000, 3, 513], [10 ** 6]])
def test_streaming_equals_batch(sizes):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000 + 77).astype(np.float32)
    ref = vqt.vqt_numpy(x)
    got = _stream(x, sizes)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9 * np.abs(ref).max())
    np.testing.assert_allclose(log_power(got), log_power(ref), atol=1e-6)


def test_frames_leave_as_soon_as_complete():
    s = StreamingVQT()
    assert len(s.push(np.zeros(HALF - 1, np.float32))) == 0
    assert len(s.push(np.zeros(1, np.float32))) == 1  # frame 0 needs samples up to HALF
    assert len(s.push(np.zeros(vqt.HOP - 1, np.float32))) == 0
    assert len(s.push(np.zeros(1, np.float32))) == 1
    assert s.delay_samples / vqt.FS == pytest.approx(0.040)


def test_short_signal_and_reset():
    x = np.ones(50, np.float32)
    s = StreamingVQT()
    got = np.concatenate([s.push(x), s.flush()])
    np.testing.assert_allclose(got, vqt.vqt_numpy(x), atol=1e-9)
    s.reset()
    assert len(s.flush()) == 1  # an empty signal still has frame 0, as the batch transform does
