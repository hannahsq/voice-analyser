"""Streaming VQT (plan §10.2, M5): audio arrives in chunks of any size, frames leave as soon as their window is
complete, and the output equals `vqt.vqt_numpy` on the whole signal.

Frame i is centred on sample i * HOP and reads samples [i * HOP - N_MAX / 2, i * HOP + N_MAX / 2), zero before the
signal starts, so it is ready once i * HOP + N_MAX / 2 samples have arrived: the front end's delay is N_MAX / 2
samples (40 ms), the longest window's half. `flush()` ends the signal with zeros, as the batch transform does, and
emits the frames up to len(signal) // HOP.
"""

from __future__ import annotations

import numpy as np

from ..features import vqt

HALF = vqt.N_MAX // 2


def log_power(spec: np.ndarray, floor_db: float = -100.0) -> np.ndarray:
    """Complex VQT (T, N_BINS) -> dB with a floor, as `vqt.TorchVQT.log_power` computes it."""
    p = spec.real ** 2 + spec.imag ** 2
    return 10.0 * np.log10(p + 10 ** (floor_db / 10))


class StreamingVQT:
    """Push audio chunks, get the complex VQT frames (n, N_BINS) completed by each push."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(HALF, np.float32)  # samples from (next frame's centre - HALF) onwards
        self._next = 0  # index of the next frame to emit
        self._received = 0  # samples pushed so far

    @property
    def delay_samples(self) -> int:
        return HALF

    def push(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, chunk])
        self._received += len(chunk)
        n = max(0, (self._received - HALF) // vqt.HOP + 1 - self._next)
        return self._emit(n)

    def flush(self) -> np.ndarray:
        """End of signal: zero-pad and emit the remaining frames, up to len(signal) // HOP (the batch count)."""
        n = self._received // vqt.HOP + 1 - self._next
        if n <= 0:
            return np.zeros((0, vqt.N_BINS), complex)
        self._buf = np.concatenate([self._buf, np.zeros(vqt.N_MAX, np.float32)])
        return self._emit(n)

    def _emit(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, vqt.N_BINS), complex)
        idx = np.arange(n)[:, None] * vqt.HOP + np.arange(vqt.N_MAX)[None, :]
        out = self._buf[idx] @ vqt.kernels()
        self._buf = self._buf[n * vqt.HOP:]
        self._next += n
        return out
