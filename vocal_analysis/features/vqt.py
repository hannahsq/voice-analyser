"""Variable-Q transform as a framed matrix product (plan §7.1).

Bin b has centre frequency f_b = F_MIN * 2^(b / BINS_PER_OCTAVE) and a Hann-windowed
complex exponential kernel of length N_b = min(Q fs / f_b, N_MAX): constant-Q above
~Q fs / N_MAX and constant bandwidth below it, so the longest window (80 ms) bounds the
front-end delay at 40 ms and still resolves harmonics for f0 >= ~60 Hz. Frames are
centred on t = i * hop and use only samples within +-N_MAX/2 of that time, so streaming
output equals batch output by construction.

Kernels are normalised so that a sinusoid of amplitude A at a bin's centre frequency gives
|X| = A / 2 in that bin (A/2 per complex exponential).
"""

from __future__ import annotations

import functools

import numpy as np

FS = 16000
HOP = 160
F_MIN = 40.0
F_MAX = 7500.0
BINS_PER_OCTAVE = 48
N_MAX = 1280  # 80 ms
N_BINS = int(np.floor(BINS_PER_OCTAVE * np.log2(F_MAX / F_MIN))) + 1
Q = 1.0 / (2 ** (1.0 / BINS_PER_OCTAVE) - 1.0)


def bin_frequencies() -> np.ndarray:
    return F_MIN * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE)


def window_lengths() -> np.ndarray:
    return np.minimum(np.round(Q * FS / bin_frequencies()), N_MAX).astype(int)


@functools.lru_cache(maxsize=1)
def kernels() -> np.ndarray:
    """Complex kernel matrix (N_MAX, N_BINS), each column zero-padded and centred."""
    k = np.zeros((N_MAX, N_BINS), complex)
    for b, (f, n) in enumerate(zip(bin_frequencies(), window_lengths())):
        w = np.hanning(n + 2)[1:-1]
        t = (np.arange(n) - (n - 1) / 2) / FS
        col = w * np.exp(-2j * np.pi * f * t) / w.sum()
        start = (N_MAX - n) // 2
        k[start : start + n, b] = col
    return k


def frames(x: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    """(T, N_MAX) frames centred at t = i * HOP; zero outside the signal."""
    x = np.asarray(x, np.float32)
    n_frames = n_frames or len(x) // HOP + 1
    pad = np.pad(x, (N_MAX // 2, N_MAX // 2 + n_frames * HOP))
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(N_MAX)[None, :]
    return pad[idx]


def vqt_numpy(x: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    """Complex VQT (T, N_BINS); reference implementation for tests."""
    return frames(x, n_frames) @ kernels()


class TorchVQT:
    """GPU version: log power with a floor, from (B, samples) audio to (B, N_BINS, T)."""

    def __init__(self, device, dtype=None):
        import torch

        k = kernels()
        self.torch = torch
        self.kr = torch.tensor(k.real, dtype=torch.float32, device=device)
        self.ki = torch.tensor(k.imag, dtype=torch.float32, device=device)
        self.device = device

    def complex_frames(self, audio, n_frames: int):
        torch = self.torch
        x = torch.nn.functional.pad(audio, (N_MAX // 2, N_MAX // 2 + n_frames * HOP))
        fr = x.unfold(-1, N_MAX, HOP)[:, :n_frames]  # (B, T, N_MAX)
        return fr @ self.kr, fr @ self.ki

    def log_power(self, audio, n_frames: int, floor_db: float = -100.0):
        """Frames centred at i * HOP of `audio`, zero-padded beyond its ends."""
        re, im = self.complex_frames(audio, n_frames)
        return self._db(re * re + im * im, floor_db)

    def log_power_with_context(self, audio, n_frames: int, floor_db: float = -100.0):
        """`audio` already carries N_MAX // 2 samples of context before frame 0 and after the last frame
        (length (n_frames - 1) * HOP + N_MAX), as produced by train.data.SynthCrops."""
        fr = audio.unfold(-1, N_MAX, HOP)[:, :n_frames]
        re, im = fr @ self.kr, fr @ self.ki
        return self._db(re * re + im * im, floor_db)

    def _db(self, p, floor_db):
        return (10.0 * self.torch.log10(p + 10 ** (floor_db / 10))).transpose(1, 2)  # (B, N_BINS, T)
