"""Channel augmentation (plan §5.1). Cheap enough to apply on the fly in DataLoader workers.

Every operation keeps timing (no added delay), so frame labels stay valid. A phone channel is
emulated (band-pass 300-3400 Hz, 8 kHz resampling, mu-law quantisation), and lossy codecs run
through ffmpeg (`codecs.py`, delay-compensated). The codec stage comes last and draws its random
numbers last, so channels drawn with p_codec = 0 are unchanged from before it existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

FS = 16000


@dataclass
class ChannelConfig:
    p_clean: float = 0.15
    p_rir: float = 0.4
    t60_s: tuple[float, float] = (0.1, 1.0)
    p_noise: float = 0.6
    snr_db: tuple[float, float] = (5.0, 40.0)
    noise_kinds: tuple[str, ...] = ("white", "pink", "brown", "hum", "babble")
    p_highpass: float = 0.5
    highpass_hz: tuple[float, float] = (40.0, 300.0)
    p_lowpass: float = 0.3
    lowpass_hz: tuple[float, float] = (4000.0, 7500.0)
    p_eq: float = 0.5
    eq_gain_db: float = 6.0
    p_phone: float = 0.05
    gain_db: tuple[float, float] = (-20.0, 0.0)
    p_clip: float = 0.03
    p_codec: float = 0.15
    codecs: tuple[str, ...] = ("opus", "aac", "mp3", "amr_wb", "amr_nb")


def _biquad_peaking(f0: float, gain_db: float, q: float, fs: float = FS) -> np.ndarray:
    """RBJ cookbook peaking EQ as a single SOS row."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / fs
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w), 1 - alpha / a])
    return np.r_[b / den[0], den / den[0]][None, :]


def synth_rir(rng: np.random.Generator, t60: float, fs: float = FS) -> np.ndarray:
    """Direct path at index 0, sparse early reflections (2-30 ms), then diffuse exponential decay
    with shorter T60 at high frequencies."""
    n = int(min(1.5 * t60, 1.2) * fs)
    t = np.arange(n) / fs
    rir = np.zeros(n)
    rir[0] = 1.0
    for _ in range(rng.integers(4, 12)):
        rir[int(rng.uniform(0.002, 0.03) * fs)] += rng.uniform(-0.6, 0.6)
    bands = [(None, 500.0, 1.2), (500.0, 3000.0, 1.0), (3000.0, None, 0.6)]
    late = np.zeros(n)
    for lo, hi, mult in bands:
        sos = butter(4, [lo, hi] if lo and hi else (hi or lo), "bandpass" if lo and hi else ("lowpass" if hi else "highpass"), fs=fs, output="sos")
        late += sosfilt(sos, rng.standard_normal(n)) * np.exp(-6.9 * t / (t60 * mult))
    onset = int(0.005 * fs)
    late[:onset] *= np.linspace(0, 1, onset)
    drr_db = rng.uniform(-5, 15)  # direct-to-reverberant ratio
    late *= np.sqrt(1.0 / (np.sum(late**2) * 10 ** (drr_db / 10)))
    return rir + late


def _noise(rng: np.random.Generator, kind: str, n: int, fs: float = FS) -> np.ndarray:
    w = rng.standard_normal(n)
    if kind == "white":
        return w
    if kind in ("pink", "brown"):
        f = np.fft.rfftfreq(n, 1 / fs)
        shape = 1 / np.maximum(f, 20.0) ** (0.5 if kind == "pink" else 1.0)
        return np.fft.irfft(np.fft.rfft(w) * shape, n)
    if kind == "hum":
        mains = rng.choice([50.0, 60.0])
        t = np.arange(n) / fs
        h = sum(rng.uniform(0.1, 1.0) / k * np.sin(2 * np.pi * k * mains * t + rng.uniform(0, 6.3)) for k in range(1, 8))
        return h + 0.05 * w
    if kind == "babble":  # several amplitude-modulated band-limited noises: speech-shaped, not speech
        out = np.zeros(n)
        for _ in range(rng.integers(3, 8)):
            lo = rng.uniform(150, 600)
            sos = butter(2, [lo, rng.uniform(lo * 3, 4000)], "bandpass", fs=fs, output="sos")
            env = np.abs(sosfilt(butter(2, rng.uniform(2, 8), fs=fs, output="sos"), rng.standard_normal(n)))
            out += sosfilt(sos, rng.standard_normal(n)) * env
        return out
    raise ValueError(kind)


def _phone(x: np.ndarray, rng: np.random.Generator, fs: float = FS) -> np.ndarray:
    y = sosfilt(butter(4, [300.0, 3400.0], "bandpass", fs=fs, output="sos"), x)
    y = resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)]
    peak = np.max(np.abs(y)) + 1e-12
    mu = 255.0
    q = np.sign(y / peak) * np.log1p(mu * np.abs(y / peak)) / np.log1p(mu)
    q = np.round(q * 127) / 127  # 8-bit companded
    return np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu * peak


def augment(x: np.ndarray, rng: np.random.Generator, cfg: ChannelConfig | None = None,
            speech_mask: np.ndarray | None = None, codec_stream: bool | None = None) -> tuple[np.ndarray, dict]:
    """Apply a random channel. speech_mask (per sample) marks where SNR is measured. codec_stream: see codecs.apply
    (False for fixed sets, whose audio must not depend on what was coded before)."""
    cfg = cfg or ChannelConfig()
    x = np.asarray(x, np.float64)
    params: dict = {"clean": False}
    if rng.random() < cfg.p_clean:
        params["clean"] = True
        return x.astype(np.float32), params
    n = len(x)
    if rng.random() < cfg.p_rir:
        t60 = rng.uniform(*cfg.t60_s)
        x = fftconvolve(x, synth_rir(rng, t60))[:n]
        params["t60_s"] = t60
    if rng.random() < cfg.p_highpass:
        f = rng.uniform(*cfg.highpass_hz)
        x = sosfilt(butter(int(rng.choice([2, 4])), f, "highpass", fs=FS, output="sos"), x)
        params["highpass_hz"] = f
    if rng.random() < cfg.p_lowpass:
        f = rng.uniform(*cfg.lowpass_hz)
        x = sosfilt(butter(int(rng.choice([4, 8])), f, "lowpass", fs=FS, output="sos"), x)
        params["lowpass_hz"] = f
    if rng.random() < cfg.p_eq:
        bands = []
        for _ in range(rng.integers(1, 4)):
            f, g, q = float(np.exp(rng.uniform(np.log(100), np.log(6000)))), rng.uniform(-cfg.eq_gain_db, cfg.eq_gain_db), rng.uniform(0.5, 4)
            x = sosfilt(_biquad_peaking(f, g, q), x)
            bands.append((f, g, q))
        params["eq"] = bands
    if rng.random() < cfg.p_noise:
        kind = str(rng.choice(cfg.noise_kinds))
        snr = rng.uniform(*cfg.snr_db)
        nz = _noise(rng, kind, n)
        m = speech_mask if speech_mask is not None and speech_mask.any() else np.ones(n, bool)
        ps, pn = np.mean(x[m] ** 2), np.mean(nz**2)
        x = x + nz * np.sqrt(ps / max(pn, 1e-20) / 10 ** (snr / 10))
        params.update(noise=kind, snr_db=snr)
    if rng.random() < cfg.p_phone:
        x = _phone(x, rng)
        params["phone"] = True
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.5 * 10 ** (rng.uniform(*cfg.gain_db) / 20)
    if rng.random() < cfg.p_clip:
        lim = np.max(np.abs(x)) * rng.uniform(0.3, 0.8)
        x = np.clip(x, -lim, lim)
        params["clip_frac"] = float(lim)
    if cfg.p_codec > 0 and rng.random() < cfg.p_codec:  # after gain and clipping: the recording is what gets coded
        from . import codecs

        name = str(rng.choice(cfg.codecs))
        kbps = float(rng.choice(codecs.CODECS[name][2]))
        x = codecs.apply(x, name, kbps, stream=codec_stream)
        params.update(codec=name, codec_kbps=kbps)
    return x.astype(np.float32), params
