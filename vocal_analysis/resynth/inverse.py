"""Inverse filtering (plan §5.2 step 2): recover the glottal flow derivative from real audio.

Two methods:
- "iaif": iterative adaptive inverse filtering (Alku 1992) per 10 ms frame. The final vocal-tract
  LPC polynomial of each frame is applied as an FIR filter and the frames are overlap-added.
- "tracks": anti-resonators at given formant tracks (F1-F4 plus uniform-tube poles above them). These
  are the exact inverses of the synthesiser's resonators, so on synthetic audio with the true poles
  the source comes back exactly.

Both inverse filters have unity gain at DC, like the resonators that re-filter the source (a vocal
tract passes DC unchanged). The residual is not integrated: speech already includes lip radiation,
so what comes back is the flow derivative, the same excitation the synthesiser uses.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, lfilter, sosfiltfilt

from ..backend.corrections import upper_poles
from ..synth import filters

FS = 16000
HOP = 160
WIN = 640  # 40 ms: at least 3 periods down to 75 Hz
GAIN_CLAMP_DB = 15.0  # DC normalisation may not move a frame's gain further than this from the median


def lpc(x: np.ndarray, order: int, floor: float = 1e-6) -> np.ndarray:
    """Autocorrelation-method LPC: A(z) = 1 + a1 z^-1 + ... (returned with a0 = 1). `floor` adds
    white noise at that fraction of the energy, which keeps the system well conditioned."""
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(np.abs(spec) ** 2)[: order + 1]
    if r[0] <= 0:
        return np.r_[1.0, np.zeros(order)]
    r[0] *= 1.0 + floor
    return np.r_[1.0, solve_toeplitz(r[:order], -r[1 : order + 1])]


def iaif_frame(x: np.ndarray, order_vt: int, order_gl: int = 4, leak: float = 0.99) -> np.ndarray:
    """Vocal-tract inverse filter A(z) for one frame (Alku 1992; the two-pass form used by Aparat).

    1. a first-order fit removes the glottal tilt, 2. a first tract estimate from what is left,
    3. the glottal contribution re-estimated (order_gl) from the integrated tract residual,
    4. the final tract estimate from the speech with that glottal contribution removed and the
    lip radiation integrated out.
    """
    w = np.hanning(len(x))
    integ = lambda s: lfilter([1.0], [1.0, -leak], s)  # noqa: E731
    g1 = lpc(x * w, 1)
    vt1 = lpc(lfilter(g1, [1.0], x) * w, order_vt)
    glot = integ(lfilter(vt1, [1.0], x))
    g2 = lpc(glot * w, order_gl)
    return lpc(integ(lfilter(g2, [1.0], x)) * w, order_vt)


def _dc_normalise(polys: np.ndarray, active: np.ndarray) -> np.ndarray:
    g = polys.sum(axis=1)
    gdb = 20 * np.log10(np.maximum(np.abs(g), 1e-12))
    ref = np.median(gdb[active]) if active.any() else 0.0
    gdb = np.clip(gdb, ref - GAIN_CLAMP_DB, ref + GAIN_CLAMP_DB)
    return polys / 10 ** (gdb / 20)[:, None]


def ola_fir(x: np.ndarray, polys: np.ndarray, hop: int = HOP, win: int = WIN) -> np.ndarray:
    """Apply FIR filter polys[i] around frame i (centred at sample i * hop) and overlap-add with Hann
    windows. FIR filters have no state, so the cross-fade between frames is clean."""
    p = polys.shape[1] - 1
    pad = win // 2 + p
    xp = np.pad(np.asarray(x, float), (pad, pad + win))
    out = np.zeros(len(xp))
    wsum = np.zeros(len(xp))
    w = np.hanning(win + 1)[:win]  # periodic: overlapping copies sum to a constant
    for i, a in enumerate(polys):
        lo = pad + i * hop - win // 2
        out[lo : lo + win] += w * np.convolve(xp[lo - p : lo + win], a, "valid")
        wsum[lo : lo + win] += w
    y = out / np.where(wsum > 1e-3, wsum, 1.0)
    return y[pad : pad + len(x)]


def iaif(x: np.ndarray, fs: int = FS, order_vt: int | None = None, order_gl: int = 4,
         hop: int = HOP, win: int = WIN, highpass_hz: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """(flow derivative, per-frame tract polynomials) for frames i = 0 .. len(x) // hop."""
    order_vt = order_vt or int(fs / 1000) + 2
    x = sosfiltfilt(butter(2, highpass_hz, "highpass", fs=fs, output="sos"), np.asarray(x, float))
    n_frames = len(x) // hop + 1
    xp = np.pad(x, (win // 2, win))
    polys = np.zeros((n_frames, order_vt + 1))
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        seg = xp[i * hop : i * hop + win]
        energy[i] = float(np.dot(seg, seg))
        polys[i] = iaif_frame(seg, order_vt, order_gl) if energy[i] > 0 else np.r_[1.0, np.zeros(order_vt)]
    active = energy > 1e-3 * np.max(energy) if energy.max() > 0 else np.zeros(n_frames, bool)
    polys = _dc_normalise(polys, active)
    return ola_fir(x, polys, hop, win), polys


def frames_to_samples(values: np.ndarray, n: int, hop: int = HOP) -> np.ndarray:
    """(T, S) per-frame values -> (S, n) per-sample, linear between frame centres (held at the ends).
    A section is absent (NaN) at a sample when it is absent at either neighbouring frame."""
    values = np.asarray(values, float)
    pos = np.arange(n) / hop
    frames = np.arange(len(values))
    out = np.empty((values.shape[1], n))
    for s in range(values.shape[1]):
        v = values[:, s]
        ok = np.isfinite(v)
        out[s] = np.interp(pos, frames, np.where(ok, v, 0.0))
        out[s, np.interp(pos, frames, (~ok).astype(float)) > 0] = np.nan
    return out


def track_inverse(x: np.ndarray, formants: np.ndarray, bandwidths: np.ndarray, fs: int = FS,
                  hop: int = HOP, upper: bool = True) -> np.ndarray:
    """Flow derivative by anti-resonators at per-frame formants (T, N) with bandwidths (T, N); NaN
    bandwidths fall back to Hawks & Miller at f0 = 132 Hz. With `upper`, uniform-tube poles above
    the given formants are removed as well (`backend.corrections.upper_poles`)."""
    from ..synth.tract import bandwidth_prior

    f = np.asarray(formants, float)
    b = np.asarray(bandwidths, float)
    if upper:
        from .tracks import whole_tracks

        up = whole_tracks(upper_poles(f, fs))  # fixed pole set: no section switching on or off (clicks)
        f, b = np.concatenate([f, up], 1), np.concatenate([b, np.full(up.shape, np.nan)], 1)
    b = np.where(np.isfinite(b), b, bandwidth_prior(f, 132.0))
    fs_ = frames_to_samples(f, len(x), hop)
    bs_ = frames_to_samples(b, len(x), hop)
    return filters.filter_tv(np.asarray(x, float), fs_, bs_, fs, np.ones(len(fs_), bool))
