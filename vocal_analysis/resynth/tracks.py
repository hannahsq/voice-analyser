"""Formant tracks for resynthesis (plan §5.2 steps 3-4): smooth the tracker's output, then perturb it.

The perturbed tracks are the labels: the source is re-filtered through resonators at exactly these
poles, so how well the original tracks fit the original voice affects how clean the source is
(QC, `render.ripple_db`), not whether the labels are right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from ..backend.vtl import C_CM_S, apparent_vtl
from ..synth import tract

N_LABEL_FORMANTS = tract.N_POLES


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def enforce_order(f: np.ndarray, min_spacing_hz: float = tract.MIN_SPACING_HZ, f_max_hz: float = tract.F_MAX_HZ) -> np.ndarray:
    """Per frame: each formant at least min_spacing above the one below; NaN above f_max."""
    out = np.array(f, float)
    for n in range(1, out.shape[1]):
        out[:, n] = np.fmax(out[:, n], out[:, n - 1] + min_spacing_hz)
    out[out > f_max_hz] = np.nan
    return out


def whole_tracks(f: np.ndarray) -> np.ndarray:
    """Drop (NaN everywhere) any pole track that is absent on some frames. A recursive section switched
    on or off mid-signal restarts from the wrong state and clicks, so a recording keeps a fixed set of
    poles: one that ever leaves the band (above F_MAX_HZ) is left out for the whole recording."""
    out = np.array(f, float)
    out[:, ~np.all(np.isfinite(out), axis=0)] = np.nan
    return out


def smooth_tracks(formants: np.ndarray, sonorant: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay in log frequency within each sonorant run; across other frames, linear in log
    frequency between the neighbouring runs (held before the first and after the last). Every frame
    gets a value, since the re-filtering runs everywhere."""
    logf = np.log(np.asarray(formants, float))
    out = np.full_like(logf, np.nan)
    son = np.asarray(sonorant, bool) & np.all(np.isfinite(logf), axis=1)
    for a, b in _runs(son):
        seg = logf[a:b]
        out[a:b] = savgol_filter(seg, window, polyorder, axis=0, mode="interp") if b - a >= window else seg
    if not son.any():
        return np.full_like(logf, np.nan)
    t = np.arange(len(logf))
    for n in range(logf.shape[1]):
        out[~son, n] = np.interp(t[~son], t[son], out[son, n])
    return enforce_order(np.exp(out), f_max_hz=np.inf)


@dataclass
class PerturbConfig:
    """Plan §5.2 step 4."""

    alpha: tuple[float, float] = (0.8, 1.25)  # global tract-length factor, log-uniform; formants scale by 1 / alpha
    offset_sd: tuple[float, float] = (0.03, 0.05)  # per-formant slow offsets: SD (fraction) drawn per utterance
    offset_corr_s: float = 0.3  # their time scale (Gaussian smoothing SD)
    bw_scale: tuple[float, float] = (0.6, 1.6)  # x Hawks & Miller, log-uniform per formant per utterance
    upper_sd: float = 0.05  # spacing jitter of F5+ around the uniform-tube spacing for the new length
    p_nasal: float = 0.15  # one nasal pole-zero pair for the whole utterance (not a label)
    hop_s: float = 0.010


def _slow_noise(rng: np.random.Generator, n: int, sd_frames: float) -> np.ndarray:
    """Zero-mean Gaussian-smoothed noise with unit SD."""
    pad = int(4 * sd_frames) + 1
    x = gaussian_filter1d(rng.standard_normal(n + 2 * pad), sd_frames)[pad : pad + n]
    return x / (x.std() + 1e-12) if n > 1 else np.zeros(n)


def perturb(formants: np.ndarray, f0: np.ndarray, voiced: np.ndarray, rng: np.random.Generator,
            cfg: PerturbConfig | None = None) -> dict:
    """New poles from smoothed tracks (T, N >= 3).

    Returns formants and bandwidths (T, N_LABEL_FORMANTS), NaN above the usable band; `extras`
    (nasal pair, kwargs for the re-filter, as in `synth.tract.sample_extras`); and `info` (the draws).
    F1..F_N move by 1 / alpha times slow per-formant offsets; F_{N+1} and up continue from F_N with
    uniform-tube spacing for the new apparent length.
    """
    cfg = cfg or PerturbConfig()
    f = np.asarray(formants, float)
    t, n = f.shape
    alpha = float(np.exp(rng.uniform(*np.log(cfg.alpha))))
    sd = float(rng.uniform(*cfg.offset_sd))
    corr = cfg.offset_corr_s / cfg.hop_s
    offsets = np.stack([_slow_noise(rng, t, corr) for _ in range(n)], 1) * sd
    low = f / alpha * np.exp(offsets)
    son = np.asarray(voiced, bool)
    length = float(np.nanmedian(apparent_vtl(low[son] if son.any() else low)))
    spacing = C_CM_S / (2 * length)
    m = np.arange(1, N_LABEL_FORMANTS - n + 1)
    high = low[:, -1:] + m * spacing * np.exp(rng.normal(0.0, cfg.upper_sd, len(m)))
    new = whole_tracks(enforce_order(np.concatenate([low, high], 1)))
    f0v = np.asarray(f0, float)
    f0_ref = float(np.nanmedian(f0v[son])) if son.any() and np.isfinite(f0v[son]).any() else 132.0
    f0_bw = np.where(son & np.isfinite(f0v), f0v, f0_ref)[:, None]
    scale = np.exp(rng.uniform(*np.log(cfg.bw_scale), N_LABEL_FORMANTS))
    bws = tract.bandwidth_prior(new, f0_bw) * scale
    extras = tract.sample_extras(rng, length, p_nasal=cfg.p_nasal, p_subglottal=0.0)
    info = {"alpha": alpha, "offset_sd": sd, "apparent_length_cm": length, "bw_scale": scale.tolist(),
            "nasal": bool(len(extras["extra_pole_f"]))}
    return {"formants": new, "bandwidths": bws, "extras": extras, "info": info}
