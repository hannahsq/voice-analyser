"""f0-invariance probe (D-020): does a formant estimate move when only the pitch does?

A vocal tract with fixed resonances is excited at a sweep of f0 values; the true formants never change, so an
ideal estimator returns the same F1 at every f0. A systematic slope of the estimate against log2 f0 measures how
much the estimator infers the tract from the pitch (the training prior's coupling, D-019), which would bias
before/after comparisons where f0 changes and the tract doesn't (and vice versa).

Tracts: Story's (2005, erratum 2009) vowels through the lossy tube (`synth.tube`), at fixed lengths, with their
physical bandwidths. Source: LF (Rd 1.0), constant f0, slight jitter, aspiration at 30 dB HNR. 1 s per clip;
estimates are the median over 0.3-0.7 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..synth import tube
from ..synth.voice import CTRL, FS, Trajectories, synthesize

VOWELS = ("i", "E", "ae", "A", "o", "u")
LENGTHS_CM = (13.0, 15.5, 18.0)
F0S_HZ = tuple(float(f) for f in np.round(np.geomspace(80.0, 400.0, 8), 1))
MAX_POLES = 8
LOW_F0_MAX_HZ = 250.0


@dataclass
class ProbeClip:
    vowel: str
    length_cm: float
    f0: float
    audio: np.ndarray  # float32, FS
    formants: np.ndarray  # (4,) true F1-F4, Hz


def _render(freqs: np.ndarray, bws: np.ndarray, f0: float, seconds: float, rng: np.random.Generator) -> np.ndarray:
    c = int(seconds * FS / CTRL) + 1
    pf = np.full((c, MAX_POLES), np.nan)
    pb = np.full((c, MAX_POLES), np.nan)
    pf[:, : len(freqs)], pb[:, : len(bws)] = freqs, bws
    traj = Trajectories(f0=np.full(c, f0), rd=np.full(c, 1.0), voice_amp=np.ones(c), whisper_amp=np.zeros(c),
                        fric_amp=np.zeros(c), poles_f=pf, poles_b=pb, zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
    y = synthesize(traj, rng, jitter=0.005, shimmer_db=0.2, hnr_db=30.0)["total"]
    return (0.9 * y / np.max(np.abs(y))).astype(np.float32)


def probe_clips(vowels=VOWELS, lengths=LENGTHS_CM, f0s=F0S_HZ, seconds: float = 1.0, seed: int = 0) -> list[ProbeClip]:
    clips = []
    for v in vowels:
        for L in lengths:
            f, b = tube.poles(tube.build_shape(*tube.STORY_VOWELS[v], L))
            keep = f < 7500.0
            f, b = f[keep][:MAX_POLES], b[keep][:MAX_POLES]
            for k, f0 in enumerate(f0s):
                rng = np.random.default_rng([seed, VOWELS.index(v) if v in VOWELS else 99, int(L * 10), k])
                clips.append(ProbeClip(v, L, f0, _render(f, b, f0, seconds, rng), f[:4].copy()))
    return clips


def run(predict, clips: list[ProbeClip] | None = None) -> dict:
    """predict(audio, sr, times, meta) -> (T, >=1) formants (a FormantBaseline's predict).

    Returns per (vowel, length): the F1 slope against log2 f0 (Hz per octave, least squares over the sweep) and
    the median F1 error at each f0; plus pooled summaries: median slope, median |slope|, and the median signed
    F1 error by f0.
    """
    clips = clips or probe_clips()
    times = np.arange(0.3, 0.7001, 0.01)
    est = np.array([np.nanmedian(predict(c.audio, FS, times, {})[:, 0]) for c in clips])
    keys = sorted({(c.vowel, c.length_cm) for c in clips})
    f0s = sorted({c.f0 for c in clips})
    slopes, err = {}, np.full((len(keys), len(f0s)), np.nan)
    for i, key in enumerate(keys):
        idx = [j for j, c in enumerate(clips) if (c.vowel, c.length_cm) == key]
        x = np.log2([clips[j].f0 for j in idx])
        y = est[idx]
        truth = clips[idx[0]].formants[0]
        ok = np.isfinite(y)
        slopes[key] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan
        for j in idx:
            err[i, f0s.index(clips[j].f0)] = est[j] - truth
    s = np.array(list(slopes.values()))
    # f0 <= 250 Hz separately (D-020 result): above it F1 of close vowels nears or falls below f0, where some slope
    # comes from the resolution limit rather than from the training prior
    x_low = np.log2([f for f in f0s if f <= LOW_F0_MAX_HZ])
    s_low = []
    for i in range(len(keys)):
        y = err[i, : len(x_low)]
        ok = np.isfinite(y)
        s_low.append(float(np.polyfit(x_low[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan)
    s_low = np.array(s_low)
    return {"slopes": slopes, "f0s": f0s, "keys": keys, "f1_error": err,
            "median_slope": float(np.nanmedian(s)), "median_abs_slope": float(np.nanmedian(np.abs(s))),
            "median_slope_low_f0": float(np.nanmedian(s_low)), "median_abs_slope_low_f0": float(np.nanmedian(np.abs(s_low))),
            "median_error_by_f0": np.nanmedian(err, axis=0)}
