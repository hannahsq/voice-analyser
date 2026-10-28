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


SHIFT_FACTORS = (0.9, 0.95, 1.05, 1.1)  # +-10 % kept every formant's coverage similar; +-20 % left few F3 / F4 shifts
SHIFT_F0S_HZ = (F0S_HZ[1], F0S_HZ[3], F0S_HZ[5])  # 100, 158, 248 Hz: the low-f0 range, below LOW_F0_MAX_HZ
SHIFT_MIN_RATIO = 1.10  # a moved formant stays at least this ratio (about 165 cents) from its neighbours


@dataclass
class ShiftClip:
    vowel: str
    length_cm: float
    f0: float
    formant: int  # 0-3 for the moved formant; -1 for the unshifted tract
    factor: float  # the moved formant's frequency ratio (1.0 unshifted)
    audio: np.ndarray
    formants: np.ndarray  # (4,) true F1-F4 after the shift, Hz


def shift_clips(vowels=VOWELS, lengths=LENGTHS_CM, f0s=SHIFT_F0S_HZ, factors=SHIFT_FACTORS, seconds: float = 1.0,
                seed: int = 0) -> list[ShiftClip]:
    """Single-formant-shift probe (D-038): the probe's tracts with one of F1-F4 moved by a factor and everything else
    (the other poles, every bandwidth, the source) unchanged. An estimator should follow the moved formant (slope 1
    in log frequency) and leave the others and the speaker's length where they are (slope 0); a head that leans on a
    length or vowel prior follows less and drags its neighbours. Shifts that would bring the moved formant within
    SHIFT_MIN_RATIO of a neighbour are skipped. Each clip's source noise is seeded by its tract and f0 only, so a
    shifted clip and its unshifted partner share the same excitation."""
    clips = []
    for v in vowels:
        for L in lengths:
            f, b = tube.poles(tube.build_shape(*tube.STORY_VOWELS[v], L))
            keep = f < 7500.0
            f, b = f[keep][:MAX_POLES], b[keep][:MAX_POLES]
            variants = [(-1, 1.0, f)]
            for n in range(4):
                for a in factors:
                    g = f.copy()
                    g[n] *= a
                    if (n > 0 and g[n] < SHIFT_MIN_RATIO * g[n - 1]) or (n + 1 < len(g) and g[n + 1] < SHIFT_MIN_RATIO * g[n]):
                        continue
                    variants.append((n, a, g))
            for k, f0 in enumerate(f0s):
                for n, a, g in variants:
                    rng = np.random.default_rng([seed, 7, VOWELS.index(v) if v in VOWELS else 99, int(L * 10), k])
                    clips.append(ShiftClip(v, L, f0, n, a, _render(g, b, f0, seconds, rng), g[:4].copy()))
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
