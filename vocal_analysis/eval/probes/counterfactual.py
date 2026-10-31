"""Counterfactual resynthesis probe (D-050, probe 1).

Held-out real recordings (a resynthesised store's non-gold validation split) keep their own inverse-filtered source
and are re-filtered through a deterministic tract: the smoothed tracker tracks scaled by 1 / alpha, prior bandwidths,
no random offsets, no extra pole-zero pairs. Arms:

- **length**: alpha in ALPHAS, f0 untouched. Formants should scale by exactly the labels' change (equivariance, ideal
  slope 1), the anatomical length by alpha (ideal 1), f0 not at all (ideal 0).
- **f0**: the source resampled by r in RATIOS (f0 x r; the clip also gets 1 / r times as long, so speaking rate moves
  with it), tracks time-warped to match, alpha 1. Formants and length should not move (ideal 0), f0 should (ideal 1).

The source keeps what the inverse filter left of the real tract (`tract_left`, beta). That residue does not scale
with alpha, and resampling moves it with r, so every slope is also reported for items with small |beta|.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ...resynth.corpus import _analysis_name, load_audio
from ...resynth.render import FS, HOP, ResynthConfig, _ramp_mask, cut_points, refilter, source_from_audio
from ...resynth.tracks import PerturbConfig, perturb
from ...synth import tract

ALPHAS = (0.85, 0.92, 1.0, 1.09, 1.18)
RATIOS = (0.84, 0.92, 1.0, 1.09, 1.19)
CLIP_S = (3.0, 6.0)
SMALL_BETA = 0.2
SHIFT_MIN_RATIO = 1.10  # D-038: a moved formant stays at least this ratio from its neighbours
SRC_KEYS = ("f0", "voiced", "sonorant", "tracks", "tract_left")


@dataclass
class Item:
    key: str  # recording path within the corpus, plus the clip's start frame
    group: str  # speaker or singer id (numeric for Zeroth)
    audio: np.ndarray  # the original clip, 16 kHz
    source: np.ndarray  # its inverse-filtered source
    src: dict  # per-frame SRC_KEYS for the clip
    beta: float  # median tract_left over the clip's voiced, measured frames (NaN if none)


def _cfg_from_store(index: dict) -> ResynthConfig:
    c = dict(index["resynth_config"])
    c.pop("perturb", None)
    if "clip_s" in c:
        c["clip_s"] = tuple(c["clip_s"])
    return ResynthConfig(**c)


def load_items(store: str | Path, corpus_root: str | Path, per_group: int = 6, group_part: int = 2,
               analysis_root: str | Path = "data/resynth/analysis") -> list[Item]:
    """per_group recordings per speaker, spread evenly through the store's file list; from each, the 3-6 s clip
    (cut as the resynthesised sets are) with the most voiced frames. group_part: the path component naming the
    speaker (Zeroth: test_data_01/003/<speaker>/...)."""
    store = Path(store)
    index = json.loads((store / "index.json").read_text(encoding="utf-8"))
    cfg = _cfg_from_store(index)
    adir = Path(analysis_root) / index["corpus"] / index["teacher"]["sha256"]
    by_group: dict[str, list[str]] = {}
    for f in index["files"]:
        by_group.setdefault(f.split("/")[group_part], []).append(f)
    items = []
    for g, files in sorted(by_group.items()):
        pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(per_group, len(files)))]
        for f in pick:
            audio = load_audio(Path(corpus_root) / f)
            a = dict(np.load(adir / _analysis_name(f)))
            src = source_from_audio(audio, a, cfg)
            n = len(src["f0"])
            cuts = cut_points(audio, n, CLIP_S)
            a0, b0 = max(zip(cuts[:-1], cuts[1:]), key=lambda ab: src["voiced"][ab[0]:ab[1]].sum())
            seg = {k: np.asarray(src[k])[a0:b0] for k in SRC_KEYS}
            m = seg["voiced"] & np.isfinite(seg["tract_left"])
            beta = float(np.median(seg["tract_left"][m])) if m.any() else np.nan
            items.append(Item(f"{f}@{a0}", g, audio[a0 * HOP : b0 * HOP], src["source"][a0 * HOP : b0 * HOP], seg, beta))
    return items


def _det_tract(alpha: float, bw: float = 1.0) -> PerturbConfig:
    return PerturbConfig(alpha=(alpha, alpha), offset_sd=(0.0, 0.0), bw_scale=(bw, bw), upper_sd=0.0, p_nasal=0.0)


def _warp(src: dict, r: float, n_new: int) -> dict:
    """Per-frame values at the resampled clip's frames: new frame j sits at old frame j * r."""
    pos = np.arange(n_new) * r
    n_old = len(src["f0"])
    near = np.clip(np.round(pos).astype(int), 0, n_old - 1)
    t = np.arange(n_old)
    tracks = np.exp(np.stack([np.interp(pos, t, np.log(src["tracks"][:, k])) for k in range(src["tracks"].shape[1])], 1))
    return {"f0": np.interp(pos, t, src["f0"]) * r, "voiced": src["voiced"][near], "sonorant": src["sonorant"][near],
            "tracks": tracks, "tract_left": src["tract_left"][near]}


UNVOICED_UNIT = 80  # samples (5 ms): overlap-add unit where there is no pitch period to keep


def _stretch(signals: list[np.ndarray], f0: np.ndarray, voiced: np.ndarray, s: float) -> list[np.ndarray]:
    """Time-stretch by s without a pitch change (probe 4, D-050 amendment 2): pitch-synchronous overlap-add.

    Analysis marks one local period apart (from the per-frame f0 where voiced, UNVOICED_UNIT elsewhere); output marks
    spaced the same way along the stretched time axis, each taking the analysis mark nearest its time / s, windowed
    by a Hann window two periods long. Whole periods are repeated or dropped, so the period, and the formants of the
    source's filter, stay. Every signal gets the same marks (the source and the clip it is blended with)."""
    n = len(signals[0])

    def period(t: float) -> int:
        i = min(int(t) // HOP, len(f0) - 1)
        return int(round(FS / f0[i])) if voiced[i] and f0[i] > 0 else UNVOICED_UNIT

    marks = [0]
    while marks[-1] < n:
        marks.append(marks[-1] + period(marks[-1]))
    marks = np.array(marks[:-1])
    n_out = int(round(n * s))
    pad = 2 * int(FS / 50)
    outs = [np.zeros(n_out + 2 * pad) for _ in signals]
    xs = [np.r_[np.zeros(pad), np.asarray(x, float), np.zeros(pad)] for x in signals]
    t = 0
    while t < n_out:
        a = int(marks[np.argmin(np.abs(marks - t / s))])
        p = period(a)
        w = np.hanning(2 * p + 1)
        for x, y in zip(xs, outs):
            y[pad + t - p : pad + t + p + 1] += w * x[pad + a - p : pad + a + p + 1]
        t += p
    return [y[pad : pad + n_out].astype(np.float32) for y in outs]


def render(item: Item, alpha: float = 1.0, r: float = 1.0, bw: float = 1.0, extras: str | None = None,
           shift: tuple[int, float] | None = None, stretch: float = 1.0, hold_bw: bool = False) -> tuple[np.ndarray, dict]:
    """-> (audio, labels): labels hold the exact formants and bandwidths (T, N), the frames that are pure re-filtered signal
    (`scored`), and the voicing and f0 carried over from the tracker (f0 times r).

    bw scales every bandwidth (probe 7). extras ("nasal" or "subglottal", probe 8) adds that pole-zero pair set, drawn
    from the training distribution (`synth.tract.sample_extras`) with a seed fixed per item; extras are not labels.
    shift (n, factor) moves formant n (0-based) alone by factor (probe 2); frames where it comes within
    SHIFT_MIN_RATIO of a neighbour are left out of `scored` (as D-038's synthetic probe).
    stretch s (probe 4) lengthens the clip by s without a pitch change (`_stretch`); the tracks are time-warped,
    f0 and the formants keep their values. hold_bw (probe 12) keeps the bandwidths at their alpha = 1 values in Hz
    while alpha moves the formants (normally they follow the formants through the prior)."""
    source, audio, src = item.source, item.audio, item.src
    if stretch != 1.0:
        assert r == 1.0, "stretch and r do not combine"
        source, audio = _stretch([source, audio], np.asarray(src["f0"]), np.asarray(src["voiced"]), stretch)
        src = _warp(src, 1.0 / stretch, len(source) // HOP + 1)
        src["f0"] = src["f0"] * stretch  # _warp scales f0 by its rate; a stretch keeps it
    if r != 1.0:
        fr = Fraction(r).limit_denominator(50)
        source = resample_poly(source, fr.denominator, fr.numerator)
        audio = resample_poly(audio, fr.denominator, fr.numerator)
        src = _warp(src, fr.numerator / fr.denominator, len(source) // HOP + 1)
    n_frames = len(source) // HOP + 1
    # frame i at i * HOP: a clip of k * HOP samples has k + 1 frames, one more than its slice of the per-frame values
    src = {k: np.concatenate([np.asarray(v)[:n_frames]] + [np.asarray(v)[-1:]] * max(0, n_frames - len(v)))
           for k, v in src.items()}
    p = perturb(src["tracks"], src["f0"], src["voiced"], np.random.default_rng(0), _det_tract(alpha, bw))
    if hold_bw and alpha != 1.0:
        held = perturb(src["tracks"], src["f0"], src["voiced"], np.random.default_rng(0), _det_tract(1.0, bw))["bandwidths"]
        p["bandwidths"] = np.where(np.isfinite(held), held, p["bandwidths"])
    if extras:
        rng = np.random.default_rng(zlib.crc32(item.key.encode()))
        p["extras"] = tract.sample_extras(rng, p["info"]["apparent_length_cm"], p_nasal=float(extras == "nasal"),
                                          p_subglottal=float(extras == "subglottal"))
    crowded = np.zeros(n_frames, bool)
    if shift is not None:
        k, factor = shift
        f = p["formants"].copy()
        f[:, k] *= factor
        with np.errstate(invalid="ignore"):
            if k > 0:
                crowded |= f[:, k] < f[:, k - 1] * SHIFT_MIN_RATIO
            if k + 1 < f.shape[1]:
                crowded |= f[:, k + 1] < f[:, k] * SHIFT_MIN_RATIO
        f[:, k] = np.where(crowded, p["formants"][:, k], f[:, k])  # never cross a neighbour
        p["formants"] = f
    y = refilter(source, p["formants"], p["bandwidths"], p["extras"])
    son = src["sonorant"]
    ws = _ramp_mask(son, len(y), int(0.02 * FS))
    on = ws > 0.99
    if on.any():
        y *= np.sqrt(np.sum(audio[on] ** 2) / max(np.sum(y[on] ** 2), 1e-20))
    y = ws * y + (1 - ws) * audio
    full = np.array([on[i * HOP : (i + 1) * HOP].all() if i * HOP < len(on) else False for i in range(n_frames)])
    peak = np.max(np.abs(y))
    labels = {"formants": p["formants"], "bandwidths": p["bandwidths"], "scored": son & full & ~crowded, "voiced": src["voiced"], "f0": src["f0"]}
    return (y / peak * 0.9 if peak > 0 else y).astype(np.float32), labels


def conditions() -> list[tuple[str, float, float]]:
    """(arm, alpha, r) for every rendered condition; the reference (1, 1) once, shared by both arms."""
    return ([("ref", 1.0, 1.0)] + [("length", a, 1.0) for a in ALPHAS if a != 1.0]
            + [("f0", 1.0, r) for r in RATIOS if r != 1.0])


def chain(audio: np.ndarray, kind: str, value: float) -> np.ndarray:
    """Probe 9's signal-chain conditions on a clip: "gain" (x value, no clipping: the peak is not renormalised),
    "dc" (offset of value times the peak), "clip" (hard clip at value times the peak), "resample" (16 kHz -> value Hz
    -> 16 kHz, polyphase both ways)."""
    x = np.asarray(audio, float)
    if kind == "gain":
        return (x * value).astype(np.float32)
    if kind == "dc":
        return (x + value * np.max(np.abs(x))).astype(np.float32)
    if kind == "clip":
        c = value * np.max(np.abs(x))
        return np.clip(x, -c, c).astype(np.float32)
    if kind == "resample":
        fr = Fraction(int(value), FS)
        up = resample_poly(x, fr.numerator, fr.denominator)
        return resample_poly(up, fr.denominator, fr.numerator)[: len(x)].astype(np.float32)
    if kind == "offset":  # probe 4: value samples of silence first (a sub-hop shift of every frame)
        return np.r_[np.zeros(int(value)), x].astype(np.float32)
    if kind == "polarity":
        return (-x).astype(np.float32)
    if kind == "pad":  # value seconds of silence both sides
        z = np.zeros(int(round(value * FS)))
        return np.r_[z, x, z].astype(np.float32)
    raise ValueError(kind)


VALLEY_SIGMA = 0.1  # peak width (SD) as a fraction of the F1-F2 distance in log frequency
VALLEY_EDGE = 0.25  # fill: flat between F1 + 0.25 d and F2 - 0.25 d ...
VALLEY_RAMP = 0.15  # ... with raised-cosine ramps of 0.15 d outside that


def valley(audio: np.ndarray, formants: np.ndarray, kind: str, pos: float, level_db: float) -> np.ndarray:
    """Probe 14 (D-055): energy added to (or taken from) the F1-F2 valley of a resynthesised clip, frame by frame,
    placed from that frame's exact F1 and F2 (labels, HOP frames), so the formant peaks themselves are left alone.
    kind "peak": a Gaussian in log frequency at F1 + pos (F2 - F1) (log), SD VALLEY_SIGMA of the distance, level_db
    at its centre; "fill": level_db over the middle of the valley (VALLEY_EDGE, VALLEY_RAMP); "null": the same STFT path at
    0 dB (the processing floor). Zero-phase gain on a 40 ms Hann STFT at the label hop; frames without F1 and F2 unchanged."""
    from scipy.signal import istft, stft

    x = np.asarray(audio, float)
    nper = 4 * HOP
    f, t, Z = stft(x, FS, window="hann", nperseg=nper, noverlap=nper - HOP, boundary="zeros", padded=True)
    u = np.log2(np.maximum(f, 1.0))
    g_db = np.zeros(Z.shape)
    if kind != "null":
        for k in range(Z.shape[1]):
            if k >= len(formants) or not np.all(np.isfinite(formants[k, :2])):
                continue
            a, b = np.log2(formants[k, 0]), np.log2(formants[k, 1])
            d = b - a
            if kind == "peak":
                g_db[:, k] = level_db * np.exp(-0.5 * ((u - (a + pos * d)) / (VALLEY_SIGMA * d)) ** 2)
            elif kind == "fill":
                lo, hi, r = a + VALLEY_EDGE * d, b - VALLEY_EDGE * d, VALLEY_RAMP * d
                w = np.clip(np.minimum(u - (lo - r), (hi + r) - u) / r, 0.0, 1.0)
                g_db[:, k] = level_db * 0.5 * (1 - np.cos(np.pi * w))
            else:
                raise ValueError(kind)
    _, y = istft(Z * 10 ** (g_db / 20), FS, window="hann", nperseg=nper, noverlap=nper - HOP, boundary=True)
    y = np.r_[y, np.zeros(max(0, len(x) - len(y)))][: len(x)]
    return y.astype(np.float32)
