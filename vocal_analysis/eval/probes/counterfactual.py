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


def render(item: Item, alpha: float = 1.0, r: float = 1.0, bw: float = 1.0, extras: str | None = None) -> tuple[np.ndarray, dict]:
    """-> (audio, labels): labels hold the exact formants and bandwidths (T, N), the frames that are pure re-filtered signal
    (`scored`), and the voicing and f0 carried over from the tracker (f0 times r).

    bw scales every bandwidth (probe 7). extras ("nasal" or "subglottal", probe 8) adds that pole-zero pair set, drawn
    from the training distribution (`synth.tract.sample_extras`) with a seed fixed per item; extras are not labels."""
    source, audio, src = item.source, item.audio, item.src
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
    if extras:
        rng = np.random.default_rng(zlib.crc32(item.key.encode()))
        p["extras"] = tract.sample_extras(rng, p["info"]["apparent_length_cm"], p_nasal=float(extras == "nasal"),
                                          p_subglottal=float(extras == "subglottal"))
    y = refilter(source, p["formants"], p["bandwidths"], p["extras"])
    son = src["sonorant"]
    ws = _ramp_mask(son, len(y), int(0.02 * FS))
    on = ws > 0.99
    if on.any():
        y *= np.sqrt(np.sum(audio[on] ** 2) / max(np.sum(y[on] ** 2), 1e-20))
    y = ws * y + (1 - ws) * audio
    full = np.array([on[i * HOP : (i + 1) * HOP].all() if i * HOP < len(on) else False for i in range(n_frames)])
    peak = np.max(np.abs(y))
    labels = {"formants": p["formants"], "bandwidths": p["bandwidths"], "scored": son & full, "voiced": src["voiced"], "f0": src["f0"]}
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
    raise ValueError(kind)
