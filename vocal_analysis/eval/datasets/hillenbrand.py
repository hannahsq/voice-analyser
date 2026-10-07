"""Hillenbrand et al. (1995) American English vowels: cross-domain formant gold set.

Layout (verified against the original distribution, 2026-10-07):
    <root>/vowdata.dat     one row per token after a free-text header
    <root>/timedata.dat    vowel start/end and two judges' steady-state times (ms)
    <root>/<token>.wav     16 kHz mono PCM, e.g. m01ae.wav

vowdata.dat columns:
    token  dur_ms  f0  F1 F2 F3 F4  F1_20 F2_20 F3_20  F1_50 F2_50 F3_50  F1_80 F2_80 F3_80
"steady state" values plus F1-F3 at 20/50/80 % of vowel duration; 0 means
"not measurable". Token names are <group><speaker 2 digits><vowel 2 letters>,
group in m/w/b/g (men, women, boys, girls).

Each token yields four labelled frames: steady state (f0, F1-F4; time = mean
of the two judges) and 20/50/80 % of the vowel nucleus (F1-F3 only).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

from ...data.manifest import ManifestRow, audio_info, speaker_key
from . import GoldItem

CORPUS = "hillenbrand"
LICENCE = "free for research; distributed by J. Hillenbrand, mirrored with permission (no formal licence)"
FRAME_KINDS = ("steady", "20%", "50%", "80%")
_TOKEN = re.compile(r"^([mwbg]\d\d)([a-z]{2})$")


def _rows(path: Path, min_fields: int) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split()
        if len(parts) >= min_fields and _TOKEN.match(parts[0]):
            out[parts[0]] = np.array([float(x) for x in parts[1:]])
    return out


@functools.lru_cache(maxsize=4)
def parse_vowdata(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for token, nums in _rows(path, 7).items():
        nums = nums.copy()
        nums[nums == 0] = np.nan
        vals = {"dur_ms": nums[0], "f0": nums[1], "formants_ss": nums[2:6]}
        if len(nums) >= 15:
            vals["formants_20_50_80"] = nums[6:15].reshape(3, 3)
        out[token] = vals
    return out


@functools.lru_cache(maxsize=4)
def parse_timedata(path: Path) -> dict[str, dict[str, float]]:
    """Times in seconds: start, end, steady (mean of the two judges)."""
    return {
        token: {"start": v[0] / 1000, "end": v[1] / 1000, "steady": (v[2] + v[3]) / 2000}
        for token, v in _rows(path, 5).items()
    }


def _wavs(root: Path) -> dict[str, Path]:
    return {p.stem.lower(): p for p in root.rglob("*.wav") if _TOKEN.match(p.stem.lower())}


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    labels = parse_vowdata(root / "vowdata.dat")
    rows = []
    for token, wav in sorted(_wavs(root).items()):
        if token not in labels:
            continue
        sr, dur = audio_info(wav)
        spk = _TOKEN.match(token).group(1)
        rows.append(ManifestRow(wav.relative_to(root).as_posix(), CORPUS, speaker_key(CORPUS, spk), "gold", sr, round(dur, 4), LICENCE))
    return rows


def load(root: str | Path, row: ManifestRow) -> GoldItem:
    root = Path(root)
    token = Path(row.path).stem.lower()
    lab = parse_vowdata(root / "vowdata.dat")[token]
    m = _TOKEN.match(token)
    meta = {"vowel": m.group(2), "group": m.group(1)[0], "dur_ms": lab["dur_ms"], "frame_kinds": FRAME_KINDS[:1]}

    timing = parse_timedata(root / "timedata.dat").get(token) if (root / "timedata.dat").exists() else None
    if timing is None or "formants_20_50_80" not in lab:
        # Steady-state frame only, time unknown.
        times = np.array([timing["steady"] if timing else np.nan])
        return GoldItem(root / row.path, row.speaker_id, times, np.array([lab["f0"]]), lab["formants_ss"][None, :], meta)

    span = timing["end"] - timing["start"]
    times = np.array([timing["steady"], *(timing["start"] + p * span for p in (0.2, 0.5, 0.8))])
    formants = np.full((4, 4), np.nan)
    formants[0] = lab["formants_ss"]
    formants[1:, :3] = lab["formants_20_50_80"]
    f0 = np.array([lab["f0"], np.nan, np.nan, np.nan])  # f0 is only given at steady state
    meta.update(frame_kinds=FRAME_KINDS, start_s=timing["start"], end_s=timing["end"])
    return GoldItem(root / row.path, row.speaker_id, times, f0, formants, meta)
