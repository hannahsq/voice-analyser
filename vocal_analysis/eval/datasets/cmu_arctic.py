"""CMU ARCTIC with EGG (Kominek & Black 2003): bdl (US male), jmk (Canadian male), slt (US female).

An analysis set, not a gold set with formant labels: the reference is computed, by closed-phase covariance LPC
on the EGG-marked closed phases (`eval.closed_phase`, D-019). Only these three voices have EGG; the other
ARCTIC voices' lar/ directories are empty.

Layout (`va download cmu_arctic`, verified 2026-10-09):
    <root>/cmu_us_<v>_arctic/orig/arctic_<a|b>NNNN.wav   32 kHz stereo: channel 0 speech, channel 1 EGG
    <root>/cmu_us_<v>_arctic/lab/arctic_<a|b>NNNN.lab    Festvox phone labels: "<end s> 125 <phone>"
The release's 16 kHz wav/ files are channel 0 of orig/ with zero lag (polarity inverted; band-passed correlation
-0.998 on arctic_a0001), so the labels' times hold for orig/.

Frames: 35, 50 and 65 % of every monophthong (ARPAbet aa ae ah ao eh ih iy uh uw er) lasting at least 70 ms.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

from ...data.manifest import ManifestRow, audio_info, speaker_key
from . import GoldItem

CORPUS = "cmu_arctic"
LICENCE = "CMU ARCTIC licence: any use, copyright notice kept (Carnegie Mellon University 2003)"
VOICES = {"bdl": "m", "jmk": "m", "slt": "f"}
MONOPHTHONGS = frozenset({"aa", "ae", "ah", "ao", "eh", "ih", "iy", "uh", "uw", "er"})
MIN_VOWEL_S = 0.07
FRAME_POSITIONS = (0.35, 0.5, 0.65)
SPEECH, EGG = 0, 1  # channels of orig/*.wav
_DIR = re.compile(r"^cmu_us_([a-z]+)_arctic$")


@functools.lru_cache(maxsize=4096)
def parse_lab(path: Path) -> tuple[tuple[str, float, float], ...]:
    """(phone, start s, end s) for every segment; a segment starts where the previous one ends."""
    out, prev, body = [], 0.0, False
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.strip() == "#":
            body = True
            continue
        parts = line.split()
        if body and len(parts) >= 3:
            end = float(parts[0])
            out.append((parts[2], prev, end))
            prev = end
    return tuple(out)


def vowel_frames(segments) -> tuple[np.ndarray, list[str]]:
    times, phones = [], []
    for ph, a, b in segments:
        if ph in MONOPHTHONGS and b - a >= MIN_VOWEL_S:
            times += [a + p * (b - a) for p in FRAME_POSITIONS]
            phones += [ph] * len(FRAME_POSITIONS)
    return np.array(times), phones


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and _DIR.match(p.name)):
        voice = _DIR.match(d.name).group(1)
        if voice not in VOICES:
            continue
        for wav in sorted((d / "orig").glob("arctic_*.wav")):
            if not (d / "lab" / f"{wav.stem}.lab").exists():
                continue
            sr, dur = audio_info(wav)
            rows.append(ManifestRow(wav.relative_to(root).as_posix(), CORPUS, speaker_key(CORPUS, voice), "gold", sr,
                                    round(dur, 4), LICENCE))
    return rows


def load(root: str | Path, row: ManifestRow) -> GoldItem:
    root = Path(root)
    wav = root / row.path
    lab = wav.parent.parent / "lab" / f"{wav.stem}.lab"
    segs = parse_lab(lab)
    times, phones = vowel_frames(segs)
    voice = _DIR.match(wav.parent.parent.name).group(1)
    meta = {"channel": SPEECH, "egg_channel": EGG, "voice": voice, "sex": VOICES[voice], "phones": phones,
            "segments": segs}
    return GoldItem(wav, row.speaker_id, times, None, None, meta)
