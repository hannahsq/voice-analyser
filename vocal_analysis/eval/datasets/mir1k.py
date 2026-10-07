"""MIR-1K: singing f0 gold set (Mandarin karaoke clips).

Layout (verified against the figshare copy, 2026-10-07):
    <root>/Wavfile/<singer>_<song>_<clip>.wav   16 kHz stereo: left = accompaniment, right = voice
    <root>/PitchLabel/<same stem>.pv            one value per frame, MIDI semitones, 0 = unvoiced
Frames: 40 ms window, 20 ms hop, no padding, so frame i is centred at 20 ms + i*20 ms
(frame count = floor((N - 640) / 320) + 1 for all 1000 clips). Right-channel energy
correlates with voicing (r = 0.82; left 0.06). Evaluate on the voice
channel only (v1 scope excludes accompaniment, plan §1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...data.manifest import ManifestRow, audio_info, speaker_key
from . import GoldItem

CORPUS = "mir1k"
LICENCE = "research-only (no explicit licence - VERIFY)"
HOP_S = 0.020
FIRST_FRAME_S = 0.020
VOICE_CHANNEL = 1
AUDIO_DIR = "Wavfile"


def midi_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, float)
    return np.where(m > 0, 440.0 * 2.0 ** ((m - 69.0) / 12.0), 0.0)


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    rows = []
    for wav in sorted((root / AUDIO_DIR).glob("*.wav")):
        if not (root / "PitchLabel" / f"{wav.stem}.pv").exists():
            continue
        sr, dur = audio_info(wav)
        singer = wav.stem.split("_")[0]
        rows.append(ManifestRow(wav.relative_to(root).as_posix(), CORPUS, speaker_key(CORPUS, singer), "gold", sr, round(dur, 4), LICENCE))
    return rows


def load(root: str | Path, row: ManifestRow) -> GoldItem:
    root = Path(root)
    semis = np.loadtxt(root / "PitchLabel" / f"{Path(row.path).stem}.pv", ndmin=1)
    return GoldItem(
        audio_path=root / row.path,
        speaker_id=row.speaker_id,
        times=FIRST_FRAME_S + HOP_S * np.arange(len(semis)),
        f0=midi_to_hz(semis),
        meta={"channel": VOICE_CHANNEL},
    )
