"""PTDB-TUG: speech f0 gold set with laryngograph-derived reference.

Layout (verified against SPEECH_DATA_ZIPPED.zip, 2026-10-07; 4718 utterances, 20 speakers):
    <root>/{FEMALE,MALE}/MIC/<spk>/mic_<spk>_<utt>.wav   48 kHz mono
    <root>/{FEMALE,MALE}/LAR/<spk>/lar_<spk>_<utt>.wav   laryngograph, same length as MIC
    <root>/{FEMALE,MALE}/REF/<spk>/ref_<spk>_<utt>.f0    4 columns, first = f0 in Hz (0 = unvoiced)
One row per 10 ms frame; every file has floor(N / 480) - 6 rows. Frame i is
centred at 20 ms + i * 10 ms: measured by aligning the reference against a
Praat f0 track of the laryngograph signal (79 files; mean |error| minimised at
20 ms, per-file IQR 19-21 ms). The documented 32 ms window would suggest 16 ms,
which is off by 4 ms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...data.manifest import ManifestRow, audio_info, speaker_key
from . import GoldItem

CORPUS = "ptdb_tug"
LICENCE = "ODbL-1.0 (VERIFY)"
HOP_S = 0.010
FRAME_OFFSET_S = 0.020


def _ref_path(root: Path, wav_rel: str) -> Path:
    wav = Path(wav_rel)
    spk_dir, mic_dir = wav.parent, wav.parent.parent
    return root / mic_dir.parent / "REF" / spk_dir.name / (wav.stem.replace("mic_", "ref_", 1) + ".f0")


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    rows = []
    for wav in sorted(root.rglob("mic_*.wav")):
        rel = wav.relative_to(root).as_posix()
        if not _ref_path(root, rel).exists():
            continue
        sr, dur = audio_info(wav)
        rows.append(ManifestRow(rel, CORPUS, speaker_key(CORPUS, wav.parent.name), "gold", sr, round(dur, 4), LICENCE))
    return rows


def frame_labels(root: str | Path, row: ManifestRow, item: GoldItem) -> np.ndarray:
    """Phonation category per reference frame, from the laryngograph channel (eval/phonation.py)."""
    import soundfile as sf

    from .. import phonation

    lar = Path(root) / row.path.replace("/MIC/", "/LAR/").replace("mic_", "lar_", 1)
    egg, sr = sf.read(lar, dtype="float64")
    if egg.ndim > 1:
        egg = egg.mean(axis=1)
    return phonation.classify(*phonation.egg_lag_correlations(egg, sr, item.times, item.f0))


def load(root: str | Path, row: ManifestRow) -> GoldItem:
    root = Path(root)
    data = np.loadtxt(_ref_path(root, row.path), ndmin=2)
    f0 = data[:, 0]
    return GoldItem(
        audio_path=root / row.path,
        speaker_id=row.speaker_id,
        times=FRAME_OFFSET_S + HOP_S * np.arange(len(f0)),
        f0=f0,
    )
