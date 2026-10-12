"""Zeroth-Korean (OpenSLR 40): read Korean speech, 115 speakers, 16 kHz FLAC (D-017).

Layout: {train,test}_data_01/003/<speaker>/<speaker>_003_<n>.flac. The corpus's own test split becomes
`val`; its train and test speakers are disjoint, and the scan refuses a corpus where they are not.

Privacy: AUDIO_INFO lists each speaker's real name. This scanner never opens it; speaker ids are the numeric
directory names only (`zeroth_korean:<id>`).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..manifest import ManifestRow, audio_info, speaker_key

CORPUS = "zeroth_korean"
LICENCE = "CC-BY-4.0"
SPLIT_DIRS = {"train_data_01": "train", "test_data_01": "val"}
_SPEAKER = re.compile(r"^\d+$")


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    rows, speakers = [], {}
    for d, split in SPLIT_DIRS.items():
        for flac in sorted((root / d).glob("*/*/*.flac")):
            spk = flac.parent.name
            if not _SPEAKER.match(spk):
                raise ValueError(f"unexpected speaker directory {spk!r} ({flac})")
            speakers.setdefault(spk, set()).add(split)
            sr, dur = audio_info(flac)
            rows.append(ManifestRow(flac.relative_to(root).as_posix(), CORPUS, speaker_key(CORPUS, spk), split, sr,
                                    round(dur, 4), LICENCE))
    both = sorted(s for s, sp in speakers.items() if len(sp) > 1)
    if both:
        raise ValueError(f"speakers in both train and test: {both}")
    if not rows:
        raise ValueError(f"no FLAC files under {root}/{{{','.join(SPLIT_DIRS)}}}")
    return rows
