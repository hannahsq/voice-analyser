"""VocalSet v1.1 (Wilkins et al. 2018): 20 professional singers, standard and extended techniques.

Layout: FULL/<singer>/<context>/<technique>/<file>.wav, singer in female1-9, male1-11. The validation
split is the authors' test singers (test_singers_technique.txt), so it is speaker-disjoint and matches
the published division.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..manifest import ManifestRow, audio_info, speaker_key

CORPUS = "vocalset"
LICENCE = "CC-BY-4.0"
_SINGER = re.compile(r"^(female|male)(\d+)$")


def _canonical(token: str) -> str | None:
    """'female3', 'F3', 'f3' -> 'female3' (the list files' format is checked when the corpus is scanned)."""
    t = token.strip().lower()
    if _SINGER.match(t):
        return t
    m = re.match(r"^([fm])(\d+)$", t)
    return f"{'female' if m.group(1) == 'f' else 'male'}{int(m.group(2))}" if m else None


def val_singers(root: Path) -> set[str]:
    path = root / "test_singers_technique.txt"
    tokens = re.split(r"[\s,;]+", path.read_text(encoding="utf-8", errors="replace"))
    singers = {c for c in map(_canonical, tokens) if c}
    if not singers:
        raise ValueError(f"no singer ids found in {path}")
    return singers


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    val = val_singers(root)
    rows = []
    for wav in sorted((root / "FULL").glob("*/*/*/*.wav")):
        singer = wav.relative_to(root / "FULL").parts[0]
        if not _SINGER.match(singer):
            raise ValueError(f"unexpected singer directory {singer!r} ({wav})")
        sr, dur = audio_info(wav)
        rows.append(ManifestRow(wav.relative_to(root).as_posix(), CORPUS, speaker_key(CORPUS, singer),
                                "val" if singer in val else "train", sr, round(dur, 4), LICENCE))
    unknown = val - {r.speaker_id.split(":", 1)[1] for r in rows}
    if unknown:
        raise ValueError(f"test singers not found in FULL/: {sorted(unknown)}")
    return rows
