"""Corpus manifests (plan §5.6, §6.4).

One CSV per corpus under data/manifests/, one row per audio file. Paths are
relative to the corpus root so manifests are portable between machines.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test", "gold")


@dataclass(frozen=True)
class ManifestRow:
    path: str
    corpus: str
    speaker_id: str  # globally unique: "<corpus>:<local id>"
    split: str
    sr: int
    duration: float
    licence: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"{self.path}: split {self.split!r} not in {SPLITS}")
        if not self.speaker_id.startswith(f"{self.corpus}:"):
            raise ValueError(f"{self.path}: speaker_id {self.speaker_id!r} must be prefixed with '{self.corpus}:'")
        if not self.licence:
            raise ValueError(f"{self.path}: licence is required")


FIELDS = [f.name for f in fields(ManifestRow)]


def speaker_key(corpus: str, local_id: str) -> str:
    return f"{corpus}:{local_id}"


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r.path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path}: columns {reader.fieldnames} != {FIELDS}")
        return [
            ManifestRow(**{**r, "sr": int(r["sr"]), "duration": float(r["duration"])})
            for r in reader
        ]


def read_manifests(directory: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for p in sorted(Path(directory).glob("*.csv")):
        rows.extend(read_manifest(p))
    return rows


def manifest_hash(path: str | Path) -> str:
    """Content hash recorded with every training run and report (plan §0)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def audio_info(path: str | Path) -> tuple[int, float]:
    import soundfile as sf

    info = sf.info(str(path))
    return info.samplerate, info.frames / info.samplerate
