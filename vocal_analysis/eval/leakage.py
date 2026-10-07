"""Leakage rules (plan §6.4).

- Gold speakers and gold audio never appear in any non-gold split.
- Splits are speaker-disjoint (a speaker belongs to exactly one split).
- Extra exclusions (e.g. TIMIT test speakers when TIMIT train is used) are
  passed in as speaker ids that must not appear in train/val.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from ..data.manifest import ManifestRow


@dataclass
class LeakageReport:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise AssertionError("data leakage detected:\n  " + "\n  ".join(self.errors))


def _audio_key(row: ManifestRow) -> str:
    return f"{row.corpus}/{PurePosixPath(row.path.replace(chr(92), '/'))}"


def check_leakage(
    rows: Iterable[ManifestRow],
    excluded_speakers: Iterable[str] = (),
    max_errors: int = 50,
) -> LeakageReport:
    rows = list(rows)
    errors: list[str] = []

    splits_by_speaker: dict[str, set[str]] = defaultdict(set)
    splits_by_audio: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        splits_by_speaker[r.speaker_id].add(r.split)
        splits_by_audio[_audio_key(r)].add(r.split)

    for spk, splits in sorted(splits_by_speaker.items()):
        if len(splits) > 1:
            kind = "gold speaker in training pool" if "gold" in splits else "speaker in multiple splits"
            errors.append(f"{kind}: {spk} -> {sorted(splits)}")

    for audio, splits in sorted(splits_by_audio.items()):
        if len(splits) > 1:
            errors.append(f"audio in multiple splits: {audio} -> {sorted(splits)}")

    excluded = set(excluded_speakers)
    for spk in sorted(excluded & {r.speaker_id for r in rows if r.split in ("train", "val")}):
        errors.append(f"excluded speaker in train/val: {spk}")

    return LeakageReport(errors[:max_errors] + ([f"... {len(errors) - max_errors} more"] if len(errors) > max_errors else []))
