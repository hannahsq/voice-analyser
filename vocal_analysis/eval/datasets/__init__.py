"""Gold-set loaders (plan §6.1).

Each corpus module exposes:
    CORPUS, LICENCE
    scan(root) -> list[ManifestRow]          # every file, split="gold"
    load(root, row) -> GoldItem              # reference labels for one file

File-format details marked "ASSUMPTION" are from the corpus documentation as
remembered when this was written and must be checked against the real data
the first time it is loaded (see docs/decisions.md, D-005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np


@dataclass
class GoldItem:
    audio_path: Path
    speaker_id: str
    times: np.ndarray  # (T,) seconds, analysed-frame times
    f0: np.ndarray | None = None  # (T,) Hz; 0 or NaN = unvoiced
    formants: np.ndarray | None = None  # (T, K) Hz, column k = F(k+1); NaN = missing
    meta: dict = field(default_factory=dict)


def registry() -> dict[str, ModuleType]:
    from . import hillenbrand, mdb_stem_synth, mir1k, ptdb_tug, vtr_timit

    return {m.CORPUS: m for m in (hillenbrand, ptdb_tug, mir1k, mdb_stem_synth, vtr_timit)}
