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


def load_audio(item: GoldItem, sr: int = 16000) -> np.ndarray:
    """Mono float32 audio at `sr`: picks meta["channel"] if set (e.g. MIR-1K voice), else averages channels."""
    import soundfile as sf
    from scipy.signal import resample_poly

    x, file_sr = sf.read(str(item.audio_path), dtype="float32", always_2d=True)
    ch = item.meta.get("channel")
    x = x[:, ch] if ch is not None else x.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, int(file_sr) // g).astype(np.float32)
    return x


def registry() -> dict[str, ModuleType]:
    from . import cmu_arctic, hillenbrand, mdb_stem_synth, mir1k, ptdb_tug, vtr_timit

    return {m.CORPUS: m for m in (hillenbrand, ptdb_tug, mir1k, mdb_stem_synth, vtr_timit, cmu_arctic)}
