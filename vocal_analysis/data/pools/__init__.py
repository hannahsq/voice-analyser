"""Training-pool corpora (plan §5.6): real audio for M3 resynthesis and later rounds, never gold.

Each corpus module exposes CORPUS, LICENCE and scan(root) -> list[ManifestRow] with speaker-disjoint
train/val splits. Speakers are corpus-prefixed, so the leakage test (§6.4) can only catch reuse within
a corpus; overlap of real people across corpora is checked by hand and recorded in docs/decisions.md.
"""

from __future__ import annotations

from types import ModuleType


def registry() -> dict[str, ModuleType]:
    from . import vocalset

    return {m.CORPUS: m for m in (vocalset,)}
