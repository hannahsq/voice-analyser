"""VTR-TIMIT formant gold set (test split only). Requires TIMIT audio (LDC licence).

Not implemented until TIMIT access is decided (plan §13, item 9). When it is:
- register rows under corpus "timit" (not "vtr_timit") with speaker ids
  "timit:<dialect>-<speaker>", so the leakage test sees the same speakers as any
  TIMIT training pool and excludes all TIMIT test speakers (plan §6.4);
- F1-F3 are hand-corrected; F4 is not and must not be scored as gold.
"""

from __future__ import annotations

from pathlib import Path

from ...data.manifest import ManifestRow
from . import GoldItem

CORPUS = "timit"
LICENCE = "LDC93S1 (TIMIT) + VTR database terms"


def scan(root: str | Path) -> list[ManifestRow]:
    raise NotImplementedError("VTR-TIMIT loader pending TIMIT licence decision (plan §13 #9)")


def load(root: str | Path, row: ManifestRow) -> GoldItem:
    raise NotImplementedError("VTR-TIMIT loader pending TIMIT licence decision (plan §13 #9)")
