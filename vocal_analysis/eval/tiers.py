"""Evaluation tiers (decisions D-024): which gold speakers may inform which decisions.

- dev: used for every comparison and decision.
- semi: semi-private. Confirms a decision made on dev ("does it replicate?"), never makes one; each use is logged in
  docs/decisions.md.
- lockbox: opened only at milestone acceptance, with the rule set beforehand; each opening is logged.

Gold sets used before 2026-10-11 are split 60 / 40 (dev / semi) by speaker, balanced within groups (Hillenbrand:
men, women, boys, girls; PTDB-TUG: women, men). Corpora too small to split by speaker, or used as analysis tools
(ARCTIC, OPENGLOT, Birkholz 2019), are dev only. Sets not yet looked at go to the lockbox, or are split semi / lockbox
when large enough. The assignment is written once (`va tiers`) to tiers.json next to this file and committed, so it
never changes silently.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..data.manifest import ManifestRow

TIERS = ("dev", "semi", "lockbox")
TIER_FILE = Path(__file__).with_name("tiers.json")
DEV_FRAC = 0.6
SEED = 20261011
DEV_ONLY = ("cmu_arctic", "openglot", "birkholz2019")  # too few speakers to split, or analysis tools
LOCKBOX = ("mocha_timit",)  # not yet looked at, too small to split
# Single speakers kept in the lockbox whatever their corpus's split (D-035): the USC 75-speaker MRI set's 16 speakers
# with hand-drawn contours in 75-Speaker Annot-16, its best anatomical (MRI midline) VTL references
LOCKBOX_SPEAKERS = frozenset(f"usc75:sub{n:03d}" for n in (9, 13, 22, 23, 28, 30, 34, 43, 47, 61, 64, 68, 70, 71, 72, 74))


def _group(speaker_id: str) -> str:
    corpus, local = speaker_id.split(":", 1)
    if corpus == "hillenbrand":
        return local[0]  # b, g, m, w
    if corpus == "ptdb_tug":
        return local[0]  # F, M
    return ""


def assign(speakers: Iterable[str], dev_frac: float = DEV_FRAC, seed: int = SEED) -> dict[str, str]:
    """Gold speakers -> tier. Within each corpus and group, a seeded shuffle; the first round(dev_frac * n) are dev,
    the rest semi. DEV_ONLY corpora are all dev, LOCKBOX corpora and LOCKBOX_SPEAKERS all lockbox."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    out: dict[str, str] = {}
    for s in sorted(set(speakers)):
        if s in LOCKBOX_SPEAKERS:
            out[s] = "lockbox"
            continue
        groups[(s.split(":", 1)[0], _group(s))].append(s)
    for (corpus, grp), members in sorted(groups.items()):
        if corpus in DEV_ONLY or corpus in LOCKBOX:
            out.update({s: "dev" if corpus in DEV_ONLY else "lockbox" for s in members})
            continue
        rng = random.Random(f"{seed}:{corpus}:{grp}")
        shuffled = members[:]
        rng.shuffle(shuffled)
        n_dev = round(dev_frac * len(shuffled))
        out.update({s: ("dev" if i < n_dev else "semi") for i, s in enumerate(shuffled)})
    return out


def load(path: str | Path = TIER_FILE) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["speakers"]


def tier_of(speaker_id: str, tiers: dict[str, str] | None = None) -> str:
    """A gold speaker's tier. Corpora missing from the file are an error, except DEV_ONLY and LOCKBOX corpora;
    LOCKBOX_SPEAKERS are lockbox whatever the file says."""
    tiers = load() if tiers is None else tiers
    if speaker_id in LOCKBOX_SPEAKERS:
        return "lockbox"
    if speaker_id in tiers:
        return tiers[speaker_id]
    corpus = speaker_id.split(":", 1)[0]
    if corpus in DEV_ONLY:
        return "dev"
    if corpus in LOCKBOX:
        return "lockbox"
    raise KeyError(f"{speaker_id}: no tier assigned; run `va tiers` and commit tiers.json")


def filter_rows(rows: Iterable[ManifestRow], allowed: Iterable[str] = ("dev",), tiers: dict[str, str] | None = None) -> list[ManifestRow]:
    """Gold rows whose speaker is in an allowed tier; non-gold rows (training pools) pass unchanged."""
    allowed = set(allowed)
    unknown = allowed - set(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers {sorted(unknown)}; expected some of {TIERS}")
    tiers = load() if tiers is None else tiers
    return [r for r in rows if r.split != "gold" or tier_of(r.speaker_id, tiers) in allowed]


def write(rows: Iterable[ManifestRow], path: str | Path = TIER_FILE) -> dict[str, str]:
    speakers = assign(r.speaker_id for r in rows if r.split == "gold")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s, t in speakers.items():
        counts[s.split(":", 1)[0]][t] += 1
    Path(path).write_text(json.dumps({"seed": SEED, "dev_frac": DEV_FRAC, "counts": counts, "speakers": dict(sorted(speakers.items()))},
                                     indent=1) + "\n", encoding="utf-8")
    return speakers
