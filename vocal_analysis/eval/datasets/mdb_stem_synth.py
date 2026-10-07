"""MDB-stem-synth: f0 labels exact by resynthesis (Salamon et al. 2017).

Layout (ASSUMPTION, from the corpus README):
    <root>/audio_stems/<Artist>_<Track>_STEM_<nn>.RESYN.wav
    <root>/annotation_stems/<same stem>.csv      rows "time,f0" (0 = unvoiced)

The corpus contains instrument stems as well as voice. Only stems listed in
<root>/vocal_stems.txt (one stem name per line, built from the MedleyDB
metadata) are used; without that file every stem is included and a warning
is logged in the manifest step (D-005).
The singer is not identified, so speaker_id = artist (conservative for leakage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...data.manifest import ManifestRow, audio_info, speaker_key
from . import GoldItem

CORPUS = "mdb_stem_synth"
LICENCE = "CC-BY-NC-4.0"
VOCAL_LIST = "vocal_stems.txt"


def _stem(wav: Path) -> str:
    return wav.name.removesuffix(".wav")


def vocal_allowlist(root: Path) -> set[str] | None:
    p = root / VOCAL_LIST
    if not p.exists():
        return None
    return {line.strip().removesuffix(".wav") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


METADATA_URL = "https://raw.githubusercontent.com/marl/medleydb/{commit}/medleydb/data/Metadata/{track}_METADATA.yaml"
MEDLEYDB_COMMIT = "537bb7bad0601fc58e4fff8404f71658b86ef60e"
SINGING_INSTRUMENTS = {"male singer", "female singer", "vocalists", "choir"}


def write_vocal_allowlist(root: str | Path, log=print) -> Path:
    """Build vocal_stems.txt from the MedleyDB per-track metadata (instrument per stem)."""
    import urllib.request

    import yaml

    root = Path(root)
    stems = sorted(_stem(w) for w in (root / "audio_stems").glob("*.wav") if not w.name.startswith("._"))
    tracks = sorted({s.split("_STEM_")[0] for s in stems})
    keep, instruments = [], {}
    for track in tracks:
        url = METADATA_URL.format(commit=MEDLEYDB_COMMIT, track=track)
        with urllib.request.urlopen(url, timeout=60) as resp:
            meta = yaml.safe_load(resp.read())
        for s in (meta.get("stems") or {}).values():
            stem = Path(s["filename"]).stem + ".RESYN"
            instruments[stem] = s.get("instrument", "")
    for stem in stems:
        inst = instruments.get(stem)
        if inst is None:
            log(f"  {stem}: not in MedleyDB metadata, excluded")
        elif inst in SINGING_INSTRUMENTS:
            keep.append(stem)
    out = root / VOCAL_LIST
    out.write_text("".join(f"{s}\n" for s in keep), encoding="utf-8")
    log(f"[{CORPUS}] {len(keep)} of {len(stems)} stems are singing voice -> {out}")
    return out


def scan(root: str | Path) -> list[ManifestRow]:
    root = Path(root)
    allow = vocal_allowlist(root)
    rows = []
    for wav in sorted(w for w in (root / "audio_stems").glob("*.wav") if not w.name.startswith("._")):
        stem = _stem(wav)
        if allow is not None and stem not in allow:
            continue
        if not (root / "annotation_stems" / f"{stem}.csv").exists():
            continue
        sr, dur = audio_info(wav)
        artist = stem.split("_")[0]
        rows.append(ManifestRow(wav.relative_to(root).as_posix(), CORPUS, speaker_key(CORPUS, artist), "gold", sr, round(dur, 4), LICENCE))
    return rows


def load(root: str | Path, row: ManifestRow) -> GoldItem:
    root = Path(root)
    data = np.loadtxt(root / "annotation_stems" / f"{_stem(Path(row.path))}.csv", delimiter=",", ndmin=2)
    return GoldItem(audio_path=root / row.path, speaker_id=row.speaker_id, times=data[:, 0], f0=data[:, 1])
