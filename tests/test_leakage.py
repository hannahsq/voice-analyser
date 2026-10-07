"""Leakage rules (plan §6.4): the synthetic cases, plus the real committed manifests."""

from pathlib import Path

import pytest

from vocal_analysis.data.manifest import ManifestRow, read_manifest, read_manifests, write_manifest
from vocal_analysis.eval.leakage import check_leakage

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "data" / "manifests"


def row(path, spk, split, corpus="c"):
    return ManifestRow(path, corpus, f"{corpus}:{spk}", split, 16000, 1.0, "CC-BY-4.0")


def test_clean_split_passes():
    rows = [row("a1.wav", "a", "train"), row("b1.wav", "b", "val"), row("g1.wav", "g", "gold")]
    assert check_leakage(rows).ok


def test_gold_speaker_in_train_fails():
    rep = check_leakage([row("g1.wav", "g", "gold"), row("g2.wav", "g", "train")])
    assert not rep.ok and "gold speaker" in rep.errors[0]


def test_speaker_across_train_val_fails():
    assert not check_leakage([row("a1.wav", "a", "train"), row("a2.wav", "a", "val")]).ok


def test_same_audio_twice_fails_even_with_different_speaker_ids():
    rep = check_leakage([row("x/a.wav", "a", "gold"), row("x\\a.wav", "b", "train")])
    assert any("audio in multiple splits" in e for e in rep.errors)


def test_excluded_speakers():
    rows = [row("a.wav", "dr1-fabc0", "train", corpus="timit")]
    assert not check_leakage(rows, excluded_speakers={"timit:dr1-fabc0"}).ok


def test_row_validation():
    with pytest.raises(ValueError):
        ManifestRow("a.wav", "c", "a", "train", 16000, 1.0, "x")  # speaker not corpus-prefixed
    with pytest.raises(ValueError):
        ManifestRow("a.wav", "c", "c:a", "holdout", 16000, 1.0, "x")
    with pytest.raises(ValueError):
        ManifestRow("a.wav", "c", "c:a", "train", 16000, 1.0, "")


def test_manifest_roundtrip(tmp_path):
    rows = [row("b.wav", "b", "train"), row("a.wav", "a", "gold")]
    p = write_manifest(rows, tmp_path / "c.csv")
    assert read_manifest(p) == sorted(rows, key=lambda r: r.path)


def test_committed_manifests_have_no_leakage():
    """The real gate: runs over every manifest in data/manifests/."""
    if not MANIFEST_DIR.exists() or not any(MANIFEST_DIR.glob("*.csv")):
        pytest.skip("no manifests built yet")
    check_leakage(read_manifests(MANIFEST_DIR)).raise_if_failed()
