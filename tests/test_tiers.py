"""Evaluation tiers (decisions D-024): fixed, speaker-disjoint, balanced, and enforced by the filter."""

from vocal_analysis.data.manifest import ManifestRow
from vocal_analysis.eval import tiers


def _rows(ids, split="gold"):
    return [ManifestRow(f"{i}.wav", i.split(":")[0], i, split, 16000, 1.0, "x") for i in ids]


def test_assignment_is_deterministic_and_balanced_within_groups():
    ids = [f"hillenbrand:{g}{n:02d}" for g in "bgmw" for n in range(1, 21)] + [f"ptdb_tug:{s}{n:02d}" for s in "FM" for n in range(1, 11)]
    a, b = tiers.assign(ids), tiers.assign(list(reversed(ids)))
    assert a == b
    for prefix in ("hillenbrand:b", "hillenbrand:w", "ptdb_tug:F", "ptdb_tug:M"):
        group = [t for s, t in a.items() if s.startswith(prefix)]
        assert group.count("dev") == round(0.6 * len(group)) and set(group) == {"dev", "semi"}


def test_dev_only_and_lockbox_corpora():
    a = tiers.assign(["cmu_arctic:bdl", "cmu_arctic:slt", "mocha_timit:fsew0"])
    assert a == {"cmu_arctic:bdl": "dev", "cmu_arctic:slt": "dev", "mocha_timit:fsew0": "lockbox"}


def test_filter_keeps_training_rows_and_drops_other_tiers():
    t = {"hillenbrand:w01": "dev", "hillenbrand:w02": "semi"}
    rows = _rows(["hillenbrand:w01", "hillenbrand:w02"]) + _rows(["vocalset:female1"], split="train")
    kept = {r.speaker_id for r in tiers.filter_rows(rows, ("dev",), t)}
    assert kept == {"hillenbrand:w01", "vocalset:female1"}
    assert {r.speaker_id for r in tiers.filter_rows(rows, ("semi",), t)} == {"hillenbrand:w02", "vocalset:female1"}


def test_committed_file_covers_every_gold_speaker():
    t = tiers.load()
    assert set(t.values()) <= set(tiers.TIERS)
    hb = [s for s in t if s.startswith("hillenbrand:")]
    assert len(hb) == 139 and sum(t[s] == "dev" for s in hb) == 83
    assert sorted(t[s] for s in t if s.startswith("ptdb_tug:")).count("dev") == 12
