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


def test_lockbox_speakers_override_their_corpus():
    a = tiers.assign(["usc75:sub009", "usc75:sub001"])
    assert a["usc75:sub009"] == "lockbox" and a["usc75:sub001"] in ("dev", "semi")
    assert tiers.tier_of("usc75:sub074", {"usc75:sub074": "dev"}) == "lockbox"


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


def test_suite_report_reads_only_the_named_tiers(tmp_path, capsys):
    """`va baselines` / `va eval teacher` score dev speakers unless told otherwise; the analysis scripts' path (tiers
    None) keeps every speaker and its old cache names."""
    from pathlib import Path

    from vocal_analysis.cli import _tiers_arg
    from vocal_analysis.data.manifest import write_manifest
    from vocal_analysis.eval import report

    t = tiers.load()
    dev = next(s for s in sorted(t) if s.startswith("hillenbrand:") and t[s] == "dev")
    semi = next(s for s in sorted(t) if s.startswith("hillenbrand:") and t[s] == "semi")
    m = write_manifest(_rows([dev, semi]), tmp_path / "hillenbrand.csv")
    assert {r.speaker_id for r in report._rows(m, None)} == {dev, semi}
    assert {r.speaker_id for r in report._rows(m, ("dev",))} == {dev}
    assert {r.speaker_id for r in report._rows(m, ("dev", "semi"))} == {dev, semi}
    assert report._cache_path(Path("c"), "praat", "hillenbrand") == Path("c/praat__hillenbrand.npz")
    assert report._cache_path(Path("c"), "praat", "hillenbrand", ("dev",)) == Path("c/praat__hillenbrand__dev.npz")
    assert report._key({"a": 1}, None) == {"a": 1} and report._key({"a": 1}, ("semi", "dev"))["tiers"] == ["dev", "semi"]
    assert "Includes" not in report._tiers_line(["dev"])
    assert "semi-private" in report._tiers_line(["dev", "semi"]) and "every gold speaker" in report._tiers_line(None)
    assert _tiers_arg("dev") == ("dev",) and "Log this use" not in capsys.readouterr().out
    assert _tiers_arg("semi,dev") == ("dev", "semi") and "Log this use" in capsys.readouterr().out
