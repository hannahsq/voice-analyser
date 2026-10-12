"""Training-pool scanners (plan §5.6) on small fake corpus trees."""

import numpy as np
import pytest
import soundfile as sf

from vocal_analysis.data.pools import registry, vocalset
from vocal_analysis.eval.leakage import check_leakage


def _vocalset_tree(root, test_list: str):
    for singer, ctx, tech in [("female1", "scales", "belt"), ("female2", "arpeggios", "breathy"),
                              ("male3", "excerpts", "spoken"), ("male11", "long_tones", "messa")]:
        d = root / "FULL" / singer / ctx / tech
        d.mkdir(parents=True)
        sf.write(d / f"{singer}_{ctx}_{tech}.wav", np.zeros(4410, np.float32), 44100)
    (root / "FULL" / ".DS_Store").write_bytes(b"")
    (root / "test_singers_technique.txt").write_text(test_list)


@pytest.mark.parametrize("test_list", ["female2\nmale11\n", "F2, M11", "f2 m11"])
def test_vocalset_scan_splits_by_singer(tmp_path, test_list):
    _vocalset_tree(tmp_path, test_list)
    rows = vocalset.scan(tmp_path)
    assert len(rows) == 4 and all(r.corpus == "vocalset" and r.licence == "CC-BY-4.0" for r in rows)
    split = {r.speaker_id: r.split for r in rows}
    assert split == {"vocalset:female1": "train", "vocalset:female2": "val", "vocalset:male3": "train", "vocalset:male11": "val"}
    assert rows[0].sr == 44100 and rows[0].duration == pytest.approx(0.1)
    assert check_leakage(rows).ok


def test_vocalset_scan_rejects_unknown_test_singer(tmp_path):
    _vocalset_tree(tmp_path, "female7")
    with pytest.raises(ValueError, match="female7"):
        vocalset.scan(tmp_path)


def test_pool_registry():
    assert {"vocalset", "zeroth_korean"} <= set(registry())


def _zeroth_tree(root, train=("106", "107"), test=("003",)):
    for d, spks in (("train_data_01", train), ("test_data_01", test)):
        for spk in spks:
            p = root / d / "003" / spk
            p.mkdir(parents=True)
            sf.write(p / f"{spk}_003_0001.flac", np.zeros(1600, np.float32), 16000)
    (root / "AUDIO_INFO").write_text("SPEAKERID|NAME|SEX|SCRIPTID|DATASET\n")


def test_zeroth_scan_splits_and_ids(tmp_path):
    from vocal_analysis.data.pools import zeroth_korean

    _zeroth_tree(tmp_path)
    rows = zeroth_korean.scan(tmp_path)
    assert {r.speaker_id: r.split for r in rows} == {"zeroth_korean:106": "train", "zeroth_korean:107": "train",
                                                     "zeroth_korean:003": "val"}
    assert rows[0].sr == 16000 and rows[0].duration == pytest.approx(0.1) and check_leakage(rows).ok


def test_zeroth_scan_rejects_shared_speaker(tmp_path):
    from vocal_analysis.data.pools import zeroth_korean

    _zeroth_tree(tmp_path, train=("106",), test=("106",))
    with pytest.raises(ValueError, match="106"):
        zeroth_korean.scan(tmp_path)


def test_vocalset_resynth_skips_lip_trills():
    from vocal_analysis.data.manifest import ManifestRow

    row = lambda tech: ManifestRow(f"FULL/female1/arpeggios/{tech}/x.wav", "vocalset", "vocalset:female1", "train", 44100, 1.0, "CC-BY-4.0")  # noqa: E731
    assert not vocalset.resynth_ok(row("lip_trill")) and vocalset.resynth_ok(row("trill")) and vocalset.resynth_ok(row("belt"))
