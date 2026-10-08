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
    assert "vocalset" in registry()
