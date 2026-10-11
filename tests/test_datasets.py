"""Gold-set loaders on tiny fixtures that mimic each corpus layout."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vocal_analysis.eval.datasets import hillenbrand, mdb_stem_synth, mir1k, ptdb_tug, registry


def wav(path: Path, sr=16000, seconds=0.1, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((int(sr * seconds), channels)).squeeze(), sr)


def test_registry_lists_all_gold_sets():
    assert set(registry()) == {"hillenbrand", "ptdb_tug", "mir1k", "mdb_stem_synth", "timit", "cmu_arctic"}


def test_hillenbrand(tmp_path):
    (tmp_path / "vowdata.dat").write_text(
        "Hillenbrand vowel data\nfile dur f0 F1 F2 F3 F4 ...\n"
        "m01ae  257  174   663  2012  2659  3933  600 2000 2600  650 2000 2650  640 1990 2640\n"
        "w02iy  300  227   437  2761  3372     0  400 2700 3300  430 2760 3370  440 2750 3380\n"
    )
    wav(tmp_path / "men" / "m01ae.wav")
    wav(tmp_path / "women" / "w02iy.wav")
    wav(tmp_path / "women" / "w99uh.wav")  # no labels -> not in manifest
    rows = hillenbrand.scan(tmp_path)
    assert [r.speaker_id for r in rows] == ["hillenbrand:m01", "hillenbrand:w02"]
    assert all(r.split == "gold" for r in rows)

    # Without timedata.dat: one steady-state frame, unknown time.
    item = hillenbrand.load(tmp_path, rows[1])
    assert item.formants.shape == (1, 4) and np.isnan(item.times[0])
    np.testing.assert_array_equal(item.formants[0, :3], [437, 2761, 3372])
    assert np.isnan(item.formants[0, 3])  # 0 = not measured
    assert item.f0[0] == 227 and item.meta["vowel"] == "iy" and item.meta["group"] == "w"

    hillenbrand.parse_timedata.cache_clear()
    (tmp_path / "timedata.dat").write_text(
        "File   Start    End   Center1  Center2\n"
        "m01ae  100.0   400.0   200.0    210.0\n"
        "w02iy  100.0   400.0   250.0    250.0\n"
    )
    item = hillenbrand.load(tmp_path, rows[0])
    np.testing.assert_allclose(item.times, [0.205, 0.16, 0.25, 0.34])
    np.testing.assert_array_equal(item.formants[2], [650, 2000, 2650, np.nan])
    assert np.isnan(item.f0[1:]).all() and item.meta["frame_kinds"][0] == "steady"


def test_ptdb_tug(tmp_path):
    base = tmp_path / "SPEECH DATA" / "FEMALE"
    wav(base / "MIC" / "F01" / "mic_F01_sa1.wav", sr=48000)
    wav(base / "MIC" / "F01" / "mic_F01_sa2.wav", sr=48000)  # no reference -> skipped
    (base / "REF" / "F01").mkdir(parents=True)
    (base / "REF" / "F01" / "ref_F01_sa1.f0").write_text("0 0 0 0\n210.5 1 0.3 0\n212.0 1 0.4 0\n")
    rows = ptdb_tug.scan(tmp_path)
    assert len(rows) == 1 and rows[0].speaker_id == "ptdb_tug:F01" and rows[0].sr == 48000
    item = ptdb_tug.load(tmp_path, rows[0])
    np.testing.assert_allclose(item.f0, [0, 210.5, 212.0])
    np.testing.assert_allclose(item.times, [0.020, 0.030, 0.040])


def test_mir1k(tmp_path):
    wav(tmp_path / "Wavfile" / "abjones_1_01.wav", channels=2)
    (tmp_path / "PitchLabel").mkdir()
    (tmp_path / "PitchLabel" / "abjones_1_01.pv").write_text("0\n69\n57.0\n")
    rows = mir1k.scan(tmp_path)
    assert rows[0].speaker_id == "mir1k:abjones"
    item = mir1k.load(tmp_path, rows[0])
    np.testing.assert_allclose(item.f0, [0, 440, 220])
    np.testing.assert_allclose(item.times, [0.02, 0.04, 0.06])


@pytest.mark.parametrize("with_allowlist", [False, True])
def test_mdb_stem_synth(tmp_path, with_allowlist):
    for stem in ("MusicDelta_Beatles_STEM_01.RESYN", "MusicDelta_Beatles_STEM_02.RESYN"):
        wav(tmp_path / "audio_stems" / f"{stem}.wav", sr=44100)
        (tmp_path / "annotation_stems").mkdir(exist_ok=True)
        (tmp_path / "annotation_stems" / f"{stem}.csv").write_text("0.0,0.0\n0.0029,261.6\n")
    if with_allowlist:
        (tmp_path / "vocal_stems.txt").write_text("MusicDelta_Beatles_STEM_02.RESYN\n")
    rows = mdb_stem_synth.scan(tmp_path)
    assert len(rows) == (1 if with_allowlist else 2)
    assert rows[0].speaker_id == "mdb_stem_synth:MusicDelta"
    item = mdb_stem_synth.load(tmp_path, rows[-1])
    np.testing.assert_allclose(item.f0, [0, 261.6])
