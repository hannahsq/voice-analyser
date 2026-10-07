from pathlib import Path

import pytest

from vocal_analysis.cli import main
from vocal_analysis.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(ROOT / "configs" / "default.yaml") == load_config(None)


def test_overrides_and_unknown_keys():
    cfg = load_config(None, ["data.root=D:/corpora", "eval.bootstrap_resamples=50"])
    assert cfg.data.root == "D:/corpora" and cfg.eval.bootstrap_resamples == 50
    with pytest.raises(KeyError):
        load_config(None, ["data.rooot=x"])


def test_manifest_and_leakage_commands(tmp_path, capsys):
    corpus = tmp_path / "mir1k"
    (corpus / "Wavfile").mkdir(parents=True)
    (corpus / "PitchLabel").mkdir()
    import numpy as np
    import soundfile as sf

    sf.write(corpus / "Wavfile" / "abjones_1_01.wav", np.zeros((1600, 2)), 16000)
    (corpus / "PitchLabel" / "abjones_1_01.pv").write_text("0\n")
    sets = ["--set", f"data.root={tmp_path.as_posix()}", "--set", f"data.manifests={(tmp_path / 'm').as_posix()}"]
    assert main(["manifest", "mir1k", *sets]) == 0
    assert (tmp_path / "m" / "mir1k.csv").exists()
    assert main(["leakage", *sets]) == 0
    assert "OK" in capsys.readouterr().out
