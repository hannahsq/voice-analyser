import json

import numpy as np
import pytest

from vocal_analysis import analyse as A


def _fake(n=50, vtl=True):
    rng = np.random.default_rng(0)
    r = {"t": np.arange(n) * 0.01, "f0": np.full(n, 200.0), "f0_sigma_cents": np.full(n, 10.0),
         "voiced_p": np.r_[np.zeros(10), np.ones(n - 10)], "sonorant_p": np.r_[np.zeros(5), np.ones(n - 5)],
         "formants": np.tile([500.0, 1500.0, 2500.0, 3500.0], (n, 1)), "formant_sigma_cents": np.full((n, 4), 30.0),
         "bandwidths": np.tile([80.0, 100.0, 150.0, 200.0], (n, 1))}
    if vtl:
        r["vtl_log2cm"] = np.log2(15.0) + rng.normal(0, 0.01, n)
        r["vtl_sigma"] = np.full(n, 0.02)
    return r


def test_parse_window():
    from vocal_analysis.models.infer import DEFAULT_WINDOW

    assert A.parse_window("crop") == DEFAULT_WINDOW
    assert A.parse_window("whole") is None
    assert A.parse_window("800") == 800
    with pytest.raises(ValueError):
        A.parse_window("10")


def test_columns_mask_and_offset():
    r = _fake()
    raw = A.columns(r)
    assert np.isfinite(raw["f0"]).all() and np.isfinite(raw["F1"]).all()
    m = A.columns(r, t0=1.5, mask=True)
    assert np.isnan(m["f0"][:10]).all() and np.isfinite(m["f0"][10:]).all()
    assert np.isnan(m["F1"][:5]).all() and np.isfinite(m["F1"][5:]).all()
    assert np.isnan(m["vtl_cm"][:5]).all()
    assert m["t"][0] == 1.5
    assert np.isfinite(m["voiced_p"]).all()  # probabilities are never masked


def test_summary_without_vtl_head():
    s = A.summary(_fake(vtl=False))
    assert "vtl_cm" not in s and s["f0_hz"]["median"] == 200.0
    assert s["formants_hz_median"] == [500.0, 1500.0, 2500.0, 3500.0]


def test_summary_vtl_is_precision_weighted():
    s = A.summary(_fake())
    assert abs(s["vtl_cm"]["value"] - 15.0) < 0.1
    assert s["vtl_cm"]["sigma_log2"] < 0.02  # 45 sonorant frames pooled


def test_write_formats(tmp_path):
    r = _fake()
    files = A.write(A.columns(r, mask=True), A.summary(r), {"file": "x.wav"}, tmp_path / "x", ("csv", "json", "npz"))
    assert {p.suffix for p in files} == {".csv", ".json", ".npz"}
    doc = json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))
    assert doc["frames"]["f0"][0] is None and doc["meta"]["file"] == "x.wav"
    header = (tmp_path / "x.csv").read_text().splitlines()[0].split(",")
    assert header[:3] == ["t", "f0", "f0_sigma_cents"] and "vtl_cm" in header
    only = A.write(A.columns(r), A.summary(r), {}, tmp_path / "y", ("csv",))
    assert "frames" not in json.loads((tmp_path / "y.json").read_text(encoding="utf-8"))  # summary always, frames only on request
    assert {p.suffix for p in only} == {".csv", ".json"}


def test_plot_writes_png(tmp_path):
    audio = np.random.default_rng(1).normal(0, 0.1, 8000).astype(np.float32)
    r = _fake(51)
    for kind in ("wide", "narrow"):
        p = A.plot(audio, r, tmp_path / f"{kind}.png", spectrogram=kind)
        assert p.exists() and p.stat().st_size > 1000
