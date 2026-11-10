"""D-085's build: paired acoustic heads, the iVTL -> aVTL link, long batches, auxiliary-loss normalisation."""

import numpy as np
import pytest
import torch

from vocal_analysis.features.vqt import N_BINS
from vocal_analysis.models.teacher import Teacher, TeacherConfig
from vocal_analysis.train.data import LengthBatches, MixedCrops
from vocal_analysis.train.losses import teacher_loss
from vocal_analysis.train.teacher import TrainConfig, _AuxWeights, _p_link_truth

SMALL = dict(channels=(8, 8, 8), f0_channels=8, embed=32, temporal="conformer", conformer_dim=32, f0_conv_dim=16,
             conformer_heads=2)


def _model(**kw):
    torch.manual_seed(0)
    return Teacher(TeacherConfig(**SMALL, **kw)).eval()


def _labels(b, t):
    son = torch.ones(b, t, dtype=torch.bool)
    fm = torch.tensor([500.0, 1500.0, 2500.0, 3500.0]).expand(b, t, 4).clone()
    bw = torch.full((b, t, 4), 80.0)
    return {"voiced": son, "sonorant": son, "f0": torch.full((b, t), 150.0), "formants": fm, "bandwidths": bw,
            "formants_acoustic": fm + 20.0, "bandwidths_acoustic": bw * 1.2, "vtl": torch.full((b, t), 4.0),
            "vtl_inst": torch.full((b, t), 4.1)}


def test_acoustic_heads_and_loss():
    m = _model(acoustic_heads=True)
    out = m(torch.randn(2, N_BINS, 20))
    assert out["formant_logits_acoustic"].shape == out["formant_logits"].shape
    assert out["logb_mean_acoustic"].shape == out["logb_mean"].shape
    _, parts = teacher_loss(out, _labels(2, 20))
    assert {"formants_acoustic", "bandwidth_acoustic"} <= parts.keys()
    assert all(torch.isfinite(v) for v in parts.values())


def test_vtl_link_starts_at_zero_and_reads_the_fed_length():
    m = _model(vtl_head=True, vtl_inst_head=True, vtl_link=True)
    ref = _model(vtl_head=True, vtl_inst_head=True)
    ref.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("vtl_link")})
    x = torch.randn(1, N_BINS, 20)
    with torch.no_grad():
        assert torch.allclose(m(x)["vtl_mu"], ref(x)["vtl_mu"])
        torch.nn.init.normal_(m.vtl_link_out[-1].weight)
        a = m(x, vtl_inst_fed=torch.full((1, 20), 3.9))["vtl_mu"]
        b = m(x, vtl_inst_fed=torch.full((1, 20), 4.3))["vtl_mu"]
    assert not torch.allclose(a, b)


def test_long_batches_go_to_their_part():
    lb = LengthBatches(200, 1000, (32, 400), 6400, seed=1, long_frames=(400, 1600), long_batch_frac=0.3, long_part=2)
    batches = list(lb)
    long = [bt for bt in batches if len(bt[0]) == 4]
    assert 0.2 < len(long) / len(batches) < 0.4
    assert all(it[3] == 2 and it[1] >= 400 and it[2] == 1 for bt in long for it in bt)
    assert all(sum(it[1] for it in bt) == pytest.approx(6400, rel=0.3) for bt in long)

    class Part:
        def __init__(self, k):
            self.k = k

        def __len__(self):
            return 10

        def __getitem__(self, item):
            return self.k

    mc = MixedCrops([Part(0), Part(1), Part(2)], [1.0, 1.0, 0.0], 100, fixed=True)
    assert {mc[(i, 50, 1)] for i in range(200)} == {0, 1}  # weight 0: never in the short mix
    assert {mc[(i, 800, 1, 2)] for i in range(20)} == {2}


def test_aux_weights_hold_the_budget():
    cfg = TrainConfig(aux_budget=0.25, aux_floor=1e-6)
    aux = _AuxWeights(cfg)
    parts = {"f0": torch.tensor(2.0), "formants": torch.tensor(3.0), "vtl": torch.tensor(-1.5), "vtl_inst": torch.tensor(0.4)}
    aux.update(parts)
    w = aux.weights()
    main = sum(w[k] * parts[k] for k in ("f0", "formants"))
    side = sum(w[k] * parts[k].abs() for k in ("vtl", "vtl_inst"))
    assert float(side / (main + side)) == pytest.approx(0.25, abs=1e-5)
    assert float(w["vtl"] * 1.5) == pytest.approx(float(w["vtl_inst"] * 0.4))  # equal within the concept


def test_link_truth_ramp():
    cfg = TrainConfig(vtl="new", vtl_inst=True, vtl_link=True, vtl_start=0, vtl_detach=1000, vtl_link_truth_steps=4000)
    assert _p_link_truth(0, cfg) == 1.0 and _p_link_truth(999, cfg) == 1.0
    assert _p_link_truth(3000, cfg) == pytest.approx(0.5)
    assert _p_link_truth(6000, cfg) == 0.0


def test_acoustic_labels_fall_back_to_closed(tmp_path):
    from vocal_analysis.train.data import LABEL_KEYS, SynthCrops

    flat = tmp_path / "flat"
    flat.mkdir()
    n = 60
    np.save(flat / "audio.npy", np.zeros(n * 160 + 2000, np.int16))
    np.save(flat / "audio_offsets.npy", np.array([0, n * 160 + 2000]))
    np.save(flat / "frame_offsets.npy", np.array([0, n]))
    shapes = {"formants": (n, 5), "bandwidths": (n, 5), "source_h_db": (n, 4)}
    for k in LABEL_KEYS:
        dt = bool if k in ("voiced", "sonorant", "whisper") else np.float32
        np.save(flat / f"label_{k}.npy", np.full(shapes.get(k, (n,)), 1 if dt is bool else 300.0, dt))
    ds = SynthCrops(flat, np.array([0]), 20, channel=None, fixed=True, acoustic=True)
    _, lab = ds[0]
    assert lab["formants_acoustic"].shape == (20, 4)
    np.testing.assert_array_equal(lab["formants_acoustic"], lab["formants"])


def test_aux_floor_caps_a_weight_near_zero():
    aux = _AuxWeights(TrainConfig(aux_budget=0.25))
    aux.update({"f0": torch.tensor(2.0), "formants": torch.tensor(3.0), "vtl": torch.tensor(1e-4), "vtl_inst": torch.tensor(0.0)})
    w = aux.weights()
    assert float(w["vtl"]) <= 0.25 / 0.75 * 5.0 / 2 + 1e-6
