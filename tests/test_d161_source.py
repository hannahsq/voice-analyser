"""D-161: the source head's levels, the source envelope channel, the widened input layer and the source loss."""

import dataclasses

import numpy as np
import pytest
import torch

from vocal_analysis.features import vqt
from vocal_analysis.models.teacher import (LOG2_VQT, SOURCE_H, Teacher, TeacherConfig, source_envelope,
                                           source_levels)
from vocal_analysis.train.losses import source_loss
from vocal_analysis.train.teacher import SOURCE_INPUT, _widen_input, _widen_opt_state

SMALL = dict(channels=(8, 8, 8), f0_channels=8, embed=32, gru_hidden=16, gru_layers=1, f0_gru_hidden=16)


def test_levels_h1_free_and_non_increasing_from_h2():
    torch.manual_seed(0)
    raw = torch.randn(3, 5, SOURCE_H - 1) * 3
    lv = source_levels(raw)
    assert lv.shape == (3, 5, SOURCE_H)
    assert torch.all(lv[..., 1] == 0)
    assert torch.all(lv[..., 2:] - lv[..., 1:-1] <= 0)
    assert (lv[..., 0] > 0).any() and (lv[..., 0] < 0).any()  # H1 - H2 takes either sign


def test_envelope_at_harmonics_and_scale():
    f0 = torch.full((1, 2), 100.0)
    lv = torch.tensor([-3.0, 0.0] + [-2.0 * k for k in range(1, SOURCE_H - 1)])[None, None].repeat(1, 2, 1)
    env = source_envelope(lv, f0, torch.tensor([[1.0, 0.0]]), LOG2_VQT)
    assert env.shape == (1, 1, vqt.N_BINS, 2)
    assert torch.all(env[..., 1] == 0)  # unvoiced: flat
    fb = vqt.bin_frequencies()
    for k in (2, 5, 10):
        b = int(np.argmin(np.abs(fb - 100.0 * k)))
        u = np.log2(fb[b] / 100.0)
        k0 = int(np.floor(2 ** u))
        lo, hi = lv[0, 0, k0 - 1].item(), lv[0, 0, k0].item()
        w = (u - np.log2(k0)) / (np.log2(k0 + 1) - np.log2(k0))
        want = (lo + w * (hi - lo)) / 20.0  # max level is H2 = 0 dB
        assert env[0, 0, b, 0].item() == pytest.approx(want, abs=1e-4)
    below = fb < 100.0
    assert torch.allclose(env[0, 0, torch.from_numpy(below), 0], torch.tensor(-3.0 / 20.0))
    assert env.min() >= -5.0


def _pair(source_env):
    base = Teacher(TeacherConfig(**SMALL)).eval()
    wide = Teacher(TeacherConfig(**SMALL, source_head=True, source_env=source_env)).eval()
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    _widen_input(sd, wide)
    missing, unexpected = wide.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("source_head.") for k in missing)
    return base, wide


@pytest.mark.parametrize("source_env", ["zero", "pred"])
def test_widened_model_starts_at_its_parent(source_env):
    torch.manual_seed(1)
    base, wide = _pair(source_env)
    for k in SOURCE_INPUT:
        assert torch.all(wide.state_dict()[k][:, 3] == 0)
    x = torch.randn(2, vqt.N_BINS, 30) * 10 - 40
    with torch.no_grad():
        a, b = base(x), wide(x)
    for k in ("formant_logits", "logb_mean", "f0_logits", "sonorant_logit"):
        assert torch.allclose(a[k], b[k], atol=1e-5), k
    assert b["source_levels"].shape == (2, 30, SOURCE_H)


def test_fed_levels_replace_the_heads_on_chosen_clips_only():
    torch.manual_seed(2)
    _, wide = _pair("pred")
    with torch.no_grad():  # a non-zero channel weight, so channel 4's content reaches the outputs
        wide.enc.convs[0].conv.weight[:, 3] = 0.5
    x = torch.randn(2, vqt.N_BINS, 30) * 10 - 40
    fed = torch.full((2, 30, SOURCE_H), float("nan"))
    fed[1] = -torch.arange(SOURCE_H, dtype=torch.float32) * 3
    with torch.no_grad():
        a, b = wide(x), wide(x, source_fed=fed)
        z = wide(x, source_zero=torch.tensor([True, False]))
    assert torch.allclose(a["formant_logits"][0], b["formant_logits"][0])
    assert not torch.allclose(a["formant_logits"][1], b["formant_logits"][1])
    assert not torch.allclose(a["formant_logits"][0], z["formant_logits"][0])
    assert torch.allclose(a["formant_logits"][1], z["formant_logits"][1])


def test_source_loss_offset_invariant_and_masked():
    torch.manual_seed(3)
    pred = source_levels(torch.randn(2, 4, SOURCE_H - 1))
    y = pred.detach() + torch.randn_like(pred) * 3
    f0 = torch.full((2, 4), 200.0)
    v = torch.ones(2, 4, dtype=torch.bool)
    w = torch.ones(2, 4)
    l0 = source_loss(pred, y, f0, v, w)
    assert source_loss(pred, y + 17.0, f0, v, w).item() == pytest.approx(l0.item(), rel=1e-5)
    assert source_loss(pred, pred.detach() - 5.0, f0, v, w).item() == pytest.approx(0.0, abs=1e-6)
    y2 = y.clone()
    y2[..., 20:] += 50.0  # above 4 kHz at 200 Hz: masked
    assert source_loss(pred, y2, f0, v, w).item() == pytest.approx(l0.item(), rel=1e-5)
    assert source_loss(pred, y, f0, v, torch.zeros(2, 4)).item() == 0.0


def test_optimiser_state_widened():
    _, wide = _pair("pred")
    params = dict(wide.named_parameters())
    opt = torch.optim.AdamW([params[k] for k in SOURCE_INPUT], lr=1e-3)
    for k in SOURCE_INPUT:
        p = params[k]
        narrow = (p.shape[0], p.shape[1] - 1, *p.shape[2:])
        opt.state[p] = {"step": torch.tensor(5.0), "exp_avg": torch.ones(narrow), "exp_avg_sq": torch.ones(narrow)}
    _widen_opt_state(opt, wide)
    for k in SOURCE_INPUT:
        st = opt.state[params[k]]
        assert st["exp_avg"].shape == params[k].shape
        assert torch.all(st["exp_avg"][:, -1] == 0) and torch.all(st["exp_avg"][:, :-1] == 1)


def test_config_round_trip():
    cfg = TeacherConfig(**SMALL, source_head=True, source_env="pred")
    assert TeacherConfig(**dataclasses.asdict(cfg)) == cfg
    with pytest.raises(ValueError):
        Teacher(TeacherConfig(**SMALL, source_env="pred"))
