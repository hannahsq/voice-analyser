"""M4 reconstruction loss: renderer, harmonic targets, source profiling, loss minimum and gradients."""
import numpy as np
import torch

from vocal_analysis.synth import filters
from vocal_analysis.train import reconstruction as R

FS = 16000


def test_render_matches_synth_filters():
    f = np.linspace(50, 7000, 300)
    pf, pb = np.array([500.0, 1500.0, 2500.0, 3500.0]), np.array([60.0, 90.0, 120.0, 200.0])
    zf, zb = np.array([1200.0]), np.array([150.0])
    ref = 20 * np.log10(np.abs(filters.response(f, pf, pb, FS, zf, zb)))
    got = R.render_db(torch.tensor(f), torch.tensor(pf), torch.tensor(pb), torch.tensor(zf), torch.tensor(zb),
                      radiation=False).numpy()
    assert np.max(np.abs(got - ref)) < 1e-6


def test_nan_sections_are_skipped():
    f = torch.linspace(100, 4000, 50)
    a = R.render_db(f, torch.tensor([500.0, float("nan")]), torch.tensor([80.0, float("nan")]), radiation=False)
    b = R.render_db(f, torch.tensor([500.0]), torch.tensor([80.0]), radiation=False)
    assert torch.allclose(a, b)


def test_harmonic_targets_recover_amplitudes():
    f0, n = 180.0, FS
    t = np.arange(n) / FS
    k = np.arange(1, 20)
    amps = 10 ** (-(k - 1) * 1.5 / 20)
    x = sum(a * np.cos(2 * np.pi * kk * f0 * t + 0.3 * kk) for kk, a in zip(k, amps))
    T = n // R.HOP + 1
    db, ok = R.harmonic_targets(x, np.full(T, f0), np.ones(T, bool), fmax=3500, k_max=24)
    i = T // 2
    assert ok[i, :19].all() and not ok[i, 19:].any()  # 20 x 180 = 3600 > fmax
    assert np.max(np.abs(db[i, :19] - 20 * np.log10(amps))) < 0.05


def test_profile_removes_offset_and_tilt():
    f = torch.arange(1, 30, dtype=torch.float64) * 150
    resid = 7.0 - 4.5 * torch.log2(f)
    out = R.profile_out(resid, f, torch.ones_like(f), "tilt")
    assert out.abs().max() < 1e-6
    assert R.profile_out(resid, f, torch.ones_like(f), "offset").abs().max() > 1


def _toy(f1, f0=200.0, K=24):
    """targets rendered from known poles, with an arbitrary source level and slope"""
    f = f0 * torch.arange(1, K + 1, dtype=torch.float64)
    pf, pb = torch.tensor([f1, 1500.0, 2600.0, 3500.0], dtype=torch.float64), torch.tensor([70.0, 90.0, 120.0, 180.0], dtype=torch.float64)
    db = R.render_db(f, pf, pb) + 3.0 - 9.0 * torch.log2(f / 100)
    return db[None, None], torch.ones(1, 1, K, dtype=torch.bool), torch.tensor([[f0]], dtype=torch.float64), pf, pb


def test_loss_minimum_at_true_f1():
    db, ok, f0, pf, pb = _toy(620.0)
    grid = np.arange(500.0, 760.0, 5.0)
    losses = []
    for g in grid:
        p = pf.clone()
        p[0] = g
        losses.append(R.reconstruction_loss(db, ok, f0, p[None, None], pb[None, None])[0].item())
    assert abs(grid[int(np.argmin(losses))] - 620.0) <= 5.0
    assert min(losses) < 1e-6


def test_loss_gradient_finite_and_points_home():
    db, ok, f0, pf, pb = _toy(620.0)
    p = pf.clone()
    p[0] = 560.0
    p = p.requires_grad_(True)
    loss, _ = R.reconstruction_loss(db, ok, f0, p[None, None], pb[None, None])
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad[0] < 0  # raising F1 lowers the loss


def test_too_few_harmonics_carry_no_loss():
    db, ok, f0, pf, pb = _toy(620.0)
    ok = ok.clone()
    ok[..., 3:] = False
    loss, per = R.reconstruction_loss(db, ok, f0, pf[None, None] * 1.2, pb[None, None])
    assert loss.item() == 0.0 and torch.isnan(per).all()


def test_nuisance_head_ranges():
    torch.manual_seed(0)
    h = R.NuisanceHead(16)
    out = h(torch.randn(2, 5, 16) * 10)
    assert out["pole_f"].shape == (2, 5, 2) and out["zero_f"].shape == (2, 5, 2)
    assert (out["zero_f"] >= 250).all() and (out["zero_f"] <= 4000).all()
    assert (out["pole_b"] >= 40).all() and (out["pole_b"] <= 1000).all()
