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


def test_lf_source_matches_glottal_table():
    from vocal_analysis.synth import glottal

    rd = np.array([0.45, 1.0, 1.73, 2.6])
    got = R.lf_source_db(torch.tensor(rd, dtype=torch.float64), 8).numpy()
    assert np.max(np.abs(got - glottal.source_h_db(rd, 8))) < 1e-6


def test_rd_source_recovers_f1_and_rd_gradient():
    f0, K, rd_true = 140.0, 30, 1.1
    f = f0 * torch.arange(1, K + 1, dtype=torch.float64)
    pf = torch.tensor([520.0, 1400.0, 2500.0, 3400.0], dtype=torch.float64)
    pb = torch.tensor([70.0, 90.0, 120.0, 180.0], dtype=torch.float64)
    db = (R.render_db(f, pf, pb, radiation=False) + R.lf_source_db(torch.tensor(rd_true, dtype=torch.float64), K) - 12.0)[None, None]
    ok = torch.ones(1, 1, K, dtype=torch.bool)
    f0t = torch.tensor([[f0]], dtype=torch.float64)
    grid = np.arange(440.0, 600.0, 5.0)
    losses = []
    for g in grid:
        p = pf.clone()
        p[0] = g
        losses.append(R.reconstruction_loss(db, ok, f0t, p[None, None], pb[None, None], source="rd", profile="offset")[0].item())
    assert abs(grid[int(np.argmin(losses))] - 520.0) <= 5.0
    rd = torch.tensor([[0.8]], dtype=torch.float64, requires_grad=True)
    loss, _ = R.reconstruction_loss(db, ok, f0t, pf[None, None], pb[None, None], source="rd", profile="offset", rd=rd)
    loss.backward()
    assert torch.isfinite(rd.grad).all() and rd.grad.item() < 0  # raising Rd towards 1.1 lowers the loss


def test_snr_gate_drops_harmonics_under_noise():
    rng = np.random.default_rng(0)
    f0, n = 120.0, FS
    t = np.arange(n) / FS
    k = np.arange(1, 30)
    amps = np.where(k <= 10, 1.0, 1e-4)  # harmonics 11+ far below the noise
    x = sum(a * np.cos(2 * np.pi * kk * f0 * t) for kk, a in zip(k, amps)) + 0.003 * rng.standard_normal(n)
    T = n // R.HOP + 1
    _, ok_all = R.harmonic_targets(x, np.full(T, f0), np.ones(T, bool), fmax=3500, k_max=32, floor_db=120)
    db, ok = R.harmonic_targets(x, np.full(T, f0), np.ones(T, bool), fmax=3500, k_max=32, floor_db=120, snr_min_db=10)
    i = T // 2
    assert ok_all[i, :29].all()
    assert ok[i, :10].all() and ok[i, 10:29].mean() <= 0.1  # a noise-only harmonic passes 10 dB about 0.7 % of the time
    assert np.max(np.abs(db[i, :10])) < 0.5  # strong harmonics keep their level (0 dB)


def test_decode_formants_matches_numpy_decode():
    from vocal_analysis.models.bins import FORMANT_AXIS, decode

    torch.manual_seed(0)
    logits = torch.randn(3, 5, 4, FORMANT_AXIS.n_bins) * 3
    got = R.decode_formants(logits).numpy()
    p = torch.softmax(logits, -1).numpy()
    ref, _ = decode(p, FORMANT_AXIS)
    assert np.max(np.abs(np.log2(got / ref))) * 1200 < 0.01


def test_model_recon_loss_runs_and_backprops():
    from vocal_analysis.models.bins import FORMANT_AXIS

    torch.manual_seed(0)
    B, T, K = 2, 6, 24
    logits = torch.randn(B, T, 4, FORMANT_AXIS.n_bins, requires_grad=True)
    out = {"formant_logits": logits, "logb_mean": torch.full((B, T, 4), float(np.log(100.0)), requires_grad=True)}
    lab = {"db": torch.randn(B, T, K), "ok": torch.ones(B, T, K, dtype=torch.bool), "f0": torch.full((B, T), 150.0),
           "mask": torch.ones(B, T, dtype=torch.bool)}
    loss = R.model_recon_loss(out, lab, rd_grid=[0.8, 1.2])
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0 and out["logb_mean"].grad.abs().sum() > 0


def test_upper_poles_match_the_prior():
    """D-109: the higher-pole correction's bandwidths are synth.tract.bandwidth_prior's, its poles the tube's"""
    from vocal_analysis.synth.tract import bandwidth_prior, tube_resonances
    from vocal_analysis.train.reconstruction import upper_poles

    tube = tube_resonances(14.5)
    f, b = upper_poles(torch.tensor([tube[3]], dtype=torch.float64), torch.tensor([210.0], dtype=torch.float64))
    want_f = np.where(tube[4:] <= 7600, tube[4:], np.nan)
    np.testing.assert_allclose(f[0].numpy(), want_f, rtol=1e-9)
    ok = np.isfinite(want_f)
    np.testing.assert_allclose(b[0].numpy()[ok], bandwidth_prior(want_f[ok], 210.0), rtol=1e-9)
