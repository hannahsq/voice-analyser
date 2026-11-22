"""D-128 (M5): the teacher's causal mode. Off it is the teacher; on, no output reads a later input than its
lookahead allows, and frame-by-frame streaming with caches equals the batch forward."""

from dataclasses import replace

import pytest
import torch

from vocal_analysis.features import vqt
from vocal_analysis.models.teacher import Teacher, TeacherConfig

SMALL = TeacherConfig(channels=(8, 8, 8), f0_channels=8, embed=32, temporal="conformer", conformer_dim=32,
                      conformer_layers=2, conformer_heads=4, conformer_kernel=5, f0_conv_dim=16,
                      f0_conv_dilations=(1, 2, 4), vtl_head=True)


def _lp(t, seed=0, b=1):
    g = torch.Generator().manual_seed(seed)
    return -60.0 + 20.0 * torch.randn(b, vqt.N_BINS, t, generator=g)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return Teacher(cfg).eval()


def test_copy_with_causality_off_is_the_teacher():
    teacher = _model(SMALL)
    student = Teacher(replace(SMALL)).eval()
    student.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        a, b = teacher(_lp(40)), student(_lp(40))
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_causal_weights_load_from_the_teacher():
    teacher = _model(SMALL)
    student = Teacher(replace(SMALL, causal=True, attn_window=16, lookahead=2))
    student.load_state_dict(teacher.state_dict())  # strict: the same parameters


@pytest.mark.parametrize("lookahead", [0, 2])
@pytest.mark.parametrize("window", [0, 7])
def test_no_output_reads_beyond_its_lookahead(lookahead, window):
    m = _model(replace(SMALL, causal=True, attn_window=window, lookahead=lookahead))
    x = _lp(50)
    t0 = 30
    y = x.clone()
    y[..., t0 + 1:] = _lp(50, seed=1)[..., t0 + 1:]
    with torch.no_grad():
        a, b = m(x), m(y)
    last = t0 - lookahead  # outputs up to here read inputs up to t0 at most
    for k in a:
        assert torch.equal(a[k][:, : last + 1], b[k][:, : last + 1]), k
        assert not torch.equal(a[k][:, last + 1:], b[k][:, last + 1:]), k  # and the next one does read t0 + 1


def test_the_teacher_is_not_causal():
    m = _model(SMALL)
    x = _lp(50)
    y = x.clone()
    y[..., 31:] = _lp(50, seed=1)[..., 31:]
    with torch.no_grad():
        assert not torch.equal(m(x)["formant_logits"][:, :30], m(y)["formant_logits"][:, :30])


@pytest.mark.parametrize("chunks", [[1], [3, 1, 7], [64]])
@pytest.mark.parametrize("lookahead", [0, 2])
@pytest.mark.parametrize("comb", [False, True])
def test_streaming_equals_batch(chunks, lookahead, comb):
    window = 9
    # float64 for a tight bound; the comb is drawn in float32 whatever the model's type (as the teacher does), so with
    # it the bound covers float32 rounding
    m = _model(replace(SMALL, causal=True, attn_window=window, lookahead=lookahead, comb=comb)).double()
    tol = 1e-4 if comb else 1e-9
    t = 45
    x = _lp(t, b=2).double()
    with torch.no_grad():
        ref = m(x)
        xs = torch.cat([x, torch.full((2, vqt.N_BINS, lookahead), -100.0, dtype=torch.float64)], -1)
        outs, state, i, k = [], None, 0, 0
        while i < xs.shape[-1]:
            n = chunks[k % len(chunks)]
            o, state = m.step(xs[..., i:i + n], state)
            outs.append(o)
            i, k = i + n, k + 1
    got = {key: torch.cat([o[key] for o in outs], 1)[:, lookahead:] for key in ref}
    for key in ref:
        assert got[key].shape == ref[key].shape, key
        torch.testing.assert_close(got[key], ref[key], atol=tol, rtol=tol, msg=key)
    # the caches stay bounded: each attention cache holds at most window - 1 frames
    for key, s in state["enc"].items():
        if isinstance(key, tuple):
            assert s[1][0].shape[2] <= window - 1


def test_causal_mode_rejects_gru_and_stray_options():
    with pytest.raises(ValueError):
        Teacher(TeacherConfig(channels=(8, 8, 8), f0_channels=8, embed=32, gru_hidden=16, gru_layers=1,
                              f0_gru_hidden=16, causal=True))
    with pytest.raises(ValueError):
        Teacher(replace(SMALL, lookahead=2))


def test_distil_loss_is_zero_for_the_teacher_itself():
    from vocal_analysis.train.distil import distil_loss

    m = _model(SMALL)
    with torch.no_grad():
        out = m(_lp(30, b=2))
    loss, parts = distil_loss(out, out, torch.ones(2, 30, dtype=torch.bool))
    assert float(loss) == pytest.approx(0.0, abs=1e-5) and set(parts) >= {"f0", "formants", "bandwidth", "vtl"}


@pytest.mark.slow
def test_distil_trains_checkpoints_and_resumes(tmp_path):
    """A tiny teacher, then its causal copy distilled for a few steps (lookahead 2), resumed, and loadable."""
    import json

    import numpy as np

    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.synth.sampler import SynthConfig
    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.distil import DistilConfig, distil
    from vocal_analysis.train.teacher import TrainConfig, train

    small = TeacherConfig(channels=(8, 8, 8), f0_channels=8, embed=32, temporal="conformer", conformer_dim=32,
                          f0_conv_dim=16)
    data = write_set(tmp_path / "s", 8, SynthConfig(duration_s=(2.5, 2.5)), seed=3, shard_size=4, workers=1, log=lambda m: None)
    tcfg = TrainConfig(data=str(data), out=str(tmp_path / "t"), steps=2, batch=4, n_frames=50, workers=1, log_every=2,
                       eval_every=2, ckpt_every=2, val_clips=4, warmup=1)
    tdir = train(tcfg, small, log=lambda *a: None)
    dc = DistilConfig(teacher=str(tdir / "ckpt_last.pt"), out=str(tmp_path / "s1"), lookahead=2, attn_window=20,
                      data=str(data), resynth=(), resynth_val=(), steps=2, batch=4, n_frames=50, crop_frames=None,
                      warmup=1, decay_start=1, workers=1, log_every=1, eval_every=2, ckpt_every=2, val_clips=4)
    out = distil(dc, log=lambda *a: None)
    dc.steps = 4
    distil(dc, log=lambda *a: None)
    recs = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs if "loss" in r] == [1, 2, 3, 4] and all(np.isfinite(r["loss"]) for r in recs if "loss" in r)
    assert any("eval_agree" in r for r in recs) and any("eval_teacher" in r for r in recs)
    s, _ = load_teacher(out / "ckpt_last.pt", torch.device("cpu"))
    assert s.cfg.causal and s.cfg.lookahead == 2 and s.cfg.attn_window == 20
