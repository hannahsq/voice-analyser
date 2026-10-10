"""Model plumbing: heat-map targets and decoding, crop alignment, teacher shapes (plan §7)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vocal_analysis.features import vqt  # noqa: E402
from vocal_analysis.models.bins import F0_AXIS, FORMANT_AXIS, decode, gaussian_targets  # noqa: E402


@pytest.mark.parametrize("hz", [123.4, 517.0, 2750.0])
def test_targets_decode_back_to_value(hz):
    t, valid = gaussian_targets(torch, torch.tensor([hz]), FORMANT_AXIS)
    assert bool(valid[0]) and float(t.sum()) == pytest.approx(1.0)
    got, sigma = decode(t.numpy(), FORMANT_AXIS)
    assert 1200 * abs(np.log2(got[0] / hz)) < 2.0  # local mean recovers the value to within 2 cents
    assert 25 < sigma[0] < 40  # target blur is 35 cents, truncated to +-80 cents


def test_targets_mask_out_of_range_and_missing():
    t, valid = gaussian_targets(torch, torch.tensor([float("nan"), 0.0, 20.0, 5000.0]), F0_AXIS)
    assert valid.tolist() == [False, False, False, False]
    assert float(t.abs().sum()) == 0.0


def test_crops_align_audio_and_labels(tmp_path):
    from vocal_analysis.synth.sampler import SynthConfig
    from vocal_analysis.synth.shards import ShardReader, write_set
    from vocal_analysis.train.data import SynthCrops, consolidate

    root = write_set(tmp_path / "s", 4, SynthConfig(duration_s=(2.5, 2.5)), seed=3, shard_size=2, workers=1, log=lambda m: None)
    flat = consolidate(root, log=lambda m: None)
    reader = ShardReader(root)
    ds = SynthCrops(flat, np.arange(4), n_frames=50, channel=None)
    for i in range(4):
        audio, lab = ds[i]
        clip_audio, clip_lab, _ = reader[i]
        # the crop start is random: find it from the audio (unique; labels can repeat over a steady stretch), then
        # check that the labels at that start are the crop's
        centres = vqt.N_MAX // 2 + np.arange(50) * vqt.HOP
        starts = [s for s in range(len(clip_lab["f0"]) - 49)
                  if np.allclose(audio[centres], clip_audio[(s + np.arange(50)) * vqt.HOP], atol=1e-6)]
        assert len(starts) == 1
        start = starts[0]
        assert np.array_equal(lab["voiced"], clip_lab["voiced"][start : start + 50])
        assert np.allclose(lab["f0"], clip_lab["f0"][start : start + 50], equal_nan=True)
        assert np.allclose(lab["formants"], clip_lab["formants"][start : start + 50, : lab["formants"].shape[1]], equal_nan=True)


def test_fixed_crops_ignore_worker_seed(tmp_path):
    """Validation items are identical across evals (new DataLoader workers reseed torch); training items vary."""
    from dataclasses import replace

    from vocal_analysis.synth.channel import ChannelConfig
    from vocal_analysis.synth.sampler import SynthConfig
    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import SynthCrops, consolidate

    root = write_set(tmp_path / "s", 2, SynthConfig(duration_s=(2.5, 2.5)), seed=3, shard_size=2, workers=1, log=lambda m: None)
    flat = consolidate(root, log=lambda m: None)
    ch = replace(ChannelConfig(), p_clean=0.0, p_codec=0.0)

    def draw(fixed):
        out = []
        for worker_seed in (1, 2):
            torch.manual_seed(worker_seed)
            out.append(SynthCrops(flat, np.arange(2), n_frames=50, channel=ch, seed=7, fixed=fixed)[1][0])
        return out

    a, b = draw(True)
    assert np.array_equal(a, b)
    a, b = draw(False)
    assert not np.array_equal(a, b)


def test_length_batches_keep_frames_per_batch(tmp_path):
    from vocal_analysis.synth.sampler import SynthConfig
    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import LengthBatches, SynthCrops, collate, consolidate

    batches = list(LengthBatches(200, 1000, (32, 400), 32 * 200, seed=1))
    lengths = np.array([b[0][1] for b in batches])
    assert set(lengths) == {32, 46, 66, 94, 136, 194, 279, 400}  # 8 shapes, log-uniform over 32-400
    frames = np.array([sum(t * k for _, t, k in b) for b in batches])
    assert np.all(np.abs(frames / 6400 - 1) < 0.1)  # same audio per step, to rounding
    assert all(len({(t, k) for _, t, k in b}) == 1 for b in batches)  # one length per batch
    assert max(len(b) for b in batches) <= 36  # about as many clips as a fixed batch of 32

    root = write_set(tmp_path / "s", 2, SynthConfig(duration_s=(2.5, 2.5)), seed=3, shard_size=2, workers=1, log=lambda m: None)
    ds = SynthCrops(consolidate(root, log=lambda m: None), np.arange(2), n_frames=200, channel=None)
    audio, lab = collate([ds[(0, 37, 5)], ds[(1, 37, 5)]])
    assert lab["f0"].shape == (10, 37) and audio.shape[1] == 36 * vqt.HOP + vqt.N_MAX
    _, lab = ds[(0, 400, 1)][0]  # past the end of a 2.5 s clip: flags off, f0 finite (the comb reads it), formants absent
    assert np.isfinite(lab["f0"]).all() and not lab["voiced"][300:].any() and np.isnan(lab["formants"][300:]).all()


def test_teacher_forward_shapes_and_comb():
    from vocal_analysis.models.teacher import Teacher, TeacherConfig

    m = Teacher(TeacherConfig(channels=(8, 8, 8), f0_channels=8, embed=32, gru_hidden=16, gru_layers=1, f0_gru_hidden=16))
    lp = torch.randn(2, vqt.N_BINS, 30)
    out = m(lp)
    assert out["f0_logits"].shape == (2, 30, F0_AXIS.n_bins)
    assert out["formant_logits"].shape == (2, 30, 4, FORMANT_AXIS.n_bins)
    comb = m.comb(torch.full((1, 3), 200.0), torch.ones(1, 3))[0, 0, :, 0].numpy()
    f = vqt.bin_frequencies()
    near = np.abs(f / 200.0 - np.round(f / 200.0)) * 200.0 / f * 1200 / np.log(2)  # cents to nearest harmonic (approx.)
    assert comb.max() <= 1.0 and np.all(comb[near < 3] > 0.95) and np.all(comb[near > 60] < 0.02)
    assert float(m.comb(torch.full((1, 3), 200.0), torch.zeros(1, 3)).abs().max()) == 0.0  # unvoiced: no comb


def test_wsd_schedule_and_branch_config(tmp_path):
    import json
    import math
    from dataclasses import asdict

    from vocal_analysis.train.teacher import TrainConfig, _lr, _p_comb_truth, branch_config

    cfg = TrainConfig()  # wsd, constant phase
    assert _lr(0, cfg) == pytest.approx(cfg.lr / cfg.warmup) and _lr(cfg.warmup - 1, cfg) == pytest.approx(cfg.lr)
    assert _lr(5000, cfg) == _lr(39999, cfg) == cfg.lr
    assert _p_comb_truth(0, cfg) == 1.0 and _p_comb_truth(2000, cfg) == 0.5 and _p_comb_truth(4000, cfg) == 0.0

    (tmp_path / "config.json").write_text(json.dumps({"train": asdict(cfg)}))
    b = branch_config(tmp_path, 16000, "runs/x")
    assert (b.decay_start, b.steps, b.out, b.data) == (16000, 20000, "runs/x", cfg.data)
    assert _lr(15999, b) == cfg.lr and _lr(16000, b) == cfg.lr and _lr(20000, b) == 0.0
    assert _lr(17000, b) == pytest.approx(cfg.lr * (1 - math.sqrt(0.25)))
    assert branch_config(tmp_path, 6000, "o", decay_steps=2000).steps == 8000

    v1 = TrainConfig(schedule="cosine", steps=20000, comb_truth_steps=None)  # teacher v1's schedule, unchanged
    assert _lr(10500, v1) == pytest.approx(cfg.lr * 0.5) and _lr(20000, v1) == pytest.approx(0.0, abs=1e-12)
    assert _p_comb_truth(3000, v1) == pytest.approx(0.5)
    (tmp_path / "config.json").write_text(json.dumps({"train": asdict(v1)}))
    b1 = branch_config(tmp_path, 6000, "o")  # decays from v1's own rate at 6000, as d06k does from its own
    assert b1.lr == pytest.approx(_lr(6000, v1)) and (b1.decay_start, b1.steps) == (6000, 10000)
    assert _lr(6000, b1) == pytest.approx(_lr(6000, v1)) and _lr(10000, b1) == 0.0
    old = {k: v for k, v in asdict(v1).items() if k not in ("schedule", "comb_truth_steps")}  # v1's saved config
    (tmp_path / "config.json").write_text(json.dumps({"train": old}))
    assert branch_config(tmp_path, 6000, "o").lr == pytest.approx(_lr(6000, v1))


def test_conformer_variant_matches_the_gru_teacher_outputs():
    """D-021 item 1: the conformer variant swaps only the temporal modules; outputs and their shapes are unchanged,
    and any crop length works (rotary positions)."""
    import torch

    from vocal_analysis.features import vqt
    from vocal_analysis.models.teacher import Teacher, TeacherConfig

    s2 = dict(channels=(8, 16, 32), f0_channels=8, embed=192, gru_hidden=96, f0_gru_hidden=64)
    gru, conf = Teacher(TeacherConfig(**s2)), Teacher(TeacherConfig(**s2, temporal="conformer"))
    for t in (57, 230):
        x = torch.randn(2, vqt.N_BINS, t)
        a, b = gru(x), conf(x)
        assert a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert abs(n(conf) / n(gru) - 1) < 0.1
