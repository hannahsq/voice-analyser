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
        # find the crop start by matching f0 labels, then check the audio at frame centres
        start = next(s for s in range(len(clip_lab["f0"]) - 49) if np.allclose(clip_lab["f0"][s : s + 50], lab["f0"]))
        for j in (0, 17, 49):
            assert audio[vqt.N_MAX // 2 + j * vqt.HOP] == pytest.approx(clip_audio[(start + j) * vqt.HOP], abs=1e-6)
        assert np.array_equal(lab["voiced"], clip_lab["voiced"][start : start + 50])


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
