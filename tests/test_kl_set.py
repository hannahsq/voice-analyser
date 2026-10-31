from dataclasses import replace

import numpy as np
import pytest

from vocal_analysis.synth import kl_set
from vocal_analysis.synth.sampler import SynthConfig, generate


@pytest.mark.slow
def test_clip_has_the_sampler_layout():
    cfg = replace(kl_set.KLSetConfig(), p_nasal_target=1.0)  # every target nasalised: the branch is exercised
    a = kl_set.generate(2, cfg, seed=11)
    ref = generate(0, replace(SynthConfig(), p_tube=1.0), seed=0)
    assert set(a.labels) == set(ref.labels) | {"formants_mixed"}
    n = len(a.labels["t"])
    assert all(len(v) == n for v in a.labels.values())
    assert a.labels["formants"].shape == (n, ref.labels["formants"].shape[1])
    assert np.isfinite(a.audio).all() and 0.4 < np.abs(a.audio).max() <= 0.5 + 1e-6
    f = a.labels["formants"][:, :4]
    assert np.all(np.isnan(f) == a.labels["formants_mixed"][:, :4])  # masked exactly where mixed
    assert np.nanmin(np.diff(f, axis=1)) > 0
    assert a.meta["tract"] == "kl" and min(a.meta["kl"]["port_cm2_targets"]) > 0
