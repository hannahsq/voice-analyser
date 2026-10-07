"""Metrics against hand-computed cases (plan §0, §6.2)."""

import numpy as np
import pytest

from vocal_analysis.eval import metrics as M

nan = np.nan


def test_cents_error():
    assert M.cents_error(440.0, 220.0) == pytest.approx(1200.0)
    assert M.cents_error(220.0 * 2 ** (50 / 1200), 220.0) == pytest.approx(50.0)


def test_detection_rate_and_vs_or():
    ref = np.array([500.0, 500.0, 2500.0, 2500.0, nan])
    est = np.array([590.0, 650.0, 2650.0, 2900.0, 1000.0])
    # errors: 90 (18%,<200) ok; 150 (30%) rel fails; 150 (6%,<200) ok; 400 (16%, >200) abs fails
    assert M.detection_rate(est, ref, 0.20, 200.0, mode="and") == pytest.approx(2 / 4)
    # "or": 150 Hz passes abs; 400 Hz passes rel (16% < 20%)
    assert M.detection_rate(est, ref, 0.20, 200.0, mode="or") == pytest.approx(4 / 4)


def test_detection_rate_missing_estimate_is_miss():
    assert M.detection_rate(np.array([nan, 500.0]), np.array([500.0, 500.0])) == pytest.approx(0.5)


def test_formant_errors():
    ref = np.array([500.0, 1000.0, 2000.0, nan])
    est = np.array([550.0, 900.0, 2800.0, 700.0])
    r = M.formant_errors(est, ref)
    assert r["n"] == 3
    assert r["mae_hz"] == pytest.approx((50 + 100 + 800) / 3)
    assert r["mae_pct"] == pytest.approx((10 + 10 + 40) / 3)
    assert r["rmse_hz"] == pytest.approx(np.sqrt((50**2 + 100**2 + 800**2) / 3))
    assert r["gross_rate"] == pytest.approx(1 / 3)  # only 40% > 30%


def test_pitch_metrics_hand_case():
    ref = np.array([200.0, 200.0, 200.0, 200.0, 200.0, 0.0, 0.0])
    est = np.array([
        200.0 * 2 ** (40 / 1200),  # within 50 cents
        400.0,  # octave up
        100.0 * 2 ** (30 / 1200),  # octave down, chroma within 50
        300.0,  # fifth: wrong pitch and chroma
        0.0,  # missed voicing
        150.0,  # false voicing
        0.0,  # correct unvoiced
    ])
    r = M.pitch_metrics(est, ref)
    assert r["n_voiced"] == 5
    assert r["rpa"] == pytest.approx(1 / 5)
    assert r["rca"] == pytest.approx(3 / 5)
    assert r["octave_error_rate"] == pytest.approx(2 / 5)
    assert r["gross_error_rate"] == pytest.approx(4 / 5)  # all but the first deviate >20% (missing counts)
    # voicing: tp=4, fp=1, fn=1
    assert r["voicing_precision"] == pytest.approx(0.8)
    assert r["voicing_recall"] == pytest.approx(0.8)
    assert r["voicing_f1"] == pytest.approx(0.8)


def test_log_bandwidth_and_db_error():
    r = M.log_bandwidth_error(np.array([100.0, 50.0]), np.array([50.0, 100.0]))
    assert r["mae_log"] == pytest.approx(np.log(2)) and r["bias_log"] == pytest.approx(0.0)
    d = M.db_error(np.array([3.0, -1.0, nan]), np.array([1.0, 0.0, 2.0]))
    assert d == {"mae_db": pytest.approx(1.5), "bias_db": pytest.approx(0.5), "n": 2}


def test_within_group_sd():
    v = np.array([15.0, 17.0, 14.0, 14.0, 99.0])
    g = np.array(["a", "a", "b", "b", "c"])  # c has one value: skipped
    assert M.within_group_sd(v, g) == pytest.approx((np.std([15, 17], ddof=1) + 0.0) / 2)


def test_interval_coverage_gaussian():
    rng = np.random.default_rng(0)
    ref = np.zeros(200_000)
    sigma = np.full_like(ref, 2.0)
    est = rng.normal(0.0, 2.0, ref.size)
    cov = M.interval_coverage(est, sigma, ref)
    for lvl, c in cov.items():
        assert c == pytest.approx(lvl, abs=0.005)
    # overconfident sigma -> under-coverage
    assert M.interval_coverage(est, sigma / 2, ref)[0.95] < 0.75


def test_reliability_ece_hand_case():
    prob = np.array([0.1, 0.1, 0.9, 0.9])
    outcome = np.array([0, 1, 1, 1])
    r = M.reliability(prob, outcome, n_bins=10)
    # bin 1: conf .1 acc .5 (n=2); bin 9: conf .9 acc 1 (n=2)
    assert r.ece == pytest.approx((2 * 0.4 + 2 * 0.1) / 4)
    assert r.counts.sum() == 4


def test_reliability_prob_one_lands_in_last_bin():
    r = M.reliability(np.array([1.0, 0.0]), np.array([1, 0]), n_bins=4)
    assert r.counts.tolist() == [1, 0, 0, 1] and r.ece == pytest.approx(0.0)


def test_f0_band_labels():
    labels = M.f0_band_labels(np.array([100.0, 150.0, 299.0, 450.0, 800.0, 0.0, nan]))
    assert labels.tolist() == ["<150", "150-300", "150-300", "300-500", ">500", "unvoiced", "unvoiced"]


def test_speaker_bootstrap_groups_whole_speakers():
    # speaker A: all correct; speaker B: all wrong; many frames for A.
    correct = np.array([1.0] * 90 + [0.0] * 10)
    spk = np.array(["A"] * 90 + ["B"] * 10)
    res = M.speaker_bootstrap(lambda idx: correct[idx].mean(), spk, n_resamples=500, seed=1)
    assert res.estimate == pytest.approx(0.9) and res.n_groups == 2
    # Resampling speakers yields only {A,A}=1.0, {A,B}=0.9, {B,B}=0.0
    assert res.low == pytest.approx(0.0) and res.high == pytest.approx(1.0)


def test_speaker_bootstrap_deterministic():
    x = np.random.default_rng(0).random(300)
    spk = np.repeat(np.arange(30), 10)
    a = M.speaker_bootstrap(lambda i: x[i].mean(), spk, 200, seed=3)
    b = M.speaker_bootstrap(lambda i: x[i].mean(), spk, 200, seed=3)
    assert a == b and a.low < a.estimate < a.high


def test_stratified():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    s = np.array(["lo", "hi", "lo", "hi"])
    out = M.stratified(lambda i: {"mean": x[i].mean()}, s)
    assert out == {"all": {"mean": 2.5}, "hi": {"mean": 3.0}, "lo": {"mean": 2.0}}


def test_frame_stats_sum_to_metric_functions():
    rng = np.random.default_rng(0)
    ref = rng.uniform(300, 3000, 500)
    ref[rng.random(500) < 0.1] = nan
    est = ref * rng.normal(1, 0.2, 500)
    est[rng.random(500) < 0.1] = nan
    fs = {k: v.sum() for k, v in M.formant_frame_stats(est, ref, [(0.2, 200.0), (0.3, 300.0)]).items()}
    fe = M.formant_errors(est, ref)
    assert fs["abs_err"] / fs["n_both"] == pytest.approx(fe["mae_hz"])
    assert fs["abs_pct"] / fs["n_both"] == pytest.approx(fe["mae_pct"])
    assert np.sqrt(fs["sq_err"] / fs["n_both"]) == pytest.approx(fe["rmse_hz"])
    assert fs["gross"] / fs["n_ref"] == pytest.approx(fe["gross_rate"])
    assert fs["det_0.2_200"] / fs["n_ref"] == pytest.approx(M.detection_rate(est, ref, 0.2, 200.0))
    assert fs["det_0.3_300"] / fs["n_ref"] == pytest.approx(M.detection_rate(est, ref, 0.3, 300.0))

    f0_ref = np.where(rng.random(500) < 0.3, 0.0, rng.uniform(80, 800, 500))
    f0_est = f0_ref * 2.0 ** rng.choice([0, 0, 0, 1, -1, 0.3], 500) * rng.normal(1, 0.01, 500)
    ev = rng.random(500) < 0.7
    ps = {k: v.sum() for k, v in M.pitch_frame_stats(f0_est, f0_ref, ev).items()}
    pm = M.pitch_metrics(f0_est, f0_ref, est_voiced=ev)
    for k in ("rpa", "rca", "gross"):
        assert ps[k] / ps["n_voiced"] == pytest.approx(pm[k if k != "gross" else "gross_error_rate"])
    assert ps["octave"] / ps["n_voiced"] == pytest.approx(pm["octave_error_rate"])
    p, r = ps["tp"] / (ps["tp"] + ps["fp"]), ps["tp"] / (ps["tp"] + ps["fn"])
    assert (p, r) == (pytest.approx(pm["voicing_precision"]), pytest.approx(pm["voicing_recall"]))


def test_bootstrap_sums_matches_index_bootstrap():
    rng = np.random.default_rng(1)
    spk = np.repeat(np.arange(25), rng.integers(5, 40, 25))
    correct = (rng.random(len(spk)) < 0.6 + 0.01 * (spk % 10)).astype(float)
    ids, sums = M.group_sums({"num": correct, "den": np.ones_like(correct)}, spk)
    fast = M.bootstrap_sums(sums, lambda s: s["num"] / s["den"], n_resamples=4000, seed=0)
    slow = M.speaker_bootstrap(lambda i: correct[i].mean(), spk, n_resamples=4000, seed=0)
    assert fast.estimate == pytest.approx(slow.estimate)
    # Different random streams, same distribution: CIs agree to Monte Carlo error.
    assert fast.low == pytest.approx(slow.low, abs=0.01) and fast.high == pytest.approx(slow.high, abs=0.01)

