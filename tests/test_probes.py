import numpy as np

from vocal_analysis.eval.probes import measures as M


def test_item_slopes_recovers_known_slopes():
    x = np.log2([0.85, 0.92, 1.0, 1.09, 1.18])
    y = np.stack([1.0 * x + 3, 0.5 * x - 1, 0 * x + 2])
    np.testing.assert_allclose(M.item_slopes(y, x), [1.0, 0.5, 0.0], atol=1e-12)


def test_item_slopes_skips_nan_and_short_rows():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([[0.0, 2.0, np.nan, 6.0], [1.0, np.nan, np.nan, 2.0]])
    s = M.item_slopes(y, x)
    assert s[0] == 2.0 and np.isnan(s[1])


def test_shuffle_null_centres_on_zero_and_real_slope_stands_out():
    rng = np.random.default_rng(1)
    x = np.log2([0.85, 0.92, 1.0, 1.09, 1.18])
    y = x[None, :] + 0.02 * rng.standard_normal((60, 5))
    groups = np.repeat(np.arange(12), 5)
    s = M.summarise_slopes(y, x, groups, n_boot=300, n_null=300)
    assert abs(s.point - 1) < 0.05 and s.low < 1 < s.high
    assert abs(s.null_median) < 0.15 and s.p_null == 0.0
    assert s.n_items == 60 and s.n_groups == 12


def test_sigma_growth_is_zero_at_reference():
    sig = np.array([[10.0, 20.0, 40.0], [5.0, 5.0, 2.5]])
    np.testing.assert_allclose(M.sigma_growth(sig, 0), [[0, 1, 2], [0, 0, -1]])


def test_change_from_reference():
    y = np.array([[1.0, 1.5, 0.5]])
    np.testing.assert_allclose(M.change_from_reference(y, 0), [[0, 0.5, -0.5]])
