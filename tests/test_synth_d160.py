"""D-160's synthesis options: wide low-f0 bandwidths and an off-line LF return phase. Off, they leave a clip unchanged;
on, they change only what they should."""

import dataclasses

import numpy as np
import pytest

from vocal_analysis.synth import glottal
from vocal_analysis.synth.sampler import SynthConfig, generate


def test_lf_refactor_matches_line():
    for rd in (0.3, 0.6, 1.0, 2.0, 2.7):
        a = glottal.lf_params(rd)
        b = glottal.lf_constants(*glottal.rd_line(rd))
        assert a == b


def test_ra_factor_one_is_the_table():
    rd = np.array([0.6, 1.0, 1.37, 2.0])
    a = glottal.source_coefficients(rd, 64)
    b = glottal.source_coefficients(rd, 64, ra_factor=1.0)
    np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-15)


def test_ra_factor_clipped_to_valid_timing():
    for rd in glottal.RD_GRID[::20]:
        ra, rk, rg = glottal.ra_scaled(float(rd), 3.0)
        glottal.lf_constants(ra, rk, rg)  # raises if invalid
        te = (1 + rk) / (2 * rg)
        assert ra <= glottal.RA_MAX_OF_CLOSED * (1 - te) + 1e-12


def test_longer_return_phase_steepens_the_source():
    lo = glottal.source_h_db(np.array([1.0]), 40, ra_factor=0.5)[0]
    hi = glottal.source_h_db(np.array([1.0]), 40, ra_factor=3.0)[0]
    assert (hi[30:] - hi[0]).mean() < (lo[30:] - lo[0]).mean() - 3.0


@pytest.mark.parametrize("index", [0, 1, 2, 3, 5, 8])
def test_off_reproduces_and_on_changes_only_its_part(index):
    base = SynthConfig()
    ref = generate(index, base, seed=2)
    off = generate(index, dataclasses.replace(base, p_wide_bw=0.0, p_ra_offline=0.0), seed=2)
    np.testing.assert_array_equal(off.audio, ref.audio)
    bw = generate(index, dataclasses.replace(base, p_wide_bw=1.0), seed=2)
    ra = generate(index, dataclasses.replace(base, p_ra_offline=1.0), seed=2)
    for c in (bw, ra):  # every other label is the clip's
        for k in ("f0", "voiced", "formants", "rd", "length_cm"):
            np.testing.assert_array_equal(c.labels[k], ref.labels[k])
        assert c.meta["segments"] == ref.meta["segments"] and c.meta["tract"] == ref.meta["tract"]
    np.testing.assert_array_equal(ra.labels["bandwidths"], ref.labels["bandwidths"])
    np.testing.assert_array_equal(bw.labels["bandwidths"][:, 4:], ref.labels["bandwidths"][:, 4:])  # F5+ untouched
    b0, b1 = ref.labels["bandwidths"][:, :4], bw.labels["bandwidths"][:, :4]
    ok = np.isfinite(b0) & np.isfinite(b1)
    assert np.all(b1[ok] >= b0[ok] - 1e-9)  # widen only
    caps = np.array(base.wide_bw_caps)
    assert np.all((b1[ok] <= np.maximum(caps[None, :], b0)[ok] + 1e-6))
    assert bw.meta["d160"]["wide_bw"] and ra.meta["d160"]["ra_factor"] is not None
    assert 0.5 <= ra.meta["d160"]["ra_factor"] <= 3.0
