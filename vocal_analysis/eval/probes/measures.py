"""Paired probe measures (D-050).

Inputs are per item and condition: `y` (items, conditions), an output summarised per item and condition (e.g. the
median log2 F1 over the item's sonorant frames), and `x` (conditions,) or (items, conditions), the imposed change in
the same log2 units. For an equivariant output the ideal slope of y on x is 1; for an invariant one, x is the
nuisance's own log change (e.g. log2 f0 factor) and the ideal slope is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def item_slopes(y: np.ndarray, x: np.ndarray, min_points: int = 3) -> np.ndarray:
    """Least-squares slope of y on x per item (row), over its finite conditions; NaN with fewer than min_points."""
    y = np.asarray(y, float)
    x = np.broadcast_to(np.asarray(x, float), y.shape)
    out = np.full(len(y), np.nan)
    for i, (yi, xi) in enumerate(zip(y, x)):
        ok = np.isfinite(yi) & np.isfinite(xi)
        if ok.sum() >= min_points and np.ptp(xi[ok]) > 0:
            xc = xi[ok] - xi[ok].mean()
            out[i] = float(np.sum(xc * (yi[ok] - yi[ok].mean())) / np.sum(xc**2))
    return out


def change_from_reference(y: np.ndarray, ref: int) -> np.ndarray:
    """y minus the item's reference condition (items, conditions); for invariance probes without an ordered x."""
    y = np.asarray(y, float)
    return y - y[:, ref : ref + 1]


def _group_index(groups: np.ndarray) -> list[np.ndarray]:
    groups = np.asarray(groups)
    return [np.flatnonzero(groups == g) for g in np.unique(groups)]


def group_bootstrap(values: np.ndarray, groups: np.ndarray, stat=np.nanmedian, n: int = 2000, seed: int = 0,
                    ci: float = 95.0) -> tuple[float, float, float]:
    """(point, low, high): stat over items, with whole groups (speakers, recordings) resampled."""
    v = np.asarray(values, float)
    idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(np.concatenate([v[idx[k]] for k in rng.integers(len(idx), size=len(idx))])) for _ in range(n)])
    lo, hi = np.nanpercentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat(v)), float(lo), float(hi)


def shuffle_null(y: np.ndarray, x: np.ndarray, stat=np.nanmedian, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Distribution of stat(item slopes) with each item's conditions permuted (pairs y with the wrong x).

    The slope an output shows when it has no relation to the imposed change: the floor any real slope is read
    against. Permutation keeps each item's own spread of y."""
    y = np.asarray(y, float)
    x = np.broadcast_to(np.asarray(x, float), y.shape)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        perm = np.argsort(rng.random(y.shape), axis=1)
        out[k] = stat(item_slopes(np.take_along_axis(y, perm, 1), x))
    return out


@dataclass
class SlopeSummary:
    """One output's slope summary for one probe."""

    point: float  # stat over items
    low: float  # group-bootstrap interval
    high: float
    null_median: float  # shuffle null
    null_low: float
    null_high: float
    p_null: float  # share of the null at least as far from 0 as the observed point (two-sided)
    n_items: int
    n_groups: int


def summarise_slopes(y: np.ndarray, x: np.ndarray, groups: np.ndarray, n_boot: int = 2000, n_null: int = 1000,
                     seed: int = 0) -> SlopeSummary:
    s = item_slopes(y, x)
    ok = np.isfinite(s)
    g = np.asarray(groups)[ok]
    point, lo, hi = group_bootstrap(s[ok], g, n=n_boot, seed=seed)
    null = shuffle_null(np.asarray(y, float)[ok], np.broadcast_to(np.asarray(x, float), np.shape(y))[ok], n=n_null,
                        seed=seed + 1)
    nl, nm, nh = np.nanpercentile(null, [2.5, 50, 97.5])
    p = float(np.mean(np.abs(null) >= abs(point)))
    return SlopeSummary(point, lo, hi, float(nm), float(nl), float(nh), p, int(ok.sum()), len(np.unique(g)))


def sigma_growth(sigma: np.ndarray, ref: int) -> np.ndarray:
    """Per item and condition: log2 of the model's sigma over its sigma in the reference condition.

    Positive where the model widens its interval; an output that loses evidence should show this rather than drift."""
    s = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(s / s[:, ref : ref + 1])
