"""Evaluation metrics (plan §6.2).

All functions take flat per-frame numpy arrays. Frames without a reference
(NaN, or unvoiced for f0) are excluded by the caller-visible masks below, so
every metric states its own denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------- units


def hz_to_cents(f: np.ndarray, ref_hz: float = 10.0) -> np.ndarray:
    return 1200.0 * np.log2(np.asarray(f, dtype=float) / ref_hz)


def cents_error(est_hz: np.ndarray, ref_hz: np.ndarray) -> np.ndarray:
    return 1200.0 * np.log2(np.asarray(est_hz, float) / np.asarray(ref_hz, float))


def _valid(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones(np.shape(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


# --------------------------------------------------------------------------- formants


def detection_rate(
    est: np.ndarray, ref: np.ndarray, rel_tol: float = 0.20, abs_tol_hz: float = 200.0, mode: str = "and"
) -> float:
    """Fraction of reference frames whose estimate is within tolerance.

    mode="and": |err| <= rel_tol*ref AND |err| <= abs_tol_hz (stricter; default).
    mode="or":  either bound suffices. The literature is inconsistent here; see
    docs/decisions.md (D-004). Frames with a reference but a missing estimate
    count as misses.
    """
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    has_ref = np.isfinite(ref)
    if not has_ref.any():
        return float("nan")
    err = np.abs(est - ref)
    rel_ok = err <= rel_tol * ref
    abs_ok = err <= abs_tol_hz
    ok = (rel_ok & abs_ok) if mode == "and" else (rel_ok | abs_ok)
    ok &= np.isfinite(est)
    return float(ok[has_ref].mean())


def formant_errors(est: np.ndarray, ref: np.ndarray, gross_rel: float = 0.30) -> dict[str, float]:
    """MAE (Hz, %), RMSE (Hz) over frames with both values; gross rate over frames with a reference."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    m = _valid(est, ref)
    has_ref = np.isfinite(ref)
    if not m.any():
        nan = float("nan")
        return {"mae_hz": nan, "mae_pct": nan, "rmse_hz": nan, "gross_rate": nan, "n": 0}
    err = est[m] - ref[m]
    gross = np.ones(has_ref.sum(), dtype=bool)  # missing estimate counts as gross
    rel = np.abs(est[has_ref] - ref[has_ref]) / ref[has_ref]
    gross[np.isfinite(rel)] = rel[np.isfinite(rel)] > gross_rel
    return {
        "mae_hz": float(np.mean(np.abs(err))),
        "mae_pct": float(100.0 * np.mean(np.abs(err) / ref[m])),
        "rmse_hz": float(np.sqrt(np.mean(err**2))),
        "gross_rate": float(gross.mean()),
        "n": int(m.sum()),
    }


# --------------------------------------------------------------------------- f0


def pitch_metrics(
    est_hz: np.ndarray,
    ref_hz: np.ndarray,
    est_voiced: np.ndarray | None = None,
    ref_voiced: np.ndarray | None = None,
    tol_cents: float = 50.0,
    gross_rel: float = 0.20,
) -> dict[str, float]:
    """Raw pitch/chroma accuracy, octave and gross error rates, voicing P/R/F1.

    Pitch metrics are computed over reference-voiced frames regardless of the
    estimated voicing decision (as in mir_eval's raw pitch accuracy). Voicing
    defaults to "f0 > 0 and finite" when not given.
    """
    est_hz, ref_hz = np.asarray(est_hz, float), np.asarray(ref_hz, float)
    if ref_voiced is None:
        ref_voiced = np.isfinite(ref_hz) & (ref_hz > 0)
    if est_voiced is None:
        est_voiced = np.isfinite(est_hz) & (est_hz > 0)
    ref_voiced, est_voiced = np.asarray(ref_voiced, bool), np.asarray(est_voiced, bool)

    out: dict[str, float] = {}
    rv = ref_voiced & np.isfinite(ref_hz) & (ref_hz > 0)
    n = int(rv.sum())
    if n:
        e, r = est_hz[rv], ref_hz[rv]
        has_est = np.isfinite(e) & (e > 0)
        c = np.full(n, np.inf)
        c[has_est] = cents_error(e[has_est], r[has_est])
        folded = np.full(n, np.inf)
        folded[has_est] = np.abs((c[has_est] + 600.0) % 1200.0 - 600.0)
        octaves = np.round(c / 1200.0)
        out["rpa"] = float(np.mean(np.abs(c) <= tol_cents))
        out["rca"] = float(np.mean(folded <= tol_cents))
        out["octave_error_rate"] = float(np.mean((folded <= tol_cents) & (np.abs(c) > tol_cents) & (octaves != 0)))
        rel = np.full(n, np.inf)
        rel[has_est] = np.abs(e[has_est] - r[has_est]) / r[has_est]
        out["gross_error_rate"] = float(np.mean(rel > gross_rel))
    else:
        out.update(rpa=float("nan"), rca=float("nan"), octave_error_rate=float("nan"), gross_error_rate=float("nan"))
    out["n_voiced"] = n

    tp = int(np.sum(est_voiced & ref_voiced))
    fp = int(np.sum(est_voiced & ~ref_voiced))
    fn = int(np.sum(~est_voiced & ref_voiced))
    p = tp / (tp + fp) if tp + fp else float("nan")
    r_ = tp / (tp + fn) if tp + fn else float("nan")
    out["voicing_precision"] = p
    out["voicing_recall"] = r_
    if tp:
        out["voicing_f1"] = 2 * p * r_ / (p + r_)
    else:
        out["voicing_f1"] = 0.0 if fp + fn else float("nan")
    return out


# --------------------------------------------------------------------------- other quantities


def log_bandwidth_error(est_hz: np.ndarray, ref_hz: np.ndarray) -> dict[str, float]:
    """Error in natural-log units (0.1 ~ 10 %)."""
    est_hz, ref_hz = np.asarray(est_hz, float), np.asarray(ref_hz, float)
    m = _valid(est_hz, ref_hz) & (est_hz > 0) & (ref_hz > 0)
    if not m.any():
        return {"mae_log": float("nan"), "bias_log": float("nan"), "n": 0}
    d = np.log(est_hz[m]) - np.log(ref_hz[m])
    return {"mae_log": float(np.mean(np.abs(d))), "bias_log": float(np.mean(d)), "n": int(m.sum())}


def db_error(est_db: np.ndarray, ref_db: np.ndarray) -> dict[str, float]:
    """For H1*-H2* and other dB-valued measures."""
    est_db, ref_db = np.asarray(est_db, float), np.asarray(ref_db, float)
    m = _valid(est_db, ref_db)
    if not m.any():
        return {"mae_db": float("nan"), "bias_db": float("nan"), "n": 0}
    d = est_db[m] - ref_db[m]
    return {"mae_db": float(np.mean(np.abs(d))), "bias_db": float(np.mean(d)), "n": int(m.sum())}


def within_group_sd(values: np.ndarray, groups: np.ndarray) -> float:
    """Mean over groups of the per-group SD (ddof=1); e.g. anatomical VTL per speaker across utterances."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    sds = [np.std(values[groups == g], ddof=1) for g in np.unique(groups) if np.sum(groups == g) >= 2]
    return float(np.mean(sds)) if sds else float("nan")


# --------------------------------------------------------------------------- calibration


def interval_coverage(
    est: np.ndarray, sigma: np.ndarray, ref: np.ndarray, levels: Sequence[float] = (0.5, 0.8, 0.95)
) -> dict[float, float]:
    """Empirical coverage of central Gaussian intervals est ± z·sigma.

    Use consistent units: e.g. est/ref in cents and sigma in cents.
    """
    est, sigma, ref = (np.asarray(a, float) for a in (est, sigma, ref))
    m = _valid(est, sigma, ref)
    z = np.abs(est[m] - ref[m]) / sigma[m]
    return {lvl: float(np.mean(z <= norm.ppf(0.5 + lvl / 2))) if m.any() else float("nan") for lvl in levels}


@dataclass
class Reliability:
    bin_edges: np.ndarray
    confidence: np.ndarray  # mean predicted probability per bin (NaN if empty)
    accuracy: np.ndarray  # observed frequency per bin (NaN if empty)
    counts: np.ndarray
    ece: float


def reliability(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> Reliability:
    """Reliability diagram data and expected calibration error for a binary probability (e.g. voicing)."""
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        conf = np.bincount(idx, weights=prob, minlength=n_bins) / counts
        acc = np.bincount(idx, weights=outcome, minlength=n_bins) / counts
    nz = counts > 0
    ece = float(np.sum(counts[nz] * np.abs(conf[nz] - acc[nz])) / max(counts.sum(), 1))
    return Reliability(edges, conf, acc, counts, ece)


# --------------------------------------------------------------------------- stratification and bootstrap


def f0_band_labels(f0_hz: np.ndarray, edges: Sequence[float] = (150.0, 300.0, 500.0)) -> np.ndarray:
    """Band label per frame: '<150', '150-300', '300-500', '>500' (plan §6.2); 'unvoiced' if f0 is missing."""
    f0_hz = np.asarray(f0_hz, float)
    names = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
    labels = np.array(names, dtype=object)[np.digitize(np.nan_to_num(f0_hz, nan=0.0), edges)]
    labels[~(np.isfinite(f0_hz) & (f0_hz > 0))] = "unvoiced"
    return labels


@dataclass
class BootstrapResult:
    estimate: float
    low: float
    high: float
    n_groups: int


def speaker_bootstrap(
    metric: Callable[[np.ndarray], float],
    speakers: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile CI where whole speakers are resampled with replacement (plan §6.2).

    `metric` receives an index array into the per-frame arrays and returns a
    scalar; speakers is the per-frame speaker id.
    """
    speakers = np.asarray(speakers)
    _, inverse = np.unique(speakers, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))
    groups = np.split(order, bounds[:-1])
    point = float(metric(np.arange(len(speakers))))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(groups), len(groups))
        stats[i] = metric(np.concatenate([groups[g] for g in pick]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(point, float(lo), float(hi), len(groups))


def group_sums(values: Mapping[str, np.ndarray], groups: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-group sums of per-frame statistics (counts, error sums, indicator flags).

    Returns (group_ids, {name: (G,) sums}). Ratio metrics computed from these
    sums are exact, and resampling groups is cheap (see bootstrap_sums).
    """
    ids, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return ids, {k: np.bincount(inverse, weights=np.asarray(v, float), minlength=len(ids)) for k, v in values.items()}


def bootstrap_sums(
    sums: Mapping[str, np.ndarray],
    stat: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Speaker bootstrap on per-group sums: equivalent to speaker_bootstrap for ratio metrics.

    `stat` maps a dict of summed statistics (scalars or (B,) arrays) to the
    metric; it is called once on the full totals and once, vectorised, on all
    resamples.
    """
    n_groups = len(next(iter(sums.values())))
    point = float(stat({k: v.sum() for k, v in sums.items()}))
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_resamples, n_groups))
    picks = rng.integers(0, n_groups, (n_resamples, n_groups))
    np.add.at(weights, (np.arange(n_resamples)[:, None], picks), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = np.asarray(stat({k: weights @ v for k, v in sums.items()}), float)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(point, float(lo), float(hi), n_groups)


def formant_frame_stats(
    est: np.ndarray, ref: np.ndarray, tolerances: Sequence[Sequence[float]] = ((0.20, 200.0),), gross_rel: float = 0.30, mode: str = "and"
) -> dict[str, np.ndarray]:
    """Per-frame quantities whose sums give formant_errors / detection_rate exactly."""
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    has_ref = np.isfinite(ref)
    both = has_ref & np.isfinite(est)
    err = np.where(both, np.abs(est - ref), 0.0)
    safe_ref = np.where(has_ref, ref, 1.0)
    out = {
        "n_ref": has_ref.astype(float),
        "n_both": both.astype(float),
        "abs_err": err,
        "abs_pct": np.where(both, 100.0 * err / safe_ref, 0.0),
        "sq_err": err**2,
        "gross": (has_ref & ~(both & (err / safe_ref <= gross_rel))).astype(float),
    }
    for rel, ab in tolerances:
        rel_ok, abs_ok = err <= rel * safe_ref, err <= ab
        ok = (rel_ok & abs_ok) if mode == "and" else (rel_ok | abs_ok)
        out[f"det_{rel:g}_{ab:g}"] = (both & ok).astype(float)
    return out


def pitch_frame_stats(
    est_hz: np.ndarray, ref_hz: np.ndarray, est_voiced: np.ndarray, tol_cents: float = 50.0, gross_rel: float = 0.20
) -> dict[str, np.ndarray]:
    """Per-frame quantities whose sums give pitch_metrics exactly."""
    est_hz, ref_hz = np.asarray(est_hz, float), np.asarray(ref_hz, float)
    rv = np.isfinite(ref_hz) & (ref_hz > 0)
    ev = np.asarray(est_voiced, bool)
    has_est = rv & np.isfinite(est_hz) & (est_hz > 0)
    c = np.full(len(ref_hz), np.inf)
    c[has_est] = cents_error(est_hz[has_est], ref_hz[has_est])
    folded = np.full(len(ref_hz), np.inf)
    folded[has_est] = np.abs((c[has_est] + 600.0) % 1200.0 - 600.0)
    rel = np.full(len(ref_hz), np.inf)
    rel[has_est] = np.abs(est_hz[has_est] - ref_hz[has_est]) / ref_hz[has_est]
    with np.errstate(invalid="ignore"):
        octave = (folded <= tol_cents) & (np.abs(c) > tol_cents) & (np.round(c / 1200.0) != 0)
    return {
        "n_voiced": rv.astype(float),
        "rpa": (rv & (np.abs(c) <= tol_cents)).astype(float),
        "rca": (rv & (folded <= tol_cents)).astype(float),
        "octave": (rv & octave).astype(float),
        "gross": (rv & (rel > gross_rel)).astype(float),
        "tp": (ev & rv).astype(float),
        "fp": (ev & ~rv).astype(float),
        "fn": (~ev & rv).astype(float),
    }


def stratified(
    metric: Callable[[np.ndarray], Mapping[str, float]], strata: np.ndarray
) -> dict[str, Mapping[str, float]]:
    """Apply `metric(indices)` per stratum and overall."""
    strata = np.asarray(strata)
    out = {"all": metric(np.arange(len(strata)))}
    for s in sorted(set(strata.tolist()), key=str):
        out[str(s)] = metric(np.flatnonzero(strata == s))
    return out
