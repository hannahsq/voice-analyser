"""D-045 exploratory addendum: do the D_a measures' correlations with teacher v3's F1 error come from the reference?

    python scripts/analysis/d045_confound.py <d045 out dir>     (CPU; reads ptdb.npz and D-044 stage 2's tsv)

Teacher v3's F1 error on PTDB-TUG dev frames against the closed-phase reference, D-044's R and QCP-FB (D-046 arm B),
and the references' own differences, each correlated with the 12 D_a inputs: pooled Spearman with a speaker bootstrap,
the mean within-speaker Spearman, and a within-speaker shuffle noise floor. Writes d045_confound.txt.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

D044_TSV = Path("reports/analysis/d044/d044_stage2.tsv")
NAMES = ["h1h2", "h1a1", "cpp", "hnr", "tilt", "hf", "jitter", "shimmer", "B1", "B2", "B3", "log2 f0"]
N_BOOT = 2000
N_SHUF = 200


def within_rho(x, y, spk):
    from scipy.stats import spearmanr

    rs = []
    for u in np.unique(spk):
        m = spk == u
        if m.sum() >= 20:
            rs.append(spearmanr(x[m], y[m])[0])
    return float(np.nanmean(rs))


def stats(x, y, spk, rng):
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    x, y, spk = x[ok], y[ok], spk[ok]
    rho = spearmanr(x, y)[0]
    us = np.unique(spk)
    by = {u: np.flatnonzero(spk == u) for u in us}
    bs = [spearmanr(x[ii], y[ii])[0] for ii in
          (np.concatenate([by[u] for u in rng.choice(us, len(us))]) for _ in range(N_BOOT))]
    wr = within_rho(x, y, spk)
    null = []
    for _ in range(N_SHUF):
        xs = x.copy()
        for u in us:
            xs[by[u]] = rng.permutation(xs[by[u]])
        null.append(abs(spearmanr(xs, y)[0]))
    return rho, np.percentile(bs, 2.5), np.percentile(bs, 97.5), wr, np.percentile(null, 95)


def main():
    out = Path(sys.argv[1])
    pt = {k: v for k, v in np.load(out / "ptdb.npz", allow_pickle=True).items()}
    with D044_TSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    tv3_d044 = np.array([float(r["tv3_F1"]) for r in rows])
    ref_d044 = np.array([float(r["ref_F1"]) for r in rows])
    spk_d044 = np.array([r["speaker"] for r in rows])
    # both files take the same frames of cache/cp/ptdb_egg_sub.npz in the same order; check rather than assume
    assert len(rows) == len(pt["tv3_F1"]), (len(rows), len(pt["tv3_F1"]))
    assert (spk_d044 == pt["speaker"].astype(str)).all()
    assert np.nanmax(np.abs(tv3_d044 - pt["tv3_F1"])) < 0.01
    assert np.nanmax(np.abs(ref_d044 - pt["cp_F1"])) < 0.01
    r_f1 = np.array([float(r["R_F1"]) if r["R_F1"] not in ("", "nan") else np.nan for r in rows])

    X = np.column_stack([pt["meas"], pt["B"], np.log2(np.maximum(pt["f0"], 1.0))])
    tv3, cp, q = pt["tv3_F1"], pt["cp_F1"], pt["qcp_F1"]
    targets = {"tv3-cp": tv3 - cp, "tv3-R": tv3 - r_f1, "tv3-QCP": tv3 - q, "cp-R": cp - r_f1, "cp-QCP": cp - q}
    rng = np.random.default_rng(4545)
    lines = [f"D-045 exploratory addendum: {len(tv3)} PTDB-TUG dev frames, aligned with D-044 stage 2 (checked)",
             "Spearman pooled [speaker bootstrap 95 %] / mean within-speaker / within-speaker shuffle null (95th pct |rho|)",
             "* = interval excludes 0 and |pooled| above the null", ""]
    for sx, label in (("f", "women"), ("m", "men")):
        m = pt["sex"] == sx
        med = {k: np.nanmedian(v[m]) for k, v in targets.items()}
        lines.append(f"{label} ({m.sum()} frames, {len(np.unique(pt['speaker'][m]))} speakers); medians (Hz): "
                     + "  ".join(f"{k} {v:+.1f}" for k, v in med.items()))
        lines.append("  " + " " * 8 + "".join(f"{k:>34s}" for k in targets))
        for j, n in enumerate(NAMES):
            cells = []
            for k, y in targets.items():
                rho, lo, hi, wr, nl = stats(X[m, j], y[m], pt["speaker"][m], rng)
                star = "*" if (lo > 0 or hi < 0) and abs(rho) > nl else " "
                cells.append(f"{rho:+.3f} [{lo:+.3f},{hi:+.3f}] {wr:+.3f} {nl:.3f}{star}")
            lines.append(f"  {n:8s}" + "".join(f"{c:>34s}" for c in cells))
            print(lines[-1], flush=True)
        lines.append("")
    # added after the table was seen: size of the HNR link, as median errors by within-speaker HNR quartile
    lines.append("Median F1 error (Hz) by within-speaker HNR quartile (Q1 least harmonic), added after the table was seen:")
    hnr = X[:, NAMES.index("hnr")]
    for sx, label in (("f", "women"), ("m", "men")):
        idx = np.flatnonzero((pt["sex"] == sx) & np.isfinite(hnr))
        q = np.zeros(len(idx), int)
        for u in np.unique(pt["speaker"][idx]):
            j = pt["speaker"][idx] == u
            q[j] = np.digitize(hnr[idx][j], np.percentile(hnr[idx][j], [25, 50, 75]))
        for k in ("tv3-cp", "tv3-R", "tv3-QCP"):
            e = targets[k][idx]
            lines.append(f"  {label:5s} {k:8s} " + "  ".join(f"Q{i + 1} {np.nanmedian(e[q == i]):+6.1f}" for i in range(4)))
    (out / "d045_confound.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
