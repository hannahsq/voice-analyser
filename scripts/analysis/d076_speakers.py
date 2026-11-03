"""D-076 addendum (reported, deciding nothing): per-speaker CQ slopes with their OLS standard errors, beside each
speaker's CQ spread and closed-phase duration (the reference's window), from D-076's data and model.

    python scripts/analysis/d076_speakers.py <windowed scratch dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("reports/analysis/d076")
V31 = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")


def main():
    d = dict(np.load(Path(sys.argv[1]) / "ptdb_egg_sub.npz", allow_pickle=True))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    e = np.mean([d[m][:, 0] for m in V31], 0) - d["cp"][:, 0]
    ok = dev & np.isfinite(d["cq"]) & np.isfinite(d["cp"][:, 0]) & (d["f0"] > 0) & np.isfinite(e)
    closed_ms = 1000 * d["cq"] / d["f0"]
    L = ["D-076 addendum (2026-11-04; reported, deciding nothing): per speaker, the CQ slope of v3.1's F1 error (Hz per "
         "0.1 CQ, controlling closed-phase F1 and log f0) with its OLS standard error; CQ median and SD; the closed phase's "
         "median duration (the reference's window); the error's median", "",
         f"{'speaker':8s} {'sex':3s} {'frames':>6s} {'slope':>7s} {'SE':>5s} {'t':>6s} {'CQ med':>7s} {'CQ SD':>6s} "
         f"{'closed ms':>9s} {'error med':>9s} {'error SD':>8s}"]
    rows = []
    for s in np.unique(d["speaker"][ok]):
        k = ok & (d["speaker"] == s)
        X = np.column_stack([d["cq"][k], np.ones(k.sum()), d["cp"][k, 0], np.log(d["f0"][k])])
        y = e[k]
        b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sigma2 = float(np.sum((y - X @ b) ** 2) / (len(y) - X.shape[1]))
        se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[0, 0])) * 0.1
        sl = float(b[0]) * 0.1
        rows.append((d["sex"][k][0], sl, se))
        L.append(f"{s:8s} {d['sex'][k][0]:3s} {k.sum():6d} {sl:+7.2f} {se:5.2f} {sl / se:+6.2f} {np.median(d['cq'][k]):7.3f} "
                 f"{np.std(d['cq'][k]):6.3f} {np.median(closed_ms[k]):9.2f} {np.median(y):+9.1f} {np.std(y):8.1f}")
    L.append("")
    for sx, name in (("f", "women"), ("m", "men")):
        r = [(sl, se) for x, sl, se in rows if x == sx]
        w = np.array([1 / se**2 for _, se in r])
        fe = float(np.sum(w * np.array([sl for sl, _ in r])) / w.sum())
        q = float(np.sum(w * (np.array([sl for sl, _ in r]) - fe) ** 2))
        L.append(f"{name}: inverse-variance mean {fe:+.2f} +- {1 / np.sqrt(w.sum()):.2f}; heterogeneity Q {q:.1f} on "
                 f"{len(r) - 1} df; slopes below 0 / above 0: {sum(sl < 0 for sl, _ in r)} / {sum(sl > 0 for sl, _ in r)}")
    L.append("Note: frames within a speaker are not independent (neighbouring frames share cycles and utterances), so the "
             "OLS SEs are too small; read the t values as relative, not as tests.")
    (OUT / "speakers.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
