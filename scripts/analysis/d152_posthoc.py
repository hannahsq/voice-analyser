"""D-152 post hoc (not registered): what the failing frames read in the primary block (men, f0 <= 150, D-141's cell by
Praat), so a Praat artefact (a spurious formant between F1 and F2) can be told from a model misreading. Reads cache/d152/.

    python scripts/analysis/d152_posthoc.py > reports/analysis/d152/d152_posthoc.txt
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")

if __name__ == "__main__":
    rows = {"fail": [], "ok": []}
    for p in sorted((ROOT / "cache/d152").glob("male*.npz")):
        z = np.load(p)
        P, f0 = z["P"], z["f0"]
        v = z[f"{MODELS[0]}|v"] > 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = v & np.all(np.isfinite(P), 1) & (P[:, 0] >= 350) & (P[:, 0] <= 650) & (P[:, 1] / P[:, 0] < 2.2) & (f0 <= 150)
            for m in MODELS:
                F = z[f"{m}|F"]
                fail = np.abs(np.log(F[:, 1] / P[:, 2])) < np.abs(np.log(F[:, 1] / P[:, 1]))
                drop = np.abs(np.log(F[:, 0] / P[:, 1])) < np.abs(np.log(F[:, 0] / P[:, 0]))
                for k, s in (("fail", cell & fail), ("ok", cell & ~fail)):
                    rows[k].append(np.c_[P[s], F[s, :3], drop[s], z["B1"][s]])
    print("men, f0 <= 150, cell frames, both seeds pooled")
    for k, r in rows.items():
        r = np.concatenate(r)
        print(f"  {k:4s} n {len(r)}: Praat F1/F2/F3 median {np.round(np.median(r[:, :3], 0))}; model F1/F2/F3 {np.round(np.median(r[:, 3:6], 0))}; "
              f"Praat B1 {np.median(r[:, 7]):.0f}")
        if k == "fail":
            print(f"       dropped F1 (model F1 nearer Praat F2) {r[:, 6].mean():.0%}; Praat F3 below 1800 Hz {np.mean(r[:, 2] < 1800):.0%}; "
                  f"Praat F3 quartiles {np.round(np.percentile(r[:, 2], [25, 50, 75]))}; model F2 quartiles {np.round(np.percentile(r[:, 4], [25, 50, 75]))}")
