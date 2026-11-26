"""D-146: F1's acoustic bandwidth in train_v2kb's coupled renders (D-080's coupled set), in D-141's cell at low f0.
train_v2kb predates D-081's acoustic bandwidth label, so a random sample of its clips is regenerated with its own config
and seed (`d078_render.config()`, seed 78), checked against the stored labels, and read for the cycle-mean bandwidth
B (1 + (bm - 1) <a>) (D-081) and the peak-opening one, B bm. Descriptive; CPU.

    python scripts/analysis/d146_coupled_bw.py [n_clips]   -> reports/analysis/d146/d146.{txt,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d146"
SET = ROOT / "data/synth/train_v2kb/flat"
SEED, N_SET = 78, 20000


def _cell(F, f0, son):
    return son & np.isfinite(f0) & (F[:, 0] >= 350) & (F[:, 0] <= 650) & (F[:, 1] / F[:, 0] < 2.2)


def _summary(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "gt250": float(np.mean(x > 250)), "gt360": float(np.mean(x > 360))}


def main(n_clips: int) -> None:
    sys.path.insert(0, str(ROOT / "scripts/analysis"))
    import d078_render

    from vocal_analysis.synth.sampler import generate

    cfg = d078_render.config()
    fo = np.load(SET / "frame_offsets.npy")
    st = {k: np.load(SET / f"label_{k}.npy", mmap_mode="r") for k in ("formants", "bandwidths", "f0", "sonorant")}
    idx = np.sort(np.random.default_rng(1460).choice(N_SET, n_clips, replace=False))
    rows = {"closed": [], "cycle_mean": [], "peak": [], "f0": [], "opening": []}
    mism = 0
    for j, i in enumerate(idx):
        c = generate(int(i), cfg, seed=SEED)
        L = c.labels
        a, b = fo[i], fo[i + 1]
        same = (len(L["f0"]) == b - a and np.array_equal(L["formants"], st["formants"][a:b], equal_nan=True)
                and np.array_equal(L["bandwidths"], st["bandwidths"][a:b], equal_nan=True))
        if not same:
            mism += 1
            continue
        lo = (L["f0"] <= 150) & _cell(L["formants"], L["f0"], L["sonorant"] > 0.5)
        if lo.any():
            B, Ba, bm = L["bandwidths"][lo, 0], L["bandwidths_acoustic"][lo, 0], L["coupling_bm"][lo, 0]
            rows["closed"].append(B)
            rows["cycle_mean"].append(Ba)
            rows["peak"].append(B * bm)
            rows["f0"].append(L["f0"][lo])
            rows["opening"].append(L["glottal_opening"][lo])
        if (j + 1) % 50 == 0:
            print(j + 1, flush=True)
    R = {k: np.concatenate(v) if v else np.zeros(0) for k, v in rows.items()}
    res = {"clips": int(n_clips), "mismatched": mism, "frames": int(len(R["closed"])),
           "closed": _summary(R["closed"]), "cycle_mean": _summary(R["cycle_mean"]), "peak": _summary(R["peak"]),
           "opening_median": float(np.nanmedian(R["opening"])) if len(R["opening"]) else float("nan")}
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d146.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = [f"D-146: F1 bandwidth in train_v2kb's coupled renders, D-141's cell (F1 350-650 Hz, F2/F1 < 2.2) at f0 <= 150 Hz",
             f"  {n_clips} clips regenerated (seed {SEED}); {mism} did not match the stored labels and were dropped; "
             f"{res['frames']} cell frames; median glottal opening <a> {res['opening_median']:.2f}"]
    for k, name in (("closed", "closed-glottis label"), ("cycle_mean", "cycle mean (acoustic, D-081)"), ("peak", "at peak opening")):
        s = res[k]
        if s["n"]:
            lines.append(f"  {name:30s} median {s['median']:.0f}, 90th {s['p90']:.0f}, 99th {s['p99']:.0f} Hz; "
                         f"above 250 Hz {s['gt250']:.2%}, above 360 Hz {s['gt360']:.2%}")
    lines.append("  for comparison (D-141 stage 0, the same cell at f0 <= 150): train_v2 0.00 % / 0.00 %, "
                 "train_bw_v1 1.68 % / 0.18 % above 250 / 360 Hz")
    s = res["cycle_mean"]
    lines.append("Reading (registered): " + ("WIDE-F1 COVERAGE: the coupled audio carries wide F1s at low f0" if s.get("gt360", 0) >= 0.01
                                             else "NO WIDE-F1 COVERAGE beyond what train_bw_v1 had" if s.get("gt360", 0) < 0.0036
                                             else "SOME: above train_bw_v1's, under the 1 % bar"))
    (REP / "d146.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
