"""D-130, descriptive extra (added after the registered tables and maps were seen): F1 error inside Story's original
(q1, q2) box against the 10 % extension, and the share of frames where F1 snapped to a harmonic (tracked within 50
cents of 1-3 x f0 while the truth is more than 100 cents from each), clean, per f0; frames whose true F1 is below f0
left out. Args: label=run name ... (reports/probes/vowelplane/<run>_raw.npz)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE, RAW = ROOT / "data/probes/vowelplane_v1", ROOT / "reports/probes/vowelplane"

if __name__ == "__main__":
    from vocal_analysis.synth import tube

    names = dict(a.split("=", 1) for a in sys.argv[1:])
    meta = json.loads((CACHE / "meta.json").read_text())
    lo = np.array([tube.Q1_RANGE[0], tube.Q2_RANGE[0]])
    hi = np.array([tube.Q1_RANGE[1], tube.Q2_RANGE[1]])
    raws = {k: np.load(RAW / f"{v}_raw.npz") for k, v in names.items()}
    print("D-130 extra: F1 signed error median (|error| median), cents, inside Story's box and in the extension; F1 snapped to a harmonic")
    for f0 in (100.0, 200.0, 300.0, 400.0):
        print(f"f0 {f0:g} Hz")
        for k, r in raws.items():
            ins, out, snap = [], [], []
            for it in meta["items"]:
                if it["kind"] != "sweep" or it["f0"] != f0:
                    continue
                d = np.load(CACHE / f"{it['name']}.npz")
                T, q = d["formants"], d["q"]
                inside = np.all((q >= lo) & (q <= hi), 1)
                M, P = r[f"{it['name']}|clean|F"], r[f"{it['name']}|clean|son"]
                ok = np.isfinite(M[:, 0]) & (P > 0.5) & (T[:, 0] >= f0)
                e = 1200 * np.log2(M[:, 0] / T[:, 0])
                ins.append(e[ok & inside])
                out.append(e[ok & ~inside])
                near = lambda x: np.min(np.abs(1200 * np.log2(x[:, None] / (f0 * np.arange(1, 4)[None]))), 1)  # noqa: E731
                snap.append(((near(M[:, 0]) < 50) & (near(T[:, 0]) > 100))[ok])
            i, o, h = np.concatenate(ins), np.concatenate(out), np.concatenate(snap)
            print(f"  {k:8s} inside {np.median(i):+4.0f} ({np.median(np.abs(i)):3.0f}); extension {np.median(o):+4.0f} "
                  f"({np.median(np.abs(o)):3.0f}), {len(o) / (len(o) + len(i)):.0%} of frames; snapped to a harmonic {100 * h.mean():4.1f} %")
