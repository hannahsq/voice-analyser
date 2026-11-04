"""D-079 decision (registered 2026-11-04): do trunks trained on coupled audio keep the glottal opening? From
d079_probe.py's fits: R² of <a> on val_v2ko voiced frames (head-seed means). D = R²(v2c2) - R²(mix), paired by trunk
seed: kept if the seed mean D >= 0.10 with all three pairs positive; not kept if the mean D < 0.03 or the pairs split
in sign; else partly. Independently, carried without coupling if mix's seed mean R² >= 0.5. Reported: v2c, v3.1, per
clip R², ridge, and the Rd-only ceiling (0.75).
"""

import json
from pathlib import Path

import numpy as np

C, O = Path("cache/d079"), Path("reports/analysis/d079")
GROUPS = {"mix": [f"mix_s{s}" for s in (0, 1, 2)], "v2c2": [f"v2c2_s{s}" for s in (0, 1, 2)],
          "v2c": [f"v2c_s{s}" for s in (0, 1, 2)], "v3.1": ["v31_s1", "v31_s2"]}

if __name__ == "__main__":
    R = {n: json.loads((C / f"{n}_fit.json").read_text()) for g in GROUPS.values() for n in g}
    L = ["D-079: R² of the frame's mean glottal opening <a> from frozen trunk features (val_v2ko voiced frames)", "",
         "trunk      frame R² (head seeds 1 / 2)   clip R²   ridge"]
    for g, names in GROUPS.items():
        for n in names:
            r = R[n]
            L.append(f"  {n:9s} {r['mlp']:.3f} ({r['mlp_s1']:.3f} / {r['mlp_s2']:.3f})        {r['mlp_clip']:.3f}    {r['ridge']:.3f}")
        L.append(f"  {g} mean: {np.mean([R[n]['mlp'] for n in names]):.3f}")
    L.append("ceiling: <a> from the frame's Rd label alone, R² 0.748 (d079_opening_share.py ceiling)")
    d = np.array([R[f"v2c2_s{s}"]["mlp"] - R[f"mix_s{s}"]["mlp"] for s in (0, 1, 2)])
    mix = np.mean([R[n]["mlp"] for n in GROUPS["mix"]])
    L += ["", f"D = R²(v2c2) - R²(mix) by seed: {', '.join(f'{x:+.3f}' for x in d)}; mean {d.mean():+.3f}"]
    if d.mean() >= 0.10 and np.all(d > 0):
        verdict = "kept"
    elif d.mean() < 0.03 or not (np.all(d > 0) or np.all(d < 0)):
        verdict = "not kept"
    else:
        verdict = "partly"
    carried = "carried without coupling" if mix >= 0.5 else "not carried without coupling"
    L += [f"mix seed mean R² {mix:.3f}: {carried}", "", f"DECISION: {verdict}; {carried}"]
    O.mkdir(parents=True, exist_ok=True)
    (O / "decide_d079.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
