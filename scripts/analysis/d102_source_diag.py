"""D-102 follow-up (after its result; descriptive): are the targets and renderer right, and is the source shape the
bias? On regenerated train_v2 validation clips (cascade without extras), per voiced sonorant frame, harmonics 1-8:
(a) target - render(all labelled poles) - the generator's source level (`source_h_db`): flat across k if the targets
and renderer are right (SD over k, after removing a frame offset); (b) the same without the source term: the source's
shape the profile has to absorb (SD after removing offset and slope); (c) F1's grid minimum (as D-102) when the
first 8 harmonics carry the exact source level and only an offset is profiled.

    python scripts/analysis/d102_source_diag.py [n_clips] > reports/analysis/d102/source_diag.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import torch

    from d102_render_check import BANDS, GRID, load_cfg
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.train import reconstruction as R
    from vocal_analysis.train.data import split_ids

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    idx, cfg = load_cfg("data/synth/train_v2")
    rows = []
    for i in split_ids(idx["n_clips"])[1][:n]:
        c = generate(int(i), cfg, idx["seed"])
        if c.meta.get("tube") or (c.meta.get("n_extra_poles") or 0) > 0:
            continue
        lab = c.labels
        db, ok = R.harmonic_targets(c.audio, lab["f0"], lab["voiced"].astype(bool), k_max=64)
        fr = np.flatnonzero(lab["voiced"].astype(bool) & lab["sonorant"].astype(bool) & np.isfinite(lab["formants"][:, 0])
                            & ok[:, :8].all(1))[::5]
        for t in fr:
            f0 = lab["f0"][t]
            k = np.arange(1, 9)
            f = torch.tensor(f0 * k, dtype=torch.float64)
            P, B = torch.tensor(lab["formants"][t], dtype=torch.float64), torch.tensor(lab["bandwidths"][t], dtype=torch.float64)
            rend = R.render_db(f, P, B, radiation=False).numpy()  # the LF derivative carries radiation
            src = lab["source_h_db"][t, :8]
            a = db[t, :8] - rend - src
            b = db[t, :8] - rend
            lf = np.log2(f0 * k)
            sb = b - np.polyval(np.polyfit(lf, b, 1), lf)
            # (c) grid with exact source on harmonics 1-8, offset profiled
            C = len(GRID)
            poles = np.repeat(lab["formants"][t][None], C, 0)
            poles[:, 0] = lab["formants"][t, 0] * 2 ** (GRID / 1200)
            tgt = torch.tensor(np.repeat((db[t, :8] - src)[None, None], C, 0), dtype=torch.float64)
            loss, per = R.reconstruction_loss(tgt, torch.ones(C, 1, 8, dtype=torch.bool), torch.full((C, 1), f0, dtype=torch.float64),
                                              torch.tensor(poles[:, None]), torch.tensor(np.repeat(lab["bandwidths"][t][None, None], C, 0)),
                                              profile="offset", radiation=False)
            j = int(np.nanargmin(per.numpy()[:, 0]))
            rows.append((f0, np.std(a - a.mean()), np.std(sb), GRID[j], j in (0, C - 1), lab["formants"][t, 0] / f0))
    r = np.array(rows, float)
    L = [f"D-102 source diagnostic: cascade clips without extras, {len(r)} frames, harmonics 1-8", ""]
    for lo, hi in BANDS:
        s = r[(r[:, 0] >= lo) & (r[:, 0] < hi)]
        if len(s) >= 20:
            L.append(f"  f0 {lo:.0f}-{hi if hi < 1e8 else 'up'}: n {len(s)}; (a) target - render - source, SD over k (dB) median {np.median(s[:, 1]):.2f}; "
                     f"(b) source shape left after offset + slope, SD {np.median(s[:, 2]):.2f} dB; (c) F1 minimum with the exact source "
                     f"{np.median(s[:, 3]):+.0f} c (|.| {np.median(np.abs(s[:, 3])):.0f}), edge {100 * np.mean(s[:, 4]):.1f} %; median F1/f0 {np.median(s[:, 5]):.1f}")
    print("\n".join(L))
