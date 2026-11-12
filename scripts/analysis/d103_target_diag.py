"""D-103 follow-up (after its result; descriptive): are the high-harmonic targets underestimated? On regenerated
train_v2 validation clips (cascade without extras), per voiced sonorant frame: target - render(label poles, no
radiation) - LF source(label Rd), minus the frame's median over harmonics 1-4, by harmonic frequency band; and the
frame's f0 change across the fit window (cents). If the targets lose high harmonics to f0 drift, the residual falls with
frequency and more so where f0 moves.

    python scripts/analysis/d103_target_diag.py [n_clips] > reports/analysis/d103/target_diag.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import torch

    from d102_render_check import load_cfg
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.train import reconstruction as R
    from vocal_analysis.train.data import split_ids

    n = int(next((a for a in sys.argv[1:] if a.isdigit()), 80))
    idx, cfg = load_cfg("data/synth/train_v2")
    edges = [0, 500, 1000, 2000, 3000, 4000, 5000]
    res = []  # (f0, drift cents, band index, residual)
    for i in split_ids(idx["n_clips"])[1][:n]:
        c = generate(int(i), cfg, idx["seed"])
        if c.meta.get("tube") or (c.meta.get("n_extra_poles") or 0) > 0:
            continue
        lab = c.labels
        db, ok = R.harmonic_targets(c.audio, lab["f0"], lab["voiced"].astype(bool), k_max=64,
                                    snr_min_db=float(sys.argv[sys.argv.index("--snr") + 1]) if "--snr" in sys.argv else None)
        fr = np.flatnonzero(lab["voiced"].astype(bool) & lab["sonorant"].astype(bool) & np.isfinite(lab["formants"][:, 0])
                            & ok[:, :4].all(1))[::5]
        for t in fr:
            f0 = lab["f0"][t]
            half = max(1, int(round(max(0.025, 3 / f0) * 100 / 2)))
            a, b = max(0, t - half), min(len(lab["f0"]) - 1, t + half)
            drift = 1200 * abs(np.log2(lab["f0"][b] / lab["f0"][a])) if lab["f0"][a] > 0 and lab["f0"][b] > 0 else np.nan
            K = int(ok[t].sum())
            kk = np.flatnonzero(ok[t])
            f = torch.tensor(f0 * (kk + 1), dtype=torch.float64)
            rend = R.render_db(f, torch.tensor(lab["formants"][t]), torch.tensor(lab["bandwidths"][t]), radiation=False).numpy()
            src = R.lf_source_db(torch.tensor(float(lab["rd"][t]), dtype=torch.float64), 64).numpy()[kk]
            r = db[t, kk] - rend - src
            r = r - np.median(r[: min(4, len(r))])
            for fk, rk in zip(f0 * (kk + 1), r):
                res.append((f0, drift, int(np.searchsorted(edges, fk) - 1), rk, float(c.meta.get('hnr_db', np.nan))))
    a = np.array(res, float)
    L = [f"D-103 target diagnostic: cascade clips without extras; residual target - render - LF source (dB, re harmonics 1-4), median", ""]
    for flo, fhi, lab_ in ((0, 150, "f0 < 150"), (150, 250, "150-250"), (250, 400, "250-400")):
        for dlo, dhi, dl in ((0, 10, "drift < 10 c"), (10, 1e9, "drift >= 10 c")):
            s = a[(a[:, 0] >= flo) & (a[:, 0] < fhi) & (a[:, 1] >= dlo) & (a[:, 1] < dhi)]
            if len(s) < 50:
                continue
            L.append(f"  {lab_:8s} {dl:13s}: " + "  ".join(
                f"{edges[b]}-{edges[b + 1]} Hz {np.median(s[s[:, 2] == b, 3]):+5.1f}" for b in range(len(edges) - 1) if (s[:, 2] == b).sum() >= 20)
                + f"  (n {len(s)})")
    L.append("")
    L.append("by the clip's HNR (f0 < 250 Hz): residual by harmonic frequency band")
    for hlo, hhi in ((0, 15), (15, 25), (25, 35), (35, 99)):
        s = a[(a[:, 0] < 250) & (a[:, 4] >= hlo) & (a[:, 4] < hhi)]
        if len(s) >= 50:
            L.append(f"  HNR {hlo}-{hhi} dB: " + "  ".join(f"{edges[b]}-{edges[b + 1]} Hz {np.median(s[s[:, 2] == b, 3]):+5.1f}"
                                                        for b in range(len(edges) - 1) if (s[:, 2] == b).sum() >= 20) + f"  (n {len(s)})")
    print("\n".join(L))
