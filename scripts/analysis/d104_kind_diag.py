"""D-104 follow-up (descriptive): which clip properties leave a render mismatch? On regenerated train_v2 validation
clips, per voiced sonorant frame (every 5th), the SNR-gated targets (10 dB) minus the render from the labelled poles
and bandwidths (no radiation) minus the LF source at the label Rd, offset removed: RMS over valid harmonics (dB), by
engine (cascade / tube) and extra sections (none / nasal / subglottal), and the same with the clip's own extra sections
added to the render (the synthesiser draws them once per clip). CPU.

    python scripts/analysis/d104_kind_diag.py [n_clips] > reports/analysis/d104/kind_diag.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import torch

    from d102_render_check import BANDS, load_cfg
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.train import reconstruction as R
    from vocal_analysis.train.data import split_ids

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    idx, cfg = load_cfg("data/synth/train_v2")
    rows = []
    for i in split_ids(idx["n_clips"])[1][:n]:
        c = generate(int(i), cfg, idx["seed"])
        lab, ex = c.labels, c.meta.get("extras") or {}
        engine = "tube" if c.meta.get("tube") else "cascade"
        pz = np.asarray(ex.get("extra_pole_f", []), float)
        extra = "none" if not len(pz) else ("nasal" if np.min(pz) < 1100 and c.meta.get("n_extra_zeros", 0) else "subglottal")
        db, ok = R.harmonic_targets(c.audio, lab["f0"], lab["voiced"].astype(bool), k_max=64, snr_min_db=10)
        fr = np.flatnonzero(lab["voiced"].astype(bool) & lab["sonorant"].astype(bool) & np.isfinite(lab["formants"][:, 0])
                            & (ok.sum(1) >= 4))[::5]
        for t in fr[fr < len(db)]:
            kk = np.flatnonzero(ok[t])
            f = torch.tensor(lab["f0"][t] * (kk + 1), dtype=torch.float64)
            P, B = torch.tensor(lab["formants"][t]), torch.tensor(lab["bandwidths"][t])
            src = R.lf_source_db(torch.tensor(float(lab["rd"][t]), dtype=torch.float64), 64).numpy()[kk]
            r0 = db[t, kk] - R.render_db(f, P, B, radiation=False).numpy() - src
            r0 = r0 - r0.mean()
            if len(pz):
                zf = torch.tensor(ex.get("extra_zero_f", []), dtype=torch.float64)
                zb = torch.tensor(ex.get("extra_zero_b", []), dtype=torch.float64)
                PP = torch.cat([P, torch.tensor(pz)]); BB = torch.cat([B, torch.tensor(ex["extra_pole_b"], dtype=torch.float64)])
                r1 = db[t, kk] - R.render_db(f, PP, BB, zf if len(zf) else None, zb if len(zb) else None, radiation=False).numpy() - src
                r1 = r1 - r1.mean()
            else:
                r1 = r0
            rows.append((engine, extra, lab["f0"][t], np.sqrt(np.mean(r0 ** 2)), np.sqrt(np.mean(r1 ** 2))))
    L = ["D-104 kind diagnostic: RMS residual (dB) of gated targets against the label render + LF source (offset removed); "
         "without | with the clip's own extra sections in the render; median over frames", ""]
    for eng in ("cascade", "tube"):
        for extra in ("none", "nasal", "subglottal"):
            s = [r for r in rows if r[0] == eng and r[1] == extra]
            if len(s) < 20:
                continue
            cells = []
            for lo, hi in BANDS:
                b = [r for r in s if lo <= r[2] < hi]
                if len(b) >= 20:
                    cells.append(f"f0 {lo:.0f}-{hi if hi < 1e8 else 'up'}: {np.median([r[3] for r in b]):.2f} | {np.median([r[4] for r in b]):.2f}")
            L.append(f"  {eng:8s} extras {extra:10s} n {len(s):5d}: " + "; ".join(cells))
    print("\n".join(L))
