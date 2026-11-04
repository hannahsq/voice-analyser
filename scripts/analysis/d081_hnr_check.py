"""D-081 note (results seen): is v3.1's h above the label ceiling because of the closed-tract aspiration? The B1 h and
h+<a> heads' glottal slope by the clip's HNR tercile (lower HNR = more aspiration), and the closed label's own slope on
B1's frame set (what full recovery reads as there).

    python scripts/analysis/d081_hnr_check.py   (VA_DEVICE=cuda)
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import torch

    from d081_label_oracle import Slope
    from d081_oracle_heads import LOG, aligned, fit

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    L = []
    for trunk in ("v31_s1", "v31_s2"):
        (ht, lt), (hv, lv) = aligned(trunk, "train"), aligned(trunk, "val")
        ok = lambda z: z["sonorant"].astype(bool) & np.isfinite(z["glottal_opening"]) & np.isfinite(z["formants"][:, 0]) & np.isfinite(z["hnr_db"])  # noqa: E731
        mt, mv = ok(lt), ok(lv)
        v = {k: lv[k][mv] for k in ("formants", "formants_acoustic", "clip")}
        S = Slope(v)
        hnr = lv["hnr_db"][mv]
        q = np.percentile(hnr, [33.3, 66.7])
        bins = np.digitize(hnr, q)

        def by(p):
            out = [S(p)]
            for b in range(3):
                keep = S.use.copy()
                S.use = keep & (bins == b)
                rest_all = S.rest
                S.rest = rest_all[(bins == b)[keep]]
                out.append(S(p))
                S.use, S.rest = keep, rest_all
            return out

        L.append(f"{trunk}: HNR terciles cut at {q[0]:.1f} / {q[1]:.1f} dB; slope all / low HNR / mid / high")
        L.append("  closed label  " + " / ".join(f"{x:+.3f}" for x in by(v["formants"][:, 0])))
        for name, keys in (("h", []), ("h+<a>", ["glottal_opening"])):
            xt = np.column_stack([ht[mt].astype(np.float32)] + [lt[k][mt] for k in keys])
            xv = np.column_stack([hv[mv].astype(np.float32)] + [lv[k][mv] for k in keys])
            p = np.exp(fit(xt, LOG(lt["formants"][mt, :1]), xv, 1, dev)[:, 0])
            L.append(f"  {name:12s} " + " / ".join(f"{x:+.3f}" for x in by(p)))
    Path("reports/analysis/d081/hnr_check.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
