"""D-154 post hoc (not registered): does the resynthesis artefact on OK files follow the tract's bandwidths? Per file,
the tract's median B1 and B2 (Praat, as D-154 built them) on the fixed frames, against R's and S's failure share.

    python scripts/analysis/d154_posthoc.py > reports/analysis/d154/d154_posthoc.txt
"""
import importlib.util
import json
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    import soundfile as sf
    from scipy.signal import resample_poly
    from scipy.stats import spearmanr

    spec = importlib.util.spec_from_file_location("d154", ROOT / "scripts/analysis/d154_resynth_decomp.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    res = {r["path"]: r for p in sorted(d.CACHE.glob("*.json")) for r in json.loads(p.read_text(encoding="utf-8"))}
    rows = []
    for singer in ("male10", "male3", "male5"):
        for f in d._files(singer):
            y, sr = sf.read(ROOT / "data/vocalset" / f["meta"]["path"], dtype="float64", always_2d=True)
            g = gcd(sr, d.FS)
            F, B = d._tract(resample_poly(y.mean(1), d.FS // g, sr // g), len(f["v"]), f["v"])
            r = res[f["meta"]["path"]]
            sm = lambda c: np.mean([r[f"{c}|{m}"] for m in d.MODELS])  # noqa: E731
            i = f["idx"]
            rows.append((sm("O"), sm("R"), sm("S"), *np.median(B[i, :3], 0), *np.median(F[i, :2], 0)))
    a = np.array(rows)
    for name, sel in (("ok", a[:, 0] < 0.2), ("failing", a[:, 0] >= 0.5)):
        x = a[sel]
        print(f"{name} files ({len(x)}): tract median B1 {np.median(x[:, 3]):.0f}, B2 {np.median(x[:, 4]):.0f}, B3 {np.median(x[:, 5]):.0f} Hz; "
              f"F1 {np.median(x[:, 6]):.0f}, F2 {np.median(x[:, 7]):.0f}")
        for c, j in (("R", 1), ("S", 2)):
            print(f"  {c}: Spearman with B1 {spearmanr(x[:, j], x[:, 3])[0]:+.2f}, B2 {spearmanr(x[:, j], x[:, 4])[0]:+.2f}, "
                  f"F2 {spearmanr(x[:, j], x[:, 7])[0]:+.2f}, F2-F1 {spearmanr(x[:, j], x[:, 7] - x[:, 6])[0]:+.2f}")
        ok = x[:, 4] < 150
        print(f"  files with B2 < 150 Hz: {ok.sum()}, mean R {x[ok, 1].mean() if ok.any() else np.nan:.2f}, S {x[ok, 2].mean() if ok.any() else np.nan:.2f}; "
              f"B2 >= 150: {(~ok).sum()}, R {x[~ok, 1].mean() if (~ok).any() else np.nan:.2f}, S {x[~ok, 2].mean() if (~ok).any() else np.nan:.2f}")
