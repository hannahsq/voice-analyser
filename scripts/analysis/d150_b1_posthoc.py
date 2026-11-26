"""D-150, post hoc (after the result): Praat Burg B1 (ceiling 5000 Hz) on the primary block's Hillenbrand dev tokens."""
import json
from pathlib import Path
import numpy as np
ROOT = Path("c:/github/voice-analyser")
if __name__ == "__main__":
    import sys; sys.path.insert(0, str(ROOT / "scripts/analysis"))
    import parselmouth
    import d150_dev_dropped as d
    T = {t["token"]: t for t in json.loads((ROOT / "reports/analysis/d150/d150.json").read_text(encoding="utf-8"))}
    out = []
    for row, it, y in d._tokens():
        t = T[Path(row.path).stem.lower()]
        F = t["ref_ss"]
        if not (350 <= F[0] <= 650 and F[1] / F[0] < 2.2 and t["f0"] <= 150):
            continue
        fm = parselmouth.Sound(y.astype(np.float64), 16000).to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000.0)
        b = [fm.get_bandwidth_at_time(1, tt) for tt in it.times if np.isfinite(tt)]
        out.append((F[0], F[1], float(np.nanmedian(b))))
    o = np.array(out)
    print("primary tokens", len(o), "Praat B1 median %.0f, 90th %.0f, >250 %d, >360 %d" % (np.nanmedian(o[:, 2]), np.nanpercentile(o[:, 2], 90), (o[:, 2] > 250).sum(), (o[:, 2] > 360).sum()))
    c = o[o[:, 1] < 900]
    print("F2<900:", len(c), "B1 median %.0f, >250 %d" % (np.nanmedian(c[:, 2]), (c[:, 2] > 250).sum()))
