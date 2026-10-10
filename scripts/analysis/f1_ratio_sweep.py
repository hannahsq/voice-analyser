"""F1 against F1/f0 for teachers with and without p_f1_band in training (train_v2: 30 % of cascade clips at f0 150-350
set F1 = f0 x 1.5-4.5). Same renders as sensitivity.py (LF Rd 1.0, Hawks & Miller at 132 Hz, women's upper formants).
Args: name=ckpt ... Run from the m3 worktree."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv
from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)
dev = torch.device("cpu")
models = {n: load_teacher(c, dev)[0] for n, c in (a.split("=", 1) for a in sys.argv[1:])}
F1 = np.arange(300.0, 1001.0, 25.0)
F2 = (1300.0, 2200.0)
BANDS = [(0, 1.75), (1.75, 2.25), (2.25, 2.75), (2.75, 3.25), (3.25, 3.75), (3.75, 4.5), (4.5, 99)]
print("F1 error (Hz) by F1/f0 band: median signed [median |err|] (n renders); F1 300-1000 step 25, F2 1300 and 2200, F3 2800")
for f0 in (120.0, 210.0, 260.0):
    sv.F0["women"] = f0
    err = {n: [] for n in models}
    ratio = []
    for i, f1 in enumerate(F1):
        for j, f2 in enumerate(F2):
            y = sv.render((f1, f2, 2800.0), "women", sv.BASE, np.random.default_rng([i, j, 11]))
            ratio.append(f1 / f0)
            for n, m in models.items():
                err[n].append(np.nanmedian(analyse(m, y, dev)["formants"][30:71, 0]) - f1)
    ratio = np.array(ratio)
    print(f"\nf0 {f0:.0f} Hz" + ("  (inside p_f1_band's 150-350 Hz)" if 150 <= f0 <= 350 else "  (outside p_f1_band)"))
    print(f"  {'F1/f0':>11s}" + "".join(f"{n:>17s}" for n in models))
    for lo, hi in BANDS:
        m_ = (ratio >= lo) & (ratio < hi)
        if not m_.any():
            continue
        cells = []
        for n in models:
            e = np.array(err[n])[m_]
            cells.append(f"{np.median(e):+5.0f} [{np.median(np.abs(e)):3.0f}] ({m_.sum():2d})")
        print(f"  {lo:4.2f}-{min(hi, 9):4.2f}" + "".join(f"{c:>17s}" for c in cells), flush=True)
    print(f"  {'all':>11s}" + "".join(f"{np.median(err[n]):+5.0f} [{np.median(np.abs(err[n])):3.0f}]     " for n in models))
