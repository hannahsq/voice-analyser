"""sensitivity_f123's renders read by Praat (Burg, defaults) instead of the teacher. Run from the m3 worktree."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv
from sensitivity_f123 import FACTORS, render
from vocal_analysis.baselines.formants import praat_default

sys.stdout.reconfigure(encoding="utf-8")
times = np.arange(0.3, 0.7001, 0.01)
print("Praat Burg defaults: median over vowels of Fk - closed-phase Fk (Hz), change from baseline in brackets")
print(f"{'factor':34s} " + "  ".join(f"{s:>5s} F{k}     " for s in ("men", "women") for k in (1, 2, 3)))
base = {}
for name, change in FACTORS:
    cells = []
    for sex in ("men", "women"):
        errs = [np.nanmedian(praat_default(render(f, sex, change, np.random.default_rng([v, 7])), sv.FS, times, {})[:, :3], axis=0)
                - np.array(f) for v, f in enumerate(sv.PB[sex])]
        e = np.median(errs, axis=0)
        for k in range(3):
            if name == "baseline":
                base[(sex, k)] = e[k]
            cells.append(f"{e[k]:+5.0f} ({e[k] - base[(sex, k)]:+4.0f})")
    print(f"{name:34s} " + "  ".join(cells), flush=True)
