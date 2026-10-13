"""Bandwidth hypothesis for v3b (fixed 132 Hz Hawks & Miller in training): do models trained without f0-scaled
bandwidths read wide B1 at high f0 as higher F1 than d06k? v2s1 gives the seed spread. Args: name=ckpt ...
Run from the m3 worktree."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import filters

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(4)
dev = torch.device(__import__("os").environ.get("VA_DEVICE", "cpu"))  # cuda: same readings, much faster
models = {n: load_teacher(c, dev)[0] for n, c in (a.split("=", 1) for a in sys.argv[1:])}
GROUPS = [("men f0 120", "men", 120.0), ("women f0 210", "women", 210.0), ("women f0 300", "women", 300.0)]
F = [("baseline", {}), ("static B1 x2", {"b1_mult": 2.0}), ("static B1 x3", {"b1_mult": 3.0}),
     ("open phase: B1 x3", {"op_b1": 3.0}), ("open phase: F1 +5 %, B1 x3", {"op_f1": 0.05, "op_b1": 3.0})]
f = np.array([300.0, 500.0, 800.0])
print("train_v2's f0-scaled B1 against 132 Hz (Hawks & Miller), F1 300 / 500 / 800 Hz:")
for f0 in (120, 210, 300):
    print(f"  f0 {f0}: " + " / ".join(f"x{r:.2f}" for r in filters.hawks_miller_bandwidth(f, f0) / filters.hawks_miller_bandwidth(f, 132.0)))
print("\nF1 error vs closed phase (Hz), median over 8 PB vowels; change from baseline in brackets")
for gname, sex, f0 in GROUPS:
    sv.F0[sex] = f0
    print(f"\n{gname}\n{'factor':30s}" + "".join(f"{n:>14s}" for n in models))
    base = {}
    for fname, ch in F:
        ys = [sv.render(fv, sex, sv.BASE | ch, np.random.default_rng([v, 7])) for v, fv in enumerate(sv.PB[sex])]
        cells = []
        for n, m in models.items():
            e = float(np.median([np.nanmedian(analyse(m, y, dev)["formants"][30:71, 0]) - fv[0] for y, fv in zip(ys, sv.PB[sex])]))
            base.setdefault(n, e)
            cells.append(f"{e:+5.0f} ({e - base[n]:+4.0f})")
        print(f"{fname:30s}" + "".join(f"{c:>14s}" for c in cells), flush=True)
