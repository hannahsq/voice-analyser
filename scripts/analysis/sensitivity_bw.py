"""Teacher vs Praat on bandwidth factors (sensitivity.py renders). Args: name=ckpt. Run from the m3 worktree."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv
from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(8)
dev = torch.device("cpu")
name, ck = sys.argv[1].split("=", 1)
model = load_teacher(ck, dev)[0]
times = np.arange(0.3, 0.7001, 0.01)
F = [("baseline", {}), ("static B1 x2", {"b1_mult": 2.0}), ("static B1 x3", {"b1_mult": 3.0}),
     ("open phase: B1 x2", {"op_b1": 2.0}), ("open phase: B1 x3", {"op_b1": 3.0}), ("open phase: B1 x4", {"op_b1": 4.0}),
     ("open phase: F1 +5 %", {"op_f1": 0.05}), ("open phase: F1 +5 %, B1 x3", {"op_f1": 0.05, "op_b1": 3.0})]
print(f"F1 error vs closed phase (Hz), median over vowels; change from baseline in brackets\n{'factor':30s}"
      f"{name + ' men':>16s}{'Praat men':>16s}{name + ' women':>16s}{'Praat women':>16s}")
base = {}
for fname, ch in F:
    cells = []
    for sex in ("men", "women"):
        for tr in (name, "praat"):
            errs = []
            for v, f in enumerate(sv.PB[sex]):
                y = sv.render(f, sex, sv.BASE | ch, np.random.default_rng([v, 7]))
                est = analyse(model, y, dev)["formants"][30:71, 0] if tr == name else praat_default(y, sv.FS, times, {})[:, 0]
                errs.append(np.nanmedian(est) - f[0])
            e = float(np.median(errs))
            base.setdefault((sex, tr), e)
            cells.append(f"{e:+6.0f} ({e - base[(sex, tr)]:+4.0f})")
    print(f"{fname:30s}" + "".join(f"{c:>16s}" for c in cells), flush=True)
