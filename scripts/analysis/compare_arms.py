"""Is arm A not worse than arm B on synthetic validation (each k35 run's last eval, seeds 0..n-1)? Per measure (F1-F4
MAE, f0 RPA) the mean difference A - B against 2 SE (SE = sqrt(2 / n) x the baseline seed SDs, as decide_recipe.py).
Prints the differences and "NOTWORSE: yes|no". Args: arm A, arm B, n (default 3)."""
import json
import sys

import numpy as np

A, B = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
SD = {"F1": 0.50, "F2": 1.08, "F3": 1.91, "F4": 3.62, "RPA": 0.07}


def ev(run):
    recs = [json.loads(x) for x in open(f"runs/small/{run}/log.jsonl")]
    e = [r["eval"] for r in recs if "eval" in r][-1]
    return {**{f"F{n}": e[f"F{n}_mae_hz"] for n in range(1, 5)}, "RPA": 100 * e["f0_rpa"]}


ok = True
for m, sd in SD.items():
    d = float(np.mean([ev(f"{A}_s{k}_k35")[m] - ev(f"{B}_s{k}_k35")[m] for k in range(N)]))
    z = (-d if m == "RPA" else d) / (np.sqrt(2 / N) * sd)
    ok &= z < 2
    print(f"{A} - {B} {m}: {d:+.2f} ({z:+.1f} SE, positive = {A} worse)")
print(f"NOTWORSE: {'yes' if ok else 'no'}")
