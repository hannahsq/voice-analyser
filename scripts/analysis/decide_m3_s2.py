"""Pre-registered rule (roadmap, 2026-10-11 night): do three S2 Conformer seeds trained with resynthesised VocalSet
(m3_conformer_s0-2) show "no degradation" against mix_conformer_s0-2 on dev speakers? Means over the three same-seed
pairs, from the paired files (reports/analysis/paired/fp32/mix_conformer_sN_vs_m3_conformer_sN.txt) and the
closed-phase frames. No degradation when all hold:
  Hillenbrand F1 MAE all <= +3 Hz, F2 MAE all <= +5 Hz, F3 MAE all <= +10 Hz;
  PTDB-TUG f0 RPA, all frames, >= -1.0 point;
  PTDB-TUG closed-phase F1, women's median |err| <= +5 Hz.
Prints the table and, last, "DECISION: m3" (no degradation) or "DECISION: m2". Args: scratch dir, paired dir."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, P = Path(sys.argv[1]), Path(sys.argv[2])
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
LIMITS = {"F1 MAE all": 3.0, "F2 MAE all": 5.0, "F3 MAE all": 10.0}
diffs = {k: [] for k in (*LIMITS, "RPA all frames", "cp women |err|")}
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
for k in range(3):
    a, b = f"mix_conformer_s{k}", f"m3_conformer_s{k}"
    txt = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    for key in LIMITS:
        diffs[key].append(float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", txt).group(1)))
    diffs["RPA all frames"].append(float(re.search(r"all frames\s*:.*?diff\s+([+-][\d.]+)", txt).group(1)))
    m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[a][:, 0]) & np.isfinite(d[b][:, 0])
    diffs["cp women |err|"].append(float(np.median(np.abs(d[b][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[a][m, 0] - d["cp"][m, 0]))))
ok = True
for key, v in diffs.items():
    mean = float(np.mean(v))
    good = mean >= -1.0 if key.startswith("RPA") else mean <= (LIMITS.get(key, 5.0))
    ok &= good
    print(f"{key:16s} per seed {', '.join(f'{x:+.1f}' for x in v)}  mean {mean:+.1f}  {'ok' if good else 'DEGRADED'}")
print(f"DECISION: {'m3' if ok else 'm2'}")
