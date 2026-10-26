"""Roadmap 3 (D-028 "windows", default from 2026-10-27): the reference models' dev numbers re-measured windowed, next
to the whole-file ones, so later comparisons use windowed references. Closed phase (PTDB-TUG dev, the _sub frames):
F1-F3 median signed error and median |error| by sex, whole (scratch S) and windowed (scratch SW, columns added with
add_model.py's windows); labeller v3 is the per-frame mean in log frequency of conformer_v2r2_s1 and _s2_d16k.
Hillenbrand and PTDB-TUG RPA come from the paired whole-vs-windowed reports (m2_paired_sub.py), quoted here.
Args: S, SW, paired report dir, model names (columns)."""
import re
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.eval import tiers as T

sys.stdout.reconfigure(encoding="utf-8")
S, SW, PD = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NAMES = sys.argv[4:]
d, w = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True)), dict(np.load(SW / "ptdb_egg_sub.npz", allow_pickle=True))
assert np.array_equal(d["t"], w["t"]) and np.array_equal(d["speaker"], w["speaker"])
tm = T.load()
dev = np.array([T.tier_of(f"ptdb_tug:{s}", tm) == "dev" for s in d["speaker"]])
cp = d["cp"][:, :3]
for z in (d, w):
    z["labeller_v3"] = np.exp2((np.log2(z["conformer_v2r2_s1_d16k"]) + np.log2(z["conformer_v2r2_s2_d16k"])) / 2)
print(f"Closed phase, PTDB-TUG dev ({len(set(d['speaker'][dev]))} speakers, {dev.sum()} frames): median signed error / "
      f"median |error| (Hz), F1 F2 F3; whole -> windowed")
for name in NAMES + ["labeller_v3"]:
    for sx in ("f", "m"):
        m = dev & (d["sex"] == sx)
        row = []
        for z in (d, w):
            e = z[name][m] - cp[m]
            row.append([(np.nanmedian(e[:, k]), np.nanmedian(np.abs(e[:, k]))) for k in range(3)])
        cells = "  ".join(f"F{k + 1} {row[0][k][0]:+6.1f}/{row[0][k][1]:5.1f} -> {row[1][k][0]:+6.1f}/{row[1][k][1]:5.1f}" for k in range(3))
        print(f"  {name:24s} {'women' if sx == 'f' else 'men  '}  {cells}")
print("\nHillenbrand (dev) and PTDB-TUG RPA, whole vs windowed (paired reports)")
for name in NAMES:
    f = PD / f"{name}_whole_vs_windowed.txt"
    if not f.exists():
        continue
    t = f.read_text(encoding="utf-8")
    for key in ("F1 MAE all", "F2 MAE all", "F3 MAE all", "all frames"):
        mm = re.search(rf"{key}\s*:\s*([\d.]+)\s+([\d.]+)\s+diff\s+([+-][\d.]+)\s+(\[[^\]]+\])", t)
        if mm:
            print(f"  {name:24s} {key:11s} {mm.group(1):>6s} -> {mm.group(2):>6s}  diff {mm.group(3)} {mm.group(4)}")
