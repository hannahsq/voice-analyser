"""D-034, reported only (pre-registered 2026-10-24): teacher v3's F1 tracking on the F1 sweep, breathy arms against modal.
Per arm and f0: tracking slope (least squares of log2 read F1 on log2 true F1 over the sweep's cells) and mean |F1 error|
(cents). Reading per breathy arm: slope higher than modal's by >= 0.2 at >= 3 of 4 f0 -> "tracks better", lower by
>= 0.2 at >= 3 -> "tracks worse", otherwise "no clear change". Students' and Parakeet's means reported alongside.
Args: out prefix, then ARM=CELLS_CSV (modal first). Writes <prefix>.txt and <prefix>_reads.csv."""
import csv
import sys
from collections import defaultdict

import numpy as np

PRE, ARMS = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
data = {}
for arm, path in ARMS:
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[r["model"]][(float(r["f0_hz"]), float(r["f1_over_f0"]))] = float(r["value"])
    data[arm] = d
out = open(f"{PRE}.txt", "w", encoding="utf-8")


def say(s=""):
    print(s)
    out.write(s + "\n")


say("D-034 F1 sweep, breathy arms against modal: teacher v3's F1 tracking (reported only; pre-registered 2026-10-24)")
stats = {}
rows = []
for arm, _ in ARMS:
    t = data[arm]["teacher_cents"]
    for f0 in sorted({f for f, _ in t}):
        cells = sorted((r, e) for (f, r), e in t.items() if f == f0)
        true = np.log2([f0 * r for r, _ in cells])
        read = true + np.array([e for _, e in cells]) / 1200
        slope = np.polyfit(true, read, 1)[0]
        stats[arm, f0] = (slope, float(np.mean([abs(e) for _, e in cells])))
        for (r, e), rd in zip(cells, read):
            rows.append([arm, f0, r, round(f0 * r, 1), round(2 ** rd, 1), round(2 ** rd / f0, 3), round(e, 1)])
f0s = sorted({f for _, f in stats})
say("\ntracking slope (1 follows F1, 0 flat) / mean |F1 error| in cents")
say("  arm            " + "  ".join(f"f0 {f:.0f} Hz      " for f in f0s))
for arm, _ in ARMS:
    say(f"  {arm:14s} " + "  ".join(f"{stats[arm, f][0]:+.2f} / {stats[arm, f][1]:5.0f}  " for f in f0s))
modal = ARMS[0][0]
for arm, _ in ARMS[1:]:
    d = [stats[arm, f][0] - stats[modal, f][0] for f in f0s]
    up, down = sum(x >= 0.2 for x in d), sum(x <= -0.2 for x in d)
    reading = "tracks better" if up >= 3 else "tracks worse" if down >= 3 else "no clear change"
    say(f"\n{arm} against {modal}: slope change " + ", ".join(f"{x:+.2f}" for x in d) + f" -> READING: {reading}")
say("\nmeans over the sweep (students' normalised error against Parakeet; Parakeet's own variation)")
for arm, _ in ARMS:
    say(f"  {arm:14s} " + "  ".join(f"{m} {np.mean(list(data[arm][m].values())):.3f}" for m in ("v2", "v4", "parakeet_variation") if m in data[arm]))
out.close()
with open(f"{PRE}_reads.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "f0_hz", "f1_over_f0", "f1_hz", "read_f1_hz", "read_over_f0", "error_cents"])
    w.writerows(rows)
