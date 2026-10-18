"""Training recipe screen (pre-registered 2026-10-18, before any arm ran; roadmap 3). One arm's S2 seeds against the
same-seed baselines (mixr_conformer_sN: train_v2, the mixed-crop S2 Conformer, today's code), dev tier.
Decided on the synthetic validation set (each k35 run's last "eval"; fixed crops and channels), where the six
baseline seeds' SDs are F1 0.50, F2 1.08, F3 1.91, F4 3.62 Hz and f0 RPA 0.07 points (mix_conformer_s0-5): on dev
Hillenbrand they are 3.4 / 7.9 / 17.6 Hz, too noisy to decide at S2. With n seeds, the mean difference's standard
error is taken as SE = sqrt(2 / n) x SD (unpaired, so conservative).
  better / worse on a measure: mean difference beyond 2 SE in that direction; grey: between 1 and 2 SE.
  guard-rails (real, dev, means over the pairs, paired files and closed phase; catch only large regressions):
    Hillenbrand F1 <= +4, F2 <= +9, F3 <= +20 Hz; PTDB-TUG RPA >= -1.0 point; women's closed-phase |error| <= +5 Hz.
DECISION: "worse" (any measure worse, or a guard-rail fails); else "grey" (any measure in the grey zone toward worse,
or none better and one in the grey zone toward better: run seeds 3-5 and decide again with n = 6); else "better" (at
least one of F1-F4 better, RPA not worse) or "same" (nothing beyond 1 SE). A cost arm (shorter decay, larger batch)
is adopted on "same" or "better"; a quality arm on "better" only.
Args: scratch dir (closed-phase frames), arm prefix, number of seeds (3 or 6), optional baseline prefix."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, ARM, N = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
BASE = sys.argv[4] if len(sys.argv) > 4 else "mixr_conformer"
P = Path("reports/analysis/paired/fp32")
SD = {"F1": 0.50, "F2": 1.08, "F3": 1.91, "F4": 3.62, "RPA": 0.07}
SE = {k: np.sqrt(2 / N) * v for k, v in SD.items()}
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]


def last_eval(run):
    recs = [json.loads(x) for x in (Path("runs/small") / run / "log.jsonl").read_text().splitlines()]
    ev = [r["eval"] for r in recs if "eval" in r][-1]
    return {**{f"F{n}": ev[f"F{n}_mae_hz"] for n in range(1, 5)}, "RPA": 100 * ev["f0_rpa"]}


diffs = {k: [] for k in SD}
print(f"{ARM} against {BASE}, {N} seeds; synthetic validation, arm minus baseline")
for k in range(N):
    a, b = last_eval(f"{ARM}_s{k}_k35"), last_eval(f"{BASE}_s{k}_k35")
    for m in SD:
        diffs[m].append(a[m] - b[m])
    print(f"  s{k}: " + "  ".join(f"{m} {a[m] - b[m]:+.2f}" for m in SD))
verdict = {}
for m in SD:
    d = float(np.mean(diffs[m]))
    sign = -1 if m == "RPA" else 1  # positive = worse
    z = sign * d / SE[m]
    if not np.isfinite(z):  # a diverged run (NaN or inf validation error) is worse, never "same"
        z = np.inf
    verdict[m] = "worse" if z >= 2 else "grey-worse" if z >= 1 else "better" if z <= -2 else "grey-better" if z <= -1 else "same"
    print(f"  mean {m}: {d:+.2f} (SE {SE[m]:.2f}, {z:+.1f} SE) -> {verdict[m]}")

print("Guard-rails (dev, means over pairs)")
LIM = {"F1 MAE all": 4.0, "F2 MAE all": 9.0, "F3 MAE all": 20.0}
guard_ok = True
g = {k: [] for k in (*LIM, "RPA all frames", "cp women |err|")}
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
for k in range(N):
    a, b = f"{ARM}_s{k}", f"{BASE}_s{k}"
    txt = (P / f"{b}_vs_{a}.txt").read_text(encoding="utf-8")
    for key in LIM:
        g[key].append(float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", txt).group(1)))
    g["RPA all frames"].append(float(re.search(r"all frames\s*:.*?diff\s+([+-][\d.]+)", txt).group(1)))
    m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[a][:, 0]) & np.isfinite(d[b][:, 0])
    g["cp women |err|"].append(float(np.median(np.abs(d[a][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[b][m, 0] - d["cp"][m, 0]))))
for key, v in g.items():
    mv = float(np.mean(v))
    ok = mv >= -1.0 if key == "RPA all frames" else mv <= (5.0 if key == "cp women |err|" else LIM[key])
    guard_ok &= ok
    print(f"  {key:16s} per seed {', '.join(f'{x:+.1f}' for x in v)}  mean {mv:+.1f}  {'ok' if ok else 'FAILS'}")

vs = list(verdict.values())
if "worse" in vs or not guard_ok:
    dec = "worse"
elif "grey-worse" in vs or ("better" not in vs and "grey-better" in vs):
    dec = "grey"
elif "better" in [verdict[f"F{n}"] for n in range(1, 5)]:
    dec = "better"
else:
    dec = "same"
print(f"DECISION: {dec}")
