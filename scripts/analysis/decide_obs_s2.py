"""Pre-registered rule (D-027, 2026-10-17, before any run): the observability head, three S2 Conformer seeds
(obs_conformer_s0-2: mix_conformer_s0-2's runs with `obs_head=true`) against their same-seed baselines, dev tier.
  A. No degradation: decide_m3_s2.py's rule on the same pairs (its file says "DECISION: m3").
  B. The head learns its label: on the synthetic validation set (the k35 runs' last "eval"), means over seeds, every
     formant F1-F4 has Spearman(o_pred, o_label) >= 0.8 and MAE <= 0.1.
  C. It transfers to real speech: PTDB-TUG dev closed-phase frames, per seed, |model - closed phase| in cents against
     the head's o; means over seeds, for F1 and F2 each: Spearman(o, |err|) <= -0.1, and the median |err| in the lowest
     fifth of o at least 1.5 x that in the highest fifth. F3 and formant sigma's Spearman are reported, not judged
     (the reference's own F3 scatter; sigma includes the prior, o does not), as is women's F3 o at f0 >= 250 Hz
     against f0 < 200 Hz (D-026's identifiability finding predicts it lower).
DECISION: "gate" (A, B and C: the head ships and may gate M4's re-tracking), "output" (A and B: it ships as an output
calibrated on synthetic data; the gate needs another route), "fix" (A or B fails).
Args: scratch dir (closed-phase frames), the no-degradation file, runs dir of the small runs (default runs/small)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
S, NOHARM = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runs/small")
SEEDS = (0, 1, 2)
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]

a = re.search(r"^DECISION: (\S+)", NOHARM.read_text(encoding="utf-8"), re.M)
A = bool(a) and a.group(1) == "m3"
print(f"A. no degradation ({NOHARM}): {a.group(1) if a else 'missing'} -> {'holds' if A else 'fails'}")

print("\nB. synthetic validation (last eval of each k35 run), per formant: Spearman, MAE")
rho, mae = np.zeros((len(SEEDS), 4)), np.zeros((len(SEEDS), 4))
for i, k in enumerate(SEEDS):
    recs = [json.loads(x) for x in (RUNS / f"obs_conformer_s{k}_k35" / "log.jsonl").read_text().splitlines()]
    ev = [r["eval"] for r in recs if "eval" in r][-1]
    rho[i] = [ev[f"F{n}_obs_rho"] for n in range(1, 5)]
    mae[i] = [ev[f"F{n}_obs_mae"] for n in range(1, 5)]
    print(f"  s{k}: " + "  ".join(f"F{n + 1} {rho[i, n]:.3f} / {mae[i, n]:.3f}" for n in range(4)))
mr, mm = rho.mean(0), mae.mean(0)
print("  mean: " + "  ".join(f"F{n + 1} {mr[n]:.3f} / {mm[n]:.3f}" for n in range(4)))
B = bool(np.all(mr >= 0.8) and np.all(mm <= 0.1))
print(f"  -> {'holds' if B else 'fails'} (every formant Spearman >= 0.8 and MAE <= 0.1)")

print("\nC. PTDB-TUG dev closed phase: Spearman(o, |err|), median |err| cents lowest / highest fifth of o, ratio;"
      " Spearman(sigma, |err|)")
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
cr, ratio = np.zeros((len(SEEDS), 3)), np.zeros((len(SEEDS), 3))
hi_f0 = []
for i, k in enumerate(SEEDS):
    m = f"obs_conformer_s{k}"
    o, sg = d[f"{m}__obs"], d[f"{m}__sigma"]
    line = []
    for j in range(3):
        ok = dev & np.isfinite(d["cp"][:, j]) & np.isfinite(d[m][:, j]) & np.isfinite(o[:, j])
        err = np.abs(1200 * np.log2(d[m][ok, j] / d["cp"][ok, j]))
        oj = o[ok, j]
        q1, q4 = np.quantile(oj, [0.2, 0.8])
        lo, hi = np.median(err[oj <= q1]), np.median(err[oj >= q4])
        cr[i, j], ratio[i, j] = spearmanr(oj, err)[0], lo / hi
        rs = spearmanr(sg[ok, j], err)[0]
        line.append(f"F{j + 1} {cr[i, j]:+.3f}, {lo:.0f} / {hi:.0f} ({ratio[i, j]:.2f}); sigma {rs:+.3f}")
    print(f"  s{k}: " + "  ".join(line))
    w = dev & (d["sex"] == "f") & np.isfinite(o[:, 2]) & np.isfinite(d["f0"])
    hi_f0.append((np.median(o[w & (d["f0"] >= 250), 2]), np.median(o[w & (d["f0"] < 200), 2])))
mc, mq = cr.mean(0), ratio.mean(0)
print("  mean: " + "  ".join(f"F{j + 1} {mc[j]:+.3f}, ratio {mq[j]:.2f}" for j in range(3)))
h = np.mean(hi_f0, 0)
print(f"  women's F3 o, median: f0 >= 250 Hz {h[0]:.3f}, f0 < 200 Hz {h[1]:.3f} (reported)")
C = bool(np.all(mc[:2] <= -0.1) and np.all(mq[:2] >= 1.5))
print(f"  -> {'holds' if C else 'fails'} (F1 and F2: Spearman <= -0.1 and ratio >= 1.5)")

print(f"\nDECISION: {'gate' if A and B and C else 'output' if A and B else 'fix'}")
