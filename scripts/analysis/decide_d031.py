"""D-031 decision (pre-registered 2026-10-22): the D-030 recipe at full size. Branches d031_<arm>_b<05|10|20>k against
teacher v3 (conformer_v2r2_s1_d16k, 20k). Synthetic validation from each run's last "eval" (F1-F4 MAE Hz, f0 RPA);
one run against one reference: SE = sqrt(2) x the full-size seed SD (F1 0.50, F2 0.65, F3 1.12, F4 1.80 Hz), RPA worse
only below -0.1 point. Guard-rails at 20k (dev, `paired/fp32/<ref>_vs_<run>.txt`, closed-phase frames): Hillenbrand
F1 <= +4, F2 <= +9, F3 <= +20 Hz MAE; PTDB-TUG RPA >= -1.0; women's closed-phase |error| <= +5 Hz.
  1. choice at 20k: among rA / rB / rC passing the guard-rails and not diverged, the lowest summed F1-F4 SE units;
  2. adoption: "better" (one of F1-F4 beyond -2 SE, none beyond +2 SE, RPA not below -0.1), else teacher v3's recipe stays;
  3. weight decay: at 5k and 10k, rB against ctl: "decay rule" / "wd 0.01" / "no difference";
  4. reported: best arm per horizon; the shortest horizon at which the chosen arm is not worse than the reference at 20k.
Arg: scratch dir. Lines "DECISION ...". Hillenbrand F1 reported at every horizon."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S = Path(sys.argv[1])
REF = "conformer_v2r2_s1_d16k"
SE = {"F1": 0.71, "F2": 0.92, "F3": 1.58, "F4": 2.55}
P = Path("reports/analysis/paired/fp32")
tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
ARMS, HOR = ["rA", "rB", "rC"], ["05", "10", "20"]


def last_eval(run):
    p = Path("runs") / run / "log.jsonl"
    if not p.exists():
        return None
    ev = [json.loads(x)["eval"] for x in p.read_text().splitlines() if '"eval"' in x]
    return {**{f"F{n}": ev[-1][f"F{n}_mae_hz"] for n in range(1, 5)}, "RPA": 100 * ev[-1]["f0_rpa"]} if ev else None


def paired(run):
    t = (P / f"{REF}_vs_{run}.txt").read_text(encoding="utf-8")
    g = lambda key: float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", t).group(1))  # noqa: E731
    return {"F1": g("F1 MAE all"), "F2": g("F2 MAE all"), "F3": g("F3 MAE all"), "RPA": g("all frames")}


d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])


def cp_women(run):
    m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[run][:, 0]) & np.isfinite(d[REF][:, 0])
    return float(np.median(np.abs(d[run][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[REF][m, 0] - d["cp"][m, 0])))


ref = last_eval(REF)
print(f"D-031 against {REF} (synthetic: {', '.join(f'{k} {v:.2f}' for k, v in ref.items())})")
Z, ROWS = {}, {}
for arm in ARMS + ["ctl"]:
    for h in HOR:
        run = f"d031_{arm}_b{h}k"
        ev = last_eval(run)
        if ev is None:
            continue
        z = {k: (ev[k] - ref[k]) / SE[k] for k in SE}
        diverged = not all(np.isfinite(list(ev.values())))
        pr = paired(run)
        cw = cp_women(run)
        guard = pr["F1"] <= 4 and pr["F2"] <= 9 and pr["F3"] <= 20 and pr["RPA"] >= -1.0 and cw <= 5
        Z[(arm, h)] = (z, ev["RPA"] - ref["RPA"], diverged, guard)
        ROWS[(arm, h)] = pr
        print(f"  {run:16s} F1-F4 {' '.join(f'{ev[k] - ref[k]:+6.2f}' for k in SE)} Hz ({' '.join(f'{z[k]:+5.1f}' for k in SE)} SE)"
              f"  RPA {ev['RPA'] - ref['RPA']:+.2f}  | dev Hillenbrand F1/F2/F3 {pr['F1']:+.1f}/{pr['F2']:+.1f}/{pr['F3']:+.1f}"
              f"  RPA {pr['RPA']:+.1f}  cp women {cw:+.1f}  guard {'ok' if guard else 'FAILS'}{'  DIVERGED' if diverged else ''}")


def better(z, drpa):
    return any(z[k] <= -2 for k in SE) and not any(z[k] >= 2 for k in SE) and drpa >= -0.1


# 1-2
cands = [a for a in ARMS if (a, "20") in Z and not Z[(a, "20")][2] and Z[(a, "20")][3]]
if cands:
    choice = min(cands, key=lambda a: sum(Z[(a, "20")][0].values()))
    z, drpa = Z[(choice, "20")][0], Z[(choice, "20")][1]
    print(f"DECISION choice: {choice} (summed SE units {sum(z.values()):+.1f})")
    print(f"DECISION adoption: {'better: ' + choice + ' becomes the full-size recipe' if better(z, drpa) else 'not better: teacher v3 recipe stays'}")
else:
    choice = None
    print("DECISION choice: none (every arm fails a guard-rail or diverged); teacher v3 recipe stays")
# 3
res = []
for h in ("05", "10"):
    if ("rB", h) in Z and ("ctl", h) in Z:
        res.append({k: Z[("rB", h)][0][k] - Z[("ctl", h)][0][k] for k in SE})  # rB minus control, in reference-SE units
if len(res) == 2:
    rb_better = any(any(r[k] <= -2 for k in SE) for r in res) and not any(all(r[k] >= 2 for k in []) for r in res) \
        and not any(any(r[k] >= 2 for k in SE) for r in res)
    ctl_better = any(any(r[k] >= 2 for k in SE) for r in res) and not any(any(r[k] <= -2 for k in SE) for r in res)
    print(f"  rB - control (SE units), 5k: {' '.join(f'{res[0][k]:+.1f}' for k in SE)}; 10k: {' '.join(f'{res[1][k]:+.1f}' for k in SE)}")
    print(f"DECISION weight decay: {'decay rule' if rb_better else 'wd 0.01' if ctl_better else 'no difference'}")
# 4
for h in HOR:
    have = [a for a in ARMS if (a, h) in Z and not Z[(a, h)][2]]
    if have:
        b = min(have, key=lambda a: sum(Z[(a, h)][0].values()))
        print(f"  best arm at {int(h)}k: {b}; Hillenbrand F1 (dev, vs teacher v3) " + ", ".join(f"{a} {ROWS[(a, h)]['F1']:+.1f}" for a in have))
if choice:
    ok = [h for h in HOR if (choice, h) in Z and not any(Z[(choice, h)][0][k] >= 2 for k in SE)]
    print(f"  shortest horizon at which {choice} is not worse than teacher v3 at 20k: {int(ok[0]) if ok else 'none'}k")
