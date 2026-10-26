"""D-036 decision (pre-registered 2026-10-27): the VTL head trained jointly in a decay branch. Arms d036_arm0 (control,
no head) .. d036_arm4, all branched from conformer_v2r2_s1/ckpt_016000. Formants against arm 0: synthetic from each
run's last "eval" (SE units F1-F4 0.71 / 0.92 / 1.58 / 2.55 Hz; RPA worse only below -0.1 point); dev guard-rails,
windowed (`paired/w400/d036_arm0_vs_<run>.txt`; closed-phase frames in the windowed scratch dir): Hillenbrand
F1 <= +4, F2 <= +9, F3 <= +20 Hz; PTDB-TUG RPA >= -1.0; women's closed-phase F1 |error| <= +5 Hz. Length from
vtl_score_joint.py's json: reliability above A (interval excluding 0), women / men ratio in [0.85, 0.92], probe excess
over the shuffle null <= 10 c/oct. The joint head joins item 5 if arm 1 passes all three; the rest is reported.
Args: windowed scratch dir, vtl json."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, VJ = Path(sys.argv[1]), json.load(open(sys.argv[2]))
REF = "d036_arm0"
SE = {"F1": 0.71, "F2": 0.92, "F3": 1.58, "F4": 2.55}
P = Path("reports/analysis/paired/w400")
tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]


def last_eval(run):
    ev = [json.loads(x)["eval"] for x in (Path("runs") / run / "log.jsonl").read_text().splitlines() if '"eval"' in x]
    return {**{f"F{n}": ev[-1][f"F{n}_mae_hz"] for n in range(1, 5)}, "RPA": 100 * ev[-1]["f0_rpa"]}


def paired(a, b):
    t = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    g = lambda key: float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", t).group(1))  # noqa: E731
    return {"F1": g("F1 MAE all"), "F2": g("F2 MAE all"), "F3": g("F3 MAE all"), "RPA": g("all frames")}


d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])


def cp_women(run, ref):
    m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[run][:, 0]) & np.isfinite(d[ref][:, 0])
    return float(np.median(np.abs(d[run][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))


def formant_check(run, ref=REF):
    a, b = last_eval(ref), last_eval(run)
    z = {k: (b[k] - a[k]) / SE[k] for k in SE}
    pr, cw = paired(ref, run), cp_women(run, ref)
    syn_ok = all(v <= 2 for v in z.values()) and b["RPA"] - a["RPA"] >= -0.1
    guard = pr["F1"] <= 4 and pr["F2"] <= 9 and pr["F3"] <= 20 and pr["RPA"] >= -1.0 and cw <= 5
    print(f"  {run} vs {ref}: synthetic F1-F4 {' '.join(f'{v:+.1f}' for v in z.values())} SE, RPA {b['RPA'] - a['RPA']:+.2f} | dev "
          f"Hillenbrand F1/F2/F3 {pr['F1']:+.1f}/{pr['F2']:+.1f}/{pr['F3']:+.1f}  RPA {pr['RPA']:+.1f}  cp women F1 {cw:+.1f} | "
          f"synthetic {'ok' if syn_ok else 'WORSE'}, guard-rails {'ok' if guard else 'FAIL'}")
    return syn_ok and guard


print("Formants, f0 (against arm 0 unless stated)")
formant_check("d036_arm0", "conformer_v2r2_s1_d16k")
ok1 = formant_check("d036_arm1")
for arm in ("d036_arm2", "d036_arm3", "d036_arm4"):
    formant_check(arm)
print("\nLength (vtl_score_joint.py)")
for n, r in VJ.items():
    print(f"  {n}: reliability - A [{r['reliability_ci_vs_A'][0]:+.3f}, {r['reliability_ci_vs_A'][1]:+.3f}]; w/m {r['w/m']:.3f}; "
          f"probe {r['probe_obs']:.1f} c/oct, excess over null {r['probe_excess']:+.1f}")
r = VJ["d036_arm1"]
ok2 = r["reliability_ci_vs_A"][0] > 0 and 0.85 <= r["w/m"] <= 0.92
ok3 = r["probe_excess"] <= 10
print(f"\narm 1: no formant cost {'yes' if ok1 else 'NO'}; length beats A {'yes' if ok2 else 'NO'}; f0 guard {'yes' if ok3 else 'NO'}")
print(f"DECISION: the joint VTL head {'joins' if ok1 and ok2 and ok3 else 'does not join'} item 5's retrain")
