"""D-040 decision (pre-registered 2026-10-28): 5b (rA + joint VTL head from scratch) against 5a (rA), seeds paired.
On seed means, 5b becomes item 5's model if D-036's three criteria hold:
(1) no formant cost: synthetic F1-F4 from each run's last "eval" no worse than +2 SE (0.71 / 0.92 / 1.58 / 2.55 Hz), RPA
    not below -0.1 point; dev guard-rails, windowed (`paired/w400/<5a>_vs_<5b>.txt`; closed-phase frames in the windowed
    scratch dir): Hillenbrand F1 <= +4, F2 <= +9, F3 <= +20 Hz, PTDB-TUG RPA >= -1.0, women's closed-phase F1 |error| <= +5 Hz;
(2) Hillenbrand reliability above A's (seed-mean paired interval excluding 0, vtl_score_joint.py) with w/m in [0.85, 0.92];
(3) probe excess over its shuffle null <= 10 c/oct.
Per seed, every criterion is also evaluated; if the seeds disagree on any, the decision is flagged NOT ROBUST TO THE SEED.
F3 and the f0 excess are reported per seed.

    python decide_d040.py <windowed scratch dir> <vtl json> <seed-mean key> <5a>:<5b>:<vtl key> [...]

The script was checked on 5a before any 5b result (registered requirement): `--check` runs the formant part on the two
5a seeds as if one were the candidate, and the length part on D-036's json."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
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


def main():
    args = [a for a in sys.argv[1:] if a != "--check"]
    S, VJ, mean_key = Path(args[0]), json.load(open(args[1])), args[2]
    pairs = [a.split(":") for a in args[3:]]
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])

    def cp_women(run, ref):
        m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[run][:, 0]) & np.isfinite(d[ref][:, 0])
        return float(np.median(np.abs(d[run][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    def formant(ref, run):
        a, b = last_eval(ref), last_eval(run)
        return {**{f"syn {k}": (b[k] - a[k]) / SE[k] for k in SE}, "syn RPA": b["RPA"] - a["RPA"],
                **{f"dev {k}": v for k, v in paired(ref, run).items()}, "cp women F1": cp_women(run, ref)}

    def formant_ok(f):
        syn = all(f[f"syn {k}"] <= 2 for k in SE) and f["syn RPA"] >= -0.1
        guard = f["dev F1"] <= 4 and f["dev F2"] <= 9 and f["dev F3"] <= 20 and f["dev RPA"] >= -1.0 and f["cp women F1"] <= 5
        return syn and guard

    def length_ok(r):
        return r["reliability_ci_vs_A"][0] > 0 and 0.85 <= r["w/m"] <= 0.92

    print("D-040: 5b against 5a, seeds paired" + ("  [CHECK MODE: not a D-040 result]" if "--check" in sys.argv else ""))
    per = []
    for ref, run, vk in pairs:
        f = formant(ref, run)
        r = VJ[vk]
        ok = (formant_ok(f), length_ok(r), r["probe_excess"] <= 10)
        per.append((f, r, ok))
        print(f"  {run} vs {ref}: synthetic F1-F4 " + " ".join(f"{f[f'syn {k}']:+.1f}" for k in SE) + f" SE, RPA {f['syn RPA']:+.2f} | dev "
              f"Hillenbrand F1/F2/F3 {f['dev F1']:+.1f}/{f['dev F2']:+.1f}/{f['dev F3']:+.1f}  RPA {f['dev RPA']:+.1f}  cp women F1 "
              f"{f['cp women F1']:+.1f} | length ({vk}): reliability - A [{r['reliability_ci_vs_A'][0]:+.3f}, {r['reliability_ci_vs_A'][1]:+.3f}], "
              f"w/m {r['w/m']:.3f}, probe excess {r['probe_excess']:+.1f} c/oct | criteria {['yes' if x else 'no' for x in ok]}")
    fm = {k: float(np.mean([p[0][k] for p in per])) for k in per[0][0]}
    r = VJ[mean_key]
    ok = (formant_ok(fm), length_ok(r), r["probe_excess"] <= 10)
    print("\nSeed means: synthetic F1-F4 " + " ".join(f"{fm[f'syn {k}']:+.1f}" for k in SE) + f" SE, RPA {fm['syn RPA']:+.2f} | dev "
          f"Hillenbrand F1/F2/F3 {fm['dev F1']:+.1f}/{fm['dev F2']:+.1f}/{fm['dev F3']:+.1f}  RPA {fm['dev RPA']:+.1f}  cp women F1 {fm['cp women F1']:+.1f}"
          f" | length ({mean_key}): reliability - A [{r['reliability_ci_vs_A'][0]:+.3f}, {r['reliability_ci_vs_A'][1]:+.3f}], w/m {r['w/m']:.3f}, "
          f"probe excess {r['probe_excess']:+.1f} c/oct")
    print(f"(1) no formant cost: {'yes' if ok[0] else 'NO'}; (2) length beats A: {'yes' if ok[1] else 'NO'}; (3) f0 guard: {'yes' if ok[2] else 'NO'}")
    robust = all(p[2] == per[0][2] for p in per)
    print("Per seed: F3 synthetic " + ", ".join(f"{p[0]['syn F3']:+.1f} SE / dev {p[0]['dev F3']:+.1f} Hz" for p in per)
          + "; f0 excess " + ", ".join(f"{p[1]['probe_excess']:+.1f}" for p in per))
    verdict = "5b becomes item 5's model (teacher v3.1)" if all(ok) else "5a is teacher v3.1; A stays the length output"
    print(f"DECISION: {verdict}" + ("" if robust else " (NOT ROBUST TO THE SEED: the seeds disagree on a criterion)"))


if __name__ == "__main__":
    main()
