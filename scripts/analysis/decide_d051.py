"""D-051 decision (pre-registered 2026-10-30): 5c (5b + 30 % long clips with sustained larynx / lip movement + an
instantaneous-length head) against 5b, seeds paired. Non-inferiority on D-040's margins; on seed means, 5c is adopted if
(1) no formant cost: synthetic F1-F4 from each run's last "eval" no worse than +2 SE (0.71 / 0.92 / 1.58 / 2.55 Hz), RPA
    not below -0.1 point; dev guard-rails, windowed (`paired/w400/<5b>_vs_<5c>.txt`; closed-phase frames in the windowed
    scratch dir): Hillenbrand F1 <= +4, F2 <= +9, F3 <= +20 Hz, PTDB-TUG RPA >= -1.0, women's closed-phase F1 |error| <= +5 Hz;
(2) length not worse: Hillenbrand reliability, 5c minus 5b (paired seed means, vtl_score_joint.py's VA_SEED_PAIRS),
    interval lower bound above -0.02, with w/m in [0.85, 0.92];
(3) f0 probe excess over its shuffle null <= 10 c/oct.
Otherwise 5b stays. Per seed every criterion is also evaluated; seeds disagreeing -> NOT ROBUST TO THE SEED (as D-040).
Reported, deciding nothing: D-050 probe 10 (the anatomical head's slope on the instantaneous length; 5c's new head's own
slope) and the instantaneous head's validation on held-out long clips.

    python decide_d051.py <windowed scratch dir> <vtl json> <seed-pair key> <5b>:<5c>:<vtl key> [...]

`--check` runs the formant part on two existing arms (5a against 5b, D-040's files) and the length part on D-040's
seed-mean json, before any 5c result, to check the script reads its inputs."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SE = {"F1": 0.71, "F2": 0.92, "F3": 1.58, "F4": 2.55}
P = Path("reports/analysis/paired/w400")
tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]


def evals(run):
    return [json.loads(x)["eval"] for x in (Path("runs") / run / "log.jsonl").read_text().splitlines() if '"eval"' in x]


def last_eval(run):
    ev = evals(run)[-1]
    return {**{f"F{n}": ev[f"F{n}_mae_hz"] for n in range(1, 5)}, "RPA": 100 * ev["f0_rpa"]}


def paired(a, b):
    t = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    g = lambda key: float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", t).group(1))  # noqa: E731
    return {"F1": g("F1 MAE all"), "F2": g("F2 MAE all"), "F3": g("F3 MAE all"), "RPA": g("all frames")}


def probe10(run):
    """D-050 probe 10 slopes for a run, if scored: anatomical VTL (ideal 0) and instantaneous VTL (ideal 1)."""
    f = Path(f"reports/probes/articulation/{run}.json")
    if not f.exists():
        return None
    a = json.loads(f.read_text(encoding="utf-8"))
    return {k: a[k]["point"] for k in a if k.split("/")[-1] in ("VTL", "VTLi") and k.split("/")[0] in ("larynx", "lips")
            and isinstance(a[k], dict) and "point" in a[k]}


def main():
    check = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--check"]
    S, VJ, pair_key = Path(args[0]), json.load(open(args[1])), args[2]
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

    def length(r):
        """(interval, w/m, probe excess): D-051's paired entry, or in check mode D-040's interval against A."""
        ci = r["reliability_ci"] if "reliability_ci" in r else r["reliability_ci_vs_A"]
        return ci, r["w/m"], r["probe_excess"]

    def length_ok(ci, wm):
        return ci[0] > -0.02 and 0.85 <= wm <= 0.92

    print("D-051: 5c against 5b, seeds paired" + ("  [CHECK MODE: not a D-051 result]" if check else ""))
    per = []
    for ref, run, vk in pairs:
        f = formant(ref, run)
        ci, wm, ex = length(VJ[vk])
        ok = (formant_ok(f), length_ok(ci, wm), ex <= 10)
        per.append((f, ex, ok))
        print(f"  {run} vs {ref}: synthetic F1-F4 " + " ".join(f"{f[f'syn {k}']:+.1f}" for k in SE) + f" SE, RPA {f['syn RPA']:+.2f} | dev "
              f"Hillenbrand F1/F2/F3 {f['dev F1']:+.1f}/{f['dev F2']:+.1f}/{f['dev F3']:+.1f}  RPA {f['dev RPA']:+.1f}  cp women F1 "
              f"{f['cp women F1']:+.1f} | length ({vk}): [{ci[0]:+.3f}, {ci[1]:+.3f}], w/m {wm:.3f}, probe excess {ex:+.1f} c/oct "
              f"| criteria {['yes' if x else 'no' for x in ok]}")
    fm = {k: float(np.mean([p[0][k] for p in per])) for k in per[0][0]}
    ci, wm, ex = length(VJ[pair_key])
    ok = (formant_ok(fm), length_ok(ci, wm), ex <= 10)
    print("\nSeed means: synthetic F1-F4 " + " ".join(f"{fm[f'syn {k}']:+.1f}" for k in SE) + f" SE, RPA {fm['syn RPA']:+.2f} | dev "
          f"Hillenbrand F1/F2/F3 {fm['dev F1']:+.1f}/{fm['dev F2']:+.1f}/{fm['dev F3']:+.1f}  RPA {fm['dev RPA']:+.1f}  cp women F1 {fm['cp women F1']:+.1f}"
          f" | length ({pair_key}): reliability 5c - 5b [{ci[0]:+.3f}, {ci[1]:+.3f}], w/m {wm:.3f}, probe excess {ex:+.1f} c/oct")
    print(f"(1) no formant cost: {'yes' if ok[0] else 'NO'}; (2) length not worse: {'yes' if ok[1] else 'NO'}; (3) f0 guard: {'yes' if ok[2] else 'NO'}")
    robust = all(p[2] == per[0][2] for p in per)
    print("\nReported (decides nothing):")
    for ref, run, _ in pairs:
        pr = probe10(run)
        print(f"  D-050 probe 10, {run}: " + (", ".join(f"{k} {v:+.2f}" for k, v in sorted(pr.items())) if pr else "not scored"))
        lines = (Path("runs") / run / "log.jsonl").read_text().splitlines()
        ex = [json.loads(x)["eval_extra:train_long_v1"] for x in lines if '"eval_extra:train_long_v1"' in x]
        if ex:
            print(f"  held-out long clips, {run} (last eval): " + ", ".join(f"{k} {v:.4g}" for k, v in ex[-1].items()
                                                                          if k.startswith("vtl") and not k.endswith("frames")))
    verdict = "5c becomes item 5's model" if all(ok) else "5b stays item 5's model"
    print(f"\nDECISION: {verdict}" + ("" if robust else " (NOT ROBUST TO THE SEED: the seeds disagree on a criterion)"))


if __name__ == "__main__":
    main()
