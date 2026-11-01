"""D-065 decision (pre-registered 2026-11-01): two arms against 5b (teacher v3.1's recipe), each with seeds 1 and 2
paired to 5b's seeds. Arm "bw": train_v2 re-rendered with bandwidths decoupled from the formants (make_bw_set.py) in
train_v2's place. Arm "gain": the channel's training gain range -40 to 0 dB instead of -20 to 0.

An arm joins the recipe if, on seed means, its primary criterion (P) and D-051's guard-rails (1)-(3) all hold:
(P bw)   probe 12 arm C (synthetic cascade clips, exact labels): the weight on the true bandwidth, mean over B1-B4, rises
         by at least +0.10 against 5b, AND probe 7 (bandwidth counterfactual on resynthesis): the bandwidth slope, mean
         over B1-B4, rises by at least +0.10;
(P gain) probe 13 at x0.03 (median peak about -44 dBFS, inside the arm's range): mean |change| over B1-B4 at most half
         of 5b's (the ratio's seed mean <= 0.5 and each seed's <= 0.75: 5c, with no level change, gave 0.51 and 1.19),
         AND mean |change| over F1-F4 no more than 5b's + 3 cents;
(1) no formant cost: synthetic F1-F4 from each run's last "eval" no worse than +2 SE (0.71 / 0.92 / 1.58 / 2.55 Hz), RPA
    not below -0.1 point; dev guard-rails, windowed: Hillenbrand F1 <= +4, F2 <= +9, F3 <= +20 Hz, PTDB-TUG RPA >= -1.0,
    women's closed-phase F1 |error| <= +5 Hz;
(2) length not worse: Hillenbrand reliability, arm minus 5b (paired seed means, vtl_score_joint.py), interval lower
    bound above -0.02, with w/m in [0.85, 0.92];
(3) f0 probe excess over its shuffle null <= 10 c/oct.
Per seed every criterion is also evaluated; seeds disagreeing -> NOT ROBUST TO THE SEED (as D-040).

    python decide_d065.py <arm> <windowed scratch dir> <vtl json> <seed-pair key> <5b>:<arm run>:<vtl key> [...]

`--check` reads 5c against 5b (D-051's files) to check the script reads its inputs; not a D-065 result."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SE = {"F1": 0.71, "F2": 0.92, "F3": 1.58, "F4": 2.55}
P = Path("reports/analysis/paired/w400")
PR = Path("reports/probes")
tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
B = [f"B{k}" for k in range(1, 5)]
F = [f"F{k}" for k in range(1, 5)]


def last_eval(run):
    ev = [json.loads(x)["eval"] for x in (Path("runs") / run / "log.jsonl").read_text().splitlines() if '"eval"' in x][-1]
    return {**{f"F{n}": ev[f"F{n}_mae_hz"] for n in range(1, 5)}, "RPA": 100 * ev["f0_rpa"]}


def paired(a, b):
    t = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    g = lambda key: float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", t).group(1))  # noqa: E731
    return {"F1": g("F1 MAE all"), "F2": g("F2 MAE all"), "F3": g("F3 MAE all"), "RPA": g("all frames")}


def probe_bw(run):
    """(arm C weight on the label, mean B1-B4; probe 7 slope, mean B1-B4), with the per-formant values."""
    c = json.loads((PR / "bw_evidence" / f"{run}.json").read_text(encoding="utf-8"))["C"]
    s = json.loads((PR / "bandwidth" / f"{run}.json").read_text(encoding="utf-8"))
    wl = [c[b]["w_label"] for b in B]
    sl = [s[f"bw/all/{b}"]["point"] for b in B]
    return float(np.mean(wl)), float(np.mean(sl)), wl, sl


def probe_gain(run, g):
    """Probe 13 at gain g: (mean |change| B1-B4, mean |change| F1-F4), cents."""
    d = json.loads((PR / "gain" / f"{run}.json").read_text(encoding="utf-8"))
    ab = lambda ks: float(np.mean([abs(1200 * d[f"gain_{g}/change/{k}"]["point"]) for k in ks]))  # noqa: E731
    return ab(B), ab(F)


def primary(arm, ref, run):
    """(values, passes, text) for the arm's primary criterion, one seed pair."""
    if arm == "bw":
        a, b = probe_bw(ref), probe_bw(run)
        v = {"w_label": b[0] - a[0], "probe7 slope": b[1] - a[1]}
        txt = (f"arm C w_label {a[0]:.3f} -> {b[0]:.3f} ({' '.join(f'{x:.2f}' for x in b[2])}); "
               f"probe 7 slope {a[1]:.3f} -> {b[1]:.3f} ({' '.join(f'{x:.2f}' for x in b[3])})")
        return v, primary_ok(arm, v), txt
    a, b = probe_gain(ref, "0.03"), probe_gain(run, "0.03")
    v = {"B ratio": b[0] / a[0], "F excess": b[1] - a[1]}
    others = "; ".join(f"x{g}: B {probe_gain(ref, g)[0]:.0f} -> {probe_gain(run, g)[0]:.0f}, F {probe_gain(ref, g)[1]:.1f} -> "
                       f"{probe_gain(run, g)[1]:.1f}" for g in ("0.3", "0.1", "0.01"))
    txt = f"x0.03 mean |B| {a[0]:.0f} -> {b[0]:.0f} cents (ratio {v['B ratio']:.2f}), mean |F| {a[1]:.1f} -> {b[1]:.1f}; reported: {others}"
    return v, primary_ok(arm, v), txt


def primary_ok(arm, v):
    if arm == "bw":
        return v["w_label"] >= 0.10 and v["probe7 slope"] >= 0.10
    return v["B ratio"] <= 0.5 and v["F excess"] <= 3


def main():
    check = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--check"]
    arm, S, VJ, pair_key = args[0], Path(args[1]), json.load(open(args[2])), args[3]
    assert arm in ("bw", "gain")
    pairs = [a.split(":") for a in args[4:]]
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])

    def women(*runs):
        return dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.all([np.isfinite(d[r][:, 0]) for r in runs], 0)

    def cp_women(run, ref):
        m = women(run, ref)
        return float(np.median(np.abs(d[run][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    def cp_signed(run):
        m = women(run)
        return float(np.median(d[run][m, 0] - d["cp"][m, 0]))

    def formant(ref, run):
        a, b = last_eval(ref), last_eval(run)
        return {**{f"syn {k}": (b[k] - a[k]) / SE[k] for k in SE}, "syn RPA": b["RPA"] - a["RPA"],
                **{f"dev {k}": v for k, v in paired(ref, run).items()}, "cp women F1": cp_women(run, ref)}

    def formant_ok(f):
        syn = all(f[f"syn {k}"] <= 2 for k in SE) and f["syn RPA"] >= -0.1
        return syn and f["dev F1"] <= 4 and f["dev F2"] <= 9 and f["dev F3"] <= 20 and f["dev RPA"] >= -1.0 and f["cp women F1"] <= 5

    def length(r):
        ci = r["reliability_ci"] if "reliability_ci" in r else r["reliability_ci_vs_A"]
        return ci, r["w/m"], r["probe_excess"]

    def fmt(f):
        return (" ".join(f"{f[f'syn {k}']:+.1f}" for k in SE) + f" SE, RPA {f['syn RPA']:+.2f} | dev Hillenbrand F1/F2/F3 "
                f"{f['dev F1']:+.1f}/{f['dev F2']:+.1f}/{f['dev F3']:+.1f}  RPA {f['dev RPA']:+.1f}  cp women F1 |error| {f['cp women F1']:+.1f}")

    print(f"D-065 arm {arm}: against 5b, seeds paired" + ("  [CHECK MODE: not a D-065 result]" if check else ""))
    per = []
    for ref, run, vk in pairs:
        f = formant(ref, run)
        pv, pok, ptxt = primary(arm, ref, run)
        ci, wm, ex = length(VJ[vk])
        ok = (pok, formant_ok(f), ci[0] > -0.02 and 0.85 <= wm <= 0.92, ex <= 10)
        per.append((f, pv, ok))
        print(f"  {run} vs {ref}: (P) {ptxt}\n    synthetic F1-F4 {fmt(f)} (signed bias {cp_signed(ref):+.1f} -> {cp_signed(run):+.1f} Hz)"
              f" | length ({vk}): [{ci[0]:+.3f}, {ci[1]:+.3f}], w/m {wm:.3f}, probe excess {ex:+.1f} c/oct"
              f"\n    criteria P/1/2/3 {['yes' if x else 'no' for x in ok]}")
    fm = {k: float(np.mean([p[0][k] for p in per])) for k in per[0][0]}
    pm = {k: float(np.mean([p[1][k] for p in per])) for k in per[0][1]}
    ci, wm, ex = length(VJ[pair_key])
    p_ok = primary_ok(arm, pm) and (arm == "bw" or max(p[1]["B ratio"] for p in per) <= 0.75)
    ok = (p_ok, formant_ok(fm), ci[0] > -0.02 and 0.85 <= wm <= 0.92, ex <= 10)
    print("\nSeed means: (P) " + ", ".join(f"{k} {v:+.3f}" for k, v in pm.items()) + f" | synthetic F1-F4 {fmt(fm)}"
          f" | length ({pair_key}): [{ci[0]:+.3f}, {ci[1]:+.3f}], w/m {wm:.3f}, probe excess {ex:+.1f} c/oct")
    print(f"(P) primary: {'yes' if ok[0] else 'NO'}; (1) no formant cost: {'yes' if ok[1] else 'NO'}; "
          f"(2) length not worse: {'yes' if ok[2] else 'NO'}; (3) f0 guard: {'yes' if ok[3] else 'NO'}")
    robust = all(p[2] == per[0][2] for p in per)
    for ref, run, _ in pairs:
        lines = (Path("runs") / run / "log.jsonl").read_text().splitlines()
        ex = [json.loads(x) for x in lines if '"eval_extra:' in x]
        if ex:
            k = next(k for k in ex[-1] if k.startswith("eval_extra:"))
            print(f"  reported: {run} {k} (last eval): " + ", ".join(f"{a} {v:.4g}" for a, v in ex[-1][k].items() if a.endswith("_mae_hz")))
    verdict = f"arm {arm} joins the recipe" if all(ok) else f"arm {arm} does not join; 5b's recipe stays"
    print(f"\nDECISION: {verdict}" + ("" if robust else " (NOT ROBUST TO THE SEED: the seeds disagree on a criterion)"))


if __name__ == "__main__":
    main()
