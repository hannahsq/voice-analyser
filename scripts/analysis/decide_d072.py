"""D-072 decision (registered 2026-11-03): arm B (retrieve beside the flatten, `d072_retrieve_s{1,2}`) against 5b's own
decay branches (`d040_5b_s{1,2}_b20k`), seeds paired, on D-056's rule: non-inferior on D-040's margins (seed means) and,
on both seeds, at least one gain (probe 11's F1 scale error halved on Hillenbrand and PTDB-TUG; synthetic F1-F4 MAE lower
by >= 3 %; women's closed-phase F1 |error| lower by >= 2 Hz); plus D-051's length guard (2) (Hillenbrand reliability
against 5b, paired seed means, interval lower bound above -0.02, w/m in [0.85, 0.92]) and f0 guard (3) (probe excess
<= 10 c/oct). Seeds disagreeing -> NOT ROBUST TO THE SEED. Reported, deciding nothing: arm A (factorised, the
parameter-matched control) by the same measures, and B against A (attribution); 5b's own seed spread beside the length
guard (D-065's note: 5b s2 is a good draw).

    python decide_d072.py <windowed scratch dir> [--check]

`--check` reads D-065's gain arm in place of both arms, to check the script reads its inputs; not a D-072 result.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import decide_d056 as D  # noqa: E402

O = Path("reports/analysis/d072")
REF = {1: "d040_5b_s1_b20k", 2: "d040_5b_s2_b20k"}
CHECK = "--check" in sys.argv


def run(arm, s):
    return f"d065_gain_s{s}_b20k" if CHECK else f"d072_{arm}_s{s}"


def vtl(arm):
    if CHECK:
        v = json.loads(Path("reports/analysis/d065/vtl_joint_gain.json").read_text())
        return {**v, f"{arm}-5b": v["gain-5b"]}
    return json.loads((O / f"vtl_joint_{arm}.json").read_text())


def main():
    S = Path([a for a in sys.argv[1:] if a != "--check"][0])
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([D.tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    women = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0])

    def cp_abs(r, ref):
        m = women & np.isfinite(d[r][:, 0]) & np.isfinite(d[ref][:, 0])
        return float(np.median(np.abs(d[r][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    def cp_bias(r):
        m = women & np.isfinite(d[r][:, 0])
        return float(np.median(d[r][m, 0] - d["cp"][m, 0]))

    def compare(ref, run):
        (fa, ra), (fb, rb) = D.syn(ref), D.syn(run)
        sa, sb = D.scale(ref), D.scale(run)
        return {"syn_se": (fb - fa) / np.array(list(D.SE.values())), "rpa": rb - ra, "dev": D.paired(ref, run),
                "cp": cp_abs(run, ref), "syn_rel": float(np.mean(fb / fa - 1)),
                "scale_halved": all(abs(1 - sb[k]) <= 0.5 * abs(1 - sa[k]) for k in sa)}

    def show(tag, p):
        print(f"  {tag}: synthetic F1-F4 {', '.join(f'{v:+.1f}' for v in p['syn_se'])} SE ({100 * p['syn_rel']:+.1f} %), "
              f"RPA {p['rpa']:+.2f}; Hillenbrand F1/F2/F3 {', '.join(f'{v:+.1f}' for v in p['dev'])} Hz; women's cp |error| "
              f"{p['cp']:+.1f} Hz; scale error halved: {p['scale_halved']}")

    def guards(r):
        ci = r["reliability_ci"] if "reliability_ci" in r else r["reliability_ci_vs_A"]
        return ci, ci[0] > -0.02 and 0.85 <= r["w/m"] <= 0.92, r["probe_excess"] <= 10

    def rule(m, gain, r):
        """the whole rule on one set of values (seed means, or one seed): pass, and its parts"""
        noninf = (all(m["syn_se"] <= 2) and m["rpa"] >= -0.1 and m["dev"][0] <= 4 and m["dev"][1] <= 9
                  and m["dev"][2] <= 20 and m["cp"] <= 5)
        _, length_ok, f0_ok = guards(r)
        return noninf and gain and length_ok and f0_ok, noninf, length_ok, f0_ok

    def verdict(per, vj, key):
        m = {k: np.mean([per[s][k] for s in per], 0) for k in ("syn_se", "rpa", "dev", "cp", "syn_rel")}
        gains = {"scale": all(per[s]["scale_halved"] for s in per), "synthetic": all(per[s]["syn_rel"] <= -0.03 for s in per),
                 "closed phase": all(per[s]["cp"] <= -2 for s in per)}
        ok, noninf, length_ok, f0_ok = rule(m, any(gains.values()), vj[key])
        ci = guards(vj[key])[0]
        print(f"  seed means: synthetic {', '.join(f'{v:+.1f}' for v in m['syn_se'])} SE, RPA {m['rpa']:+.2f}, Hillenbrand "
              f"{', '.join(f'{v:+.1f}' for v in m['dev'])} Hz, women's cp |error| {m['cp']:+.1f} Hz -> non-inferior: {noninf}")
        print(f"  gains on both seeds: {gains}")
        print(f"  (2) length ({key}): reliability interval [{ci[0]:+.3f}, {ci[1]:+.3f}], w/m {vj[key]['w/m']:.3f} -> {length_ok}; "
              f"(3) f0 probe excess {vj[key]['probe_excess']:+.1f} c/oct -> {f0_ok}")
        seeds = {s: rule(per[s], per[s]["scale_halved"] or per[s]["syn_rel"] <= -0.03 or per[s]["cp"] <= -2, vj[f"s{s}"])[0]
                 for s in per}
        print(f"  the rule on each seed alone: {seeds}")
        return ok, len(set(seeds.values())) == 1

    print("D-072: a path beside the trunk's flatten, against 5b's own decay branches, seeds paired"
          + ("  [CHECK MODE: D-065's gain arm in both arms' place; not a D-072 result]" if CHECK else "") + "\n")
    print("per run: synthetic F1-F4 MAE Hz | RPA | resynth F1-F3 | probe 11 F1 slope (Hb, PTDB) | probe 12 w_label | "
          "women's cp F1 bias Hz")
    runs = [*REF.values()] + list(dict.fromkeys(run(a, s) for a in ("retrieve", "factorised") for s in (1, 2)))
    for r in runs:
        f, rpa = D.syn(r)
        sc = D.scale(r)
        print(f"  {r:20s} {np.round(f, 1)} | {rpa:.2f} | {np.round(D.resynth(r), 1)} | {sc['hillenbrand']:.3f}, "
              f"{sc['ptdb_tug']:.3f} | {np.round(D.w_label(r), 2)} | {cp_bias(r):+.1f}")
    print()
    out = {}
    for arm in ("retrieve", "factorised"):
        vj = vtl(arm)
        print(f"arm {'B' if arm == 'retrieve' else 'A'} ({arm}){' — decides' if arm == 'retrieve' else ' — reported (control)'}")
        per = {s: compare(REF[s], run(arm, s)) for s in REF}
        for s in per:
            show(f"seed {s}", per[s])
        out[arm] = verdict(per, vj, f"{arm}-5b")
        print()
    print("attribution, B against A (Hillenbrand F1/F2/F3 Hz, paired; synthetic F1-F4 MAE %):")
    for s in REF:
        a, b = run("factorised", s), run("retrieve", s)
        fa, fb = D.syn(a)[0], D.syn(b)[0]
        dv = np.round(D.paired(a, b), 1) if (D.P / f"{a}_vs_{b}.txt").exists() else "(no paired file)"
        print(f"  seed {s}: {dv}; {100 * float(np.mean(fb / fa - 1)):+.1f} %")
    print("reported beside the length guard: 5b's seeds differ by +0.034 [+0.005, +0.060] in Hillenbrand reliability "
          "(s2 - s1; D-065's note), so a paired seed mean against 5b carries s2's good draw")
    ok, robust = out["retrieve"]
    a_ok = out["factorised"][0]
    print(f"\nreported: arm A would {'' if a_ok else 'not '}pass the same rule"
          + (" (a gain in both: the late-added capacity's)" if ok and a_ok else
             " (a gain in B only: the content-dependent selection's)" if ok else ""))
    print("DECISION: " + ("arm B is a candidate (a from-scratch arm and the write-back screen next)" if ok else
                          "arm B is not a candidate") + ("" if robust else " (NOT ROBUST TO THE SEED)"))


if __name__ == "__main__":
    main()
