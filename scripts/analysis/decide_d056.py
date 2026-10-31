"""D-056 decision (pre-registered 2026-10-31): the anchored per-bin read-out (phase 2, `d056_s{1,2}`) against 5b's own
decay branches (`d040_5b_s{1,2}_b20k`), seeds paired. A candidate if non-inferior on D-040's margins (seed means:
synthetic F1-F4 no worse than +2 SE, RPA >= -0.1 point; Hillenbrand dev F1 <= +4, F2 <= +9, F3 <= +20 Hz; women's
closed-phase F1 |error| <= +5 Hz) and, on both seeds, at least one gain: probe 11's F1 scale error |1 - slope| at least
halved on Hillenbrand and PTDB-TUG; synthetic F1-F4 MAE lower by >= 3 %; women's closed-phase F1 |error| lower by >= 2 Hz.
Reported: phase 1 (frozen) beside D-054's arms, probe 14's fill +6 dB F1 change, probe 12, resynthesised validation, and
D-054's film seed 2 follow-up.

    python decide_d056.py <windowed scratch dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SE = {"F1": 0.71, "F2": 0.92, "F3": 1.58, "F4": 2.55}  # D-040's synthetic standard errors (decide_d051.py)
P = Path("reports/analysis/paired/w400")
RESYNTH = ("eval_resynth:zeroth_korean_val_d16k", "eval_resynth:vocalset_val_d16k")
tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
PAIRS = {1: ("d040_5b_s1_b20k", "d056_s1"), 2: ("d040_5b_s2_b20k", "d056_s2")}


def last(r, key="eval"):
    return [json.loads(x)[key] for x in (Path("runs") / r / "log.jsonl").read_text().splitlines() if f'"{key}"' in x][-1]


def syn(r):
    ev = last(r)
    return np.array([ev[f"F{n}_mae_hz"] for n in range(1, 5)]), 100 * ev["f0_rpa"]


def resynth(r):
    return np.array([[last(r, k)[f"F{n}_mae_hz"] for n in range(1, 4)] for k in RESYNTH]).mean(0)


def scale(r):
    a = json.loads(Path(f"reports/probes/scale/{r}.json").read_text(encoding="utf-8"))
    return {k: a[f"{k}/slope/F1"]["point"] for k in ("hillenbrand", "ptdb_tug")}


def valley(r):
    a = json.loads(Path(f"reports/probes/valley/{r}.json").read_text(encoding="utf-8"))
    return 1200 * a["fill_+6/change/F1"]["point"]


def w_label(r):
    a = json.loads(Path(f"reports/probes/bw_evidence/{r}.json").read_text(encoding="utf-8"))["C"]
    return np.array([a[f"B{n}"]["w_label"] for n in range(1, 5)])


def paired(a, b):
    t = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    g = lambda key: float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", t).group(1))  # noqa: E731
    return np.array([g("F1 MAE all"), g("F2 MAE all"), g("F3 MAE all")])


def main():
    S = Path(sys.argv[1])
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    women = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0])

    def cp_abs(r, ref):
        m = women & np.isfinite(d[r][:, 0]) & np.isfinite(d[ref][:, 0])
        return float(np.median(np.abs(d[r][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    def cp_bias(r):
        m = women & np.isfinite(d[r][:, 0])
        return float(np.median(d[r][m, 0] - d["cp"][m, 0]))

    print("D-056: anchored per-bin read-out (phase 2) against 5b's own decay branches, seeds paired\n")
    print("per run: synthetic F1-F4 MAE Hz | RPA | resynth F1-F3 | probe 11 F1 slope (Hb, PTDB) | probe 14 fill+6 F1 c | "
          "probe 12 w_label | women's cp F1 bias Hz")
    for r in ("d040_5b_s1_b20k", "d040_5b_s2_b20k", "d056_s1", "d056_s2", "d056_p1_s1", "d056_p1_s2", "d054_mlp_s2", "d054_film_s1", "d054_film_s2"):
        f, rpa = syn(r)
        sc = scale(r)
        print(f"  {r:16s} {np.round(f, 1)} | {rpa:.2f} | {np.round(resynth(r), 1)} | {sc['hillenbrand']:.3f}, {sc['ptdb_tug']:.3f} | "
              f"{valley(r):+.1f} | {np.round(w_label(r), 2)} | {cp_bias(r) if r in d else float('nan'):+.1f}")
    print(f"  D-054 follow-up, film s2 against mlp s2 (Hillenbrand F1/F2/F3 Hz): {np.round(paired('d054_mlp_s2', 'd054_film_s2'), 1)}\n")

    per = {}
    for s, (ref, run) in PAIRS.items():
        (fa, ra), (fb, rb) = syn(ref), syn(run)
        sa, sb = scale(ref), scale(run)
        per[s] = {"syn_se": (fb - fa) / np.array(list(SE.values())), "rpa": rb - ra, "dev": paired(ref, run), "cp": cp_abs(run, ref),
                  "syn_rel": float(np.mean(fb / fa - 1)),
                  "scale_halved": all(abs(1 - sb[k]) <= 0.5 * abs(1 - sa[k]) for k in sa)}
        p = per[s]
        print(f"seed {s}: synthetic F1-F4 {', '.join(f'{v:+.1f}' for v in p['syn_se'])} SE ({100 * p['syn_rel']:+.1f} %), RPA {p['rpa']:+.2f}; "
              f"Hillenbrand F1/F2/F3 {', '.join(f'{v:+.1f}' for v in p['dev'])} Hz; women's cp |error| {p['cp']:+.1f} Hz; "
              f"scale error halved: {p['scale_halved']}")
    m = {k: np.mean([per[s][k] for s in per], 0) for k in ("syn_se", "rpa", "dev", "cp", "syn_rel")}
    noninf = (all(m["syn_se"] <= 2) and m["rpa"] >= -0.1 and m["dev"][0] <= 4 and m["dev"][1] <= 9 and m["dev"][2] <= 20 and m["cp"] <= 5)
    gains = {"scale": all(per[s]["scale_halved"] for s in per), "synthetic": all(per[s]["syn_rel"] <= -0.03 for s in per),
             "closed phase": all(per[s]["cp"] <= -2 for s in per)}
    print(f"\nseed means: synthetic {', '.join(f'{v:+.1f}' for v in m['syn_se'])} SE, RPA {m['rpa']:+.2f}, Hillenbrand "
          f"{', '.join(f'{v:+.1f}' for v in m['dev'])} Hz, women's cp |error| {m['cp']:+.1f} Hz -> non-inferior: {noninf}")
    print(f"gains on both seeds: {gains}")
    print()
    print("DECISION: " + ("a candidate" if noninf and any(gains.values()) else "not a candidate"))


if __name__ == "__main__":
    main()
