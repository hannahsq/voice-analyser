"""D-073 decision (registered 2026-11-03): learned formant queries as the read-out on teacher v3.1's frozen trunk, each
arm ("queries": content-only keys; "queries_abs": with an absolute position per bin) against D-054's mlp heads, seed means
of head seeds 1 and 2, both seeds in the same direction. An arm is supported if
(1) its synthetic F1-F4 MAE (mean of the four, each run's last "eval") is at most 1.15 x mlp's, and
(2) probe 11's F1 scale error |1 - slope| is at most half mlp's on both Hillenbrand and PTDB-TUG,
with D-054's guards (arm - mlp): Hillenbrand dev F1 <= +4, F2 <= +9, F3 <= +20 Hz; resynthesised F1-F3 MAE no more than
5 % worse. Futility stop (as D-054 amendment 1): seed 1 over 1.5 x mlp seed 1's synthetic F1-F4 MAE ends the arm as "not
supported (futility stop)". Reported: women's closed-phase F1 bias with a speaker-bootstrap interval against mlp; probe
12; bandwidth error; D-054's film beside them.

    python decide_d073.py <windowed scratch dir> [--check]

`--check` reads D-054's film arm (seed 1 only) in place of both arms, to check the script reads its inputs; not a D-073
result.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import decide_d054 as D  # noqa: E402

O = Path("reports/analysis/d073")
SEEDS = (1, 2)
ARMS = ("queries", "queries_abs")
CHECK = "--check" in sys.argv
SEED = 5073


def run(arm, s):
    if arm == "mlp" or arm == "film":
        return D.run(arm, s)
    return D.run("film", s) if CHECK else f"d073_{arm}_s{s}"


def trained(r):
    return (Path("runs") / r / "ckpt_008000.pt").exists()


def main():
    S = Path([a for a in sys.argv[1:] if a != "--check"][0])
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([D.tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    rng = np.random.default_rng(SEED)

    def cp_bias_diff(r, ref):
        """median signed F1 bias against the closed phase, arm - mlp, Hz, with a 95 % interval over dev women speakers"""
        m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[r][:, 0]) & np.isfinite(d[ref][:, 0])
        spk = d["speaker"][m]
        e_r, e_m = d[r][m, 0] - d["cp"][m, 0], d[ref][m, 0] - d["cp"][m, 0]
        point = float(np.median(e_r) - np.median(e_m))
        sp = np.unique(spk)
        idx = {s: np.flatnonzero(spk == s) for s in sp}
        bs = []
        for _ in range(1000):
            j = np.concatenate([idx[s] for s in rng.choice(sp, len(sp))])
            bs.append(np.median(e_r[j]) - np.median(e_m[j]))
        return point, *np.percentile(bs, [2.5, 97.5])

    print("D-073: learned formant queries on teacher v3.1's frozen trunk, against D-054's mlp heads"
          + ("  [CHECK MODE: D-054's film in both arms' place; not a D-073 result]" if CHECK else "") + "\n")
    print("per run: synthetic F1-F4 MAE Hz | resynth F1-F3 | synthetic B1-B4 error c | probe 12 w_label | probe 11 F1 slope | "
          "women's cp F1 bias Hz")
    runs = [r for r in dict.fromkeys([run(a, s) for a in ("mlp", "film", *ARMS) for s in SEEDS]) if trained(r)]
    for r in runs:
        sc = D.scale(r)
        print(f"  {r:22s} {np.round(D.syn_f(r), 1)} | {np.round(D.resynth_f(r), 1)} | {np.round(D.bw_err(r), 0)} | "
              f"{np.round(D.w_label(r), 2)} | {', '.join(f'{k} {v:.3f}' for k, v in sc.items())} | "
              f"{float(np.median((d[r][:, 0] - d['cp'][:, 0])[dev & (d['sex'] == 'f') & np.isfinite(d['cp'][:, 0]) & np.isfinite(d[r][:, 0])])) if r in d else float('nan'):+.1f}")
    print()

    def rule(arm, seeds):
        mf = np.mean([D.syn_f(run("mlp", s)).mean() for s in seeds])
        af = np.mean([D.syn_f(run(arm, s)).mean() for s in seeds])
        sa = {k: np.mean([D.scale(run(arm, s))[k] for s in seeds]) for k in ("hillenbrand", "ptdb_tug")}
        sm = {k: np.mean([D.scale(run("mlp", s))[k] for s in seeds]) for k in ("hillenbrand", "ptdb_tug")}
        c1 = af <= 1.15 * mf
        c2 = all(abs(1 - sa[k]) <= 0.5 * abs(1 - sm[k]) for k in sa)
        dv = np.mean([D.paired(run("mlp", s), run(arm, s)) for s in seeds], 0)
        rs = float(np.max(np.mean([D.resynth_f(run(arm, s)) for s in seeds], 0)
                          / np.mean([D.resynth_f(run("mlp", s)) for s in seeds], 0) - 1))
        guard = dv[0] <= 4 and dv[1] <= 9 and dv[2] <= 20 and rs <= 0.05
        return c1 and c2 and guard, {"mae ratio": af / mf, "scale": sa, "scale mlp": sm, "c1": c1, "c2": c2, "dev": dv,
                                     "resynth worst": rs, "guard": guard}

    for arm in ARMS:
        print(f"arm {arm}")
        if (O / f"futility_stop_{arm}").exists():
            f1, m1 = D.syn_f(run(arm, 1)), D.syn_f(run("mlp", 1))
            print(f"  futility stop: seed 1's synthetic F1-F4 MAE {f1.mean():.1f} Hz against mlp seed 1's {m1.mean():.1f} "
                  f"({f1.mean() / m1.mean():.2f} x, stop above 1.5 x); seed 2 not trained")
            print(f"  DECISION ({arm}): not supported (futility stop)\n")
            continue
        seeds = [s for s in SEEDS if trained(run(arm, s))]
        ok, v = rule(arm, seeds)
        per = {s: rule(arm, [s])[0] for s in seeds}
        print(f"  (1) synthetic F1-F4 MAE / mlp's: {v['mae ratio']:.2f} (<= 1.15) -> {v['c1']}")
        print(f"  (2) probe 11 F1 slope, Hillenbrand / PTDB-TUG: {v['scale']['hillenbrand']:.3f} / {v['scale']['ptdb_tug']:.3f} "
              f"against mlp's {v['scale mlp']['hillenbrand']:.3f} / {v['scale mlp']['ptdb_tug']:.3f} (error at most half) -> {v['c2']}")
        print(f"  guards: Hillenbrand F1/F2/F3 {', '.join(f'{x:+.1f}' for x in v['dev'])} Hz; resynthesised F1-F3 worst "
              f"{100 * v['resynth worst']:+.1f} % -> {'pass' if v['guard'] else 'FAIL'}")
        for s in seeds:
            if run(arm, s) in d:
                p, lo, hi = cp_bias_diff(run(arm, s), run("mlp", s))
                print(f"  reported, seed {s}: women's closed-phase F1 bias, arm - mlp {p:+.1f} Hz [{lo:+.1f}, {hi:+.1f}]")
        print(f"  the rule on each seed alone: {per}")
        robust = len(set(per.values())) == 1
        print(f"  DECISION ({arm}): " + ("supported" if ok else "not supported")
              + ("" if robust else " (NOT ROBUST TO THE SEED)") + "\n")
    print("DECISION: see each arm above")


if __name__ == "__main__":
    main()
