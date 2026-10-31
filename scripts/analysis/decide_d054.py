"""D-054 decision (pre-registered 2026-10-31): the frequency-resolved read-out ("film") against MLP heads ("mlp") on
teacher v3.1's frozen trunk, seed means of head seeds 1 and 2. Supported if (A) or (B) holds and no guard fails:
(A) synthetic F1-F4 MAE (each run's last "eval") lower by >= 5 % (mean relative change), both film seeds better than both
    mlp seeds on that mean;
(B) synthetic bandwidth error (median |cents|, D-053's synthetic frames) lower by >= 10 % (mean over B1-B4) and probe 12's
    arm C weight on the label up by >= 0.05 (mean over B1-B4), both on both seeds;
guards (film - mlp, seed means): Hillenbrand dev F1 <= +4, F2 <= +9, F3 <= +20 Hz (`paired/w400/d054_mlp_s*_vs_d054_film_s*`);
women's closed-phase F1 |error| <= +5 Hz (as decide_d051.py); resynthesised validation F1-F3 MAE no more than 5 % worse.
Amendment 1 (2026-10-31): a futility stop (film seed 1 over 1.5 x mlp seed 1's synthetic
F1-F4 MAE) ends it as "not supported (futility stop)". Reported: the flat arm against mlp and against v3.1's own heads, probe 11's slopes, bandwidth errors on the resynthesised
sets, head-seed spreads.

    python decide_d054.py <windowed scratch dir>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
P = Path("reports/analysis/paired/w400")
C = Path("cache/d053")
TRUNK = "d040_5b_s2_b20k"
SEEDS = (1, 2)
RESYNTH = ("eval_resynth:zeroth_korean_val_d16k", "eval_resynth:vocalset_val_d16k")
tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]


def run(arm, s):
    return f"d054_{arm}_s{s}"


def last(r, key="eval"):
    rows = [json.loads(x)[key] for x in (Path("runs") / r / "log.jsonl").read_text().splitlines() if f'"{key}"' in x]
    return rows[-1]


def syn_f(r):
    ev = last(r)
    return np.array([ev[f"F{n}_mae_hz"] for n in range(1, 5)])


def resynth_f(r):
    return np.array([[last(r, k)[f"F{n}_mae_hz"] for n in range(1, 4)] for k in RESYNTH]).mean(0)


def bw_err(r, name="synthetic"):
    """Median |cents| of predicted against true B1-B4 on D-053's frames."""
    with np.load(C / f"{name}_truth.npz", allow_pickle=True) as t, np.load(C / f"{name}_{r}.npz") as m:
        e = 1200 * np.abs(np.log2(m["pB"] / t["B"]))
    return np.nanmedian(e, 0)


def w_label(r):
    a = json.loads(Path(f"reports/probes/bw_evidence/{r}.json").read_text(encoding="utf-8"))["C"]
    return np.array([a[f"B{n}"]["w_label"] for n in range(1, 5)])


def scale(r):
    a = json.loads(Path(f"reports/probes/scale/{r}.json").read_text(encoding="utf-8"))
    return {k: a[f"{k}/slope/F1"]["point"] for k in ("hillenbrand", "ptdb_tug") if f"{k}/slope/F1" in a}


def paired(a, b):
    t = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    g = lambda key: float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", t).group(1))  # noqa: E731
    return np.array([g("F1 MAE all"), g("F2 MAE all"), g("F3 MAE all")])


def main():
    S = Path(sys.argv[1])
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])

    def cp_women(r, ref):
        m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[r][:, 0]) & np.isfinite(d[ref][:, 0])
        return float(np.median(np.abs(d[r][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    def cp_bias(r):
        m = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.isfinite(d[r][:, 0])
        return float(np.median(d[r][m, 0] - d["cp"][m, 0]))

    print("D-054: frequency-resolved read-out (film) against MLP heads (mlp) on teacher v3.1's frozen trunk, head seeds 1, 2\n")
    print("per run: synthetic F1-F4 MAE Hz | resynth F1-F3 MAE Hz | synthetic B1-B4 error c | probe 12 w_label B1-B4 | "
          "probe 11 F1 slope | women's cp F1 bias Hz")
    have_cp = TRUNK in d
    runs = [TRUNK] + [run(a, s) for a in ("flat", "mlp", "film") for s in SEEDS if (Path("runs") / run(a, s) / "ckpt_008000.pt").exists()]
    for r in runs:
        sc = scale(r) if Path(f"reports/probes/scale/{r}.json").exists() else {}
        wl = w_label(r) if Path(f"reports/probes/bw_evidence/{r}.json").exists() else None
        print(f"  {r:18s} {np.round(syn_f(r), 1)} | {np.round(resynth_f(r), 1)} | {np.round(bw_err(r), 0)} | "
              f"{np.round(wl, 2) if wl is not None else '-'} | {', '.join(f'{k} {v:.3f}' for k, v in sc.items())} | "
              f"{cp_bias(r) if r in d else float('nan'):+.1f}")
    for name in ("zeroth", "vocalset"):
        print(f"  bandwidth error c, {name}: " + "; ".join(f"{r} {np.round(bw_err(r, name), 0)}" for r in
                                                       runs))
    print()
    if Path("reports/analysis/d054/futility_stop").exists():  # amendment 1
        f1, m1 = syn_f(run("film", 1)), syn_f(run("mlp", 1))
        print(f"amendment 1: film seed 1's synthetic F1-F4 MAE {f1.mean():.1f} Hz against mlp seed 1's {m1.mean():.1f} "
              f"({f1.mean() / m1.mean():.2f} x, stop above 1.5 x); film seed 2 not trained
")
        print("DECISION: not supported (futility stop)")
        return

    rel = {s: float(np.mean(syn_f(run("film", s)) / syn_f(run("mlp", s)) - 1)) for s in SEEDS}
    mean_f = {a: np.mean([syn_f(run(a, s)) for s in SEEDS], 0) for a in ("film", "mlp")}
    rel_mean = float(np.mean(mean_f["film"] / mean_f["mlp"] - 1))
    both = max(np.mean(syn_f(run("film", s))) for s in SEEDS) < min(np.mean(syn_f(run("mlp", s))) for s in SEEDS)
    A = rel_mean <= -0.05 and both
    print(f"(A) synthetic F1-F4 MAE, film / mlp - 1 (seed means): {100 * rel_mean:+.1f} % (per seed "
          f"{', '.join(f'{100 * v:+.1f} %' for v in rel.values())}); both film seeds better than both mlp: {both} -> {A}")
    bw_rel = {s: float(np.mean(bw_err(run("film", s)) / bw_err(run("mlp", s)) - 1)) for s in SEEDS}
    wl_d = {s: float(np.mean(w_label(run("film", s)) - w_label(run("mlp", s)))) for s in SEEDS}
    B = all(v <= -0.10 for v in bw_rel.values()) and all(v >= 0.05 for v in wl_d.values())
    print(f"(B) synthetic bandwidth error, film / mlp - 1: {', '.join(f'{100 * v:+.1f} %' for v in bw_rel.values())}; "
          f"probe 12 w_label change: {', '.join(f'{v:+.3f}' for v in wl_d.values())} -> {B}")

    dev_d = np.mean([paired(run("mlp", s), run("film", s)) for s in SEEDS], 0)
    cp_d = float(np.mean([cp_women(run("film", s), run("mlp", s)) for s in SEEDS])) if have_cp else float("nan")
    rs = float(np.max(np.mean([resynth_f(run("film", s)) for s in SEEDS], 0) / np.mean([resynth_f(run("mlp", s)) for s in SEEDS], 0) - 1))
    guard = dev_d[0] <= 4 and dev_d[1] <= 9 and dev_d[2] <= 20 and cp_d <= 5 and rs <= 0.05
    print(f"guards (film - mlp): Hillenbrand F1/F2/F3 {dev_d[0]:+.1f} / {dev_d[1]:+.1f} / {dev_d[2]:+.1f} Hz; women's cp F1 "
          f"|error| {cp_d:+.1f} Hz; resynthesised F1-F3 worst {100 * rs:+.1f} % -> {'pass' if guard else 'FAIL'}")
    flat_dev = np.mean([paired(run("mlp", s), run("flat", s)) for s in SEEDS], 0)
    print(f"reported: flat - mlp Hillenbrand F1/F2/F3 {flat_dev[0]:+.1f} / {flat_dev[1]:+.1f} / {flat_dev[2]:+.1f} Hz")
    print()
    print("DECISION: " + ("frequency-resolved read-out supported; a full-size registration follows" if (A or B) and guard
                          else "not supported on v3.1's frozen trunk" + ("" if guard else " (a guard failed)")))


if __name__ == "__main__":
    main()
