"""D-034, reported only: readings for the ripple check's follow-ups (pre-registered 2026-10-26, amended before results).
  jitter <dir> <prefix stem>: arms regen, jit0, jit2, hnr0, whisper (subhop.py VA_ARM, v4 only). Ripple P (ripple_arms.py)
      on crops with median voiced f0 >= 250 Hz and on all crops; jit2 / jit0 and hnr0 / regen with 95 % paired bootstrap
      intervals over crops; whisper against regen unpaired; Spearman correlation of each crop's own P with its clip's
      drawn jitter (clips regenerated for their metadata; crop i is val clip i).
  step <dir> <prefix stem>: arms end0, end40, endR. R^2 by offset with the 10 ms pattern removed; the step at offset o is
      the mean of the four offsets from o minus the four before (cyclic over the 80 ms); o* the most negative.
Prints the report."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 1000
MODE, DIR, STEM = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(2026)


def load(arm):
    d = defaultdict(dict)
    for r in csv.DictReader(open(DIR / f"{STEM}_{arm}_crops.csv", encoding="utf-8")):
        if r["model"] == "v4":
            d[r["set"]].setdefault(int(r["crop"]), np.zeros(40))[int(r["offset_ms"]) // 2] = float(r["r2"])
    f0 = {(r["set"], int(r["crop"])): float(r["f0_hz"]) for r in csv.DictReader(open(DIR / f"{STEM}_{arm}_f0.csv", encoding="utf-8"))}
    return {s: (sorted(v), np.stack([v[c] for c in sorted(v)])) for s, v in d.items()}, f0


def ptp(y):
    by = y.mean(0).reshape(8, 5).mean(0)
    return float(by.max() - by.min())


def ratio(a, b, paired=True):
    r = ptp(a) / ptp(b)
    if paired:
        idx = rng.integers(0, len(a), (B, len(a)))
        bs = [ptp(a[i]) / ptp(b[i]) for i in idx]
    else:
        bs = [ptp(a[rng.integers(0, len(a), len(a))]) / ptp(b[rng.integers(0, len(b), len(b))]) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return r, lo, hi


def call(lo, hi, low, high):
    return low if hi < 0.5 else high if lo > 0.75 else "partly"


if MODE == "jitter":
    A = {a: load(a) for a in ("regen", "jit0", "jit2", "hnr0", "whisper")}
    base, base_f0 = load("base") if (DIR / f"{STEM}_base_crops.csv").exists() else (None, None)
    for s in sorted(A["regen"][0]):
        crops = A["regen"][0][s][0]
        f0 = np.array([A["regen"][1][(s, c)] for c in crops])
        hi_f0 = f0 >= 250
        Y = {a: A[a][0][s][1] for a in A}
        print(f"== v4 on {s} ({len(crops)} crops, {hi_f0.sum()} with median f0 >= 250 Hz)")
        if base is not None and s in base and base[s][0] == crops:
            print(f"  regen reproduces base: max |R2 difference| {np.abs(Y['regen'] - base[s][1]).max():.5f}")
        for lab, k in (("f0 >= 250 Hz", hi_f0), ("all crops", np.ones(len(crops), bool))):
            print(f"  {lab}: P " + ", ".join(f"{a} {ptp(Y[a][k]):.4f}" for a in A if a != "whisper") + f", whisper (all crops) {ptp(Y['whisper']):.4f}")
            for num, den, low, high in (("jit2", "jit0", "jitter removes it (hop-pulse locking)", "not jitter"),
                                        ("hnr0", "regen", "aspiration removes it", "not aspiration")):
                r, lo, hi = ratio(Y[num][k], Y[den][k])
                print(f"    {num} / {den}: {r:.2f} [{lo:.2f}, {hi:.2f}] -> {call(lo, hi, low, high)}")
        r, lo, hi = ratio(Y["whisper"], Y["regen"], paired=False)
        print(f"  whisper / regen, all crops, unpaired: {r:.2f} [{lo:.2f}, {hi:.2f}] (reported, no reading)")
        # each crop's own ripple against its clip's drawn jitter
        from scipy.stats import spearmanr

        from vocal_analysis.synth.sampler import SynthConfig, generate
        from vocal_analysis.train.data import split_ids
        name = "train_v2" if s == "held_synth_v2" else "train_v2s"
        ix = json.loads((Path("data/synth") / name / "index.json").read_text())
        cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ix["synth_config"].items()})
        val = split_ids(ix["n_clips"])[1][:512]
        jit = np.array([generate(int(val[c]), cfg, ix["seed"]).meta["jitter"] for c in crops])
        own = np.array([ptp(Y["regen"][[i]]) for i in range(len(crops))])
        for lab, k in (("f0 >= 250 Hz", hi_f0), ("all crops", np.ones(len(crops), bool))):
            rho = spearmanr(own[k], jit[k])[0]
            idx = rng.integers(0, k.sum(), (B, k.sum()))
            bs = [spearmanr(own[k][i], jit[k][i])[0] for i in idx]
            print(f"  Spearman(own P, drawn jitter), {lab}: {rho:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")

elif MODE == "step":
    def steps(y):
        m = y.mean(0)
        res = m - np.tile(m.reshape(8, 5).mean(0), 8)
        return np.array([res[(np.arange(k, k + 4)) % 40].mean() - res[(np.arange(k - 4, k)) % 40].mean() for k in range(40)])
    A = {a: load(a)[0] for a in ("end0", "end40", "endR")}
    for s in sorted(A["end0"]):
        st = {a: steps(A[a][s][1]) for a in A}
        o = {a: 2 * int(np.argmin(st[a])) for a in A}
        pred = (o["end0"] + 40) % 80
        d = lambda a, b: min(abs(a - b), 80 - abs(a - b))  # noqa: E731
        v = "tied to the input length" if d(o["end40"], pred) <= 4 else "not the input length" if d(o["end40"], o["end0"]) <= 4 else "unclear"
        print(f"== v4 on {s}")
        for a in A:
            print(f"  {a:5s} o* {o[a]:2d} ms, step {st[a].min() * 1000:+.1f}e-3   steps by offset (x1e-3): {' '.join(f'{x * 1000:+.1f}' for x in st[a])}")
        print(f"  end40's o* {o['end40']} ms against predicted {pred} ms (end0 {o['end0']} ms) -> {v}")
        print(f"  endR's largest step {st['endR'].min() * 1000:+.1f}e-3 against end0's {st['end0'].min() * 1000:+.1f}e-3")
