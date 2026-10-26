"""D-034, reported only: the 10 ms ripple check (pre-registered 2026-10-26). Reads subhop.py's per-crop R^2 for the arms
base, edge and clock (VA_ARM) and applies the readings: P = peak-to-peak of R^2 averaged by position inside the 10 ms
hop (sub-offsets 0-8 ms, each over the eight hops and the crops); ratios edge / base and clock / edge with 95 % paired
bootstrap intervals over crops (1,000 resamples); the share of variation at the 10 ms period; P by median voiced f0.
Args: the three output prefixes (base, edge, clock). Prints the report."""
import csv
import sys
from collections import defaultdict

import numpy as np

PRE = dict(zip(("base", "edge", "clock"), sys.argv[1:4]))
B = 1000


def load(prefix):
    d = defaultdict(dict)  # (model, set) -> crop -> 40 values
    for r in csv.DictReader(open(f"{prefix}_crops.csv", encoding="utf-8")):
        d[(r["model"], r["set"])].setdefault(int(r["crop"]), np.zeros(40))[int(r["offset_ms"]) // 2] = float(r["r2"])
    f0 = {(r["set"], int(r["crop"])): float(r["f0_hz"]) for r in csv.DictReader(open(f"{prefix}_f0.csv", encoding="utf-8"))}
    return {k: (sorted(v), np.stack([v[c] for c in sorted(v)])) for k, v in d.items()}, f0


def ptp(y):  # y: (crops, 40) -> P
    by = y.mean(0).reshape(8, 5).mean(0)
    return float(by.max() - by.min())


def share10(y):
    m = y.mean(0)
    sp = np.abs(np.fft.rfft(m - m.mean())) ** 2
    return float(sp[8] / sp[1:].sum())


def verdict(lo, hi, low, high):
    return low if hi < 0.5 else high if lo > 0.75 else "partly"


A = {a: load(p) for a, p in PRE.items()}
rng = np.random.default_rng(2026)
calls = defaultdict(list)
for key in sorted(A["base"][0]):
    crops = A["base"][0][key][0]
    assert all(A[a][0][key][0] == crops for a in A), "arms must hold the same crops"
    Y = {a: A[a][0][key][1] for a in A}
    print(f"== {key[0]} on {key[1]} ({len(crops)} crops)")
    for a in A:
        by = Y[a].mean(0).reshape(8, 5).mean(0)
        print(f"  {a:5s} P {ptp(Y[a]):.4f}  10 ms share {share10(Y[a]):.2f}  by sub-offset 0,2,4,6,8 ms: {' '.join(f'{v - by.mean():+.4f}' for v in by)}")
    idx = rng.integers(0, len(crops), (B, len(crops)))
    for num, den, low, high in (("edge", "base", "padding accounts for most", "not the padding"), ("clock", "edge", "locked to the clip's clock", "not clip-locked")):
        r = ptp(Y[num]) / ptp(Y[den])
        bs = np.array([ptp(Y[num][i]) / ptp(Y[den][i]) for i in idx])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        v = verdict(lo, hi, low, high)
        calls[num].append(v)
        print(f"  {num} / {den}: {r:.2f} [{lo:.2f}, {hi:.2f}] -> {v}")
    f0 = np.array([A["base"][1].get((key[1], c), np.nan) for c in crops])
    for lo, hi in ((0, 250), (250, 400), (400, 1e9)):
        k = (f0 >= lo) & (f0 < hi)
        if k.sum() >= 10:
            print(f"  f0 {lo}-{hi if hi < 1e9 else ''} Hz ({k.sum()} crops): P " + ", ".join(f"{a} {ptp(Y[a][k]):.4f}" for a in A))
print("\n== Overall")
e, c = set(calls["edge"]), set(calls["clock"])
print(f"  edge / base: {sorted(e)}; clock / edge: {sorted(c)}")
if e == {"padding accounts for most"}:
    print("  -> (1) an artefact of the zero padding")
elif c == {"locked to the clip's clock"}:
    print("  -> (2) structure locked to the clip's 10 ms frame clock")
elif e <= {"not the padding", "partly"} and c <= {"not clip-locked", "partly"} and "padding accounts for most" not in e and "locked to the clip's clock" not in c:
    print("  -> (3) neither ratio falls below 0.5 anywhere" if not (e | c) & {"padding accounts for most", "locked to the clip's clock"} else "  -> mixed")
else:
    print("  -> mixed")
