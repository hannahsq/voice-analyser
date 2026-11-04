"""D-080 decision (registered 2026-11-04): a glottal-opening head on v3.1's final stage with coupled audio in the mix.
A0 = the coupled set only, A1 = A0 + the opening head; branch seeds 1 and 2 (v3.1 = d040_5b_s{1,2}_b20k, the
unchanged reference). On val_v2ko (400-frame crops):
  (1) learned if A1's opening R² >= 0.5 on voiced frames (seed mean), not learned if <= 0.3, else partly;
  (2) used if A1's F1 glottal slope exceeds A0's by >= 0.2 with the crop-bootstrap interval of the difference above 0,
      not used if the difference is < 0.1, else partly.
Both seeds the same way. Guards (A1 against A0): train_v2 validation F1-F4 MAE within +3 %; signed F1 on uncoupled
train_v2 validation within +-3 Hz. Reported: opening R² by crop length and by the crop's F1 spread; v3.1; the
resynthesised sets; Hillenbrand / PTDB-TUG / KL from the chain's other outputs.

    python decide_d080.py [windowed scratch dir]
"""

import json
import sys
from pathlib import Path

import numpy as np

C, O = Path("cache/d080"), Path("reports/analysis/d080")
SEEDS = (1, 2)
RUNS = {"A0": "d080_a0_s{}", "A1": "d080_a1_s{}", "v3.1": "d040_5b_s{}_b20k"}


def ok(z, name):
    return z[f"{name}|son"].astype(bool)


def gap_split(z0):
    """The label gap's formant-predictable part, fitted once (half the crops) and applied to every model."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    c, a = z0["v2ko400:formants|ref"], z0["v2ko400:formants_acoustic|ref"]
    item = z0["v2ko400:formants|item"]
    m = ok(z0, "v2ko400:formants") & np.isfinite(c[:, 0]) & np.isfinite(a[:, 0]) & np.all(np.isfinite(a[:, :4]), 1)
    g = a[:, 0] - c[:, 0]
    fit_on = m & (item % 2 == 0)
    f = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(a[fit_on, :4], g[fit_on])
    use = m & (item % 2 == 1)
    gp = np.full(len(g), np.nan)
    gp[use] = f.predict(a[use, :4])
    return use, g, gp, item


def glottal_slope(z, use, g, gp, item, n_boot=500):
    a = z["v2ko400:formants_acoustic|ref"][:, 0]
    p = z["v2ko400:formants|pred"][:, 0]
    m = use & np.isfinite(p)
    corr, rest = (a - p)[m], (g - gp)[m]
    it = item[m]

    def slope(idx):
        x, y = rest[idx] - rest[idx].mean(), corr[idx] - corr[idx].mean()
        return float(x @ y / (x @ x))

    crops = np.unique(it)
    by = {c: np.flatnonzero(it == c) for c in crops}
    rng = np.random.default_rng(0)
    boot = [rng.choice(crops, len(crops)) for _ in range(n_boot)]
    return slope(np.arange(len(rest))), boot, by, slope


def r2(y, p):
    return float(1 - np.var(y - p) / np.var(y))


def opening_r2(z, name):
    y, p = z[f"{name}|op_ref"], z[f"{name}|op"]
    m = ok(z, name) & np.isfinite(y) & np.isfinite(p)
    return r2(y[m], p[m]) if m.sum() > 1 else float("nan")


def last_eval(run):
    rows = [json.loads(ln) for ln in open(f"runs/{run}/log.jsonl") if '"eval"' in ln or "F1_mae_hz" in ln]
    rows = [r for r in rows if "F1_mae_hz" in r.get("eval", r)]
    return rows[-1].get("eval", rows[-1]) if rows else None


if __name__ == "__main__":
    Z = {(arm, s): dict(np.load(C / f"{RUNS[arm].format(s)}.npz")) for arm in RUNS for s in SEEDS}
    use, g, gp, item = gap_split(Z["A0", 1])
    L = ["D-080: a glottal-opening head on v3.1's final stage, coupled audio in the mix (val_v2ko, 400-frame crops)", ""]
    # (1) learned
    r = {s: opening_r2(Z["A1", s], "v2ko400:formants") for s in SEEDS}
    rm = float(np.mean(list(r.values())))
    v1 = lambda x: "learned" if x >= 0.5 else "not learned" if x <= 0.3 else "partly"  # noqa: E731
    L.append(f"(1) A1 opening R² by seed {', '.join(f'{s}: {v:.3f}' for s, v in r.items())}; mean {rm:.3f} -> {v1(rm)}"
             f"{'' if len({v1(v) for v in r.values()}) == 1 else ' (NOT ROBUST TO THE SEED)'}")
    L.append("    reported, by crop length (frames 400 / 100 / 50), seed means: "
             + " / ".join(f"{np.mean([opening_r2(Z['A1', s], f'v2ko{t}:formants') for s in SEEDS]):.3f}" for t in (400, 100, 50)))
    # by the crop's F1 spread (10-90 % range of the closed label, cents), 400-frame crops, terciles
    z = Z["A1", 1]
    name = "v2ko400:formants"
    f1, it = z[f"{name}|ref"][:, 0], z[f"{name}|item"]
    spread = {}
    for c in np.unique(it):
        x = f1[(it == c) & ok(z, name) & np.isfinite(f1)]
        spread[c] = 1200 * np.log2(np.percentile(x, 90) / np.percentile(x, 10)) if len(x) > 20 else np.nan
    sp = np.array([spread[c] for c in it])
    qs = np.nanpercentile(list(spread.values()), [33.3, 66.7])
    parts = []
    for lo, hi in ((-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)):
        vals = []
        for s in SEEDS:
            zz = Z["A1", s]
            m = ok(zz, name) & np.isfinite(zz[f"{name}|op_ref"]) & (sp > lo) & (sp <= hi)
            vals.append(r2(zz[f"{name}|op_ref"][m], zz[f"{name}|op"][m]))
        parts.append(f"{np.mean(vals):.3f}")
    L.append(f"    reported, by the crop's F1 spread terciles (cut at {qs[0]:.0f} / {qs[1]:.0f} cents), seed means: {' / '.join(parts)}")
    # (2) used
    sl = {}
    for arm in RUNS:
        for s in SEEDS:
            sl[arm, s] = glottal_slope(Z[arm, s], use, g, gp, item)
    L.append("")
    L.append("(2) F1 glottal slope (1 = full recovery of the closed-glottis F1's glottal part):")
    for arm in RUNS:
        L.append(f"    {arm:5s} " + ", ".join(f"s{s} {sl[arm, s][0]:+.3f}" for s in SEEDS))
    verdicts = []
    for s in SEEDS:
        (b1, boot, by, f1_), (b0, _, by0, f0_) = sl["A1", s], sl["A0", s]
        diffs = [f1_(np.concatenate([by[c] for c in pick if c in by])) - f0_(np.concatenate([by0[c] for c in pick if c in by0]))
                 for pick in boot]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        d = b1 - b0
        v = "used" if d >= 0.2 and lo > 0 else "not used" if d < 0.1 else "partly"
        verdicts.append(v)
        L.append(f"    seed {s}: A1 - A0 {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
    # guards
    L.append("")
    L.append("guards (A1 against A0): train_v2 validation F1-F4 MAE (log's last eval), signed F1 on uncoupled train_v2 val")
    gfail = []
    for s in SEEDS:
        e1, e0 = last_eval(RUNS["A1"].format(s)), last_eval(RUNS["A0"].format(s))
        ch = [100 * (e1[f"F{k}_mae_hz"] / e0[f"F{k}_mae_hz"] - 1) for k in (1, 2, 3, 4)]
        sg = {arm: float(np.median((Z[arm, s]["v2:formants|pred"] - Z[arm, s]["v2:formants|ref"])[ok(Z[arm, s], "v2:formants"), 0]))
              for arm in RUNS}
        fails = [f"F{k} +{c:.1f} %" for k, c in zip((1, 2, 3, 4), ch) if c > 3] + ([f"offset {sg['A1'] - sg['A0']:+.1f} Hz"]
                                                                                if abs(sg["A1"] - sg["A0"]) > 3 else [])
        gfail += fails
        L.append(f"    seed {s}: MAE change {', '.join(f'{c:+.1f} %' for c in ch)}; signed F1 A1 {sg['A1']:+.1f}, A0 {sg['A0']:+.1f}, "
                 f"v3.1 {sg['v3.1']:+.1f} Hz" + (f"  FAIL: {'; '.join(fails)}" if fails else ""))
    L.append("reported: resynthesised validation sets, F1 MAE / signed median, seed means (A0 / A1 / v3.1)")
    for nm in ("vocalset_val_d16k:formants", "zeroth_korean_val_d16k:formants"):
        cells = []
        for arm in RUNS:
            e = [(Z[arm, s][f"{nm}|pred"] - Z[arm, s][f"{nm}|ref"])[ok(Z[arm, s], nm), 0] for s in SEEDS]
            e = [x[np.isfinite(x)] for x in e]
            cells.append(f"{np.mean([np.abs(x).mean() for x in e]):.1f} ({np.mean([np.median(x) for x in e]):+.1f})")
        L.append(f"    {nm.split(':')[0]:24s} " + " / ".join(cells))
    S = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if S is not None and (S / "ptdb_egg_sub.npz").exists():
        d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
        tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
        dev = np.array([tiers.get(f"ptdb_tug:{x}") == "dev" for x in d["speaker"]])
        L.append("reported: PTDB-TUG dev closed-phase F1 bias, median Hz (women / men), seed means")
        for arm in RUNS:
            names = [RUNS[arm].format(s) for s in SEEDS]
            if not all(n in d for n in names):
                L.append(f"    {arm}: not scored")
                continue
            b = [np.mean([np.nanmedian(d[n][dev & (d["sex"] == sx) & np.isfinite(d["cp"][:, 0]), 0]
                                       - d["cp"][dev & (d["sex"] == sx) & np.isfinite(d["cp"][:, 0]), 0]) for n in names]) for sx in "fm"]
            L.append(f"    {arm:5s} {b[0]:+.1f} / {b[1]:+.1f}")
    hs = O / "hillenbrand_signed.txt"
    if hs.exists():
        L.append("reported: Hillenbrand dev signed error: see hillenbrand_signed.txt")
    v2 = "used" if all(v == "used" for v in verdicts) else "not used" if all(v == "not used" for v in verdicts) else (
        "partly" if len(set(verdicts)) == 1 else "partly (NOT ROBUST TO THE SEED)")
    L += ["", f"DECISION: (1) {v1(rm)}; (2) {v2}" + (f"; GUARD FAILED: {'; '.join(gfail)}" if gfail else "")]
    O.mkdir(parents=True, exist_ok=True)
    (O / "decide_d080.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
