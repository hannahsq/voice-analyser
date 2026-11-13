"""D-110: is D-109's upward F1 shift on real voices shaped like the open-phase (coupling) shift? Descriptive, CPU only.

The shift S = d109 seed mean - 5b seed mean (d040_5b_s{1,2}_b20k), paired per frame or token, read on two corpora:
- PTDB-TUG dev, D-091's frames (the windowed closed-phase cache, cache/m3w/ptdb_egg_sub.npz): per sex, F1-F3.
- Hillenbrand dev (the paired-score caches, cache/m3w/m2cache_sub): per group, F1-F3, and the signed bias against the
  whole-cycle reference before and after.
Shape: per speaker, OLS of S's F1 on the reference F1 (closed-phase F1 on PTDB-TUG, the label on Hillenbrand) and log f0;
the F1 coefficient in Hz per 100 Hz of F1, mean over speakers. On PTDB-TUG also the EGG CQ coefficient (D-076's model)
and S by closed-phase F1 band. Beside S, the same measures for the seed null (5b s1 - s2) and D-078's coupling-trained
pair (acoustic - closed heads, seed means), and on PTDB-TUG the references' own gaps.
Medians over frames or tokens, 95 % speaker bootstrap (2000).

    python scripts/analysis/d110_shift_signature.py > reports/analysis/d110/d110.txt
"""

import json
from pathlib import Path

import numpy as np

CACHE = Path("cache/m3w")
B, R = ("d040_5b_s1_b20k", "d040_5b_s2_b20k"), ("d109_s1", "d109_s2")
SEED = 5110


def boot(per, rng, n=2000):
    keys = list(per)
    bs = [np.median(np.concatenate([per[s] for s in rng.choice(keys, len(keys))])) for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def slopes(e, f1, f0, spk, extra=None):
    """Per-speaker OLS of e on [extra,] F1 and log f0; returns (mean F1 coefficient per 100 Hz, mean extra per 0.1)."""
    out_f, out_x = [], []
    for s in np.unique(spk):
        q = spk == s
        if q.sum() < 20:
            continue
        cols = ([extra[q]] if extra is not None else []) + [np.ones(q.sum()), f1[q], np.log(f0[q])]
        b = np.linalg.lstsq(np.column_stack(cols), e[q], rcond=None)[0]
        out_f.append(b[-2] * 100)
        if extra is not None:
            out_x.append(b[0] * 0.1)
    return float(np.mean(out_f)), (float(np.mean(out_x)) if out_x else np.nan)


def ptdb(lines, rng):
    d = dict(np.load(CACHE / "ptdb_egg_sub.npz", allow_pickle=False))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    mean = lambda ms: np.mean([d[m] for m in ms], 0)  # noqa: E731
    base = mean(B)
    pairs = {
        "S = d109 - 5b (seed means)": mean(R) - base,
        "  seed 1 alone": d[R[0]] - d[B[0]],
        "  seed 2 alone": d[R[1]] - d[B[1]],
        "seed null (5b s1 - s2)": d[B[0]] - d[B[1]],
        "D-078 acoustic - closed heads": mean(("d078_acoustic_s1", "d078_acoustic_s2")) - mean(("d078_closed_s1", "d078_closed_s2")),
        "ref: whole-cycle LPC - closed phase": d["cycle"] - d["cp"],
        "ref: Praat - closed phase": d["praat"] - d["cp"],
    }
    # D-091's frames: dev, all three references' F1 and both 5b and both d109 runs finite
    ok = dev & np.isfinite(d["cp"][:, 0]) & np.isfinite(d["cycle"][:, 0]) & np.isfinite(d["praat"][:, 0]) & (d["f0"] > 0)
    ok &= np.all(np.isfinite(base), 1) & np.all(np.isfinite(mean(R)), 1)
    lines.append("PTDB-TUG dev, D-091's frames (closed-phase cache); Hz")
    for sx, name in (("f", "women"), ("m", "men")):
        k = ok & (d["sex"] == sx)
        spk = np.unique(d["speaker"][k])
        lines.append(f"  {name}: {len(spk)} speakers, {int(k.sum())} frames; 5b's F1 - closed phase "
                     f"{np.median(base[k, 0] - d['cp'][k, 0]):+.1f}, d109's {np.median(mean(R)[k, 0] - d['cp'][k, 0]):+.1f}")
        for label, e in pairs.items():
            kk = k & np.all(np.isfinite(e), 1)
            per = {s: e[kk & (d["speaker"] == s), 0] for s in spk}
            lo, hi = boot(per, rng)
            kc = kk & np.isfinite(d["cq"])
            fs, _ = slopes(e[kk, 0], d["cp"][kk, 0], d["f0"][kk], d["speaker"][kk])
            _, cq = slopes(e[kc, 0], d["cp"][kc, 0], d["f0"][kc], d["speaker"][kc], d["cq"][kc])
            f123 = np.median(e[kk, :3], 0)
            lines.append(f"    {label:38s} F1 {f123[0]:+6.1f} [{lo:+.1f}, {hi:+.1f}]  F2 {f123[1]:+6.1f}  F3 {f123[2]:+6.1f}  "
                         f"F1 slope {fs:+5.2f} /100 Hz  CQ {cq:+5.2f} /0.1")
            if label.startswith("S ="):
                lines.append("      per speaker F1: " + ", ".join(f"{s} {np.median(per[s]):+.1f}" for s in spk))
                cells = []
                for lo_, hi_ in ((0, 450), (450, 600), (600, 2000)) if sx == "f" else ((0, 400), (400, 550), (550, 2000)):
                    q = kk & (d["cp"][:, 0] >= lo_) & (d["cp"][:, 0] < hi_)
                    cells.append(f"{lo_}-{hi_} Hz {np.median(e[q, 0]):+.1f} (n {int(q.sum())})")
                lines.append("      by closed-phase F1: " + "; ".join(cells))
    lines.append("")


def hillenbrand(lines, rng):
    from vocal_analysis.eval import tiers as T

    tm = T.load()
    L = {m: np.load(CACHE / "m2cache_sub" / f"{m}__hillenbrand.npz", allow_pickle=True) for m in B + R}
    ref, grp, spk, f0 = L[B[0]]["ref"], L[B[0]]["group"], L[B[0]]["speaker"], L[B[0]]["token_f0"]
    for m in L:
        assert np.array_equal(L[m]["ref"], ref, equal_nan=True)
    dev = np.array([T.tier_of(str(x), tm) == "dev" for x in spk])
    est = {m: L[m]["est"] for m in L}
    base, new = (est[B[0]] + est[B[1]]) / 2, (est[R[0]] + est[R[1]]) / 2
    pairs = {"S = d109 - 5b (seed means)": new - base, "  seed 1 alone": est[R[0]] - est[B[0]],
             "  seed 2 alone": est[R[1]] - est[B[1]], "seed null (5b s1 - s2)": est[B[0]] - est[B[1]]}
    lines.append("Hillenbrand dev (paired-score caches; tokens x frame kinds); Hz")
    for name, gs in (("women", ("women",)), ("men", ("men",)), ("children", ("boys", "girls"))):
        k = dev & np.isin(grp, gs) & np.isfinite(ref[:, 0]) & (f0 > 0)
        sp = np.unique(spk[k])
        lines.append(f"  {name}: {len(sp)} speakers, {int(k.sum())} rows; F1 - reference: 5b {np.median(base[k, 0] - ref[k, 0]):+.1f}, "
                     f"d109 {np.median(new[k, 0] - ref[k, 0]):+.1f}")
        for label, e in pairs.items():
            per = {s: e[k & (spk == s), 0] for s in sp}
            lo, hi = boot(per, rng)
            fs, _ = slopes(e[k, 0], ref[k, 0], f0[k], spk[k])
            f123 = np.nanmedian(e[k, :3], 0)
            lines.append(f"    {label:38s} F1 {f123[0]:+6.1f} [{lo:+.1f}, {hi:+.1f}]  F2 {f123[1]:+6.1f}  F3 {f123[2]:+6.1f}  "
                         f"F1 slope {fs:+5.2f} /100 Hz")
            if label.startswith("S ="):
                cells = []
                for lo_, hi_ in ((0, 450), (450, 650), (650, 3000)):
                    q = k & (ref[:, 0] >= lo_) & (ref[:, 0] < hi_)
                    cells.append(f"{lo_}-{hi_} Hz {np.median(e[q, 0]):+.1f} (n {int(q.sum())})")
                lines.append("      by reference F1: " + "; ".join(cells))
    lines.append("")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    lines = ["D-110: the shape of D-109's F1 shift on real voices (S = d109 - 5b, seed means unless noted). F1 slope: "
             "per-speaker OLS on the reference F1 and log f0, Hz of S per 100 Hz of F1, mean over speakers; CQ: D-076's "
             "model, Hz per 0.1 CQ. Medians, [95 % speaker bootstrap]", ""]
    ptdb(lines, rng)
    hillenbrand(lines, rng)
    print("\n".join(lines))
