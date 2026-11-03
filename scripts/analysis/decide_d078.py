"""D-078 decision (registered 2026-11-04): can the closed-glottis F1 be recovered from coupled phonation? Two heads on
teacher v3.1's frozen trunk, trained on train_v2k against the closed-glottis ("closed") or cycle-mean acoustic
("acoustic") label, head seeds 1 and 2. A = the acoustic head's F1 MAE against the closed label on val_v2k, C = the
closed head's; seed means, both seeds the same way: separable if C <= 0.7 A, not separable if C >= 0.9 A, else partly.
Reported: each head against both labels; the uncoupled and resynthesised validation sets; F2-F4; PTDB-TUG dev women's
and men's closed-phase F1 bias; KL (D-060's files) against the label and D-077's response.

    python decide_d078.py <windowed scratch dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

SEEDS = (1, 2)
C_DIR, O = Path("cache/d078"), Path("reports/analysis/d078")


def run(label, s):
    return f"d078_{label}_s{s}"


def mae(z, name, k=0, signed=False):
    p, r, son = z[f"{name}|pred"][:, k], z[f"{name}|ref"][:, k], z[f"{name}|son"]
    m = son & np.isfinite(r) & np.isfinite(p)
    e = p[m] - r[m]
    return float(np.median(e)) if signed else float(np.mean(np.abs(e)))


def main():
    Z = {(lab, s): dict(np.load(C_DIR / f"{run(lab, s)}.npz")) for lab in ("closed", "acoustic") for s in SEEDS}
    L = ["D-078: can the closed-glottis F1 be recovered from coupled phonation? (frozen v3.1 trunk, heads on train_v2k)", ""]
    any_z = next(iter(Z.values()))
    gap = np.abs(any_z["v2k:formants_acoustic|ref"][:, 0] - any_z["v2k:formants|ref"][:, 0])
    gap = gap[any_z["v2k:formants|son"] & np.isfinite(gap)]
    L.append(f"label gap on val_v2k, |acoustic - closed| F1: median {np.median(gap):.1f} Hz, mean {gap.mean():.1f}")
    L.append("")
    L.append("F1 MAE on val_v2k, Hz (signed median in brackets): head -> against closed label | against acoustic label")
    for (lab, s), z in Z.items():
        L.append(f"  {run(lab, s):18s} {mae(z, 'v2k:formants'):6.1f} ({mae(z, 'v2k:formants', signed=True):+5.1f}) | "
                 f"{mae(z, 'v2k:formants_acoustic'):6.1f} ({mae(z, 'v2k:formants_acoustic', signed=True):+5.1f})")
    L.append("")
    per = {s: mae(Z["closed", s], "v2k:formants") / mae(Z["acoustic", s], "v2k:formants") for s in SEEDS}
    C = np.mean([mae(Z["closed", s], "v2k:formants") for s in SEEDS])
    A = np.mean([mae(Z["acoustic", s], "v2k:formants") for s in SEEDS])
    ratio = C / A
    verdict = lambda r: "separable" if r <= 0.7 else "not separable" if r >= 0.9 else "partly"  # noqa: E731
    L.append(f"C = {C:.1f} Hz, A = {A:.1f} Hz, C / A = {ratio:.3f} (seed means); per seed {', '.join(f'{s}: {r:.3f}' for s, r in per.items())}")
    same = len({verdict(r) for r in per.values()}) == 1
    L.append("")
    L.append("reported: F2-F4 MAE against the closed label on val_v2k (closed head / acoustic head), seed means")
    for k in (1, 2, 3):
        L.append(f"  F{k + 1}: {np.mean([mae(Z['closed', s], 'v2k:formants', k) for s in SEEDS]):.1f} / "
                 f"{np.mean([mae(Z['acoustic', s], 'v2k:formants', k) for s in SEEDS]):.1f}")
    L.append("reported: other sets (labels: closed-glottis = acoustic there), F1 MAE and signed median, seed means, closed / acoustic head")
    for name in [k[:-5] for k in any_z if k.endswith("|pred") and not k.startswith("v2k")]:
        c = [(mae(Z["closed", s], name), mae(Z["closed", s], name, signed=True)) for s in SEEDS]
        a = [(mae(Z["acoustic", s], name), mae(Z["acoustic", s], name, signed=True)) for s in SEEDS]
        L.append(f"  {name:28s} {np.mean([x[0] for x in c]):6.1f} ({np.mean([x[1] for x in c]):+5.1f}) / "
                 f"{np.mean([x[0] for x in a]):6.1f} ({np.mean([x[1] for x in a]):+5.1f})")
    S = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if S is not None and (S / "ptdb_egg_sub.npz").exists():
        d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
        tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
        dev = np.array([tiers.get(f"ptdb_tug:{x}") == "dev" for x in d["speaker"]])
        L.append("reported: PTDB-TUG dev closed-phase F1 bias, median Hz (women / men), seed means; v3.1 for scale")
        for lab, names in (("closed", [run("closed", s) for s in SEEDS]), ("acoustic", [run("acoustic", s) for s in SEEDS]),
                           ("v3.1", ["d040_5b_s1_b20k", "d040_5b_s2_b20k"])):
            if not all(n in d for n in names):
                L.append(f"  {lab}: not scored")
                continue
            b = []
            for sx in "fm":
                m = dev & (d["sex"] == sx) & np.isfinite(d["cp"][:, 0])
                b.append(np.mean([np.nanmedian(d[n][m, 0] - d["cp"][m, 0]) for n in names]))
            L.append(f"  {lab:8s} {b[0]:+.1f} / {b[1]:+.1f}")
    kl = O / "kl" / "main_per_file.tsv"
    if kl.exists():
        rows = [ln.split("\t") for ln in kl.read_text().splitlines()]
        head, rows = rows[0], rows[1:]
        L.append("reported: KL (D-060's files), F1 error against the closed label, median by gap 0 / 0.1 / 0.2 / 0.3 (men; women)")
        for lab in ("closed", "acoustic"):
            cols = [head.index(run(lab, s)) for s in SEEDS]
            cell = lambda sx, g: np.median([np.mean([float(r[c]) for c in cols]) for r in rows if r[0] == sx and float(r[1]) == g])  # noqa: E731
            L.append(f"  {lab:8s} " + "; ".join(" / ".join(f"{cell(sx, g):+.1f}" for g in (0.0, 0.1, 0.2, 0.3)) for sx in ("Male", "Female")))
        L.append("reported: the same against D-077's measured response (error - dE), median by gap (men; women)")
        for lab in ("closed", "acoustic"):
            cols = [head.index(run(lab, s)) for s in SEEDS]

            def vs_resp(sx, g):
                out = []
                for r in rows:
                    if r[0] != sx or float(r[1]) != g:
                        continue
                    tag = "".join(c + "^" if c.isupper() else c for c in r[2])
                    p = Path("cache/d077") / f"{sx}_{tag}_{r[3]}_{g:.1f}.npy"
                    if p.exists():
                        v = np.load(p)
                        out.append(np.mean([float(r[c]) for c in cols]) - (v[1] - v[4]))
                out = np.array(out)
                return np.nanmedian(out) if len(out) else np.nan

            L.append(f"  {lab:8s} " + "; ".join(" / ".join(f"{vs_resp(sx, g):+.1f}" for g in (0.0, 0.1, 0.2, 0.3)) for sx in ("Male", "Female")))
    L.append("")
    L.append(f"DECISION: {verdict(ratio)}" + ("" if same else " (NOT ROBUST TO THE SEED)"))
    O.mkdir(parents=True, exist_ok=True)
    (O / "decide_d078.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
