"""D-080, reported: KL (D-060's files) by arm. d060_kl_gap.py pools every model it is given, so the chain's
reports/analysis/d080/kl/main.txt is the mean of A0 and A1; this splits it, with v3.1 from D-060's own files, against
the closed label and against D-077's measured response (D-078's layout).

    python scripts/analysis/d080_kl_arms.py
"""

from pathlib import Path

import numpy as np

ARMS = {"A0": ("reports/analysis/d080/kl/main_per_file.tsv", ["d080_a0_s1", "d080_a0_s2"]),
        "A1": ("reports/analysis/d080/kl/main_per_file.tsv", ["d080_a1_s1", "d080_a1_s2"]),
        "v3.1": ("reports/analysis/d060/main_per_file.tsv", ["v31_s1", "v31_s2"])}
GAPS = (0.0, 0.1, 0.2, 0.3)


def table(path, names):
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()]
    head, rows = rows[0], rows[1:]
    cols = [head.index(n) for n in names]
    return {(r[0], float(r[1]), r[2], r[3]): np.mean([float(r[c]) for c in cols]) for r in rows}


def resp(sx, g, v, f0):
    tag = "".join(c + "^" if c.isupper() else c for c in v)
    p = Path("cache/d077") / f"{sx}_{tag}_{f0}_{g:.1f}.npy"
    if not p.exists():
        return np.nan
    x = np.load(p)
    return x[1] - x[4]


if __name__ == "__main__":
    T = {a: table(*v) for a, v in ARMS.items()}
    L = ["D-080, reported: KL F1 error by arm (seed means), median over files by gap 0 / 0.1 / 0.2 / 0.3; "
         "growth = paired gap 0.3 - 0, median [95 % bootstrap]", ""]
    rng = np.random.default_rng(0)
    for ref in ("closed label", "D-077 response (error - dE)"):
        L.append(f"against the {ref}:")
        for sx in ("Male", "Female"):
            for a, t in T.items():
                e = {k: v - (resp(*k) if ref != "closed label" else 0) for k, v in t.items() if k[0] == sx}
                lv = [np.nanmedian([v for k, v in e.items() if k[1] == g]) for g in GAPS]
                d = np.array([e[(sx, 0.3, k[2], k[3])] - e[k] for k in e if k[1] == 0.0 and (sx, 0.3, k[2], k[3]) in e])
                d = d[np.isfinite(d)]
                bs = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                L.append(f"  {sx:6s} {a:5s} " + " / ".join(f"{x:+.1f}" for x in lv)
                         + f"   growth {np.median(d):+.1f} [{lo:+.1f}, {hi:+.1f}] ({len(d)} pairs)")
        L.append("")
    out = Path("reports/analysis/d080/kl_by_arm.txt")
    out.write_text("\n".join(L))
    print("\n".join(L))
