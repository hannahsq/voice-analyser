"""D-076: does v3.1's real-voice F1 error (against the closed-phase reference) track the EGG closed quotient within
speakers? PTDB-TUG dev frames from the windowed closed-phase cache; per speaker, OLS of the error on CQ, closed-phase F1
and log f0; the CQ slope in Hz per 0.1 CQ; mean over the 12 dev speakers against a within-speaker permutation null.

    python scripts/analysis/d076_egg_closure.py <windowed scratch dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("reports/analysis/d076")
V31 = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
V3 = ("conformer_v2r2_s1_d16k",)
N_PERM, SEED = 2000, 5076


def main():
    d = dict(np.load(Path(sys.argv[1]) / "ptdb_egg_sub.npz", allow_pickle=True))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    base = dev & np.isfinite(d["cq"]) & np.isfinite(d["cp"][:, 0]) & (d["f0"] > 0)
    rng = np.random.default_rng(SEED)

    def setup(models, extra=None):
        e = np.mean([d[m][:, 0] for m in models], 0) - d["cp"][:, 0]
        ok = base & np.isfinite(e) & (np.ones_like(base) if extra is None else extra)
        groups = []
        for s in np.unique(d["speaker"][ok]):
            k = ok & (d["speaker"] == s)
            if k.sum() < 20:
                continue
            X = np.column_stack([np.ones(k.sum()), d["cp"][k, 0], np.log(d["f0"][k])])
            groups.append((s, d["sex"][k][0], e[k], d["cq"][k], X))
        return groups

    def slope(y, c, X):
        return float(np.linalg.lstsq(np.column_stack([c, X]), y, rcond=None)[0][0] * 0.1)

    def test(groups, sexes="fm"):
        g = [x for x in groups if x[1] in sexes]
        sl = np.array([slope(y, c, X) for _, _, y, c, X in g])
        null = np.array([np.mean([slope(y, rng.permutation(c), X) for _, _, y, c, X in g]) for _ in range(N_PERM)])
        p = float((np.sum(null <= sl.mean()) + 1) / (N_PERM + 1))
        return sl, float(sl.mean()), p, float(null.std())

    lines = ["D-076: v3.1's F1 error against the closed-phase reference, on the EGG closed quotient within speakers "
             "(PTDB-TUG dev; slope in Hz per 0.1 CQ, controlling closed-phase F1 and log f0)", ""]
    g = setup(V31)
    sl, mean, p, sd = test(g)
    ok = mean <= -1 and p < 0.05
    lines.append(f"primary, 12 dev speakers, v3.1 seed mean: mean slope {mean:+.2f} Hz per 0.1 CQ, permutation p (one-sided) "
                 f"{p:.4f}, null SD {sd:.2f} (the predicted -2 is {2 / sd:.1f} null SDs)")
    lines.append("  per speaker: " + ", ".join(f"{s}({x}) {v:+.1f}" for (s, x, *_), v in zip(g, sl)))
    lines.append("")
    lines.append("reported:")
    for sx, name in (("f", "women"), ("m", "men")):
        s2, m2, p2, sd2 = test(g, sx)
        lines.append(f"  {name}: mean {m2:+.2f}, median {np.median(s2):+.2f}, p {p2:.4f}, null SD {sd2:.2f}")
    for m in (*V31, *V3):
        _, m2, p2, _ = test(setup((m,)))
        lines.append(f"  {m} alone: mean {m2:+.2f}, p {p2:.4f}")
    for lab, extra in (("closed-phase F1 < 500 Hz", d["cp"][:, 0] < 500), ("closed-phase F1 >= 500 Hz", d["cp"][:, 0] >= 500)):
        _, m2, p2, _ = test(setup(V31, extra))
        lines.append(f"  {lab}: mean {m2:+.2f}, p {p2:.4f}")
    lines.append("")
    lines.append("DECISION: " + ("tracks" if ok else "not detected"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d076.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
