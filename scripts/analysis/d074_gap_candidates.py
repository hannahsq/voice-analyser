"""D-074: which resonance does teacher v3.1's gap growth on KL renders follow? Candidates per (sex, vowel, gap), from
the quasi-static open-glottis poles (`coupling.open_poles`) on D-060's Story shapes and glottis: C0 the label (0), CL the
leaky closed phase (F1 at gap x peak), CM the cycle mean (F1 at each sample's area over one Rosenberg cycle). Compared
with D-060's per-file errors (seed mean); no render, no model.

    python scripts/analysis/d074_gap_candidates.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TSV = Path("reports/analysis/d060/fixed/main_per_file.tsv")
OUT = Path("reports/analysis/d074")
VOWELS = ("i", "I", "E", "ae", "V", "A", "O", "U", "o", "u")
SIZE = {"Male": (17.46, 0.12), "Female": (15.6, 0.07)}  # as D-060
GAPS = (0.0, 0.1, 0.2, 0.3)
AREAS = np.concatenate([[0.0], np.linspace(0.02, 1.0, 25)])  # fractions of the peak
SEED = 5074


def candidates():
    """{(sex, vowel, gap): (CL, CM)} in Hz, and each vowel's F1 continuation per sex"""
    from vocal_analysis.synth import coupling, kl, tube

    out, curves = {}, {}
    for sex, (L, peak) in SIZE.items():
        rel = L / tube.STORY_LENGTH_CM
        g = coupling.Glottis(peak, 0.0, 1.4 * rel, 0.3 * rel, 8000.0, 1.0 / rel)
        for v in VOWELS:
            s = tube.build_shape(*tube.STORY_VOWELS[v], L, area_scale=rel**2)
            F, B = tube.poles(s)
            f1 = np.array([F[0]] + [coupling.open_poles(s, peak * x, g, F, B, n=1)[0][0] for x in AREAS[1:]])
            curves[sex, v] = f1
            for gap in GAPS:
                cyc = kl.FS // 200  # one cycle at 200 Hz; the shape, not f0, sets the mean
                a = kl.glottal_area(np.full(10, 200.0), kl.FS // 10 * 10, kl.FS, peak, oq=0.6, sq=1.3, gap=gap)[cyc:2 * cyc] / peak
                cl = np.interp(gap, AREAS, f1) - F[0]
                cm = float(np.mean(np.interp(np.clip(a, 0, 1), AREAS, f1))) - F[0]
                out[sex, v, gap] = (cl, cm)
    return out, curves


def boot(x, rng, n=2000):
    b = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.median(x)), *np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    rows = [ln.split("\t") for ln in TSV.read_text().splitlines()[1:]]
    err = {(r[0], float(r[1]), r[2], int(r[3])): (float(r[4]) + float(r[5])) / 2 for r in rows}
    cand, curves = candidates()
    lines = ["D-074: which resonance does v3.1's gap growth on KL renders follow? (seed mean of v3.1 s2 / s1; D-060's files)", ""]
    lines.append("candidate shifts, Hz (CL leaky closed phase / CM cycle mean), gaps 0 / 0.1 / 0.2 / 0.3")
    for sex in SIZE:
        for v in VOWELS:
            lines.append(f"  {sex:6s} {v:3s} " + "  ".join(f"{cand[sex, v, g][0]:+5.1f}/{cand[sex, v, g][1]:+5.1f}" for g in GAPS)
                         + ("  (continuation not monotonic)" if np.any(np.diff(curves[sex, v]) < 0) else ""))
    lines.append("")
    verdict = {}
    for subset, vs in (("all ten vowels (decides)", VOWELS), ("without V and O (reported)", tuple(v for v in VOWELS if v not in "VO"))):
        lines.append(f"growth residual (teacher gap 0.3 - 0, minus the candidate's), median [95 %], {subset}")
        for sex in SIZE:
            pairs = sorted({(v, f) for (s, gp, v, f) in err if s == sex and v in vs})
            tg = np.array([err[sex, 0.3, v, f] - err[sex, 0.0, v, f] for v, f in pairs])
            for name, k in (("C0", None), ("CL", 0), ("CM", 1)):
                cg = np.array([0.0 if k is None else cand[sex, v, 0.3][k] - cand[sex, v, 0.0][k] for v, _ in pairs])
                m, lo, hi = boot(tg - cg, rng)
                ok = lo <= 0 <= hi and abs(m) <= 5
                if subset.endswith("(decides)"):
                    verdict[sex, name] = ok
                lines.append(f"  {sex:6s} {name}: {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] Hz -> {'accounts' if ok else 'does not'}")
        lines.append("")
    lines.append("level residual (teacher error - candidate), median over files, gaps 0 / 0.1 / 0.2 / 0.3 (reported)")
    for sex in SIZE:
        for name, k in (("C0", None), ("CL", 0), ("CM", 1)):
            med = [np.median([e - (0 if k is None else cand[s, v, gp][k]) for (s, gp, v, f), e in err.items() if s == sex and gp == g])
                   for g in GAPS]
            lines.append(f"  {sex:6s} {name}: " + " / ".join(f"{x:+.1f}" for x in med))
    lines.append("")
    lines.append("Spearman over vowels, the teacher's mean growth against the candidate's (reported)")
    for sex in SIZE:
        tg = np.array([np.mean([err[sex, 0.3, v, f] - err[sex, 0.0, v, f] for (s, gp, vv, f) in err
                                if s == sex and gp == 0.0 and vv == v]) for v in VOWELS])
        rk = lambda x: np.argsort(np.argsort(x))
        for name, k in (("CL", 0), ("CM", 1)):
            cg = np.array([cand[sex, v, 0.3][k] - cand[sex, v, 0.0][k] for v in VOWELS])
            lines.append(f"  {sex:6s} {name}: {np.corrcoef(rk(tg), rk(cg))[0, 1]:+.2f}")
    lines.append("")
    both = {n: verdict["Male", n] and verdict["Female", n] for n in ("CL", "CM")}
    agree = all(verdict["Male", n] == verdict["Female", n] for n in ("CL", "CM"))
    d = ("the sexes disagree; reported, no reading" if not agree else
         "the leaky closed phase's resonance" if both["CL"] and not both["CM"] else
         "the cycle-mean pole" if both["CM"] and not both["CL"] else
         "both account: not separable here" if both["CL"] else
         "neither: the growth is not the quasi-static pole movement (bandwidth or spectral shape next)")
    lines.append(f"DECISION: {d}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d074.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
