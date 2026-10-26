"""D-033 pooled estimate (pre-registered 2026-10-26, reported only; computed because semi-private replicated): the full
arm's F1 measures on dev and semi-private speakers together, from `full_arm.py`'s VA_DUMP files. Paired speaker
bootstrap resampling within tier and sex (PTDB-TUG closed phase) or tier and group (Hillenbrand), 2,000 resamples.
Decides nothing; the semi-private estimate alone stays the unbiased one.
Args: dev dump (.npz), semi dump (.npz), out txt."""
import sys

import numpy as np

D, S, OUTF = sys.argv[1:4]
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
EST = ["teacher_v3", "labeller_v3", "arm", "arm_t"] + [f"arm#{k}" for k in range(6)]
d, s = np.load(D, allow_pickle=True), np.load(S, allow_pickle=True)
cat = lambda k: np.concatenate([d[k], s[k]])  # noqa: E731
tier = lambda k: np.concatenate([np.full(len(d[k]), "dev"), np.full(len(s[k]), "semi")])  # noqa: E731
sex, okc, spk_c, tc = cat("sex"), cat("okc"), cat("spk_c"), tier("sex")
grp, okh, spk_h, th = cat("group"), cat("okh"), cat("spk_h"), tier("group")
errc = {k: cat(f"c_{k}") for k in EST}
errh = {k: cat(f"h_{k}") for k in EST}


def strata(spk, lab, t):
    out = []
    for st in sorted({(a, b) for a, b in zip(t, lab)}):
        m = (t == st[0]) & (lab == st[1])
        out.append([np.flatnonzero(m & (spk == x)) for x in sorted(set(spk[m]))])
    return out


def measure(ic, ih):
    r = {}
    for k in EST:
        ec, eh = errc[k][ic], errh[k][ih]
        fc, fh = okc[ic] & np.isfinite(ec), okh[ih] & np.isfinite(eh)
        r[k] = (abs(np.median(ec[fc & (sex[ic] == "f")])), abs(np.median(ec[fc & (sex[ic] == "m")])), np.mean(np.abs(eh[fh])))
    r["single-seed mean"] = tuple(np.mean([r[f"arm#{k}"][j] for k in range(6)]) for j in range(3))
    return r


SC, SH = strata(spk_c, sex, tc), strata(spk_h, grp, th)
p(f"D-033 full arm, dev + semi-private pooled (reported only): PTDB-TUG {len(set(spk_c))} speakers "
  f"({len(set(spk_c[sex == 'f']))} women), Hillenbrand {len(set(spk_h))} speakers")
pt = measure(np.arange(len(sex)), np.arange(len(grp)))
p("\nPoint estimates: women's |F1 bias| / men's |F1 bias| / Hillenbrand F1 MAE (Hz)")
for k in ["teacher_v3", "labeller_v3", "arm", "arm_t", "single-seed mean"]:
    p(f"  {k:18s} {pt[k][0]:6.1f} / {pt[k][1]:6.1f} / {pt[k][2]:6.1f}")
rng, B = np.random.default_rng(20261026), 2000
bs = []
for _ in range(B):
    ic = np.concatenate([np.concatenate([st[j] for j in rng.integers(0, len(st), len(st))]) for st in SC])
    ih = np.concatenate([np.concatenate([st[j] for j in rng.integers(0, len(st), len(st))]) for st in SH])
    bs.append(measure(ic, ih))
p(f"\nPaired speaker bootstrap ({B} resamples within tier and sex / group): mean [95 % interval], Hz")
for j, title in enumerate(("women's closed-phase |F1 median signed error|", "men's closed-phase |F1 median signed error|",
                           "Hillenbrand F1 MAE (all)")):
    p(f"  {title}")
    for a, b in (("arm", "labeller_v3"), ("arm", "teacher_v3"), ("arm", "single-seed mean"), ("arm_t", "teacher_v3")):
        v = np.array([x[a][j] - x[b][j] for x in bs])
        lo, hi = np.percentile(v, [2.5, 97.5])
        p(f"    {a + ' - ' + b:30s} {v.mean():6.1f} [{lo:6.1f}, {hi:6.1f}]")
