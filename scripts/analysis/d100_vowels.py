"""D-100 addendum: the vowel ordering of iVTL (and aVTL) on Hillenbrand dev, from d100_context.py's cached reads.

Each word's estimate centred on its speaker's mean over the 12 words (cents); per vowel the median over speakers.
Primary: rounded (uw oo oa aw er) minus unrounded (iy ih ei eh ae ah uh) mean of the centred iVTL, speaker
bootstrap (2000, within sex). Reported: the same for aVTL, the per-vowel table, the rank correlation of iVTL and aVTL
vowel means, the within-speaker correlation of iVTL with the word's f0.

    python scripts/analysis/d100_vowels.py > reports/analysis/d100/vowels.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CACHE = Path("cache/d100")
ROUNDED = ("uw", "oo", "oa", "aw", "er")
UNROUNDED = ("iy", "ih", "ei", "eh", "ae", "ah", "uh")
ORDER = ROUNDED + UNROUNDED
GROUPS = {"5b": ("d040_5b_s1_b20k", "d040_5b_s2_b20k"), "5c": ("d051_5c_s1_b20k", "d051_5c_s2_b20k"),
          "A0": ("d080_a0_s1", "d080_a0_s2"), "D-098 base": ("d098_base_b20k",), "D-098 heads": ("d098_heads_b20k",),
          "D-099 narrow": ("d099_narrow_b20k",), "D-099 base s2": ("d099_base_s2_b20k",)}


def items():
    """d100_context's Hillenbrand list, in its order: speaker, sex, vowel, median reference f0"""
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import registry

    mod = registry()["hillenbrand"]
    out = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        it = mod.load(Path("data/hillenbrand"), row)
        g = it.meta.get("group", "?")
        ok = np.isfinite(it.times)
        if g in "mw" and ok.any():
            f0 = np.asarray(it.f0, float)[ok] if it.f0 is not None else np.array([np.nan])
            f0 = f0[np.isfinite(f0) & (f0 > 0)]
            out.append((row.speaker_id, g, Path(row.path).stem[-2:], float(np.median(f0)) if len(f0) else np.nan))
    return out


def centre(e, spk):
    c = e.copy()
    for s in np.unique(spk):
        c[spk == s] -= np.nanmean(e[spk == s])
    return 1200 * c


if __name__ == "__main__":
    from scipy.stats import spearmanr

    it = items()
    spk = np.array([i[0] for i in it])
    sex = np.array([i[1] for i in it])
    vow = np.array([i[2] for i in it])
    lf0 = np.log2([i[3] for i in it])
    assert set(vow) == set(ORDER), sorted(set(vow))
    rd = np.isin(vow, ROUNDED)
    spks = {G: sorted(set(spk[sex == G])) for G in "mw"}
    rng = np.random.default_rng(0)
    draws = [[s for G in "mw" for s in rng.choice(spks[G], len(spks[G]))] for _ in range(2000)]

    def contrast(c, draw=None):
        """mean over speakers of (mean rounded - mean unrounded)"""
        ss = draw if draw is not None else [s for G in "mw" for s in spks[G]]
        per = {s: np.nanmean(c[(spk == s) & rd]) - np.nanmean(c[(spk == s) & ~rd]) for s in set(ss)}
        return float(np.mean([per[s] for s in ss]))

    L = ["D-100 addendum: vowel ordering on Hillenbrand dev (27 men, 29 women; each word centred on its speaker's mean, cents)",
         f"rounded {' '.join(ROUNDED)} | unrounded {' '.join(UNROUNDED)}; primary: iVTL rounded - unrounded, predicted > 0", ""]
    tab = {}
    for g, ms in GROUPS.items():
        if not all((CACHE / f"{m}.npz").exists() for m in ms):
            continue
        R = [dict(np.load(CACHE / f"{m}.npz")) for m in ms]
        L.append(f"== {g} ({' + '.join(ms)})")
        for read in ("R1", "R3"):
            has_i = all(np.isfinite(r[f"hillenbrand/{read}/i"]).any() for r in R)
            ca = np.mean([centre(r[f"hillenbrand/{read}/a"], spk) for r in R], 0)
            ci_ = np.mean([centre(r[f"hillenbrand/{read}/i"], spk) for r in R], 0) if has_i else None
            parts = []
            for name, c in (("iVTL", ci_), ("aVTL", ca)):
                if c is None:
                    continue
                d = contrast(c)
                lo, hi = np.percentile([contrast(c, dr) for dr in draws], [2.5, 97.5])
                parts.append(f"{name} rounded - unrounded {d:+.0f} c [{lo:+.0f}, {hi:+.0f}]")
            L.append(f"  {read}: " + "; ".join(parts))
            for name, c in (("iVTL", ci_), ("aVTL", ca)):
                if c is None:
                    continue
                med = {v: float(np.median([np.nanmean(c[(spk == s) & (vow == v)]) for s in set(spk)])) for v in ORDER}
                tab[g, read, name] = med
                L.append(f"    {name} by vowel: " + "  ".join(f"{v} {med[v]:+.0f}" for v in sorted(ORDER, key=lambda v: -med[v])))
                for G in "mw":
                    k = sex == G
                    dG = np.mean([np.nanmean(c[(spk == s) & rd]) - np.nanmean(c[(spk == s) & ~rd]) for s in spks[G]])
                    L.append(f"      {'men' if G == 'm' else 'women'}: rounded - unrounded {dG:+.0f} c")
            if ci_ is not None:
                rho = spearmanr([tab[g, read, 'iVTL'][v] for v in ORDER], [tab[g, read, 'aVTL'][v] for v in ORDER])[0]
                cf = centre(lf0, spk) / 1200
                k = np.isfinite(cf) & np.isfinite(ci_)
                r_f0 = np.corrcoef(cf[k], ci_[k])[0, 1]
                cav = np.isfinite(ca) & np.isfinite(ci_)
                b = (ci_[cav] @ ca[cav]) / (ci_[cav] @ ci_[cav])
                r2 = 1 - np.sum((ca[cav] - b * ci_[cav]) ** 2) / np.sum(ca[cav] ** 2)
                L.append(f"    iVTL vs aVTL vowel means: Spearman {rho:+.2f}; aVTL on iVTL within speakers R^2 {r2:.2f} (slope {b:.2f}); "
                         f"iVTL vs word f0 within speakers r {r_f0:+.2f}")
        L.append("")
    print("\n".join(L))
