"""D-118: do the 5b branch arms' Hillenbrand length-reliability costs remain when the words are read in context?

D-100's reads (R1 alone, R2 in a same-speaker concatenation through 4 s windows, R3 the concatenation in one pass;
PTDB-TUG also R0), imported unchanged from `d100_context.py`, on the branch arms of D-115 (d114, d109, D-090's f57 and
f5), seeds paired with 5b's own branches (D-100's caches reused). Reliability is D-051's guard as `vtl_score_joint.py`
computes it; the decomposition is D-115's (`d115_reliability.parts`). Both are recomputed here from per-speaker sums
so the bootstrap is fast, and checked against the original functions on the undrawn sample.

    score + report:  VA_DEVICE=cuda python scripts/analysis/d118_context_arms.py > reports/analysis/d118/d118.txt

Per-model reads are cached in cache/d100/<name>.npz (skipped if present).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

BASE = {1: "d040_5b_s1_b20k", 2: "d040_5b_s2_b20k"}
ARMS = {"d114": "d114_s{}", "d109": "d109_s{}", "f57": "d090_f57_s{}", "f5": "d090_f5_s{}"}
REPORTED = {"d040_5b_s1_b20k": 0.228, "d040_5b_s2_b20k": 0.261, "d114_s1": 0.215, "d114_s2": 0.236,
            "d109_s1": 0.223, "d109_s2": 0.241}
N_BOOT = 2000


def hb_vowels():
    """(speaker, vowel) of Hillenbrand's dev words in `d100_context.load_items`' order"""
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import registry

    mods = registry()
    out = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        if item.meta.get("group", "?") in "mw" and np.isfinite(item.times).any():
            out.append((row.speaker_id, item.meta.get("vowel", "?")))
    return out


def sums(e, si, n_spk, vi=None, n_vow=0):
    """per-speaker sufficient statistics of per-item estimates e (log2 cm): mean, sum of squared deviations, count,
    and (with vowels) the per-speaker-per-vowel sums of deviations and counts"""
    ok = np.isfinite(e)
    n = np.bincount(si[ok], minlength=n_spk).astype(float)
    mean = np.bincount(si[ok], e[ok], n_spk) / n
    d = e - mean[si]
    st = {"mean": mean, "ssd": np.bincount(si[ok], d[ok] ** 2, n_spk), "n": n}
    if vi is not None:
        S, N = np.zeros((n_spk, n_vow)), np.zeros((n_spk, n_vow))
        np.add.at(S, (si[ok], vi[ok]), d[ok])
        np.add.at(N, (si[ok], vi[ok]), 1.0)
        st["S"], st["N"] = S, N
    return st


def bwv(st, draw):
    """between-speaker variance B, within-speaker variance W and its vowel part V over a draw of speaker indices"""
    B = np.var(st["mean"][draw])
    W = st["ssd"][draw].sum() / st["n"][draw].sum()
    V = np.nan
    if "S" in st:
        Su, Nu = st["S"][draw].sum(0), st["N"][draw].sum(0)
        vm = Su / np.maximum(Nu, 1.0)
        V = (Nu * vm ** 2).sum() / Nu.sum()
    return B, W, V


def rel(st, draw):
    """D-051's reliability over a draw {sex: speaker indices}: the mean of men's and women's B / (B + W)"""
    r = []
    for G in "mw":
        B, W, _ = bwv(st, draw[G])
        r.append(B / (B + W))
    return float(np.mean(r))


def within_sd_wm(st, draw):
    ssd = sum(st["ssd"][draw[G]].sum() for G in "mw")
    n = sum(st["n"][draw[G]].sum() for G in "mw")
    return 1200 * np.sqrt(ssd / n), 2 ** (np.median(st["mean"][draw["w"]]) - np.median(st["mean"][draw["m"]]))


def fmt(p, b):
    lo, hi = np.nanpercentile(b, [2.5, 97.5])  # a draw of one speaker six times has no between-speaker variance (PTDB-TUG): left out
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]", lo, hi


if __name__ == "__main__":
    import d100_context as D100
    import d115_reliability as D115

    models = {n: f"runs/{n}/ckpt_last.pt" for n in list(BASE.values()) + [p.format(s) for p in ARMS.values() for s in (1, 2)]}
    D100.CACHE.mkdir(parents=True, exist_ok=True)
    sets = D100.load_items()
    cats = {"hillenbrand": D100.concatenations(sets["hillenbrand"], None), "ptdb": D100.concatenations(sets["ptdb"], 2)}
    todo = [k for k in models if not (D100.CACHE / f"{k}.npz").exists()]
    if todo:
        from vocal_analysis.device import setup_device

        dev = setup_device()
        for k in todo:
            np.savez(D100.CACHE / f"{k}.npz", **D100.score_model(k, models[k], sets, cats, dev))
    R = {k: dict(np.load(D100.CACHE / f"{k}.npz")) for k in models}

    hv = hb_vowels()
    assert [s for s, _ in hv] == [it[2] for it in sets["hillenbrand"]], "vowel list out of step with D-100's items"
    L = ["D-118: the 5b branch arms' length reliability with the words read in context (dev; D-100's reads, D-051's reliability)",
         "reads: " + "; ".join(f"{k} {v}" for k, v in D100.READ_NAMES.items()) + ". Same frames and averaging in every read.",
         f"arm - 5b: seeds paired with 5b's own branches, seed means, 95 % paired speaker bootstrap ({N_BOOT})", ""]
    verdict = None
    for ds, items in sets.items():
        spk = np.array([it[2] for it in items])
        names = sorted(set(spk))
        sex = {it[2]: it[3] for it in items}
        si = np.array([names.index(s) for s in spk])
        full = {G: np.array([j for j, s in enumerate(names) if sex[s] == G]) for G in "mw"}
        vi, n_vow, vow = None, 0, None
        if ds == "hillenbrand":
            vow = np.array([v for _, v in hv])
            vl = sorted(set(vow))
            vi, n_vow = np.array([vl.index(v) for v in vow]), len(vl)
        rng = np.random.default_rng(118)
        draws = [{G: full[G][rng.integers(0, len(full[G]), len(full[G]))] for G in "mw"} for _ in range(N_BOOT)]
        ST = {m: {rd: sums(R[m][f"{ds}/{rd}/a"], si, len(names), vi, n_vow) for rd in D100.READS[ds]} for m in models}
        L.append(f"== {ds} dev: {len(full['m'])} men, {len(full['w'])} women, {len(items)} items; items without an estimate: "
                 + ", ".join(f"{rd} {max(int((~np.isfinite(R[m][f'{ds}/{rd}/a'])).sum()) for m in models)}" for rd in D100.READS[ds]))

        # the fast sums against the original functions, undrawn sample, every model and read
        worst = 0.0
        spks = {G: [names[j] for j in full[G]] for G in "mw"}
        for m in models:
            for rd in D100.READS[ds]:
                e = R[m][f"{ds}/{rd}/a"]
                worst = max(worst, abs(rel(ST[m][rd], full) - D100.reliability(e, spk, spks)[0]))
                if ds == "hillenbrand":
                    for G in "mw":
                        p = D115.parts(e, spk, vow, spks[G])
                        B, W, V = bwv(ST[m][rd], full[G])
                        worst = max(worst, abs(B / p["B"] - 1), abs(W / p["W"] - 1), abs(V / p["V"] - 1))
        L.append(f"  code check: this file's sums against d100_context.reliability" + (" and d115_reliability.parts" if ds == "hillenbrand" else "")
                 + f", largest difference {worst:.1e}")
        ok_pre = True
        if ds == "hillenbrand":
            pre = {m: rel(ST[m]["R1"], full) for m in REPORTED}
            ok_pre = all(abs(pre[m] - REPORTED[m]) <= 0.001 + 5e-4 for m in REPORTED)  # reported values are rounded to 0.001
            L.append("  precondition (R1 against the reported guard reliability): " + "; ".join(
                f"{m} {pre[m]:.3f} ({REPORTED[m]:.3f})" for m in REPORTED) + f" -> {'holds' if ok_pre else 'FAILS: nothing below is read'}")
        L.append("")
        L.append(f"  {'model':8s} " + " | ".join(f"{rd:>22s}" for rd in D100.READS[ds]) + "   (reliability; within SD c; w/m; seed means)")
        groups = {"5b": [BASE[1], BASE[2]], **{a: [p.format(1), p.format(2)] for a, p in ARMS.items()}}
        for g, ms in groups.items():
            cells = []
            for rd in D100.READS[ds]:
                sd, wm = np.mean([within_sd_wm(ST[m][rd], full) for m in ms], 0)
                cells.append(f"{np.mean([rel(ST[m][rd], full) for m in ms]):.3f} {sd:6.0f} c {wm:.3f}")
            L.append(f"  {g:8s} " + " | ".join(f"{c:>22s}" for c in cells))
        L.append("")
        L.append("  arm - 5b, reliability:")
        D = {}
        for a, p in ARMS.items():
            parts_ = []
            for rd in D100.READS[ds]:
                def diff(draw, s=None):
                    ss = (1, 2) if s is None else (s,)
                    return float(np.mean([rel(ST[p.format(k)][rd], draw) - rel(ST[BASE[k]][rd], draw) for k in ss]))
                txt, lo, hi = fmt(diff(full), [diff(d) for d in draws])
                D[a, rd] = (diff(full), lo, hi)
                parts_.append(f"{rd} {txt} (seeds {diff(full, 1):+.3f} / {diff(full, 2):+.3f})")
            L.append(f"    {a:5s} " + "; ".join(parts_))
        s21 = "; ".join(f"{rd} {rel(ST[BASE[2]][rd], full) - rel(ST[BASE[1]][rd], full):+.3f}" for rd in D100.READS[ds])
        L.append(f"    scale: 5b's own seeds (s2 - s1, separate trunks): {s21}")
        L.append("")
        L.append("  decomposition, arm - 5b (seed means): d rel; d ln B (between speakers); d ln W (within); d ln V (vowel part of W)")
        for a, p in ARMS.items():
            for rd in ("R1", "R3"):
                for G in "mw":
                    def dec(draw):
                        o = []
                        for k in (1, 2):
                            Ba, Wa, Va = bwv(ST[p.format(k)][rd], draw[G])
                            Bb, Wb, Vb = bwv(ST[BASE[k]][rd], draw[G])
                            o.append((Ba / (Ba + Wa) - Bb / (Bb + Wb), np.log(Ba / Bb), np.log(Wa / Wb),
                                      np.log(Va / Vb) if ds == "hillenbrand" else np.nan))
                        return np.mean(o, 0)
                    pt = dec(full)
                    bs = np.array([dec(d) for d in draws])
                    cells = [fmt(pt[j], bs[:, j])[0] for j in range(4 if ds == "hillenbrand" else 3)]
                    L.append(f"    {a:5s} {rd} {'men  ' if G == 'm' else 'women'}: d rel {cells[0]}; d ln B {cells[1]}; d ln W {cells[2]}"
                             + (f"; d ln V {cells[3]}" if ds == "hillenbrand" else ""))
        L.append("")
        if ds == "hillenbrand":
            c1, c3, lo3, hi3 = D["d114", "R1"][0], *D["d114", "R3"]
            f_lo = D["f5", "R3"][1]
            if not ok_pre:
                verdict = "NOT READ (precondition fails)"
            elif f_lo <= -0.02:
                verdict = f"UNREADABLE (f5 - 5b's lower bound at R3 {f_lo:+.3f} is at or below -0.02)"
            elif lo3 > -0.02 and c3 >= c1 / 2:
                verdict = "GONE IN CONTEXT"
            elif hi3 < 0 and c3 <= c1 / 2:
                verdict = "KEPT"
            else:
                verdict = "PARTIAL"
            L.append(f"  READING (registered): d114 - 5b on words alone C1 {c1:+.3f}; in context, one pass C3 {c3:+.3f} [{lo3:+.3f}, {hi3:+.3f}]; "
                     f"C1 / 2 = {c1 / 2:+.3f}; f5 - 5b at R3 lower bound {f_lo:+.3f} -> {verdict}")
            L.append("")
    print("\n".join(L))
