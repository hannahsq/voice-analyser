"""D-093 (with amendment 1): is Hillenbrand's group gradient in F1 explained by f0 and the vowel?

Hillenbrand dev, every group; v3.1 (seed mean in log frequency of d040_5b_s{1,2}_b20k) read at each token's
steady-state time (windowed analysis, as the suite). Error = model F1 - reference F1 (cents; Hz reported).
Predictors: the reference steady-state f0; the predicted F1 = the vowel's pooled reference F1 (median over all other
dev speakers) x the speaker's scale (geometric mean over the speaker's other vowels and F1-F3 of reference / that
vowel-formant's pooled median over the other speakers). Never the token's own reference formants.
Conditioned contrasts on a quarter-octave grid in f0 and predicted F1 (cells with >= 10 tokens of each group, weights
n1 n2 / (n1 + n2)); if under 30 % of women's tokens are in cells shared with men, amendment 1 item 3's f0-slope
route decides instead. Speaker bootstrap (2000, within group).

    python scripts/analysis/d093_hillenbrand_surface.py > reports/analysis/d093/surface.txt

GPU for the model reads (cached in cache/d093/tokens.npz), CPU afterwards.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODELS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
CACHE = Path("cache/d093/tokens.npz")
NB = 2000


def read_tokens():
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry

    mod = registry()["hillenbrand"]
    rows = TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",))
    items = [mod.load(Path("data/hillenbrand"), r) for r in rows]
    out = {"speaker": np.array([r.speaker_id for r in rows]), "group": np.array([it.meta["group"] for it in items]),
           "vowel": np.array([it.meta["vowel"] for it in items]), "t": np.array([it.times[0] for it in items]),
           "f0": np.array([it.f0[0] for it in items], float), "ref": np.array([it.formants[0, :3] for it in items], float)}
    if CACHE.exists():
        c = np.load(CACHE)
        if np.array_equal(c["speaker"], out["speaker"]):
            out["model"] = c["model"]
            return out
    from vocal_analysis.device import setup_device
    from vocal_analysis.models.infer import analyse, load_teacher

    dev = setup_device()
    logs = []
    for m in MODELS:
        model = load_teacher(f"runs/{m}/ckpt_last.pt", dev)[0]
        v = []
        for it, t in zip(items, out["t"]):
            r = analyse(model, load_audio(it, 16000), dev)
            v.append(r["formants"][min(int(round(t / 0.01)), len(r["formants"]) - 1), :3])
        logs.append(np.log2(np.array(v, float)))
        del model
    out["model"] = 2 ** np.mean(logs, 0)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, speaker=out["speaker"], model=out["model"])
    return out


def predicted_f1(d):
    """the vowel's pooled F1 (other speakers) x the speaker's scale (other vowels, F1-F3, against other speakers)"""
    spk, vow, ref = d["speaker"], d["vowel"], d["ref"]
    pred = np.full(len(spk), np.nan)
    for s in np.unique(spk):
        other = spk != s
        pooled = {v: np.nanmedian(ref[other & (vow == v)], 0) for v in np.unique(vow)}
        mine = np.flatnonzero(spk == s)
        for j in mine:
            ratios = [np.log2(ref[k, f] / pooled[vow[k]][f]) for k in mine if k != j for f in range(3)
                      if np.isfinite(ref[k, f]) and ref[k, f] > 0]
            pred[j] = pooled[vow[j]][0] * 2 ** np.mean(ratios)
    return pred


def cells(lf0, lf1):
    return np.floor(lf0 / 0.25).astype(int) * 1000 + np.floor(lf1 / 0.25).astype(int)


if __name__ == "__main__":
    d = read_tokens()
    ok = np.isfinite(d["f0"]) & (d["f0"] > 0) & np.isfinite(d["ref"][:, 0]) & np.isfinite(d["model"][:, 0])
    pred = predicted_f1(d)
    ok &= np.isfinite(pred)
    spk, grp, vow = d["speaker"][ok], d["group"][ok], d["vowel"][ok]
    grp = np.where(np.isin(grp, ("b", "g")), "c", grp)  # children pooled for the contrast; boys / girls reported
    grp4 = d["group"][ok]
    err = 1200 * np.log2(d["model"][ok, 0] / d["ref"][ok, 0])
    errhz = d["model"][ok, 0] - d["ref"][ok, 0]
    lf0, lf1 = np.log2(d["f0"][ok]), np.log2(pred[ok])
    cell = cells(lf0, lf1)
    spks = {g: np.unique(spk[grp == g]) for g in "mwc"}
    idx_of = {s: np.flatnonzero(spk == s) for s in np.unique(spk)}
    rng = np.random.default_rng(93)
    draws = [np.concatenate([idx_of[s] for g in "mwc" for s in rng.choice(spks[g], len(spks[g]))]) for _ in range(NB)]
    allidx = np.arange(len(err))

    def uncond(a, b, ix):
        return err[ix][grp[ix] == a].mean() - err[ix][grp[ix] == b].mean()

    def cond(a, b, ix):
        e, g, c = err[ix], grp[ix], cell[ix]
        num = den = 0.0
        n_a_in = 0
        for k in np.unique(c):
            ia, ib = (c == k) & (g == a), (c == k) & (g == b)
            na, nb = ia.sum(), ib.sum()
            if na >= 10 and nb >= 10:
                w = na * nb / (na + nb)
                num += w * (e[ia].mean() - e[ib].mean())
                den += w
                n_a_in += na
        return (num / den if den else np.nan), n_a_in / max(1, (g == a).sum())

    def f0_route(a, b, ix):
        """within-group slope of error on log2 f0 (vowel and group terms) x the groups' mean log2 f0 difference"""
        e, f, v, g = err[ix], lf0[ix], vow[ix], grp[ix]
        X = [f] + [(v == u).astype(float) for u in np.unique(vow)] + [(g == h).astype(float) for h in "wc"]
        beta = np.linalg.lstsq(np.column_stack(X), e, rcond=None)[0]
        return beta[0] * (f[g == a].mean() - f[g == b].mean()), beta[0]

    def ci(f):
        b = np.array([f(ix) for ix in draws], float)
        return np.nanpercentile(b, [2.5, 97.5])

    L = [f"D-093: Hillenbrand dev F1 error of v3.1 (seed mean of {' + '.join(MODELS)}), steady-state frames, against f0 and the "
         f"predicted F1; {len(err)} tokens, men {len(spks['m'])}, women {len(spks['w'])}, children {len(spks['c'])} speakers", ""]
    L.append("error by group (mean cents / Hz; median Hz):")
    for g in "mwbg":
        k = grp4 == g
        L.append(f"  {g}: {err[k].mean():+.1f} c / {errhz[k].mean():+.1f} Hz; median {np.median(errhz[k]):+.1f} Hz; "
                 f"median f0 {np.median(d['f0'][ok][k]):.0f} Hz; n {k.sum()}")
    L.append("")
    res = {}
    for a, b, lab in (("w", "m", "women - men"), ("c", "m", "children - men")):
        u = uncond(a, b, allidx)
        ulo, uhi = ci(lambda ix: uncond(a, b, ix))
        c_, share = cond(a, b, allidx)
        clo, chi = ci(lambda ix: cond(a, b, ix)[0])
        p, slope = f0_route(a, b, allidx)
        r_ = p / u
        rlo, rhi = ci(lambda ix: f0_route(a, b, ix)[0] / uncond(a, b, ix))
        res[lab] = dict(u=u, uci=(ulo, uhi), c=c_, cci=(clo, chi), share=share, ratio=r_, rci=(rlo, rhi))
        L.append(f"{lab}:")
        L.append(f"  unconditioned {u:+.1f} c [{ulo:+.1f}, {uhi:+.1f}]")
        L.append(f"  conditioned on the (f0, predicted F1) grid {c_:+.1f} c [{clo:+.1f}, {chi:+.1f}]; "
                 f"{100 * share:.0f} % of {lab.split(' ')[0]}'s tokens in shared cells; ratio to unconditioned {c_ / u:+.2f}")
        L.append(f"  f0 route (amendment item 3): within-group slope {slope:+.1f} c per octave; predicted gap {p:+.1f} c; "
                 f"ratio {r_:+.2f} [{rlo:+.2f}, {rhi:+.2f}]")
        L.append("")

    women_share = res["women - men"]["share"]
    verdicts = {}
    if women_share >= 0.30:
        route = "grid"
        for lab, r in res.items():
            frac = r["c"] / r["u"]
            inc0 = r["cci"][0] <= 0 <= r["cci"][1]
            exc0 = not inc0
            verdicts[lab] = ("explained" if abs(frac) <= 1 / 3 and inc0 else
                             "not explained" if frac >= 2 / 3 and exc0 else "partly")
        both_expl = all(v == "explained" for v in verdicts.values())
        any_not = any(v == "not explained" for v in verdicts.values())
        verdict = "EXPLAINED" if both_expl else "NOT EXPLAINED" if any_not else "PARTLY"
    else:
        route = "f0 slope (women's shared-cell share under 30 %)"
        for lab, r in res.items():  # amendment item 3: the point ratio decides; its interval is reported beside it
            lo, hi = r["rci"]
            verdicts[lab] = ("explained" if r["ratio"] >= 2 / 3 else "not explained" if r["ratio"] <= 1 / 3 else "partly")
            if lo < 1 / 3 < hi or lo < 2 / 3 < hi:
                verdicts[lab] += f" (interval [{lo:+.2f}, {hi:+.2f}] crosses a threshold)"
        v = [x.split(" (")[0] for x in verdicts.values()]
        verdict = "EXPLAINED" if all(x == "explained" for x in v) else "NOT EXPLAINED" if any(x == "not explained" for x in v) else "PARTLY"
    L.append(f"rule route: {route}; per contrast: " + "; ".join(f"{k} {v}" for k, v in verdicts.items()))

    # reported: regression, with and without group terms
    def ols(ix, groups):
        e, f, p_ = err[ix], lf0[ix] - lf0.mean(), lf1[ix] - lf1.mean()
        X = [np.ones(len(e)), f, p_, f * p_] + ([(grp[ix] == h).astype(float) for h in "wc"] if groups else [])
        return np.linalg.lstsq(np.column_stack(X), e, rcond=None)[0]

    L.append("")
    L.append("reported: OLS of the error (cents) on centred log2 f0, log2 predicted F1 and their product [speaker bootstrap]")
    names = ["intercept", "log2 f0", "log2 F1", "product", "women", "children"]
    for groups in (False, True):
        b0 = ols(allidx, groups)
        bb = np.array([ols(ix, groups) for ix in draws])
        lo, hi = np.percentile(bb, [2.5, 97.5], 0)
        L.append(f"  {'with' if groups else 'without'} group terms: " + "; ".join(
            f"{n} {b0[i]:+.1f} [{lo[i]:+.1f}, {hi[i]:+.1f}]" for i, n in enumerate(names[:len(b0)])))
    L.append("")
    L.append("reported: error by vowel and group (mean cents)")
    vs = sorted(np.unique(vow), key=lambda v: np.median(lf1[vow == v]))
    L.append("  " + " ".join(f"{v:>6s}" for v in ["grp"] + vs))
    for g in "mwbg":
        L.append("  " + f"{g:>6s}" + " ".join(f"{err[(grp4 == g) & (vow == v)].mean():+6.0f}" for v in vs))

    # reported: the surface (no group terms) on PTDB-TUG dev frames, Praat F1 as the predictor, closed phase as reference
    try:
        import json

        q = np.load("cache/m3w/ptdb_egg_sub.npz", allow_pickle=True)

        tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
        dv = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in q["speaker"]])
        mf1 = 2 ** np.mean([np.log2(q[m][:, 0]) for m in MODELS], 0)
        k = dv & np.isfinite(q["cp"][:, 0]) & np.isfinite(q["praat"][:, 0]) & np.isfinite(mf1) & (q["f0"] > 0)
        b0 = ols(allidx, False)
        f = np.log2(q["f0"][k]) - lf0.mean()
        p_ = np.log2(q["praat"][k, 0]) - lf1.mean()
        pr_c = b0[0] + b0[1] * f + b0[2] * p_ + b0[3] * f * p_
        pr_hz = q["cp"][k, 0] * (2 ** (pr_c / 1200) - 1)
        act_hz = mf1[k] - q["cp"][k, 0]
        L.append("")
        L.append("reported: the surface on PTDB-TUG dev frames (Praat F1 as the predictor; closed-phase reference)")
        for sx, lab in (("f", "women"), ("m", "men")):
            m_ = q["sex"][k] == sx
            L.append(f"  {lab}: predicted median {np.median(pr_hz[m_]):+.1f} Hz, actual median {np.median(act_hz[m_]):+.1f} Hz "
                     f"({m_.sum()} frames; f0 {np.median(q['f0'][k][m_]):.0f} Hz)")
    except Exception as ex:  # reported only
        L.append(f"  PTDB-TUG surface: not computed ({ex})")
    L.append("")
    L.append(f"DECISION: {verdict} (route: {route}; " + "; ".join(
        f"{k}: unconditioned {r['u']:+.1f} c, conditioned {r['c']:+.1f} c [{r['cci'][0]:+.1f}, {r['cci'][1]:+.1f}], "
        f"f0 route ratio {r['ratio']:+.2f}" for k, r in res.items()) + ")")
    print("\n".join(L))
