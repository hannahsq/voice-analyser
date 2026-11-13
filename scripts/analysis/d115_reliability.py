"""D-115: where does the Hillenbrand length-reliability cost of 5b's decay-branch arms come from? Descriptive, dev only.

Reliability is D-051's length guard as `vtl_score_joint.py` computes it: per Hillenbrand dev file, the VTL head's
precision-weighted mean of log2 length over the reference frames (analyse(), windowed); per sex (men, women),
var(speaker means) / (var(speaker means) + mean within-speaker squared deviation); the guard is the mean of the two.

    score:  python scripts/analysis/d115_reliability.py score name=ckpt ...     (VA_DEVICE=cuda; per-file caches in cache/d115/)
    report: python scripts/analysis/d115_reliability.py report > reports/analysis/d115/d115.txt

Per arm (seeds paired with 5b's own branch, then seed means), for men and women:
  (1) the reliability change, with the between-speaker variance B and within-speaker variance W as log changes, and
      their first-order shares of the change (d rel ~ rel (1 - rel) (d ln B - d ln W));
  (2) W split into the vowel's part V (variance of the vowel means of within-speaker deviations) and the rest;
  (3) speaker concentration: the share of the paired change in sum of squared deviations from the arm's top 3 speakers;
  (4) the reliability change with an unweighted frame mean (the head's sigma not used);
  (5) per vowel: the arm's change in the vowel's mean deviation (cents), and its correlation with the vowel's mean
      reference F1 across the 12 vowels.
Speaker bootstrap (2000, paired across arms) for (1), (2) and (4).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CACHE = Path(os.environ.get("D115_CACHE", "cache/d115"))  # D-116 points this at its own per-frame caches
AGG = os.environ.get("D115_AGG", "weighted")  # "median": the per-file median over frames (A in vtl_score_joint.py)
BASE = {1: "d040_5b_s1_b20k", 2: "d040_5b_s2_b20k"}
ARMS = {"d114": "d114_s{}", "d109": "d109_s{}", "f57": "d090_f57_s{}", "f5": "d090_f5_s{}"}
REPORTED = {"d040_5b_s1_b20k": 0.228, "d040_5b_s2_b20k": 0.261, "d114_s1": 0.215, "d114_s2": 0.236} if AGG == "weighted" else {}


def hillenbrand_dev():
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry

    mods = registry()
    hb = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        ok = np.isfinite(item.times)
        if ok.any():
            f1 = np.asarray(item.formants)[ok][:, 0] if getattr(item, "formants", None) is not None else np.full(ok.sum(), np.nan)
            hb.append((load_audio(item, 16000), item.times[ok], row.speaker_id, item.meta.get("group", "?"),
                       item.meta.get("vowel", "?"), float(np.nanmedian(np.where(f1 > 0, f1, np.nan)))))
    return hb


def score(pairs):
    import torch

    from vocal_analysis.device import setup_device
    from vocal_analysis.models.infer import analyse, load_teacher

    dev = setup_device()
    hb = hillenbrand_dev()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "files.npz", spk=np.array([h[2] for h in hb]), group=np.array([h[3] for h in hb]),
             vowel=np.array([h[4] for h in hb]), ref_f1=np.array([h[5] for h in hb]))
    for name, ck in pairs.items():
        out = CACHE / f"{name}.npz"
        if out.exists():
            print(f"{name}: cached")
            continue
        model, _ = load_teacher(ck, dev)
        mu, sg, off = [], [], [0]
        for x, t, *_ in hb:
            r = analyse(model, x, dev)
            idx = np.clip(np.round(t / 0.01).astype(int), 0, len(r["f0"]) - 1)
            mu.append(r["vtl_log2cm"][idx]), sg.append(r["vtl_sigma"][idx])
            off.append(off[-1] + len(idx))
        np.savez(out, mu=np.concatenate(mu), sigma=np.concatenate(sg), off=np.array(off))
        print(f"{name}: {len(hb)} files")
        del model
        torch.cuda.empty_cache()


def per_file(name, weighted=True):
    z = np.load(CACHE / f"{name}.npz")
    mu, sg, off = z["mu"], z["sigma"], z["off"]
    e = []
    for j in range(len(off) - 1):
        m, s = mu[off[j]:off[j + 1]], sg[off[j]:off[j + 1]]
        if AGG == "median":
            e.append(float(np.nanmedian(m)))
            continue
        w = 1 / s ** 2 if weighted else np.ones_like(s)
        e.append(float((w * m).sum() / w.sum()))
    return np.array(e)


def parts(e, fs, fv, spk_list):
    """B, W, V (vowel part of W), per-speaker sum of squared deviations; e in log2 units."""
    means = np.array([np.nanmean(e[fs == s]) for s in spk_list])
    dev_, vow, ssd = [], [], []
    for s in spk_list:
        m = fs == s
        d = e[m] - np.nanmean(e[m])
        dev_.append(d), vow.append(fv[m]), ssd.append(np.nansum(d ** 2))
    d, v = np.concatenate(dev_), np.concatenate(vow)
    W = np.nanmean(d ** 2)
    vm = {u: np.nanmean(d[v == u]) for u in np.unique(v)}
    V = np.nanmean(np.array([vm[u] for u in v]) ** 2)
    B = np.var(means)
    return {"rel": B / (B + W), "B": B, "W": W, "V": V, "ssd": np.array(ssd), "vowel_dev": vm}


def report():
    f = np.load(CACHE / "files.npz")
    fs, fg, fv, rf1 = f["spk"], f["group"], f["vowel"], f["ref_f1"]
    spks = {G: sorted(set(fs[fg == G])) for G in "mw"}
    names = list(BASE.values()) + [a.format(s) for a in ARMS.values() for s in (1, 2)]
    names = [n for n in names if (CACHE / f"{n}.npz").exists()]
    E = {n: per_file(n) for n in names}
    EU = {n: per_file(n, weighted=False) for n in names}
    rng = np.random.default_rng(20261115)

    def rel(e, draw):
        return np.mean([parts(e, fs, fv, draw[G])["rel"] for G in "mw"])

    print("D-115: where the Hillenbrand length-reliability cost of 5b's decay-branch arms comes from (dev; men and women)")
    print("\nInstrument check (reliability, mean of men and women): " + "; ".join(
        f"{n} {rel(E[n], spks):.3f}" + (f" (reported {REPORTED[n]:.3f})" if n in REPORTED else "") for n in names))
    boots = [{G: [spks[G][j] for j in rng.integers(0, len(spks[G]), len(spks[G]))] for G in "mw"} for _ in range(2000)]
    vowels = sorted(set(fv))
    vf1 = {u: np.nanmean(rf1[fv == u]) for u in vowels}

    def arm_stats(e_arm, e_base, draw):
        out = {}
        for G in "mw":
            a, b = parts(e_arm, fs, fv, draw[G]), parts(e_base, fs, fv, draw[G])
            out[G] = {"drel": a["rel"] - b["rel"], "dlnB": np.log(a["B"] / b["B"]), "dlnW": np.log(a["W"] / b["W"]),
                      "dlnV": np.log(a["V"] / b["V"]), "dlnR": np.log((a["W"] - a["V"]) / (b["W"] - b["V"])),
                      "rel0": b["rel"], "Vshare0": b["V"] / b["W"]}
        return out

    def seedmean(fn):
        r1, r2 = fn(1), fn(2)
        return {G: {k: (r1[G][k] + r2[G][k]) / 2 for k in r1[G]} for G in "mw"}

    for arm, pat in ARMS.items():
        if not all((CACHE / f"{pat.format(s)}.npz").exists() for s in (1, 2)):
            continue
        print(f"\n== {arm} - 5b (seeds paired; seed means unless noted) ==")
        pt = seedmean(lambda s: arm_stats(E[pat.format(s)], E[BASE[s]], spks))
        bs = [seedmean(lambda s: arm_stats(E[pat.format(s)], E[BASE[s]], d)) for d in boots]
        for G in "mw":
            p = pt[G]
            ci = {k: np.percentile([b[G][k] for b in bs], [2.5, 97.5]) for k in ("drel", "dlnB", "dlnW", "dlnV", "dlnR")}
            k0 = p["rel0"] * (1 - p["rel0"])
            fmt = lambda k: f"{p[k]:+.3f} [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]"  # noqa: E731
            print(f"  {'men' if G == 'm' else 'women'}: d rel {fmt('drel')} (5b {p['rel0']:.3f}); d ln B {fmt('dlnB')}; d ln W {fmt('dlnW')}; "
                  f"first-order shares: B {k0 * p['dlnB']:+.3f}, W {-k0 * p['dlnW']:+.3f}")
            print(f"         W's parts: vowel d ln V {fmt('dlnV')} (vowel share of 5b's W {p['Vshare0']:.2f}); rest d ln R {fmt('dlnR')}")
        for s in (1, 2):
            r = arm_stats(E[pat.format(s)], E[BASE[s]], spks)
            print(f"  seed {s} alone: " + "; ".join(f"{'men' if G == 'm' else 'women'} d rel {r[G]['drel']:+.3f}, d ln B {r[G]['dlnB']:+.3f}, "
                                                    f"d ln W {r[G]['dlnW']:+.3f}, d ln V {r[G]['dlnV']:+.3f}" for G in "mw"))
        # (3) concentration of the change in within-speaker spread
        for G in "mw":
            sh = []
            for s in (1, 2):
                a, b = parts(E[pat.format(s)], fs, fv, spks[G]), parts(E[BASE[s]], fs, fv, spks[G])
                d = a["ssd"] - b["ssd"]
                tot = d.sum()
                top = np.sort(d)[::-1][:3].sum() if tot > 0 else np.sort(d)[:3].sum()
                sh.append(f"seed {s}: total {1200 ** 2 * tot / len(d):+.0f} c^2 per speaker, top 3 speakers {top / tot:.0%}, "
                          f"speakers with more spread {np.mean(d > 0):.0%}")
            print(f"  (3) {'men' if G == 'm' else 'women'} within-speaker spread change: " + "; ".join(sh))
        # (4) unweighted frame mean
        du = np.mean([rel(EU[pat.format(s)], spks) - rel(EU[BASE[s]], spks) for s in (1, 2)])
        dw = np.mean([rel(E[pat.format(s)], spks) - rel(E[BASE[s]], spks) for s in (1, 2)])
        bu = np.percentile([np.mean([rel(EU[pat.format(s)], d) - rel(EU[BASE[s]], d) for s in (1, 2)]) for d in boots[:500]], [2.5, 97.5])
        print(f"  (4) reliability change, precision-weighted {dw:+.3f}; unweighted frames {du:+.3f} [{bu[0]:+.3f}, {bu[1]:+.3f}] (500 draws)")
        # (5) per vowel
        for G in "mw":
            dv = {u: np.mean([parts(E[pat.format(s)], fs, fv, spks[G])["vowel_dev"][u] - parts(E[BASE[s]], fs, fv, spks[G])["vowel_dev"][u]
                              for s in (1, 2)]) for u in vowels}
            x, y = np.array([vf1[u] for u in vowels]), np.array([1200 * dv[u] for u in vowels])
            r = np.corrcoef(x, y)[0, 1]
            print(f"  (5) {'men' if G == 'm' else 'women'} vowel deviation change (c), r with vowel F1 {r:+.2f}: "
                  + " ".join(f"{u} {1200 * dv[u]:+.0f}" for u in sorted(vowels, key=lambda u: vf1[u])))
    print("\nScale: 5b's seeds (s2 - s1, unpaired arms): " + "; ".join(
        f"{'men' if G == 'm' else 'women'} d rel {parts(E[BASE[2]], fs, fv, spks[G])['rel'] - parts(E[BASE[1]], fs, fv, spks[G])['rel']:+.3f}, "
        f"d ln B {np.log(parts(E[BASE[2]], fs, fv, spks[G])['B'] / parts(E[BASE[1]], fs, fv, spks[G])['B']):+.3f}, "
        f"d ln W {np.log(parts(E[BASE[2]], fs, fv, spks[G])['W'] / parts(E[BASE[1]], fs, fv, spks[G])['W']):+.3f}" for G in "mw"))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(dict(a.split("=", 1) for a in sys.argv[2:]))
    else:
        report()
