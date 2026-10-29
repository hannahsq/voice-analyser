"""D-037 scoring (pre-registered in docs/decisions.md): VTL heads refitted on frozen trunks (d037_head.py) and each joint
trunk's own head (D-036, from the cached output), on d037_cache.py's parts. Estimators: "<arm>" = a refit arm's seed
mean (mu the mean of the seeds' means, sigma^2 the mean of their variances plus the variance of their means, as D-035);
"J<trunk>" = the trunk's own joint head; "A" = apparent length from labeller v3's F1-F4 (as D-035 / D-036).
Per item, the precision-weighted mean of log2 length over its frames.
- Synthetic validation (train_v2's first 512 held-out clips; sonorant frames; tube clips with length not from f0 for
  NLL, SNR and coverage): per-clip median |error| by kind, validation NLL and SNR (20 log10(label SD / sigma)).
- Probe (eval.invariance, frames 30-70 as D-036): median |slope| at f0 <= 250 Hz, shuffle null (1000), excess; vowel
  spread; bias by length.
- Hillenbrand dev (reference frames): D-036's measures, paired speaker bootstrap (2000).
- Vowel classification (reported only): 12 vowels from the trunk's own F1-F3 (per-file median of log2 Hz at the
  reference frames): raw; + log2 of the speaker's length from an estimator (median over the speaker's other files);
  Lobanov (per speaker and formant z-score). LDA (pooled covariance), 5 folds grouped by speaker; accuracy with a speaker
  bootstrap.
- Decisions: Q1 (R3 against J3: SNR gain >= 0.5 dB with its paired clip-bootstrap interval excluding 0, and reliability
  better with its paired interval excluding 0; reported unreliable if an R3 seed is flagged as overfitting) and Q3 (R3c
  against R3: reliability better; probe excess <= 10 and rising by at most 3; w/m in [0.85, 0.92]; vowel spread rising by
  at most 10 c).
Args: out txt, then arm=<trunk cache dir>:<head dir prefix> pairs for refit arms (seeds _s0, _s1, _s2), then
J<name>=<trunk cache dir> for joint heads. Run from the main checkout."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parakeet"))
from vtl_head import Head  # noqa: E402

from vocal_analysis.backend.vtl import apparent_vtl  # noqa: E402
from vocal_analysis.eval import invariance as inv  # noqa: E402
from vocal_analysis.train.data import split_ids  # noqa: E402

LABEL_SD = 0.2852  # as d037_head.py
SEEDS = (0, 1, 2)
VOWELS12 = None  # Hillenbrand's vowel codes, from the file names (speaker id + two letters)


def load_heads(prefix):
    hs = []
    for s in SEEDS:
        d = Path(f"{prefix}_s{s}")
        ck = torch.load(d / "head.pt", map_location="cpu", weights_only=False)
        h = Head(ck["in_dim"])
        h.load_state_dict(ck["state"])
        hs.append((h.eval(), ck["kind"], ck["stats"], json.loads((d / "done").read_text())))
    return hs


def refit_frames(hs, feats, log2F):
    """per-frame seed-mean (mu, var) for frames (n, 128) and their F1-F4 (n, 4)"""
    mus, vs = [], []
    for h, kind, stats, _ in hs:
        x = np.asarray(feats, np.float32)
        if kind == "hF":
            x = np.concatenate([x, (np.asarray(log2F, np.float32) - stats[0]) / stats[1]], -1)
        with torch.inference_mode():
            m, lv = h(torch.from_numpy(x))
        mus.append(m.numpy())
        vs.append(np.exp(lv.numpy()))
    mus, vs = np.stack(mus), np.stack(vs)
    return mus.mean(0), vs.mean(0) + mus.var(0)


def agg(mu, var):
    w = 1 / var
    return float((w * mu).sum() / w.sum())


def frame_nll(mu, var, y):
    return float(np.mean(0.5 * (np.log(var) + (y - mu) ** 2 / var)))


def snr(nll):
    return 20 * np.log10(LABEL_SD / np.exp(nll - 0.5))


def lda_cv(X, y, groups, folds=5, seed=0):
    """leave-speakers-out LDA accuracy per item (1 = correct)"""
    rng = np.random.default_rng(seed)
    gs = np.array(sorted(set(groups)))
    fold_of = dict(zip(gs, rng.permutation(len(gs)) % folds))
    f = np.array([fold_of[g] for g in groups])
    correct = np.zeros(len(y), bool)
    classes = np.array(sorted(set(y)))
    for k in range(folds):
        tr, te = f != k, f == k
        mu = np.stack([X[tr & (y == c)].mean(0) for c in classes])
        resid = np.concatenate([X[tr & (y == c)] - mu[i] for i, c in enumerate(classes)])
        S = np.cov(resid.T) + 1e-6 * np.eye(X.shape[1])
        Si = np.linalg.inv(S)
        d = X[te] @ Si @ mu.T - 0.5 * np.sum(mu @ Si * mu, 1)
        correct[te] = classes[d.argmax(1)] == y[te]
    return correct


def main():
    outf = sys.argv[1]
    refit = {a.split("=", 1)[0]: a.split("=", 1)[1].rsplit(":", 1) for a in sys.argv[2:] if not a.startswith("J")}
    joint = {a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in sys.argv[2:] if a.startswith("J")}
    OUT = open(outf, "w", encoding="utf-8")
    p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
    rng = np.random.default_rng(20261028)

    # ---- A's calibration (as D-035 / D-036)
    W = Path("cache")
    v = np.load(W / "heads_l3_6" / "vtl.npz")
    son = np.load(W / "heads_l3_6" / "sonorant.npy", mmap_mode="r")
    fm = np.load(W / "heads_l3_6" / "formants.npy", mmap_mode="r")
    a4, yc = [], []
    for i in np.flatnonzero(v["tube"] & ~v["from_f0"]):
        s = np.asarray(son[i]) > 0.5
        if s.sum() >= 3:
            a4.append(np.nanmedian(apparent_vtl(np.asarray(fm[i])[s])))
            yc.append(np.log2(v["length_cm"][i]))
    cal = np.polyfit(np.log2(a4), yc, 1)
    A_of = lambda log2F: np.polyval(cal, np.log2(apparent_vtl(2 ** log2F)))  # noqa: E731

    # ---- shared inputs
    flat = Path("data/synth/train_v2/flat")
    _, val_ids = split_ids(json.loads(Path("data/synth/train_v2/index.json").read_text())["n_clips"])
    clip = val_ids[:512]
    L = np.log2(np.load(flat / "clip_length_cm.npy")[clip])
    F0c, TU = np.load(flat / "clip_length_from_f0.npy")[clip], np.load(flat / "clip_tube.npy")[clip]
    kinds = {"tube, not from f0": TU & ~F0c, "cascade, not from f0": ~TU & ~F0c, "length from f0": F0c}
    vson = np.load(W / "emb_v1" / "synth_train_v2" / "sonorant.npy").reshape(512, 400) > 0.5
    clips = inv.probe_clips()
    pf0 = np.log2([c.f0 for c in clips])
    ptruth = np.log2([c.length_cm for c in clips])
    plen = np.array([c.length_cm for c in clips])
    keys = sorted({(c.vowel, c.length_cm) for c in clips})
    kid = np.array([keys.index((c.vowel, c.length_cm)) for c in clips])
    low = pf0 <= np.log2(250.0)
    HB = W / "emb_v1" / "gold_hillenbrand"
    ro = np.load(HB / "ref_off.npy")

    trunks = sorted({Path(c) for c, _ in refit.values()} | set(joint.values()))
    files = None
    for t in trunks:
        z = np.load(t / "hb" / "files.npz")
        assert np.array_equal(z["off"], ro), t  # the same files and frames as D-036's A
        files = z if files is None else files
    fs, fg, paths = files["speaker"], files["group"], files["path"]
    nf = len(fs)
    spks = {G: sorted(set(fs[fg == G])) for G in "mwbg"}

    def slope_med(e, f):
        return np.median([abs(1200 * np.polyfit(f[(kid == k) & low], e[(kid == k) & low], 1)[0]) for k in range(len(keys))])

    # ---- per estimator: per-frame (mu, var) on each part
    def parts_of(name):
        if name in joint:
            t = joint[name]
            fr = lambda part: (np.load(t / part / "vtl_mu.npy"), np.exp(np.load(t / part / "vtl_logvar.npy")))  # noqa: E731
            return fr("val"), fr("probe"), fr("hb")
        t, prefix = refit[name]
        t, hs = Path(t), load_heads(prefix)
        out = []
        for part in ("val", "probe", "hb"):
            f, F = np.load(t / part / "feats.npy"), np.load(t / part / "log2F.npy")
            sh = f.shape[:-1]
            mu, var = refit_frames(hs, f.reshape(-1, 128), F.reshape(-1, 4))
            out.append((mu.reshape(sh), var.reshape(sh)))
        return out

    E, VAL_NLL, PROBE, flags = {}, {}, {}, {}
    names = list(refit) + list(joint)
    p("D-037: VTL heads refitted on frozen trunks; joint heads J*; A = apparent length")
    for name in names:
        (vm, vv), (pm, pv), (hm, hv) = parts_of(name)
        est = np.full(512, np.nan)
        nll = {}
        zs = []
        for c in range(len(clip)):
            s = vson[c]
            if not s.any():
                continue
            est[c] = agg(vm[c][s], vv[c][s])
            if kinds["tube, not from f0"][c]:
                nll[c] = frame_nll(vm[c][s], vv[c][s], L[c])
                zs.append(np.abs(vm[c][s] - L[c]) / np.sqrt(vv[c][s]))
        VAL_NLL[name] = nll
        err = 1200 * np.abs(est - L)
        z = np.concatenate(zs)
        mean_nll = np.mean(list(nll.values()))
        line = f"\n[{name}] validation NLL {mean_nll:+.4f}, SNR {snr(mean_nll):.2f} dB; per-clip median |error| (c): " + "  ".join(
            f"{k} {np.nanmedian(err[m]):.0f}" for k, m in kinds.items()) + f"; coverage frames {np.mean(z < 1):.2f} / {np.mean(z < 2):.2f}"
        if name in refit:
            hs = load_heads(refit[name][1])
            flags[name] = [h[3]["overfit_flag"] for h in hs]
            line += "\n    seeds: " + "; ".join(f"s{s} final val {h[3]['final_val_snr_db']:.2f} dB (min NLL {h[3]['min_val_nll']:+.4f}), overfit flag {h[3]['overfit_flag']}"
                                             for s, h in zip(SEEDS, hs))
        p(line)
        e = np.array([agg(pm[k][30:71], pv[k][30:71]) for k in range(len(clips))])
        obs = slope_med(e, pf0)
        nul = []
        for _ in range(1000):
            f = pf0.copy()
            for kk in range(len(keys)):
                m = (kid == kk) & low
                f[m] = rng.permutation(f[m])
            nul.append(slope_med(e, f))
        spread = np.mean([np.std([1200 * e[(kid == keys.index((vw, Lc))) & low].mean() for vw in inv.VOWELS]) for Lc in inv.LENGTHS_CM])
        PROBE[name] = {"obs": obs, "null": float(np.median(nul)), "excess": obs - float(np.median(nul)), "spread": spread}
        p(f"[{name}] probe: median |slope| f0 <= 250 Hz {obs:.1f} c/oct; shuffle null median {np.median(nul):.1f} (95 % {np.percentile(nul, 95):.1f}); "
          f"excess {obs - np.median(nul):+.1f}; P(null >= obs) {np.mean(np.array(nul) >= obs):.3f}; vowel spread {spread:.0f} c; bias by length "
          + "  ".join(f"{Lc:g}: {1200 * np.median((e - ptruth)[low & (plen == Lc)]):+.0f}" for Lc in inv.LENGTHS_CM))
        E[name] = np.array([agg(hm[ro[j]:ro[j + 1]], hv[ro[j]:ro[j + 1]]) for j in range(nf)])

    lf = (np.load(HB / "m_tv3_s1_log2F.npy") + np.load(HB / "m_tv3_s2_log2F.npy")) / 2
    a = A_of(lf)
    E["A"] = np.array([np.nanmedian(a[ro[j]:ro[j + 1]]) for j in range(nf)])
    lf0 = np.load(HB / "t_log2f0.npy")
    f0_file = np.array([np.nanmedian(lf0[ro[j]:ro[j + 1]]) for j in range(nf)])
    f0_spk = {s: np.nanmedian(f0_file[fs == s]) for s in set(fs)}

    def measures(e, draw):
        sm = {G: np.array([np.nanmedian(e[fs == s]) for s in draw[G]]) for G in draw}
        within = np.concatenate([e[fs == s] - np.nanmean(e[fs == s]) for G in draw for s in draw[G]])
        rel = {}
        for G in "mw":
            means = np.array([np.nanmean(e[fs == s]) for s in draw[G]])
            wv = np.nanmean(np.concatenate([(e[fs == s] - np.nanmean(e[fs == s])) ** 2 for s in draw[G]]))
            rel[G] = np.var(means) / (np.var(means) + wv)
        cx = np.concatenate([np.array([np.nanmean(e[fs == s]) for s in draw[G]]) - np.mean([np.nanmean(e[fs == s]) for s in draw[G]]) for G in "mw"])
        cy = np.concatenate([np.array([f0_spk[s] for s in draw[G]]) - np.mean([f0_spk[s] for s in draw[G]]) for G in "mw"])
        return {"reliability": (rel["m"] + rel["w"]) / 2, "rel men": rel["m"], "rel women": rel["w"],
                "w/m": 2 ** (np.median(sm["w"]) - np.median(sm["m"])), "within SD c": 1200 * np.sqrt(np.nanmean(within ** 2)),
                "men cm": 2 ** np.median(sm["m"]), "women cm": 2 ** np.median(sm["w"]),
                "children cm": 2 ** np.median(np.concatenate([sm["b"], sm["g"]])), "r(len, f0) in sex": np.corrcoef(cx, cy)[0, 1]}

    pt = {n: measures(E[n], spks) for n in E}
    bs = []
    for _ in range(2000):
        dr = {G: [spks[G][j] for j in rng.integers(0, len(spks[G]), len(spks[G]))] for G in spks}
        bs.append({n: measures(E[n], dr) for n in E})
    ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))  # noqa: E731
    p(f"\nHillenbrand dev ({', '.join(f'{G} {len(spks[G])}' for G in 'mwbg')} speakers, {nf} files): point [95 % speaker bootstrap]")
    for m in pt["A"]:
        p(f"  {m}")
        for n in E:
            lo, hi = ci([b[n][m] for b in bs])
            p(f"    {n:10s} {pt[n][m]:8.3f} [{lo:8.3f}, {hi:8.3f}]")

    def paired(a_, b_, m):
        x = [r[a_][m] - r[b_][m] for r in bs]
        return pt[a_][m] - pt[b_][m], ci(x)

    p("\nPaired differences (speaker bootstrap)")
    pairs = [(n, "A") for n in names] + [(x, y) for x, y in (("R3", "J3"), ("R1", "J1"), ("R2", "J2"), ("R3c", "R3"), ("R1c", "R1"),
                                                             ("R3", "RT"), ("R1", "RT"), ("R2", "RT"), ("R1", "R3"), ("R2", "R3"), ("R3a", "R3"))
                                                             + tuple(tuple(q.split(":")) for q in filter(None, os.environ.get("VA_PAIRS", "").split(",")))  # D-040
                                                             if x in E and y in E]
    for a_, b_ in pairs:
        for m in ("reliability", "w/m", "within SD c"):
            d, (lo, hi) = paired(a_, b_, m)
            p(f"  {a_ + ' - ' + b_:12s} {m:12s} {d:+8.3f} [{lo:+8.3f}, {hi:+8.3f}]")

    # ---- validation SNR, paired clip bootstrap
    def snr_gain(a_, b_):
        cs = sorted(set(VAL_NLL[a_]) & set(VAL_NLL[b_]))
        na, nb = np.array([VAL_NLL[a_][c] for c in cs]), np.array([VAL_NLL[b_][c] for c in cs])
        g = snr(na.mean()) - snr(nb.mean())
        r = np.random.default_rng(20261029)
        boot = []
        for _ in range(2000):
            j = r.integers(0, len(cs), len(cs))
            boot.append(snr(na[j].mean()) - snr(nb[j].mean()))
        return g, ci(boot), len(cs)

    p("\nValidation SNR gain (dB), paired clip bootstrap (2000)")
    gains = {}
    for a_, b_ in (("R3", "J3"), ("R1", "J1"), ("R2", "J2"), ("R3c", "R3"), ("R1c", "R1"), ("RT", "J3")):
        if a_ in VAL_NLL and b_ in VAL_NLL:
            g, (lo, hi), n = snr_gain(a_, b_)
            gains[(a_, b_)] = (g, lo, hi)
            p(f"  {a_ + ' - ' + b_:12s} {g:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  ({n} clips)")

    # ---- vowel classification (reported only)
    vowel = np.array([Path(x).stem[3:] for x in paths])
    tr_dirs = {n: Path(refit[n][0]) if n in refit else joint[n] for n in names}
    p("\nVowel classification (Hillenbrand dev, 12 vowels, LDA, 5 folds grouped by speaker): accuracy [95 % speaker bootstrap]")
    vc = {}
    for t in sorted(set(tr_dirs.values())):
        F = np.load(t / "hb" / "log2F.npy")[:, :3]
        X = np.stack([np.nanmedian(F[ro[j]:ro[j + 1]], 0) for j in range(nf)])
        ok = np.isfinite(X).all(1)

        def spk_len(e):
            return np.array([np.nanmedian(e[(fs == fs[j]) & (np.arange(nf) != j)]) for j in range(nf)])

        ways = {"raw": X}
        for n in ["A"] + [n for n in names if tr_dirs[n] == t]:
            ways[f"+ log2 L ({n})"] = X + spk_len(E[n])[:, None]
        ways["Lobanov"] = np.stack([(X[:, i] - np.array([np.nanmean(X[fs == s, i]) for s in fs])) / np.array([np.nanstd(X[fs == s, i]) for s in fs])
                                    for i in range(3)], 1)
        p(f"  trunk {t.name} (its own F1-F3)")
        for w, Xw in ways.items():
            m = ok & np.isfinite(Xw).all(1)
            corr = lda_cv(Xw[m], vowel[m], fs[m])
            spk_list = sorted(set(fs[m]))
            boot = []
            for _ in range(2000):
                dr = rng.integers(0, len(spk_list), len(spk_list))
                boot.append(np.mean(np.concatenate([corr[fs[m] == spk_list[j]] for j in dr])))
            lo, hi = ci(boot)
            vc[f"{t.name} {w}"] = float(corr.mean())
            p(f"    {w:22s} {corr.mean():.3f} [{lo:.3f}, {hi:.3f}]  ({m.sum()} files)")

    # ---- decisions
    p("\nDecisions (arm 3's trunk)")
    res = {}
    if "R3" in E and "J3" in E:
        g, lo, hi = gains[("R3", "J3")]
        d, (rlo, rhi) = paired("R3", "J3", "reliability")
        under = g >= 0.5 and lo > 0 and rlo > 0
        unreliable = any(flags.get("R3", []))
        res["Q1"] = {"snr_gain": g, "snr_ci": [lo, hi], "rel_diff": d, "rel_ci": [rlo, rhi], "under_trained": under, "unreliable": unreliable}
        p(f"  Q1: SNR gain {g:+.2f} dB [{lo:+.2f}, {hi:+.2f}] (needs >= 0.5 and > 0); reliability {d:+.3f} [{rlo:+.3f}, {rhi:+.3f}] (needs > 0)"
          + f"{'; R3 seed flagged as overfitting: UNRELIABLE' if unreliable else ''}")
        p("  DECISION Q1: " + ("unreliable (overfitting flag); item 5 keeps D-036's setup until a rerun settles it" if unreliable else
                               "the joint head is under-trained: item 5 holds the VTL head's rate at its peak until the last 25 %" if under else
                               "not under-trained: item 5 keeps D-036's setup"))
    if "R3c" in E and "R3" in E:
        d, (rlo, rhi) = paired("R3c", "R3", "reliability")
        ex, ex0 = PROBE["R3c"]["excess"], PROBE["R3"]["excess"]
        wm = pt["R3c"]["w/m"]
        sp, sp0 = PROBE["R3c"]["spread"], PROBE["R3"]["spread"]
        ok3 = rlo > 0 and ex <= 10 and ex - ex0 <= 3 and 0.85 <= wm <= 0.92 and sp - sp0 <= 10
        res["Q3"] = {"rel_diff": d, "rel_ci": [rlo, rhi], "excess": ex, "excess_rise": ex - ex0, "w/m": wm, "spread_rise": sp - sp0, "joins": ok3}
        p(f"  Q3: reliability {d:+.3f} [{rlo:+.3f}, {rhi:+.3f}] (> 0); probe excess {ex:+.1f} (<= 10), rise {ex - ex0:+.1f} (<= 3); "
          f"w/m {wm:.3f} ([0.85, 0.92]); vowel spread rise {sp - sp0:+.0f} c (<= 10)")
        p("  DECISION Q3: " + ("the formant inputs join item 5's VTL head" if ok3 else "the VTL head reads h alone"))
    json.dump({"decisions": res, "probe": PROBE, "flags": flags, "vowel_accuracy": vc,
               "hillenbrand": {n: pt[n] for n in E}}, open(Path(outf).with_suffix(".json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
