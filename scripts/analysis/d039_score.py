"""D-039 scoring (exploratory, reported only): arms T and P (D-035's heads), E, ED, F, FD (d039_fusion.py) and W
(precision-weighted late fusion of T and P), with D-035 / D-036's measures and D-037's vowel classification.
Seed mean per arm (mu the mean of the seeds' means, sigma^2 the mean of their variances plus the variance of their means).
Frames: T, W and the fused arms at the teacher's 10 ms frames (each taking the Parakeet frame that covers it: crops
e = round((t + 4) / 8), whole files e = round(t / 8)); P alone at Parakeet's own frames, as D-035.
Args: out txt, runs dir of the fused arms (<dir>/<arm>_s<seed>/model.pt). Run from the main checkout."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parakeet"))
from d037_score import agg, frame_nll, lda_cv, snr  # noqa: E402
from d039_fusion import Early, Penultimate, load_d035  # noqa: E402
from vtl_head import p_frames  # noqa: E402

from vocal_analysis.backend.vtl import apparent_vtl  # noqa: E402
from vocal_analysis.eval import invariance as inv  # noqa: E402
from vocal_analysis.train.data import split_ids  # noqa: E402

SEEDS = (0, 1, 2)
FUSED = ("E", "ED", "F", "FD")
ARMS = ("T", "P") + FUSED + ("W",)


def seed_mean(rs):
    mus, vs = np.stack([m for m, _ in rs]), np.stack([v for _, v in rs])
    return mus.mean(0), vs.mean(0) + mus.var(0)


def main():
    outf, rdir = sys.argv[1], Path(sys.argv[2])
    OUT = open(outf, "w", encoding="utf-8")
    p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
    rng = np.random.default_rng(20261028)
    W_, VTL, EMB = Path("cache"), Path("cache/vtl"), Path("cache/emb_v1")

    heads = {a: [load_d035(a, s) for s in SEEDS] for a in ("T", "P")}
    fused = {}
    for a in FUSED:
        fused[a] = []
        for s in SEEDS:
            m = Early() if a in ("E", "ED") else Penultimate(s)
            m.load_state_dict(torch.load(rdir / f"{a}_s{s}" / "model.pt", map_location="cpu", weights_only=False)["state"])
            fused[a].append(m.eval())

    def run(module, *xs):
        with torch.inference_mode():
            m, lv = module(*[torch.from_numpy(np.asarray(x, np.float32)) for x in xs])
        return m.numpy(), np.exp(lv.numpy())

    def predict(arm, t, pt, pown=None):
        """per-frame (mu, var): t (n, 128) teacher frames, pt (n, 1024) Parakeet frames aligned to them; pown (m, 1024)
        Parakeet at its own frames (arm P only)"""
        if arm == "T":
            return seed_mean([run(h, t) for h in heads["T"]])
        if arm == "P":
            return seed_mean([run(h, pown) for h in heads["P"]])
        if arm == "W":
            mt, vt = seed_mean([run(h, t) for h in heads["T"]])
            mp, vp = seed_mean([run(h, pt) for h in heads["P"]])
            w = 1 / vt + 1 / vp
            return (mt / vt + mp / vp) / w, 1 / w
        keep = np.ones((len(t), 2), np.float32)
        return seed_mean([run(m, t, pt, keep) for m in fused[arm]])

    # ---- A's calibration (as D-035)
    v = np.load(W_ / "heads_l3_6" / "vtl.npz")
    son = np.load(W_ / "heads_l3_6" / "sonorant.npy", mmap_mode="r")
    fm = np.load(W_ / "heads_l3_6" / "formants.npy", mmap_mode="r")
    a4, yc = [], []
    for i in np.flatnonzero(v["tube"] & ~v["from_f0"]):
        s = np.asarray(son[i]) > 0.5
        if s.sum() >= 3:
            a4.append(np.nanmedian(apparent_vtl(np.asarray(fm[i])[s])))
            yc.append(np.log2(v["length_cm"][i]))
    cal = np.polyfit(np.log2(a4), yc, 1)
    A_of = lambda log2F: np.polyval(cal, np.log2(apparent_vtl(2 ** log2F)))  # noqa: E731

    # ---- synthetic validation
    flat = Path("data/synth/train_v2/flat")
    _, val_ids = split_ids(json.loads(Path("data/synth/train_v2/index.json").read_text())["n_clips"])
    clip = val_ids[:512]
    L = np.log2(np.load(flat / "clip_length_cm.npy")[clip])
    F0c, TU = np.load(flat / "clip_length_from_f0.npy")[clip], np.load(flat / "clip_tube.npy")[clip]
    kinds = {"tube, not from f0": TU & ~F0c, "cascade, not from f0": ~TU & ~F0c, "length from f0": F0c}
    vs = np.load(EMB / "synth_train_v2" / "sonorant.npy").reshape(512, 400) > 0.5
    vT = np.load(VTL / "val" / "feats.npy")
    vP = np.load(EMB / "synth_train_v2" / "emb.npy", mmap_mode="r")
    ve, vj = p_frames(51)
    p("D-039 (exploratory, reported only): VTL heads over teacher and Parakeet features together")
    p(f"\nSynthetic validation: 512 held-out clips ({', '.join(f'{k} {m.sum()}' for k, m in kinds.items())}); NLL and SNR on tube clips, length not from f0")
    for arm in ARMS:
        est, nlls = np.full(512, np.nan), []
        for c in range(512):
            if arm == "P":
                sel = vs[c][vj]
                if not sel.any():
                    continue
                mu, var = predict(arm, None, None, np.asarray(vP[51 * c + ve[sel], 5]))
                y = L[c]
            else:
                t = np.flatnonzero(vs[c])
                if not len(t):
                    continue
                e = np.clip(np.round((t + 4) / 8).astype(int), 1, 50)
                mu, var = predict(arm, vT[c][t], np.asarray(vP[51 * c + e, 5]))
                y = L[c]
            est[c] = agg(mu, var)
            if kinds["tube, not from f0"][c]:
                nlls.append(frame_nll(mu, var, y))
        err = 1200 * np.abs(est - L)
        n = float(np.mean(nlls))
        p(f"  {arm:3s} per-clip median |error| (c) " + "  ".join(f"{k} {np.nanmedian(err[m]):4.0f}" for k, m in kinds.items())
          + f"   NLL {n:+.3f}, SNR {snr(n):5.2f} dB" + ("   (W's sigma is overconfident by construction)" if arm == "W" else ""))

    # ---- probe
    meta = np.load(VTL / "probe" / "meta.npz")
    pT, pP = np.load(VTL / "probe" / "T.npy"), np.load(VTL / "probe" / "P.npy")
    pf0, ptruth, plen = np.log2(meta["f0"]), np.log2(meta["length_cm"]), meta["length_cm"]
    keys = sorted({(a, b) for a, b in zip(meta["vowel"], meta["length_cm"])})
    kid = np.array([keys.index((a, b)) for a, b in zip(meta["vowel"], meta["length_cm"])])
    low = pf0 <= np.log2(250.0)
    pe = np.arange(pP.shape[1])
    pe = pe[(8 * pe >= 30) & (8 * pe <= 70)]
    tt = np.arange(30, 71)
    te = np.clip(np.round(tt / 8).astype(int), 0, pP.shape[1] - 1)

    def slope_med(e, f):
        return np.median([abs(1200 * np.polyfit(f[(kid == k) & low], e[(kid == k) & low], 1)[0]) for k in range(len(keys))])

    p("\nf0-invariance probe (144 clips; f0 <= 250 Hz for the slopes; D-036's shuffle null, 1000)")
    PROBE = {}
    for arm in ARMS:
        e = np.array([agg(*(predict(arm, None, None, pP[c, pe]) if arm == "P" else predict(arm, pT[c, tt], pP[c, te]))) for c in range(len(pf0))])
        obs = slope_med(e, pf0)
        nul = []
        for _ in range(1000):
            f = pf0.copy()
            for kk in range(len(keys)):
                m = (kid == kk) & low
                f[m] = rng.permutation(f[m])
            nul.append(slope_med(e, f))
        spread = np.mean([np.std([1200 * e[(kid == keys.index((vw, Lc))) & low].mean() for vw in sorted(set(meta["vowel"]))])
                          for Lc in sorted(set(plen))])
        PROBE[arm] = {"obs": obs, "excess": obs - float(np.median(nul)), "spread": spread}
        p(f"  {arm:3s} median |slope| {obs:5.1f} c/oct; null median {np.median(nul):5.1f} (95 % {np.percentile(nul, 95):5.1f}); excess {obs - np.median(nul):+5.1f}; "
          f"P(null >= obs) {np.mean(np.array(nul) >= obs):.3f}; vowel spread {spread:3.0f} c; bias by length "
          + "  ".join(f"{Lc:g}: {1200 * np.median((e - ptruth)[low & (plen == Lc)]):+4.0f}" for Lc in sorted(set(plen))))

    # ---- Hillenbrand dev (as D-035)
    HB = EMB / "gold_hillenbrand"
    g, spk = np.load(HB / "group.npy"), np.load(HB / "speaker.npy")
    ro, eo, mel = np.load(HB / "ref_off.npy"), np.load(HB / "emb_off.npy"), np.load(HB / "mel.npy")
    hP = np.load(HB / "emb.npy", mmap_mode="r")
    hT = np.load(VTL / "hb" / "feats.npy")
    nf = len(ro) - 1
    fg, fs = g[ro[:-1]], spk[ro[:-1]]
    Pfr = np.concatenate([np.asarray(hP[eo[k] + np.clip(np.round(mel[ro[k]:ro[k + 1]] / 8).astype(int), 0, eo[k + 1] - eo[k] - 1), 5]) for k in range(nf)])
    E = {}
    for arm in ARMS:
        mu, var = predict(arm, None, None, Pfr) if arm == "P" else predict(arm, hT, Pfr)
        E[arm] = np.array([agg(mu[ro[j]:ro[j + 1]], var[ro[j]:ro[j + 1]]) for j in range(nf)])
    lf = (np.load(HB / "m_tv3_s1_log2F.npy") + np.load(HB / "m_tv3_s2_log2F.npy")) / 2
    a = A_of(lf)
    E["A"] = np.array([np.nanmedian(a[ro[j]:ro[j + 1]]) for j in range(nf)])
    lf0 = np.load(HB / "t_log2f0.npy")
    f0_file = np.array([np.nanmedian(lf0[ro[j]:ro[j + 1]]) for j in range(nf)])
    f0_spk = {s: np.nanmedian(f0_file[fs == s]) for s in set(fs)}
    spks = {G: sorted(set(fs[fg == G])) for G in "mwbg"}

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
            p(f"    {n:3s} {pt[n][m]:8.3f} [{lo:8.3f}, {hi:8.3f}]")
    p("\nPaired differences (speaker bootstrap)")
    for a_ in ARMS:
        for b_ in ("A", "T", "P"):
            if a_ == b_:
                continue
            for m in ("reliability", "w/m", "within SD c"):
                lo, hi = ci([r[a_][m] - r[b_][m] for r in bs])
                p(f"  {a_ + ' - ' + b_:8s} {m:12s} {pt[a_][m] - pt[b_][m]:+8.3f} [{lo:+8.3f}, {hi:+8.3f}]")

    # ---- vowel classification (teacher v3's F1-F3 at the reference frames)
    paths = np.load(HB / "speaker.npy")  # speaker per frame; vowels from the manifest order below
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    z = np.load("cache/d037/arm3/hb/files.npz") if Path("cache/d037/arm3/hb/files.npz").exists() else None
    if z is not None and np.array_equal(z["off"], ro):
        vowel = np.array([Path(x).stem[3:] for x in z["path"]])
        F = np.load(HB / "m_tv3_s1_log2F.npy")[:, :3]
        X = np.stack([np.nanmedian(F[ro[j]:ro[j + 1]], 0) for j in range(nf)])
        ok = np.isfinite(X).all(1)
        p("\nVowel classification (Hillenbrand dev, 12 vowels, teacher v3's F1-F3, LDA, 5 folds grouped by speaker): accuracy [95 % speaker bootstrap]")
        ways = {"raw": X}
        for n in ["A"] + list(ARMS):
            sl = np.array([np.nanmedian(E[n][(fs == fs[j]) & (np.arange(nf) != j)]) for j in range(nf)])
            ways[f"+ log2 L ({n})"] = X + sl[:, None]
        ways["Lobanov"] = np.stack([(X[:, i] - np.array([np.nanmean(X[fs == s, i]) for s in fs])) / np.array([np.nanstd(X[fs == s, i]) for s in fs])
                                    for i in range(3)], 1)
        for w, Xw in ways.items():
            m = ok & np.isfinite(Xw).all(1)
            corr = lda_cv(Xw[m], vowel[m], fs[m])
            sl_ = sorted(set(fs[m]))
            boot = [np.mean(np.concatenate([corr[fs[m] == sl_[j]] for j in rng.integers(0, len(sl_), len(sl_))])) for _ in range(2000)]
            lo, hi = ci(boot)
            p(f"  {w:16s} {corr.mean():.3f} [{lo:.3f}, {hi:.3f}]")
    json.dump({"probe": PROBE, "hillenbrand": pt}, open(Path(outf).with_suffix(".json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
