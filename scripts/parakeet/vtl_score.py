"""D-035 scoring (pre-registered in docs/decisions.md): the VTL heads T, P, Tc (three seeds each; an arm's estimate is the
seed mean, sigma^2 = mean of sigma^2 + variance of the means) against the formant-derived apparent length A.
Per item (clip or file) the estimate is the precision-weighted mean of log2 length over its frames.
- Synthetic validation (train_v2's first 512 held-out clips, cache.py's crops): per-clip |error| (cents) by clip kind;
  sigma coverage within 1 and 2 sigma, per frame and per clip.
- f0-invariance probe (eval.invariance; 0.3-0.7 s): slope of log2 length against log2 f0 per (vowel, length), cents per
  octave (guard: median |slope| at f0 <= 250 Hz <= 10); vowel spread; bias by length.
- Hillenbrand dev (reference frames): reliability (men, women, mean; primary), women / men ratio of speaker medians (guard
  [0.85, 0.92]), within-speaker SD, group medians, the within-group correlation of speaker length with speaker median f0.
  Paired speaker bootstrap within group, 2000 resamples.
Decision: the head joins item 5 if T's mean reliability exceeds A's (interval excluding 0), T passes both guards, and
every T seed's reliability exceeds A's. Args: vtl cache dir, out txt. Run from the main checkout."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vtl_head import Head, p_frames  # noqa: E402

from vocal_analysis.backend.vtl import apparent_vtl  # noqa: E402

VTL, OUTF = Path(sys.argv[1]), sys.argv[2]
WT = Path(__file__).resolve().parents[2]
HEADS, EMB = WT / "cache" / "heads_l3_6", WT / "cache" / "emb_v1"
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
ARMS = ("T", "P", "Tc")
SEEDS = (0, 1, 2)
FEAT = {"T": "T", "Tc": "T", "P": "P"}
rng = np.random.default_rng(20261026)


def heads(arm):
    out = []
    for s in SEEDS:
        ck = torch.load(VTL / "heads" / f"{arm}_s{s}" / "head.pt", map_location="cpu", weights_only=False)
        h = Head(ck["in_dim"])
        h.load_state_dict(ck["state"])
        out.append(h.eval())
    return out


H = {a: heads(a) for a in ARMS}


def predict(arm, x, seed=None):
    """x (n, in) -> mean, variance of log2 cm: the seed mean (or one seed's)."""
    with torch.inference_mode():
        r = [h(torch.from_numpy(np.asarray(x, np.float32))) for h in (H[arm] if seed is None else [H[arm][seed]])]
    mu = np.stack([m.numpy() for m, _ in r])
    var = np.stack([np.exp(v.numpy()) for _, v in r])
    return mu.mean(0), var.mean(0) + mu.var(0)


def agg(mu, var):
    w = 1 / var
    return float((w * mu).sum() / w.sum()), float(np.sqrt(1 / w.sum()))


# ---- A's calibration: tube crops' true F1-F4 (per-crop median of the uniform-tube length), length not from f0
v = np.load(HEADS / "vtl.npz")
son = np.load(HEADS / "sonorant.npy", mmap_mode="r")
fm = np.load(HEADS / "formants.npy", mmap_mode="r")
a4, yc = [], []
for i in np.flatnonzero(v["tube"] & ~v["from_f0"]):
    s = np.asarray(son[i]) > 0.5
    if s.sum() >= 3:
        a4.append(np.nanmedian(apparent_vtl(np.asarray(fm[i])[s])))
        yc.append(np.log2(v["length_cm"][i]))
CAL = np.polyfit(np.log2(a4), yc, 1)
A_of = lambda log2F: np.polyval(CAL, np.log2(apparent_vtl(2 ** log2F)))  # noqa: E731  per frame, log2 cm
p(f"D-035 VTL probe. A's calibration: slope {CAL[0]:.3f} on {len(yc)} tube crops")

# ---- synthetic validation
flat = Path("data/synth/train_v2/flat")
clip = np.load(VTL / "val" / "clip.npy")
L = np.log2(np.load(flat / "clip_length_cm.npy")[clip])
F0c = np.load(flat / "clip_length_from_f0.npy")[clip]
TU = np.load(flat / "clip_tube.npy")[clip]
vs = np.load(EMB / "synth_train_v2" / "sonorant.npy").reshape(512, 400) > 0.5
vT = np.load(VTL / "val" / "feats.npy")
vP = np.load(EMB / "synth_train_v2" / "emb.npy", mmap_mode="r")
ve, vj = p_frames(51)
vlf = ((np.load(EMB / "synth_train_v2" / "m_tv3_s1_log2F.npy") + np.load(EMB / "synth_train_v2" / "m_tv3_s2_log2F.npy")) / 2).reshape(512, 400, 4)
kinds = {"tube, not from f0": TU & ~F0c, "cascade, not from f0": ~TU & ~F0c, "length from f0": F0c}
p(f"\nSynthetic validation: 512 held-out train_v2 clips ({', '.join(f'{k} {m.sum()}' for k, m in kinds.items())})")
p("  per-clip median |error| (cents) by kind; coverage within 1 / 2 sigma on tube, not-from-f0 clips (frames; clips)")
for arm in ARMS + ("A",):
    est, cov_f, sig = np.full(512, np.nan), [], np.full(512, np.nan)
    for c in range(512):
        if arm == "A":
            s = vs[c]
            if s.sum() >= 3:
                est[c] = np.nanmedian(A_of(vlf[c][s]))
            continue
        if FEAT[arm] == "T":
            x = vT[c][vs[c]]
        else:
            sel = vs[c][vj]
            x = np.asarray(vP[51 * c + ve[sel], 5])
        if len(x) == 0:
            continue
        mu, var = predict(arm, x)
        est[c], sig[c] = agg(mu, var)
        if TU[c] and not F0c[c]:
            z = np.abs(mu - L[c]) / np.sqrt(var)
            cov_f.append(z)
    err = 1200 * np.abs(est - L)
    line = "  ".join(f"{k} {np.nanmedian(err[m]):5.0f}" for k, m in kinds.items())
    if arm != "A":
        z = np.concatenate(cov_f)
        zc = (np.abs(est - L) / sig)[kinds["tube, not from f0"]]
        line += f"   coverage frames {np.mean(z < 1):.2f} / {np.mean(z < 2):.2f}; clips {np.nanmean(zc < 1):.2f} / {np.nanmean(zc < 2):.2f}"
    p(f"  {arm:3s} {line}")

# ---- probe
meta = np.load(VTL / "probe" / "meta.npz")
pT, pP = np.load(VTL / "probe" / "T.npy"), np.load(VTL / "probe" / "P.npy")
f0 = np.log2(meta["f0"])
truth = np.log2(meta["length_cm"])
keys = sorted({(a, b) for a, b in zip(meta["vowel"], meta["length_cm"])})
kid = np.array([keys.index((a, b)) for a, b in zip(meta["vowel"], meta["length_cm"])])
low = f0 <= np.log2(250.0)
pe = np.arange(pP.shape[1])
pe = pe[(8 * pe >= 30) & (8 * pe <= 70)]


def probe_est(arm, seed=None):
    if arm == "A":
        from vocal_analysis.eval import invariance as inv
        from vocal_analysis.eval.teacher_eval import teacher_baselines
        prs = [teacher_baselines(f"runs/conformer_v2r2_s{k}_d16k/ckpt_last.pt", f"s{k}")[0].predict for k in (1, 2)]
        times = np.arange(0.3, 0.7001, 0.01)
        return np.array([np.nanmedian(A_of(np.mean([np.log2(pr(c.audio, inv.FS, times, {})[:, :4]) for pr in prs], 0)))
                         for c in inv.probe_clips()])
    return np.array([agg(*predict(arm, pT[c, 30:71] if FEAT[arm] == "T" else pP[c, pe], seed))[0] for c in range(len(f0))])


p("\nf0-invariance probe (144 clips; Story's vowels at 13 / 15.5 / 18 cm, f0 80-400 Hz): slope of log2 length on log2 f0")
probe_slope = {}
for arm in ARMS + ("A",):
    e = probe_est(arm)
    sl = np.array([1200 * np.polyfit(f0[kid == k], e[kid == k], 1)[0] for k in range(len(keys))])
    sl_lo = np.array([1200 * np.polyfit(f0[(kid == k) & low], e[(kid == k) & low], 1)[0] for k in range(len(keys))])
    probe_slope[arm] = np.median(np.abs(sl_lo))
    spread = np.mean([np.std([1200 * e[(kid == keys.index((vw, Lc))) & low].mean() for vw in sorted(set(meta["vowel"]))])
                      for Lc in sorted(set(meta["length_cm"]))])
    bias = "  ".join(f"{Lc:g}: {1200 * np.median((e - truth)[low & (meta['length_cm'] == Lc)]):+.0f}" for Lc in sorted(set(meta["length_cm"])))
    p(f"  {arm:3s} median |slope| f0 <= 250 Hz {probe_slope[arm]:5.1f} c/oct (median {np.median(sl_lo):+.1f}); whole sweep "
      f"{np.median(np.abs(sl)):5.1f}; vowel spread {spread:4.0f} c; bias by length (c) {bias}")

# ---- Hillenbrand dev
HB = EMB / "gold_hillenbrand"
g, spk = np.load(HB / "group.npy"), np.load(HB / "speaker.npy")
ro, eo, mel = np.load(HB / "ref_off.npy"), np.load(HB / "emb_off.npy"), np.load(HB / "mel.npy")
hP = np.load(HB / "emb.npy", mmap_mode="r")
hT = np.load(VTL / "hb" / "feats.npy")
lf = (np.load(HB / "m_tv3_s1_log2F.npy") + np.load(HB / "m_tv3_s2_log2F.npy")) / 2
lf0 = np.load(HB / "t_log2f0.npy")
nf = len(ro) - 1
fg, fs = g[ro[:-1]], spk[ro[:-1]]
Pfr = np.concatenate([np.asarray(hP[eo[k] + np.clip(np.round(mel[ro[k]:ro[k + 1]] / 8).astype(int), 0, eo[k + 1] - eo[k] - 1), 5])
                      for k in range(nf)])


def hb_files(arm, seed=None):
    if arm == "A":
        a = A_of(lf)
        return np.array([np.nanmedian(a[ro[k]:ro[k + 1]]) for k in range(nf)])
    mu, var = predict(arm, hT if FEAT[arm] == "T" else Pfr, seed)
    return np.array([agg(mu[ro[k]:ro[k + 1]], var[ro[k]:ro[k + 1]])[0] for k in range(nf)])


spks = {G: sorted(set(fs[fg == G])) for G in "mwbg"}
f0_spk = {s: np.nanmedian(lf0[spk == s]) for s in set(spk)}


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


E = {a: hb_files(a) for a in ARMS + ("A",)}
pt = {a: measures(E[a], spks) for a in E}
bs = []
for _ in range(2000):
    d = {G: [spks[G][j] for j in rng.integers(0, len(spks[G]), len(spks[G]))] for G in spks}
    bs.append({a: measures(E[a], d) for a in E})
p(f"\nHillenbrand dev ({', '.join(f'{G} {len(spks[G])}' for G in 'mwbg')} speakers, {nf} files): point [95 % speaker bootstrap]")
for m in pt["A"]:
    p(f"  {m}")
    for a in ARMS + ("A",):
        x = np.array([b[a][m] for b in bs])
        p(f"    {a:3s} {pt[a][m]:8.3f} [{np.percentile(x, 2.5):8.3f}, {np.percentile(x, 97.5):8.3f}]")
p("\nPaired differences (speaker bootstrap)")
ci = {}
for a, b in (("T", "A"), ("P", "A"), ("P", "T"), ("Tc", "T")):
    for m in ("reliability", "w/m", "within SD c"):
        x = np.array([r[a][m] - r[b][m] for r in bs])
        ci[(a, b, m)] = np.percentile(x, [2.5, 97.5])
        p(f"  {a + ' - ' + b:7s} {m:12s} {pt[a][m] - pt[b][m]:+8.3f} [{ci[(a, b, m)][0]:+8.3f}, {ci[(a, b, m)][1]:+8.3f}]")
seed_rel = {a: [measures(hb_files(a, s), spks)["reliability"] for s in range(len(SEEDS))] for a in ("T", "P")}
p("\nPer-seed reliability: " + "; ".join(f"{a} " + " ".join(f"{r:.3f}" for r in seed_rel[a]) for a in seed_rel) + f"; A {pt['A']['reliability']:.3f}")


def passes(a):
    c = [("reliability above A", ci[(a, "A", "reliability")][0] > 0), ("w/m in [0.85, 0.92]", 0.85 <= pt[a]["w/m"] <= 0.92),
         ("probe |slope| <= 10", probe_slope[a] <= 10), ("every seed above A", all(r > pt["A"]["reliability"] for r in seed_rel[a]))]
    return all(ok for _, ok in c), ", ".join(f"{n}: {'yes' if ok else 'NO'}" for n, ok in c)


tp, tw = passes("T")
pp, pw = passes("P")
p(f"\nT: {tw}\nP: {pw}")
p(f"DECISION: the VTL head {'joins' if tp else 'does not join'} item 5's retrain"
  + ("" if tp or not pp else "; P passes where T fails: teacher v3's features limit VTL (a case for VTL on the D-034 student)"))
