"""D-036 scoring: D-035's VTL measures for models with a jointly trained VTL head, read from the model's own output
("vtl_log2cm", "vtl_sigma"); per item the precision-weighted mean of log2 length over its frames.
- Synthetic validation: train_v2's first 512 held-out clips as cache.py's crops (seed 12345, 400 frames, the run's
  channel), sonorant frames: per-clip median |error| (cents) by kind; coverage within 1 / 2 sigma (frames).
- f0-invariance probe (eval.invariance; 0.3-0.7 s; analyse()): median |slope| of log2 length on log2 f0 at f0 <= 250 Hz,
  its shuffle null (f0 permuted within each (vowel, length), 1000 times) and the excess over the null's median; vowel
  spread; bias by length.
- Hillenbrand dev (reference frames; analyse(), windowed by default, the words are < 4 s): reliability (mean of men
  and women), women / men ratio, within-speaker SD, group medians, within-sex r(length, f0); paired speaker bootstrap
  (2000) against A (apparent length from labeller v3's F1-F4, as D-035) and against any other model given.
Args: out txt, then name=checkpoint pairs (VTL-head models), [--ref name=<d035 vtl_score cache dir>:<arm>].
Run from the main checkout."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vocal_analysis.backend.vtl import apparent_vtl
from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.device import setup_device
from vocal_analysis.eval import invariance as inv
from vocal_analysis.eval import tiers as TI
from vocal_analysis.eval.datasets import load_audio, registry
from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.infer import analyse, decode_outputs, load_teacher
from vocal_analysis.train.data import SynthCrops, collate, split_ids
from vocal_analysis.train.teacher import _saved_channel

WT = Path("../voice-analyser-parakeet/cache")
CROP = 400


def main():
    outf, models = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    OUT = open(outf, "w", encoding="utf-8")
    p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
    dev = setup_device()
    rng = np.random.default_rng(20261027)

    # ---- A's calibration (as D-035): tube crops' true F1-F4, length not from f0
    v = np.load(WT / "heads_l3_6" / "vtl.npz")
    son = np.load(WT / "heads_l3_6" / "sonorant.npy", mmap_mode="r")
    fm = np.load(WT / "heads_l3_6" / "formants.npy", mmap_mode="r")
    a4, yc = [], []
    for i in np.flatnonzero(v["tube"] & ~v["from_f0"]):
        s = np.asarray(son[i]) > 0.5
        if s.sum() >= 3:
            a4.append(np.nanmedian(apparent_vtl(np.asarray(fm[i])[s])))
            yc.append(np.log2(v["length_cm"][i]))
    cal = np.polyfit(np.log2(a4), yc, 1)
    A_of = lambda log2F: np.polyval(cal, np.log2(apparent_vtl(2 ** log2F)))  # noqa: E731

    def agg(mu, sig):
        w = 1 / sig ** 2
        return float((w * mu).sum() / w.sum())

    # ---- inputs shared by every model
    flat = Path("data/synth/train_v2/flat")
    _, val_ids = split_ids(json.loads(Path("data/synth/train_v2/index.json").read_text())["n_clips"])
    clip = val_ids[:512]
    L = np.log2(np.load(flat / "clip_length_cm.npy")[clip])
    F0c, TU = np.load(flat / "clip_length_from_f0.npy")[clip], np.load(flat / "clip_tube.npy")[clip]
    kinds = {"tube, not from f0": TU & ~F0c, "cascade, not from f0": ~TU & ~F0c, "length from f0": F0c}
    channel = _saved_channel(Path("runs/conformer_v2r2_s1"))
    val_batches = [(a, lab["sonorant"].numpy() > 0.5) for a, lab in
                   DataLoader(SynthCrops(flat, clip, CROP, channel, seed=12345, fixed=True), 16, num_workers=0, collate_fn=collate)]
    clips = inv.probe_clips()
    pf0 = np.log2([c.f0 for c in clips])
    ptruth = np.log2([c.length_cm for c in clips])
    keys = sorted({(c.vowel, c.length_cm) for c in clips})
    kid = np.array([keys.index((c.vowel, c.length_cm)) for c in clips])
    low = pf0 <= np.log2(250.0)
    mods = registry()
    hb = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        ok = np.isfinite(item.times)
        if ok.any():
            hb.append((load_audio(item, 16000), item.times[ok], row.speaker_id, item.meta.get("group", "?")))
    fs = np.array([s for _, _, s, _ in hb])
    fg = np.array([g for _, _, _, g in hb])
    spks = {G: sorted(set(fs[fg == G])) for G in "mwbg"}

    def slope_med(e, f):
        return np.median([abs(1200 * np.polyfit(f[(kid == k) & low], e[(kid == k) & low], 1)[0]) for k in range(len(keys))])

    # ---- per model: synthetic, probe, Hillenbrand files
    E, F0S, probe = {}, {}, {}
    vq = TorchVQT(dev)
    for name, ck in models.items():
        model = load_teacher(ck, dev)[0]
        est, zs = np.full(512, np.nan), []
        k = 0
        for audio, sm in val_batches:
            with torch.inference_mode():
                out = model(vq.log_power_with_context(audio.to(dev), CROP))
            for b in range(len(audio)):
                r = decode_outputs({kk: vv[b].float().cpu().numpy() for kk, vv in out.items()})
                if sm[b].sum():
                    mu, sg = r["vtl_log2cm"][sm[b]], r["vtl_sigma"][sm[b]]
                    est[k] = agg(mu, sg)
                    if kinds["tube, not from f0"][k]:
                        zs.append(np.abs(mu - L[k]) / sg)
                k += 1
        err = 1200 * np.abs(est - L)
        z = np.concatenate(zs)
        p(f"\n[{name}] synthetic per-clip median |error| (c): " + "  ".join(f"{kk} {np.nanmedian(err[m]):.0f}" for kk, m in kinds.items())
          + f"; coverage frames {np.mean(z < 1):.2f} / {np.mean(z < 2):.2f}")
        e = np.array([agg(*(lambda r: (r["vtl_log2cm"][30:71], r["vtl_sigma"][30:71]))(analyse(model, c.audio, dev))) for c in clips])
        obs = slope_med(e, pf0)
        nul = []
        for _ in range(1000):
            f = pf0.copy()
            for kk in range(len(keys)):
                m = (kid == kk) & low
                f[m] = rng.permutation(f[m])
            nul.append(slope_med(e, f))
        spread = np.mean([np.std([1200 * e[(kid == keys.index((vw, Lc))) & low].mean() for vw in inv.VOWELS]) for Lc in inv.LENGTHS_CM])
        probe[name] = (obs, np.median(nul))
        p(f"[{name}] probe: median |slope| f0 <= 250 Hz {obs:.1f} c/oct; shuffle null median {np.median(nul):.1f} (95 % {np.percentile(nul, 95):.1f}); "
          f"excess {obs - np.median(nul):+.1f}; P(null >= obs) {np.mean(np.array(nul) >= obs):.3f}; vowel spread {spread:.0f} c; bias by length "
          + "  ".join(f"{Lc:g}: {1200 * np.median((e - ptruth)[low & (np.array([c.length_cm for c in clips]) == Lc)]):+.0f}" for Lc in inv.LENGTHS_CM))
        fe, ff0 = [], []
        for x, t, _, _ in hb:
            r = analyse(model, x, dev)
            idx = np.clip(np.round(t / 0.01).astype(int), 0, len(r["f0"]) - 1)
            fe.append(agg(r["vtl_log2cm"][idx], r["vtl_sigma"][idx]))
            ff0.append(np.nanmedian(np.log2(r["f0"][idx])))
        E[name], F0S[name] = np.array(fe), np.array(ff0)
        del model
        torch.cuda.empty_cache()
    # A on Hillenbrand: labeller v3's formants at the reference frames (D-033's cache, dev; same file order checked)
    HB = WT / "emb_v1" / "gold_hillenbrand"
    ro = np.load(HB / "ref_off.npy")
    assert len(ro) - 1 == len(hb), (len(ro) - 1, len(hb))
    lf = (np.load(HB / "m_tv3_s1_log2F.npy") + np.load(HB / "m_tv3_s2_log2F.npy")) / 2
    a = A_of(lf)
    E["A"] = np.array([np.nanmedian(a[ro[j]:ro[j + 1]]) for j in range(len(hb))])
    lf0 = np.load(HB / "t_log2f0.npy")
    F0S["A"] = np.array([np.nanmedian(lf0[ro[j]:ro[j + 1]]) for j in range(len(hb))])
    f0_spk = {s: np.nanmedian(F0S["A"][fs == s]) for s in set(fs)}

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
    p(f"\nHillenbrand dev ({', '.join(f'{G} {len(spks[G])}' for G in 'mwbg')} speakers, {len(hb)} files): point [95 % speaker bootstrap]")
    for m in pt["A"]:
        p(f"  {m}")
        for n in E:
            x = np.array([b[n][m] for b in bs])
            p(f"    {n:28s} {pt[n][m]:8.3f} [{np.percentile(x, 2.5):8.3f}, {np.percentile(x, 97.5):8.3f}]")
    p("\nPaired differences (speaker bootstrap)")
    names = [n for n in E if n != "A"]
    for a_ in names:
        for b_ in ["A"] + [n for n in names if n != a_]:
            for m in ("reliability", "w/m", "within SD c"):
                x = np.array([r[a_][m] - r[b_][m] for r in bs])
                p(f"  {a_ + ' - ' + b_:44s} {m:12s} {pt[a_][m] - pt[b_][m]:+8.3f} [{np.percentile(x, 2.5):+8.3f}, {np.percentile(x, 97.5):+8.3f}]")
    json.dump({n: {"reliability_ci_vs_A": list(np.percentile([r[n]["reliability"] - r["A"]["reliability"] for r in bs], [2.5, 97.5])),
                   "w/m": pt[n]["w/m"], "probe_excess": probe[n][0] - probe[n][1], "probe_obs": probe[n][0]} for n in names},
              open(Path(outf).with_suffix(".json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
