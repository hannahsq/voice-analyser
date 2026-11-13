"""D-113: does Hillenbrand's recording chain block D-109's F1 shift? Inference only, dev speakers only.

The chain curves are measured from the two corpora's own audio (women, dev): the long-term spectrum of speech frames
(frames within 25 dB of the 95th-percentile frame level) and the noise floor (frames within 3 dB of the 5th percentile),
1/3-octave smoothed, at 16 kHz after the usual resampling. D(f) = Hillenbrand - PTDB-TUG speech spectrum.
Conditions on PTDB-TUG (D-110's closed-phase frames, `cache/m3w/ptdb_egg_sub.npz`):
  base   as scored before (must reproduce the cache)
  high   D(f) above 3 kHz (tapered in from 2.5 kHz): Hillenbrand's roll-off
  low    D(f) below 300 Hz (tapered out by 400 Hz): Hillenbrand's weaker low end
  noise  noise with Hillenbrand's floor spectrum at its level below each file's speech
  all    D(f) at every frequency, then the noise
Conditions on Hillenbrand (the tokens' labelled frames, as the paired scores):
  base, rev_high, rev_low, rev_all   the same curves with the sign reversed (towards PTDB-TUG's chain)
EQ: linear-phase FIR (firwin2, 1023 taps at 16 kHz), applied centred. Per frame: F1-F3 of each model.

    python scripts/analysis/d113_chain.py <out dir> [--null-only]   (VA_DEVICE=cuda for speed)
"""

import glob
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

B = {"b1": "runs/d040_5b_s1_b20k/ckpt_last.pt", "b2": "runs/d040_5b_s2_b20k/ckpt_last.pt"}
N = {"n1": "runs/d065_gain_s1_b20k/ckpt_last.pt", "n2": "runs/d065_gain_s2_b20k/ckpt_last.pt"}
R = {"r1": "runs/d109_s1/ckpt_last.pt", "r2": "runs/d109_s2/ckpt_last.pt"}
FS, NPER = 16000, 1024
F = np.fft.rfftfreq(NPER, 1 / FS)
SEED = 5113


def to16(x, sr):
    from scipy.signal import resample_poly

    if sr == FS:
        return x.astype(np.float32)
    g = np.gcd(int(sr), FS)
    return resample_poly(x, FS // g, int(sr) // g).astype(np.float32)


def frames_power(x):
    fr = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), NPER)[:: NPER // 2] * np.hanning(NPER)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    return P, 10 * np.log10(P.sum(1) + 1e-20)


def third_oct(db):
    out = np.empty_like(db)
    for i, fc in enumerate(F):
        m = (F >= max(fc, 20) * 2 ** (-1 / 6)) & (F <= max(fc, 20) * 2 ** (1 / 6))
        out[i] = 10 * np.log10(np.mean(10 ** (db[m] / 10))) if m.any() else db[i]
    return out


def chain_curves(hb_files, pt_files):
    def ltas(files, load):
        Ps, es = [], []
        for p in files:
            P, e = frames_power(load(p))
            Ps.append(P), es.append(e)
        P, e = np.concatenate(Ps), np.concatenate(es)
        lo, hi = np.percentile(e, 5), np.percentile(e, 95)
        return third_oct(10 * np.log10(P[e > hi - 25].mean(0))), third_oct(10 * np.log10(P[e < lo + 3].mean(0)))

    import soundfile as sf

    hs, hn = ltas(hb_files, lambda p: sf.read(p)[0])
    ps, _ = ltas(pt_files, lambda p: to16(sf.read(p)[0], 48000))
    band = (F >= 800) & (F < 1600)
    ref = lambda s: 10 * np.log10(np.mean(10 ** (s[band] / 10)))  # noqa: E731
    D = (hs - ref(hs)) - (ps - ref(ps))
    floor_rel = hn - ref(hs)  # Hillenbrand's floor relative to its speech at 800-1600 Hz
    return D, floor_rel


def eq(x, gain_db):
    from scipy.signal import fftconvolve, firwin2

    g = 10 ** (gain_db / 20)
    h = firwin2(1023, F, g, fs=FS)
    return fftconvolve(x, h, mode="same").astype(np.float32)


def add_noise(x, floor_rel, rng):
    """Noise with Hillenbrand's floor spectrum, placed below this file's speech as Hillenbrand's floor is below its."""
    P, e = frames_power(x)
    band = (F >= 800) & (F < 1600)
    sp = 10 * np.log10(np.mean(P[e > np.percentile(e, 95) - 25][:, band]))
    target = sp + floor_rel  # dB per bin, in frames_power's units
    n = rng.standard_normal(len(x) + NPER)
    Pn, _ = frames_power(n)
    white = 10 * np.log10(Pn.mean(0))
    gain_db = target - white
    y = eq(n, gain_db)[: len(x)]
    return (x + y).astype(np.float32)


def taper(lo_on, hi_on):
    """Weight 0 below lo_on rising to 1 at hi_on (or the reverse when lo_on > hi_on)."""
    w = np.clip((F - lo_on) / (hi_on - lo_on), 0, 1)
    return w


def conditions(D, floor_rel):
    high = D * taper(2500, 3000)
    low = D * taper(400, 300)
    return {
        "ptdb": {"base": None, "high": ("eq", high), "low": ("eq", low), "noise": ("noise", None), "all": ("eq+noise", D)},
        "hillenbrand": {"base": None, "rev_high": ("eq", -high), "rev_low": ("eq", -low), "rev_all": ("eq", -D)},
    }


def degrade(x, spec, floor_rel, rng):
    if spec is None:
        return x
    kind, curve = spec
    if "eq" in kind:
        x = eq(x, curve)
    if "noise" in kind:
        x = add_noise(x, floor_rel, rng)
    return x


def ptdb_items():
    import soundfile as sf

    d = dict(np.load("cache/m3w/ptdb_egg_sub.npz"))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    groups = {}
    for i in np.flatnonzero(dev):
        groups.setdefault((d["speaker"][i], d["utt"][i]), []).append(i)
    items = []
    for (spk, utt), idx in groups.items():
        p = Path("data/ptdb_tug") / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{utt}.wav"
        x, sr = sf.read(str(p))
        items.append({"key": f"{spk}/{utt}", "audio": to16(x[:, 0] if x.ndim == 2 else x, sr), "idx": np.array(idx),
                      "times": d["t"][idx]})
    return d, items


def hb_items():
    import soundfile as sf

    from vocal_analysis.eval.datasets import hillenbrand as H

    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    root = Path("data/hillenbrand")
    items = []
    for p in sorted(root.glob("*.wav")):
        tok = p.stem.lower()
        spk = f"hillenbrand:{tok[:3]}"
        if tiers.get(spk) != "dev" or tok[0] not in "wm":
            continue
        try:
            it = H.load(root, SimpleNamespace(path=p.name, speaker_id=spk))
        except KeyError:
            continue
        ok = np.isfinite(it.times)
        if not ok.any():
            continue
        x, sr = sf.read(str(p))
        items.append({"key": tok, "audio": to16(x, sr), "times": it.times[ok], "ref": it.formants[ok, :3], "spk": spk,
                      "group": tok[0]})
    return items


def run(items, models, conds, floor_rel, dev):
    from vocal_analysis.models.infer import analyse, load_teacher

    out = {}
    for name, ck in models.items():
        model, _ = load_teacher(ck, dev)
        for cname, spec in conds.items():
            rng = np.random.default_rng(SEED)  # the same noise for every model
            res = []
            for it in items:
                y = degrade(it["audio"], spec, floor_rel, rng)
                r = analyse(model, y, dev)
                j = np.clip(np.round(it["times"] / 0.010).astype(int), 0, len(r["formants"]) - 1)
                res.append(r["formants"][j, :3])
            out[(name, cname)] = res
        del model
    return out


def contrast(S, f1, spk, lo, hi, rng, n=2000):
    """median S where f1 >= hi minus median S where f1 < lo; 95 % speaker bootstrap."""
    def stat(sel):
        up, dn = sel & (f1 >= hi), sel & (f1 < lo)
        return np.median(S[up]) - np.median(S[dn]) if up.any() and dn.any() else np.nan
    sp = np.unique(spk)
    bs = []
    for _ in range(n):
        pick = rng.choice(sp, len(sp))
        sel = np.concatenate([np.flatnonzero(spk == s) for s in pick])
        m = np.zeros(len(S), bool)
        m[sel] = True
        # repeated speakers: use index lists to keep multiplicity
        Ss, ff = S[sel], f1[sel]
        up, dn = ff >= hi, ff < lo
        bs.append(np.median(Ss[up]) - np.median(Ss[dn]) if up.any() and dn.any() else np.nan)
    return stat(np.ones(len(S), bool)), *np.nanpercentile(bs, [2.5, 97.5])


if __name__ == "__main__":
    import torch

    out_dir = Path(sys.argv[1])
    null_only = "--null-only" in sys.argv
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    hb_w = [p for p in sorted(glob.glob("data/hillenbrand/w*.wav")) if tiers.get("hillenbrand:" + Path(p).stem.lower()[:3]) == "dev"]
    pt_w = [p for p in sorted(glob.glob("data/ptdb_tug/FEMALE/MIC/*/mic_*.wav")) if tiers.get("ptdb_tug:" + Path(p).parent.name) == "dev"][::3]
    D, floor_rel = chain_curves(hb_w, pt_w)
    conds = conditions(D, floor_rel)
    edges = [100, 150, 200, 300, 400, 800, 1600, 3200, 5000, 6500, 7200, 7600]
    print("chain curve D(f) = Hillenbrand - PTDB-TUG speech spectrum (dB, women, dev): " +
          " ".join(f"{e}:{np.interp(e, F, D):+.0f}" for e in edges))
    print("Hillenbrand floor re its speech at 800-1600 Hz (dB): " + " ".join(f"{e}:{np.interp(e, F, floor_rel):+.0f}" for e in edges))
    models = {**B, **N} if null_only else {**B, **N, **R}
    d, pitems = ptdb_items()
    hitems = hb_items()
    pres = run(pitems, models, conds["ptdb"], floor_rel, dev)
    hres = run(hitems, models, conds["hillenbrand"], floor_rel, dev)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PTDB-TUG: per-frame arrays in the cache's order
    nfr = len(d["t"])
    P = {}
    for (m, c), res in pres.items():
        a = np.full((nfr, 3), np.nan)
        for it, r in zip(pitems, res):
            a[it["idx"]] = r
        P[(m, c)] = a
    chk = np.nanmedian(np.abs(P[("b1", "base")][:, 0] - d["d040_5b_s1_b20k"][:, 0]))
    print(f"\ncheck: base-condition 5b s1 against the cache, median |F1 difference| {chk:.2f} Hz")
    np.savez(out_dir / ("d113_ptdb_null.npz" if null_only else "d113_ptdb.npz"), **{f"{m}__{c}": v for (m, c), v in P.items()})
    hsave = {f"{m}__{c}": np.concatenate(v) for (m, c), v in hres.items()}
    hsave["ref"] = np.concatenate([it["ref"] for it in hitems])
    hsave["spk"] = np.concatenate([[it["spk"]] * len(it["times"]) for it in hitems])
    hsave["group"] = np.concatenate([[it["group"]] * len(it["times"]) for it in hitems])
    np.savez(out_dir / ("d113_hb_null.npz" if null_only else "d113_hb.npz"), **hsave)

    rng = np.random.default_rng(SEED)
    ok = np.isfinite(d["cp"][:, 0]) & np.isfinite(d["cycle"][:, 0]) & np.isfinite(d["praat"][:, 0]) & (d["f0"] > 0)
    for sx, name, lo, hi in (("f", "women", 450, 600), ("m", "men", 400, 550)):
        k = ok & (d["sex"] == sx) & np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
        print(f"\nPTDB-TUG {name} (D-110's frames, {int(k.sum())}); contrast = median over closed-phase F1 >= {hi} minus < {lo} Hz")
        for c in conds["ptdb"]:
            mean = lambda a, b: (P[(a, c)][:, 0] + P[(b, c)][:, 0]) / 2  # noqa: E731
            pairs = {"S0 (D-065 - 5b)": mean("n1", "n2") - mean("b1", "b2")}
            if not null_only:
                pairs = {"S (d109 - 5b)": mean("r1", "r2") - mean("b1", "b2"), **pairs}
            cells = []
            for lab, S in pairs.items():
                kk = k & np.isfinite(S)
                c0, l0, h0 = contrast(S[kk], d["cp"][kk, 0], d["speaker"][kk], lo, hi, rng)
                cells.append(f"{lab} median {np.median(S[kk]):+5.1f}, contrast {c0:+5.1f} [{l0:+.1f}, {h0:+.1f}]")
            print(f"  {c:6s} " + " | ".join(cells))
    ref, spk, grp = hsave["ref"], hsave["spk"], hsave["group"]
    for g, name in (("w", "women"), ("m", "men")):
        k = (grp == g) & np.isfinite(ref[:, 0])
        print(f"\nHillenbrand {name} (dev, labelled frames, {int(k.sum())}); contrast = median over label F1 >= 650 minus < 450 Hz")
        for c in conds["hillenbrand"]:
            mean = lambda a, b: (hsave[f"{a}__{c}"][:, 0] + hsave[f"{b}__{c}"][:, 0]) / 2  # noqa: E731
            pairs = {"S0 (D-065 - 5b)": mean("n1", "n2") - mean("b1", "b2")}
            if not null_only:
                pairs = {"S (d109 - 5b)": mean("r1", "r2") - mean("b1", "b2"), **pairs}
            cells = []
            for lab, S in pairs.items():
                kk = k & np.isfinite(S)
                c0, l0, h0 = contrast(S[kk], ref[kk, 0], spk[kk], 450, 650, rng)
                cells.append(f"{lab} median {np.median(S[kk]):+5.1f}, contrast {c0:+5.1f} [{l0:+.1f}, {h0:+.1f}]")
            print(f"  {c:8s} " + " | ".join(cells))
