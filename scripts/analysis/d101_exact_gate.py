"""D-101: D-094's gate with the exact envelope. The same 60 clean train_v2 validation clips, Harvest f0 and D4C
aperiodicity, the same factors and frames; WORLD synthesises from the labelled tract's eight poles (unit DC gain)
times a log-log line fitted to CheapTrick's envelope over 100-5000 Hz per frame (level and tilt), CheapTrick's own
envelope where a frame has no formant labels. Teacher v3.1, F1 slope on log2 factor, group bootstrap and shuffle null.

    python scripts/analysis/d101_exact_gate.py > reports/analysis/d101/d101.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = "d040_5b_s2_b20k"
O = Path("reports/analysis/d101")
CACHE = Path("cache/d101")
FS = 16000


def exact_envelope(sp_ct, formants, bandwidths, nfft):
    f = np.arange(sp_ct.shape[1]) * FS / nfft
    w = 2 * np.pi * f / FS
    z1, z2 = np.exp(-1j * w), np.exp(-2j * w)
    band = (f >= 100) & (f <= 5000)
    lf = np.log(np.maximum(f, f[1]))
    out = sp_ct.copy()
    for i in range(min(len(sp_ct), len(formants))):
        F, B = formants[i], bandwidths[i]
        ok = np.isfinite(F) & np.isfinite(B) & (F > 0) & (F < FS / 2)
        if not (ok.any() and np.isfinite(F[0])):
            continue
        h2 = np.ones_like(f)
        for Fk, Bk in zip(F[ok], B[ok]):
            r, th = np.exp(-np.pi * Bk / FS), 2 * np.pi * Fk / FS
            den = 1 - 2 * r * np.cos(th) * z1 + r * r * z2
            h2 *= (np.abs(1 - 2 * r * np.cos(th) + r * r) / np.abs(den)) ** 2
        b, a = np.polyfit(lf[band], np.log(sp_ct[i, band] + 1e-30), 1)
        out[i] = np.exp(a + b * lf) * h2
    return out


def peak_stats(sp, f1, nfft):
    """per frame: the largest envelope peak within 0.6-1.6 x F1 (cents against F1) and its -3 dB width (Hz)"""
    f = np.arange(sp.shape[1]) * FS / nfft
    c, wd = [], []
    for i in range(min(len(sp), len(f1))):
        if not np.isfinite(f1[i]):
            continue
        k = np.flatnonzero((f >= 0.6 * f1[i]) & (f <= 1.6 * f1[i]))
        if len(k) < 3:
            continue
        j = k[np.argmax(sp[i, k])]
        half = sp[i, j] / 2
        lo = j
        while lo > 0 and sp[i, lo - 1] >= half:
            lo -= 1
        hi = j
        while hi < sp.shape[1] - 1 and sp[i, hi + 1] >= half:
            hi += 1
        c.append(1200 * np.log2(max(f[j], 1) / f1[i]))
        wd.append(f[hi] - f[lo] + FS / nfft)
    return (float(np.median(c)), float(np.median(wd))) if c else (np.nan, np.nan)


if __name__ == "__main__":
    import pyworld as pw
    import torch

    from vocal_analysis.eval.probes import measures as M
    from vocal_analysis.eval.probes import world as W
    from vocal_analysis.eval.probes.run import _summary
    from vocal_analysis.models.infer import analyse, load_teacher
    from vocal_analysis.synth.sampler import SynthConfig, generate

    meta = json.loads((W.OUT / "meta.json").read_text(encoding="utf-8"))
    gidx = [i for i, it in enumerate(meta["items"]) if it["set"] == "gate"]
    gate = [meta["items"][i] for i in gidx]
    order = meta["factors"]
    x_log = np.log2(order)
    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in idx["synth_config"].items()
                         if k in SynthConfig.__dataclass_fields__})
    nfft = pw.get_cheaptrick_fft_size(FS)
    CACHE.mkdir(parents=True, exist_ok=True)
    pk = {"ct": [], "ex": []}
    for k, it in enumerate(gate):
        path = CACHE / f"gate_{k:03d}.npz"
        clip = generate(int(it["key"].split(":")[1]), cfg, idx["seed"])
        x = clip.audio.astype(np.float64)
        a = W.analyse_world(x)
        sp_ex = exact_envelope(a["sp"], clip.labels["formants"], clip.labels["bandwidths"], nfft)
        d0 = np.load(W.OUT / it["file"])
        scored = d0["scored_0"]
        f1 = np.where(clip.labels["sonorant"], clip.labels["formants"][:, 0], np.nan)
        n = min(len(scored), len(f1))
        f1s = np.where(scored[:n], f1[:n], np.nan)
        pk["ct"].append(peak_stats(a["sp"], f1s, nfft))
        pk["ex"].append(peak_stats(sp_ex, f1s, nfft))
        if not path.exists():
            arr = {}
            for c, r in enumerate(order):
                y = pw.synthesize(a["f0"] * r, sp_ex, a["ap"], FS, frame_period=W.FRAME_MS)[: len(x)]
                y = np.r_[y, np.zeros(max(0, len(x) - len(y)))]
                arr[f"audio_{c}"] = (0.9 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
            np.savez_compressed(path, **arr)
    print(f"rendered {len(gate)} clips", file=sys.stderr, flush=True)

    raw = O / f"raw_{MODEL}.npz"
    O.mkdir(parents=True, exist_ok=True)
    keys = ("F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "VTL", "f0")
    if not raw.exists():
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_teacher(f"runs/{MODEL}/ckpt_last.pt", dev)[0]
        Y = {o: np.full((len(gate), len(order)), np.nan) for o in keys}
        for k, it in enumerate(gate):
            d = np.load(CACHE / f"gate_{k:03d}.npz")
            d0 = np.load(W.OUT / it["file"])
            for c in range(len(order)):
                s = _summary(analyse(model, d[f"audio_{c}"], dev), d0[f"scored_{c}"], d0[f"voiced_{c}"])
                for o in keys:
                    Y[o][k, c] = s[o]
        np.savez(raw, **Y)
    Y = dict(np.load(raw))
    Y94 = {k[2:]: v[gidx] for k, v in np.load("reports/analysis/d094/raw_d040_5b_s2_b20k.npz").items()}
    grp = np.array([it["group"] for it in gate])
    bnd = np.array([W.band(it["f0_median"]) for it in gate])

    def fmt(s):
        return (f"{s.point:+.3f} [{s.low:+.3f}, {s.high:+.3f}]  null {s.null_median:+.3f} [{s.null_low:+.3f}, "
                f"{s.null_high:+.3f}]  n {s.n_items}")

    L = [f"D-101: D-094's gate with the exact envelope; teacher v3.1 ({MODEL}); slope of log2 output on log2 f0 factor; "
         "point [95 % bootstrap over clips], shuffle null; ideal 0 (f0: 1)", ""]
    s_all = M.summarise_slopes(Y["F1"], x_log, grp)
    L.append(f"exact envelope, F1, all {len(gate)} clips: {fmt(s_all)}   (D-094 CheapTrick: {fmt(M.summarise_slopes(Y94['F1'], x_log, grp))})")
    band_ok, mid = True, None
    for b, (lo, hi) in enumerate(W.BANDS):
        k = bnd == b
        if not k.any():
            continue
        s = M.summarise_slopes(Y["F1"][k], x_log, grp[k])
        s94 = M.summarise_slopes(Y94["F1"][k], x_log, grp[k])
        d = M.item_slopes(Y["F1"][k], x_log) - M.item_slopes(Y94["F1"][k], x_log)
        dp = M.group_bootstrap(d[np.isfinite(d)], grp[k][np.isfinite(d)])
        ok = k.sum() < 10 or abs(s.point) <= 0.05
        band_ok &= ok
        if b == 1:
            mid, mid94 = s, s94
        L.append(f"  f0 {lo:.0f}-{hi if hi < 1e8 else 'up'} Hz: exact {fmt(s)}{'' if ok else '  OUTSIDE +-0.05'}")
        L.append(f"      CheapTrick {fmt(s94)}; exact - CheapTrick per clip {dp[0]:+.3f} [{dp[1]:+.3f}, {dp[2]:+.3f}]")
    for o in keys[1:]:
        L.append(f"  {o}: exact {fmt(M.summarise_slopes(Y[o], x_log, grp))}")
    L.append("")
    L.append("reported: the envelope's F1 peak (largest peak within 0.6-1.6 x F1 on scored frames): median cents from the "
             "label / median -3 dB width Hz, by band")
    for b, (lo, hi) in enumerate(W.BANDS):
        k = np.flatnonzero(bnd == b)
        if len(k):
            ct = np.array([pk["ct"][i] for i in k])
            ex = np.array([pk["ex"][i] for i in k])
            L.append(f"  f0 {lo:.0f}-{hi if hi < 1e8 else 'up'} Hz: CheapTrick {np.nanmedian(ct[:, 0]):+.0f} c / {np.nanmedian(ct[:, 1]):.0f} Hz; "
                     f"exact {np.nanmedian(ex[:, 0]):+.0f} c / {np.nanmedian(ex[:, 1]):.0f} Hz ({len(k)} clips)")
    L.append("")
    passes = abs(s_all.point) <= 0.03 and band_ok
    if passes and mid.point <= 0.080:
        verdict = "CHEAPTRICK'S ARTEFACT"
    elif mid.point >= 0.05 and mid.low > 0:
        verdict = "V3.1'S OWN"
    else:
        verdict = "UNRESOLVED"
    L.append(f"DECISION: {verdict} (exact-envelope gate {'passes' if passes else 'fails'}: all {s_all.point:+.3f}; "
             f"150-250 Hz {mid.point:+.3f} [{mid.low:+.3f}, {mid.high:+.3f}] against CheapTrick's {mid94.point:+.3f})")
    print("\n".join(L))
