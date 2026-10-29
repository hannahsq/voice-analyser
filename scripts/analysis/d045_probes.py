"""D-045 follow-ups: band periodic fraction (renders against real voices) and discriminator probes.

    python scripts/analysis/d045_probes.py bpf_check <out dir>   (the measure on D-047's paired renders, 5 against 40 dB)
    python scripts/analysis/d045_probes.py bpf <out dir>         (CPU: renders, clean and channelled, against real voices)
    python scripts/analysis/d045_probes.py feat <out dir>        (GPU in short bursts: teacher v3 features for the probes)
    python scripts/analysis/d045_probes.py fit <out dir>         (CPU: refit D_h, score, occlusion AUCs; writes probes.txt)

Band periodic fraction (Meka et al. 2026, eqs 11-12): per 50 ms Hann frame, the lag of the strongest autocorrelation
peak of the 0-1.5 kHz component in 60-400 Hz, then per band (0-1, 1-2, 2-4, 4-6, 6-8 kHz) the band's autocorrelation
at that lag over its value at 0, corrected for the window's own autocorrelation. Frames: Praat-voiced (cc, 60-400 Hz),
the same rule for every source; synthetic frames reweighted to each real sex's f0 distribution (1/6-octave bins).
Real voices: Zeroth-Korean test and speechocean762 adults (training pools; never an evaluation set).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d045_discriminator as D  # noqa: E402

FS = 16000
WIN = 800  # 50 ms
NFFT = 2048
BANDS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000))
LAGS = (FS // 400, FS // 60)
SEED_RENDER = 4548  # fresh: train_v2 is seed 2, D-047 used 4547
N_RENDER = 400
PER_FILE = 40
PER_SEX = 12000
OCCLUSION = ("full", "lp4k", "lp2k", "noise30")
N_BOOT = 2000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- band periodic fraction

_freqs = np.fft.rfftfreq(NFFT, 1 / FS)
_win = np.hanning(WIN)
_wac = np.fft.irfft(np.abs(np.fft.rfft(_win, NFFT)) ** 2)
_wac = _wac / _wac[0]
_low = _freqs <= 1500
_masks = [(_freqs >= lo) & (_freqs < hi) for lo, hi in BANDS]


def bpf(x, centres):
    """(n, 5) band periodic fractions at sample centres; NaN where the frame does not fit."""
    out = np.full((len(centres), len(BANDS)), np.nan)
    for i, c in enumerate(centres):
        a0 = int(c) - WIN // 2
        if a0 < 0 or a0 + WIN > len(x):
            continue
        p = np.abs(np.fft.rfft(x[a0:a0 + WIN] * _win, NFFT)) ** 2
        rl = np.fft.irfft(p * _low)
        if rl[0] <= 0:
            continue
        r = rl / rl[0] / np.maximum(_wac, 1e-6)
        lag = LAGS[0] + int(np.argmax(r[LAGS[0]:LAGS[1] + 1]))
        for b, m in enumerate(_masks):
            rb = np.fft.irfft(p * m)
            if rb[0] > 0:
                out[i, b] = rb[lag] / (rb[0] * _wac[lag])
    return out


def praat_frames(x, max_n, rng):
    """Praat-voiced frame centres (samples) and f0, at most max_n, 60-400 Hz."""
    import parselmouth

    snd = parselmouth.Sound(np.asarray(x, np.float64), FS)
    pt = snd.to_pitch_cc(time_step=0.01, pitch_floor=60.0, pitch_ceiling=400.0)
    f = pt.selected_array["frequency"]
    t = pt.xs()
    ok = np.flatnonzero(f > 0)
    if len(ok) > max_n:
        ok = np.sort(rng.choice(ok, max_n, replace=False))
    return np.round(t[ok] * FS).astype(int), f[ok]


def config_v2(hnr=None):
    from vocal_analysis.synth.sampler import SynthConfig

    raw = json.loads(Path("docs/synth_sets/train_v2.index.json").read_text(encoding="utf-8"))["synth_config"]
    names = {f.name for f in __import__("dataclasses").fields(SynthConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names}
    if hnr is not None:
        kw["hnr_db"] = (hnr, hnr)
    return SynthConfig(**kw)


def channel_cfg():
    from vocal_analysis.synth.channel import ChannelConfig

    c = json.load(open(D.TEACHER_V3_CFG))["channel"]
    return ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.items()})


def render(i, channelled, cfg=None, seed=SEED_RENDER):
    """(audio, meta, channel params) of fresh clip i; channelled as teacher v3's training presents synthetic clips."""
    from vocal_analysis.features.vqt import HOP
    from vocal_analysis.synth.channel import augment
    from vocal_analysis.synth.sampler import generate

    c = generate(i, cfg or config_v2(), seed=seed)
    x = c.audio.astype(np.float64)
    params = {}
    if channelled:
        mask = np.repeat(c.labels["voiced"], HOP)[: len(x)]
        mask = np.pad(mask, (0, len(x) - len(mask)))
        x, params = augment(x, np.random.default_rng([seed, i, 45]), channel_cfg(), mask, codec_stream=False)
    meta = {"hnr_db": c.meta["hnr_db"], "jitter": c.meta["jitter"], "shimmer_db": c.meta["shimmer_db"],
            "tube": float(c.meta["tract"] == "tube"), "length_cm": c.meta["length_cm"],
            "nasal_or_subglottal": float(c.meta["n_extra_poles"] > 0), "whisper_clip": float(c.labels["whisper"].mean() > 0.9),
            "rd_median": float(np.nanmedian(c.labels["rd"][c.labels["voiced"]])) if c.labels["voiced"].any() else np.nan,
            "f0_median": float(np.nanmedian(c.labels["f0"][c.labels["voiced"]])) if c.labels["voiced"].any() else np.nan}
    return x, meta, params


def real_files():
    """{(corpus, speaker): (sex, [files])} for Zeroth test and speechocean762 adults."""
    out = {}
    for s, sx in D.zeroth_speakers().items():
        f = sorted(map(str, (D.ZEROTH / "test_data_01").glob(f"*/{s}/*.flac")))
        if f and sx in "fm":
            out["zeroth", s] = (sx, f)
    for s, (sx, a) in D.so_speakers().items():
        f = sorted(map(str, (D.SO / "WAVE" / s).glob("*.WAV")))
        if f and a >= 18 and sx in "fm":
            out["so_adult", s] = (sx, f)
    return out


def read16(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path)
    x = x if x.ndim == 1 else x.mean(1)
    return resample_poly(x, FS, sr) if sr != FS else x


def bpf_check(out):
    """the measure must order D-047's paired renders: higher periodic fraction at 40 dB than at 5 dB, every band"""
    sel = json.loads(Path("runs/d047/selection.json").read_text())["women"][:20]
    import d047_hnr_renders as R

    from vocal_analysis.synth.sampler import generate

    res = {}
    rng = np.random.default_rng(0)
    for h in (5.0, 40.0):
        vals = []
        for i in sel:
            c = generate(i, R.config(h), seed=R.SEED)
            cs, _ = praat_frames(c.audio, 60, np.random.default_rng(i))
            vals.append(np.nanmean(bpf(c.audio.astype(np.float64), cs), 0))
        res[h] = np.array(vals)
    d = res[40.0] - res[5.0]
    lines = ["bpf check: D-047 women's clips 0-19, mean band periodic fraction at 5 / 40 dB, and clips where 40 dB is higher"]
    for b, (lo, hi) in enumerate(BANDS):
        lines.append(f"  {lo / 1000:.0f}-{hi / 1000:.0f} kHz: {np.nanmean(res[5.0][:, b]):.3f} / {np.nanmean(res[40.0][:, b]):.3f}; "
                     f"higher at 40 dB in {int((d[:, b] > 0).sum())} of {len(sel)}")
    (out / "bpf_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def bpf_run(out):
    rng = np.random.default_rng(4549)
    rows = []
    for channelled in (False, True):
        src = "synth_channel" if channelled else "synth_clean"
        for i in range(N_RENDER):
            x, meta, _ = render(i, channelled)
            if meta["whisper_clip"]:
                continue
            cs, f0 = praat_frames(x, PER_FILE, rng)
            if len(cs):
                rows.append((src, f"clip{i}", "?", f0, bpf(x, cs)))
        log(f"{src} done")
    files = real_files()
    for (corpus, spk), (sx, fl) in files.items():
        n_spk = sum(1 for (c, _), (s, _) in files.items() if c == corpus and s == sx)
        quota = max(PER_FILE, PER_SEX // n_spk)
        got = 0
        for f in rng.permutation(fl):
            if got >= quota:
                break
            x = read16(f)
            cs, f0 = praat_frames(x, min(PER_FILE, quota - got), rng)
            if len(cs):
                rows.append((corpus, spk, sx, f0, bpf(x, cs)))
                got += len(cs)
    log("real done")
    np.savez(out / "bpf.npz", src=np.concatenate([[r[0]] * len(r[3]) for r in rows]),
             spk=np.concatenate([[r[1]] * len(r[3]) for r in rows]), sex=np.concatenate([[r[2]] * len(r[3]) for r in rows]),
             f0=np.concatenate([r[3] for r in rows]), rho=np.concatenate([r[4] for r in rows]))
    bpf_report(out)


def wmean_boot(v, w, g, rng):
    ok = np.isfinite(v)
    v, w, g = v[ok], w[ok], g[ok]
    us = np.unique(g)
    by = {u: np.flatnonzero(g == u) for u in us}
    m = np.average(v, weights=w)
    bs = []
    for _ in range(N_BOOT):
        ii = np.concatenate([by[u] for u in rng.choice(us, len(us))])
        bs.append(np.average(v[ii], weights=w[ii]) if w[ii].sum() > 0 else np.nan)
    return m, np.array(bs)


def bpf_report(out):
    z = dict(np.load(out / "bpf.npz", allow_pickle=True))
    rng = np.random.default_rng(45)
    lines = ["Band periodic fraction (mean over frames; synthetic reweighted to each real sex's f0 distribution); 95 % bootstrap over speakers / clips",
             "bands: " + ", ".join(f"{lo / 1000:.0f}-{hi / 1000:.0f} kHz" for lo, hi in BANDS), ""]
    verdict = {}
    for sx, label in (("f", "women"), ("m", "men")):
        lines.append(label)
        means, boots = {}, {}
        for corpus in ("zeroth", "so_adult"):
            m = (z["src"] == corpus) & (z["sex"] == sx)
            for b in range(len(BANDS)):
                means[corpus, b], boots[corpus, b] = wmean_boot(z["rho"][m, b], np.ones(m.sum()), z["spk"][m], rng)
            for syn in ("synth_clean", "synth_channel"):
                ms = z["src"] == syn
                w = D.f0_weights(z["f0"][ms], z["f0"][m])
                for b in range(len(BANDS)):
                    means[syn, corpus, b], boots[syn, corpus, b] = wmean_boot(z["rho"][ms, b], w, z["spk"][ms], rng)
            lines.append(f"  {corpus:8s} " + "  ".join(f"{means[corpus, b]:.3f}" for b in range(len(BANDS))))
            for syn in ("synth_clean", "synth_channel"):
                lines.append(f"  {syn:14s} matched to {corpus:8s} " + "  ".join(f"{means[syn, corpus, b]:.3f}" for b in range(len(BANDS))))
        for syn in ("synth_clean", "synth_channel"):
            for b, (lo, hi) in enumerate(BANDS):
                cells, signs, big = [], [], []
                for corpus in ("zeroth", "so_adult"):
                    d = boots[corpus, b] - boots[syn, corpus, b]
                    dm = means[corpus, b] - means[syn, corpus, b]
                    lo_, hi_ = np.nanpercentile(d, [2.5, 97.5])
                    cells.append(f"{corpus} {dm:+.3f} [{lo_:+.3f}, {hi_:+.3f}]")
                    signs.append(np.sign(dm) if (lo_ > 0 or hi_ < 0) else 0)
                    big.append(abs(dm))
                between = abs(means["zeroth", b] - means["so_adult", b])
                differs = signs[0] == signs[1] != 0 and min(big) > between
                verdict[label, syn, b] = differs
                lines.append(f"    real - {syn} {lo / 1000:.0f}-{hi / 1000:.0f} kHz: " + "; ".join(cells)
                             + f"; between corpora {between:.3f}" + ("  DIFFERS" if differs else ""))
        lines.append("")
    for label in ("women", "men"):
        d = [f"{BANDS[b][0] / 1000:.0f}-{BANDS[b][1] / 1000:.0f}" for b in range(len(BANDS)) if verdict[label, "synth_clean", b]]
        lines.append(f"READING ({label}, clean renders): " + ("profiles agree" if not d else "differ in " + ", ".join(d) + " kHz"))
    (out / "bpf.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------- discriminator probes

def occlude(x, kind, rng):
    from scipy.signal import butter, sosfiltfilt

    if kind == "full":
        return x
    if kind in ("lp4k", "lp2k"):
        sos = butter(8, 4000 if kind == "lp4k" else 2000, fs=FS, output="sos")
        return sosfiltfilt(sos, x)
    if kind == "noise30":  # white noise 30 dB below the file's voiced-frame power: masks low-level differences
        p = np.mean(x ** 2) + 1e-12
        return x + rng.standard_normal(len(x)) * np.sqrt(p / 1e3)
    raise ValueError(kind)


def file_rows(T, x, idx, extra, cond_rng):
    """teacher v3 features at frames idx (chosen on the unoccluded file) under every occlusion"""
    out = []
    for kind in OCCLUSION:
        r = T.file(occlude(x, kind, cond_rng).astype(np.float32))
        n = min(len(r["f0"]), idx.max() + 1) if len(idx) else 0
        ii = idx[idx < n]
        out.append({"h": r["h"][ii].astype(np.float16), "f0": r["f0"][ii], "cond": np.full(len(ii), kind),
                    **{k: np.full(len(ii), v) for k, v in extra.items()}})
    return out


def feat(out):
    from vocal_analysis.device import DeviceConfig, setup_device

    dev = setup_device(DeviceConfig(device="cuda"))
    T = D.Trunk(dev)
    rng = np.random.default_rng(45046)

    if not (out / "synth_probe.npz").exists():  # fresh channelled renders: settings probe and the occlusion's synthetic side
        rows, metas = [], []
        for i in range(N_RENDER):
            x, meta, params = render(i, True)
            r = T.file(x.astype(np.float32))
            idx = D.select(r, rng, 30)
            if not len(idx):
                continue
            extra = {"clip": i, **{k: float(v) for k, v in meta.items()},
                     **{f"ch_{k}": float(v) for k, v in params.items() if isinstance(v, (int, float, bool, np.floating))}}
            metas.append(extra)
            rows += file_rows(T, x, idx, {"clip": i}, np.random.default_rng([i, 7]))
        np.savez(out / "synth_probe.npz", **D.stack(rows))
        (out / "synth_meta.json").write_text(json.dumps(metas), encoding="utf-8")
        log(f"synth probe: {len(metas)} clips")

    if not (out / "real_probe.npz").exists():
        files = real_files()
        rows = []
        for (corpus, spk), (sx, fl) in files.items():
            n_spk = sum(1 for (c, _), (s, _) in files.items() if c == corpus and s == sx)
            quota = max(30, 4000 // n_spk)
            got = 0
            for f in rng.permutation(fl):
                if got >= quota:
                    break
                x = read16(f)
                r = T.file(x.astype(np.float32))
                idx = D.select(r, rng, min(30, quota - got))
                if not len(idx):
                    continue
                rows += file_rows(T, x, idx, {"corpus": corpus, "speaker": spk, "sex": sx}, np.random.default_rng([len(rows), 9]))
                got += len(idx)
        np.savez(out / "real_probe.npz", **D.stack(rows))
        log("real probe done")

    if not (out / "resynth_probe.npz").exists():  # resynthesised training clips, channelled as trained
        from torch.utils.data import DataLoader, Subset

        from vocal_analysis.train.data import SynthCrops, collate

        rows = []
        for name in ("zeroth_korean_train_d16k", "vocalset_train_d16k"):
            flat = Path("data/resynth") / name / "flat"
            n = json.loads((Path("data/resynth") / name / "index.json").read_text())["n_clips"]
            ids = np.sort(rng.choice(n, 200, replace=False))
            ds = SynthCrops(flat, ids, D.CROP, channel_cfg(), seed=45047, fixed=True)
            k = 0
            for audio, _ in DataLoader(Subset(ds, range(len(ids))), 16, num_workers=2, collate_fn=collate):
                for b, r in enumerate(T.crops(audio)):
                    idx = D.select(r, rng, 30)
                    if len(idx):
                        rows.append({"h": r["h"][idx].astype(np.float16), "f0": r["f0"][idx],
                                     "set": np.full(len(idx), name), "clip": np.full(len(idx), int(ids[k + b]))})
                k += len(audio)
        np.savez(out / "resynth_probe.npz", **D.stack(rows))
        log("resynth probe done")

    if not (out / "hnr_probe.npz").exists():  # D-047's paired arms, channelled like the synthetic training data
        import d047_hnr_renders as R

        sel = json.loads(Path("runs/d047/selection.json").read_text())["women"][:60]
        rows = []
        for i in sel:
            for h in (5.0, 40.0):
                x, _, _ = render(i, True, cfg=R.config(h), seed=R.SEED)
                r = T.file(x.astype(np.float32))
                ok = np.flatnonzero((r["voiced_p"] > 0.5) & (r["sonorant_p"] > 0.5) & (r["f0"] > 0))
                rows.append({"h": r["h"][ok].astype(np.float16), "f0": r["f0"][ok], "clip": np.full(len(ok), i),
                             "hnr": np.full(len(ok), h)})
        np.savez(out / "hnr_probe.npz", **D.stack(rows))
        log("hnr probe done")


def fit(out):
    from scipy.stats import spearmanr

    load = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}  # noqa: E731
    cache = Path("cache/d045")
    syn, zer, so = load(cache / "synth.npz"), load(cache / "zeroth.npz"), load(cache / "so.npz")
    adult = so["group"] == "adult"
    corpora = {"zeroth": zer, "so_adult": {k: v[adult] for k, v in so.items()}}
    sp, rp = load(out / "synth_probe.npz"), load(out / "real_probe.npz")
    rs, hp = load(out / "resynth_probe.npz"), load(out / "hnr_probe.npz")
    metas = {m["clip"]: m for m in json.loads((out / "synth_meta.json").read_text())}
    rng = np.random.default_rng(4545)
    lines = [f"D-045 probes ({time.strftime('%Y-%m-%d %H:%M')}); D_h refitted on D-045's cached frames as registered; "
             "new frames scored with it (logit, real = +)", ""]

    # D_h as in D-045, then scores
    models = {}
    for cname, d in corpora.items():
        for sx in "fm":
            r = {k: v[d["sex"] == sx] for k, v in d.items()}
            models[cname, sx], auc = D.fit_one(D.design(syn, "h"), D.f0_weights(syn["f0"], r["f0"]), syn["speaker"],
                                               D.design(r, "h"), r["speaker"])
            lines.append(f"D_h {cname} {sx}: CV AUC {auc:.3f} (as D-045)")
    score = lambda m, h: m.decision_function(h.astype(np.float64))  # noqa: E731

    # (1) occlusion: refit per condition on the probe frames (same frames in every condition)
    lines.append("\n(1) Occlusion: D_h refitted per condition on the probe frames (fresh channelled renders against real), CV AUC")
    occ = {}
    for corpus in ("zeroth", "so_adult"):
        for sx in "fm":
            cells = []
            for kind in OCCLUSION:
                ms = sp["cond"] == kind
                mr = (rp["cond"] == kind) & (rp["corpus"] == corpus) & (rp["sex"] == sx)
                if mr.sum() < 50:
                    continue
                _, auc = D.fit_one(sp["h"][ms].astype(np.float64), D.f0_weights(sp["f0"][ms], rp["f0"][mr]),
                                   np.array([f"c{c}" for c in sp["clip"][ms]]), rp["h"][mr].astype(np.float64), rp["speaker"][mr])
                occ[corpus, sx, kind] = auc
                cells.append(f"{kind} {auc:.3f}")
            lines.append(f"  {corpus:8s} {sx}: " + "  ".join(cells))
    carried = [k for k in OCCLUSION[1:] if all(occ.get((c, "f", k), 1.0) < 0.95 for c in corpora)]
    lines.append("  READING (women, both corpora, AUC < 0.95 once removed): "
                 + (", ".join(carried) + " removes the separation" if carried else "no occlusion removes the separation"))

    # (2) generator settings against real-ness (full condition), clip means, women's D_h of each corpus
    lines.append("\n(2) Fresh channelled renders: clip-mean real-ness against generator and channel settings (Spearman, clip bootstrap)")
    ms = sp["cond"] == "full"
    clips = np.unique(sp["clip"][ms])
    keys = sorted({k for m in metas.values() for k in m if k != "clip"})
    for cname in corpora:
        sc = score(models[cname, "f"], sp["h"][ms])
        cm = np.array([sc[sp["clip"][ms] == c].mean() for c in clips])
        lines.append(f"  {cname} women's D_h: clip real-ness median {np.median(cm):+.2f} (real frames' median "
                     f"{np.median(score(models[cname, 'f'], D.design({k: v[corpora[cname]['sex'] == 'f'] for k, v in corpora[cname].items()}, 'h'))):+.2f})")
        res = []
        for k in keys:
            v = np.array([metas[int(c)].get(k, np.nan) for c in clips], float)
            ok = np.isfinite(v) & np.isfinite(cm)
            if ok.sum() < 20 or np.nanstd(v[ok]) == 0:
                continue
            rho = spearmanr(v[ok], cm[ok])[0]
            bs = []
            idx = np.flatnonzero(ok)
            for _ in range(1000):
                ii = rng.choice(idx, len(idx))
                bs.append(spearmanr(v[ii], cm[ii])[0])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            res.append((abs(rho), f"    {k:22s} {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]" + ("*" if lo > 0 or hi < 0 else "")))
        lines += [s for _, s in sorted(res, reverse=True)]

    # (3) resynthesised training clips
    lines.append("\n(3) Resynthesised training clips (channelled as trained): real-ness logit, median [IQR]; share scored real")
    for cname in corpora:
        for sx in "fm":
            m = models[cname, sx]
            rr = score(m, D.design({k: v[corpora[cname]["sex"] == sx] for k, v in corpora[cname].items()}, "h"))
            ss = score(m, sp["h"][sp["cond"] == "full"])
            cells = [f"real {np.median(rr):+.1f}", f"synth {np.median(ss):+.1f}"]
            for name in np.unique(rs["set"]):
                v = score(m, rs["h"][rs["set"] == name])
                cells.append(f"{name.split('_train')[0]} {np.median(v):+.1f} [{np.percentile(v, 25):+.1f}, {np.percentile(v, 75):+.1f}] "
                             f"real {np.mean(v > 0):.2f}")
            lines.append(f"  {cname} {sx}: " + "; ".join(cells))

    # (4) D-047's arms: does aspiration level move real-ness?
    lines.append("\n(4) D-047's paired renders (women's range, channelled): clip-mean real-ness, 40 - 5 dB (clip bootstrap)")
    for cname in corpora:
        sc = score(models[cname, "f"], hp["h"])
        cl = np.unique(hp["clip"])
        d = np.array([sc[(hp["clip"] == c) & (hp["hnr"] == 40.0)].mean() - sc[(hp["clip"] == c) & (hp["hnr"] == 5.0)].mean() for c in cl])
        d = d[np.isfinite(d)]
        bs = [rng.choice(d, len(d)).mean() for _ in range(N_BOOT)]
        lines.append(f"  {cname}: {d.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] "
                     f"(women's D_h; logit units; real frames minus synthetic frames median "
                     f"{np.median(score(models[cname, 'f'], D.design({k: v[corpora[cname]['sex'] == 'f'] for k, v in corpora[cname].items()}, 'h'))) - np.median(score(models[cname, 'f'], sp['h'][sp['cond'] == 'full'])):+.1f})")
    (out / "probes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"bpf_check": bpf_check, "bpf": bpf_run, "bpf_report": bpf_report, "feat": feat, "fit": fit}[stage](out)


if __name__ == "__main__":
    main()
