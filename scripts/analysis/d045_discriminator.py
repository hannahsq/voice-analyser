"""D-045: a synthetic-against-real domain discriminator, applied to PTDB-TUG dev frames.

    python scripts/analysis/d045_discriminator.py features <out dir> [--smoke]   (GPU in short bursts, then CPU)
    python scripts/analysis/d045_discriminator.py fit <out dir>                  (CPU)

features: per source, voiced and sonorant frames (teacher v3's probabilities > 0.5) with teacher v3's encoder output h
(128), its f0, F1-F3 and B1-B3, and interpretable measures from the audio (40 ms windows): H1-H2, H1-A1 (A1 at the
harmonic nearest teacher v3's F1), cepstral peak prominence, HNR (Praat), spectral tilt (dB/oct, 100 Hz-4 kHz),
energy above 4 kHz relative to 0-4 kHz, local jitter and shimmer (Praat, +-50 ms). Sources:
  synth      train_v2 clips with teacher v3's channel augmentation (as training presents them);
  zeroth     Zeroth-Korean test split (numeric ids and recorded sex only; AUDIO_INFO's names are never kept);
  so_adult   speechocean762 speakers aged >= 18; so_child (< 18) is kept for the children's arm;
  ptdb       PTDB-TUG dev frames of cache/cp/ptdb_egg_sub.npz (scoring only), with the closed-phase F1, teacher v3's
             cached F1 and QCP-FB's F1 (D-046 arm B settings, report only).
fit: D-045's registered rules (see docs/decisions.md D-045); writes d045.txt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

TEACHER_V3 = Path("runs/conformer_v2r2_s1_d16k/ckpt_020000.pt")
TEACHER_V3_CFG = Path("runs/conformer_v2r2_s1/config.json")
DATASETS = Path("C:/github/datasets")
ZEROTH = DATASETS / "zeroth_korean"
SO = DATASETS / "speechocean762" / "speechocean762"
PTDB_ROOT = Path("data/ptdb_tug")
CACHE = Path("cache/cp/ptdb_egg_sub.npz")
FS = 16000
CROP = 400
PER_CLASS_SEX = 20000
MEASURES = ("h1h2", "h1a1", "cpp", "hnr", "tilt", "hf", "jitter", "shimmer")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- teacher features

class Trunk:
    """teacher v3 with a hook on its formant encoder: per frame h, f0, voiced and sonorant probabilities, F1-F3, B1-B3"""

    def __init__(self, dev):
        from vocal_analysis.features.vqt import TorchVQT
        from vocal_analysis.models.infer import load_teacher

        self.dev = dev
        self.model = load_teacher(TEACHER_V3, dev)[0]
        self.vq = TorchVQT(dev)
        self._h = {}
        self.model.enc.register_forward_hook(lambda m, i, o: self._h.__setitem__("h", o))

    def _decode(self, out):
        from vocal_analysis.models.infer import decode_outputs

        h = self._h["h"].float().cpu().numpy()
        o = {k: v.float().cpu().numpy() for k, v in out.items()}
        res = []
        for b in range(h.shape[0]):
            d = decode_outputs({k: v[b] for k, v in o.items()})
            res.append({"h": h[b], "f0": d["f0"], "voiced_p": d["voiced_p"], "sonorant_p": d["sonorant_p"],
                        "F": d["formants"][:, :3], "B": d["bandwidths"][:, :3]})
        return res

    def crops(self, audio):
        import torch

        with torch.inference_mode():
            return self._decode(self.model(self.vq.log_power_with_context(audio.to(self.dev), CROP)))

    def file(self, x):  # frame i centred at sample i * HOP
        import torch

        from vocal_analysis.features.vqt import HOP

        n = len(x) // HOP + 1
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(self.dev)
            return self._decode(self.model(self.vq.log_power(t, n)))[0]


# ---------------------------------------------------------------- interpretable measures

def _harm(db, freqs, f, half):
    m = (freqs > f - half) & (freqs < f + half)
    return float(db[m].max()) if m.any() else np.nan


def measures(x, centres, f0, f1):
    """(T, 8) MEASURES at sample centres of x (16 kHz), given f0 and F1 (Hz) per frame."""
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(np.asarray(x, np.float64), FS)
    try:
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0)
    except Exception:
        harm = None
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
    except Exception:
        pp = None
    nwin = int(0.04 * FS)
    win = np.hanning(nwin)
    nfft = 4096
    freqs = np.fft.rfftfreq(nfft, 1 / FS)
    out = np.full((len(centres), len(MEASURES)), np.nan)
    for i, (c, f, F1) in enumerate(zip(centres, f0, f1)):
        a0 = int(c) - nwin // 2
        if a0 < 0 or a0 + nwin > len(x) or not (f > 0):
            continue
        seg = x[a0:a0 + nwin] * win
        db = 20 * np.log10(np.abs(np.fft.rfft(seg, nfft)) + 1e-9)
        h1, h2 = _harm(db, freqs, f, f / 4), _harm(db, freqs, 2 * f, f / 4)
        k1 = max(1, int(round(F1 / f))) if F1 > 0 else 1
        a1 = _harm(db, freqs, k1 * f, f / 4)
        # cepstral peak prominence (Hillenbrand et al. 1994): cepstrum of the dB spectrum, peak near 1 / f0 above a
        # regression line over 1 ms upwards
        ceps = np.abs(np.fft.irfft(db))
        q = np.arange(len(ceps)) / FS
        sel = (q >= 0.001) & (q <= 0.025)
        pk = (q >= 0.8 / f) & (q <= 1.25 / f)
        cpp = np.nan
        if pk.any() and sel.sum() > 2:
            line = np.polyfit(q[sel], 20 * np.log10(ceps[sel] + 1e-12), 1)
            j = np.flatnonzero(pk)[np.argmax(ceps[pk])]
            cpp = 20 * np.log10(ceps[j] + 1e-12) - np.polyval(line, q[j])
        band = (freqs >= 100) & (freqs <= 4000)
        tilt = np.polyfit(np.log2(freqs[band]), db[band], 1)[0]
        p = np.abs(np.fft.rfft(seg, nfft)) ** 2
        hf = 10 * np.log10(p[freqs >= 4000].sum() / max(p[(freqs > 0) & (freqs < 4000)].sum(), 1e-20) + 1e-20)
        t = c / FS
        hnr = harm.get_value(t) if harm is not None else np.nan
        jit = shim = np.nan
        if pp is not None:
            try:
                jit = call(pp, "Get jitter (local)", t - 0.05, t + 0.05, 0.0001, 0.02, 1.3)
                shim = call([snd, pp], "Get shimmer (local)", t - 0.05, t + 0.05, 0.0001, 0.02, 1.3, 1.6)
            except Exception:
                pass
        out[i] = (h1 - h2, h1 - a1, cpp, hnr, tilt, hf, jit, shim)
    return out


# ---------------------------------------------------------------- sources

def zeroth_speakers():
    """{speaker id: sex} for the test split, from AUDIO_INFO's SPEAKERID, SEX and DATASET columns only."""
    out = {}
    with open(ZEROTH / "AUDIO_INFO", encoding="utf-8", errors="replace") as f:
        cols = [c.strip() for c in f.readline().rstrip("\n").split("|")]
        si, xi, di = cols.index("SPEAKERID"), cols.index("SEX"), cols.index("DATASET")
        for line in f:
            p = [c.strip() for c in line.rstrip("\n").split("|")]
            if len(p) > max(si, xi, di) and p[di] == "test_data_01":
                out[p[si]] = p[xi]
    return out


def so_speakers():
    """{speaker dir: (sex, age)} over speechocean762's train and test metadata."""
    out = {}
    for split in ("train", "test"):
        age = dict(line.split() for line in (SO / split / "spk2age").read_text().splitlines() if line.strip())
        sex = dict(line.split() for line in (SO / split / "spk2gender").read_text().splitlines() if line.strip())
        for k, a in age.items():
            out[f"SPEAKER{int(k):04d}"] = (sex.get(k, "?"), int(a))
    return out


def select(r, rng, k):
    ok = np.flatnonzero((r["voiced_p"] > 0.5) & (r["sonorant_p"] > 0.5) & (r["f0"] > 0))
    return np.sort(rng.choice(ok, min(k, len(ok)), replace=False)) if len(ok) else ok


def rows_from(r, idx, x, centres, extra):
    m = measures(x, centres, r["f0"][idx], r["F"][idx, 0])
    return {"h": r["h"][idx].astype(np.float16), "f0": r["f0"][idx], "F": r["F"][idx], "B": r["B"][idx], "meas": m,
            **{k: np.full(len(idx), v) if np.isscalar(v) or isinstance(v, str) else v for k, v in extra.items()}}


def stack(rows):
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def real_source(T, files_by_spk, meta, per_spk, rng, reader):
    from vocal_analysis.features.vqt import HOP

    rows = []
    for spk, files in files_by_spk.items():
        got = 0
        for f in rng.permutation(files):
            if got >= per_spk:
                break
            x = reader(f)
            r = T.file(x)
            idx = select(r, rng, min(40, per_spk - got))
            if not len(idx):
                continue
            rows.append(rows_from(r, idx, x, idx * HOP, {"speaker": spk, **meta[spk]}))
            got += len(idx)
    return stack(rows)


def features(out: Path, smoke: bool):
    import soundfile as sf
    from scipy.signal import resample_poly
    from torch.utils.data import DataLoader, Subset

    from vocal_analysis.device import DeviceConfig, setup_device
    from vocal_analysis.features.vqt import HOP, N_MAX
    from vocal_analysis.synth.channel import ChannelConfig
    from vocal_analysis.train.data import SynthCrops, collate, split_ids

    dev = setup_device(DeviceConfig(device="cuda"))
    T = Trunk(dev)
    rng = np.random.default_rng(45045)
    scale = 0.02 if smoke else 1.0

    def read16(path):
        x, sr = sf.read(path)
        x = x if x.ndim == 1 else x.mean(1)
        return resample_poly(x, FS, sr) if sr != FS else x

    if not (out / "synth.npz").exists():
        cfg3 = json.load(open(TEACHER_V3_CFG))
        channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
        flat = Path(cfg3["train"]["data"]) / "flat"
        n_clips = json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"]
        train_ids, _ = split_ids(n_clips)
        n = max(8, int(800 * scale))
        ids = np.sort(rng.choice(train_ids, n, replace=False))
        ds = SynthCrops(flat, ids, CROP, channel, seed=45045, fixed=True)
        rows, k = [], 0
        for audio, _ in DataLoader(Subset(ds, range(n)), 16, num_workers=4, collate_fn=collate):
            for b, r in enumerate(T.crops(audio)):
                idx = select(r, rng, 60)
                if len(idx):
                    x = audio[b].numpy().astype(np.float64)
                    rows.append(rows_from(r, idx, x, N_MAX // 2 + idx * HOP, {"speaker": f"clip{ids[k + b]}", "sex": "?", "age": -1}))
            k += len(audio)
        np.savez(out / "synth.npz", **stack(rows))
        log(f"synth: {sum(len(r['f0']) for r in rows)} frames from {n} clips")

    if not (out / "zeroth.npz").exists():
        meta = {s: {"sex": x, "age": -1} for s, x in zeroth_speakers().items()}
        files = {s: sorted(map(str, (ZEROTH / "test_data_01").glob(f"*/{s}/*.flac"))) for s in meta}
        files = {s: f for s, f in files.items() if f}
        per = {sx: int(PER_CLASS_SEX * scale / max(1, sum(meta[s]["sex"] == sx for s in files))) for sx in "fm"}
        rows = []
        for sx in "fm":
            sub = {s: f for s, f in files.items() if meta[s]["sex"] == sx}
            rows.append(real_source(T, sub, meta, per[sx], rng, read16))
        np.savez(out / "zeroth.npz", **stack(rows))
        log(f"zeroth: {len(stack(rows)['f0'])} frames, speakers {len(files)}")

    if not (out / "so.npz").exists():
        meta = {s: {"sex": sx, "age": a} for s, (sx, a) in so_speakers().items()}
        files = {s: sorted(map(str, (SO / "WAVE" / s).glob("*.WAV"))) for s in meta}
        files = {s: f for s, f in files.items() if f}
        if smoke:
            files = dict(list(files.items())[:6])
        rows = []
        for group, keep in (("adult", lambda a: a >= 18), ("child", lambda a: a < 18)):
            for sx in "fm":
                sub = {s: f for s, f in files.items() if meta[s]["sex"] == sx and keep(meta[s]["age"])}
                if not sub:
                    continue
                per = max(1, int(PER_CLASS_SEX * scale / len(sub)))
                r = real_source(T, sub, meta, per, rng, read16)
                r["group"] = np.full(len(r["f0"]), group)
                rows.append(r)
        np.savez(out / "so.npz", **stack(rows))
        log(f"speechocean762: {len(stack(rows)['f0'])} frames, speakers {len(files)}")

    if not (out / "ptdb.npz").exists():
        from vocal_analysis.baselines.formants import FORMANT_BASELINES
        from vocal_analysis.eval.datasets import ptdb_tug as pt
        from vocal_analysis.eval.tiers import load as load_tiers

        z = np.load(CACHE, allow_pickle=True)
        tiers = load_tiers()
        keep = np.array([tiers.get("ptdb_tug:" + str(s)) == "dev" for s in z["speaker"]])
        keep &= np.isfinite(z["cp"][:, 0]) & np.isfinite(z["conformer_v2r2_s1_d16k"][:, 0])
        idx_all = np.flatnonzero(keep)
        utts = list(dict.fromkeys(z["utt"][idx_all]))
        if smoke:
            utts = utts[:4]
        paths = {Path(r.path).stem: PTDB_ROOT / r.path for r in pt.scan(PTDB_ROOT)}
        qcp_fb = FORMANT_BASELINES["qcp_fb"].predict
        rows = []
        for u in utts:
            ii = idx_all[z["utt"][idx_all] == u]
            x = read16(paths[u])
            r = T.file(x)
            fr = np.clip(np.round(z["t"][ii] / 0.010).astype(int), 0, len(r["f0"]) - 1)
            q = qcp_fb(x.astype(np.float32), FS, z["t"][ii], {})[:, 0]
            rows.append(rows_from(r, fr, x, fr * HOP, {
                "speaker": z["speaker"][ii], "sex": z["sex"][ii], "age": -1, "cp_F1": z["cp"][ii, 0],
                "tv3_F1": z["conformer_v2r2_s1_d16k"][ii, 0], "qcp_F1": q, "cq": z["cq"][ii], "f0_ref": z["f0"][ii]}))
        np.savez(out / "ptdb.npz", **stack(rows))
        log(f"ptdb dev: {len(stack(rows)['f0'])} frames from {len(utts)} utterances")


# ---------------------------------------------------------------- fit and rules

def f0_weights(f_syn, f_real):
    """Importance weights for synthetic frames so their f0 distribution matches the real one (1/6-octave bins)."""
    edges = 2.0 ** np.arange(np.log2(50), np.log2(1000), 1 / 6)
    hs, _ = np.histogram(f_syn, edges)
    hr, _ = np.histogram(f_real, edges)
    ratio = np.where(hs > 0, (hr / max(hr.sum(), 1)) / np.maximum(hs / max(hs.sum(), 1), 1e-12), 0.0)
    b = np.clip(np.digitize(f_syn, edges) - 1, 0, len(ratio) - 1)
    w = ratio[b]
    return w * (len(f_real) / max(w.sum(), 1e-12))


def design(d, kind):
    if kind == "h":
        return d["h"].astype(np.float64)
    return np.column_stack([d["meas"], d["B"], np.log2(np.maximum(d["f0"], 1.0))])


def fit_one(Xs, ws, gs, Xr, gr, seed=0, shuffle=False):
    """Weighted logistic regression, real = 1. Returns (model, cross-validated AUC)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    X = np.vstack([Xs, Xr])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xr))]
    w = np.r_[ws, np.ones(len(Xr))]
    g = np.r_[gs, gr]
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    make = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                                 LogisticRegressionCV(Cs=8, cv=3, max_iter=2000, scoring="roc_auc"))
    auc = np.nan
    if not shuffle:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make().fit(X[tr], y[tr], logisticregressioncv__sample_weight=w[tr])
            pred[te] = m.decision_function(X[te])
        auc = roc_auc_score(y, pred, sample_weight=w)
    return make().fit(X, y, logisticregressioncv__sample_weight=w), auc


def spearman_boot(x, y, spk, rng, n=2000):
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    x, y, spk = x[ok], y[ok], spk[ok]
    rho = spearmanr(x, y)[0]
    us = np.unique(spk)
    by = {u: np.flatnonzero(spk == u) for u in us}
    bs = []
    for _ in range(n):
        ii = np.concatenate([by[u] for u in rng.choice(us, len(us))])
        bs.append(spearmanr(x[ii], y[ii])[0])
    return rho, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fit(out: Path):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    load = lambda n: {k: v for k, v in np.load(out / f"{n}.npz", allow_pickle=True).items()}  # noqa: E731
    syn, zer, so, pt = load("synth"), load("zeroth"), load("so"), load("ptdb")
    adult = so["group"] == "adult"
    corpora = {"zeroth": zer, "so_adult": {k: v[adult] for k, v in so.items()}}
    rng = np.random.default_rng(45)
    lines = [f"D-045 ({time.strftime('%Y-%m-%d %H:%M')}): synthetic {len(syn['f0'])} frames; " + "; ".join(
        f"{c} {sx} {int((d['sex'] == sx).sum())} frames / {len(np.unique(d['speaker'][d['sex'] == sx]))} speakers"
        for c, d in corpora.items() for sx in "fm"), ""]
    err_cp = pt["tv3_F1"] - pt["cp_F1"]
    err_q = pt["tv3_F1"] - pt["qcp_F1"]
    rule1 = {}
    coefs = {}
    for kind in ("h", "a"):
        for cname, d in corpora.items():
            for sx in "fm":
                r = {k: v[d["sex"] == sx] for k, v in d.items()}
                if len(r["f0"]) < 50 or len(np.unique(r["speaker"])) < 2:
                    lines.append(f"D_{kind} {cname:8s} {sx}: too few frames or speakers ({len(r['f0'])}); skipped")
                    if kind == "h" and sx == "f":
                        rule1[cname] = False
                    continue
                ws = f0_weights(syn["f0"], r["f0"])
                model, auc = fit_one(design(syn, kind), ws, syn["speaker"], design(r, kind), r["speaker"])
                coefs[kind, cname, sx] = model[-1].coef_.ravel()
                m = pt["sex"] == sx
                if m.sum() < 20:
                    lines.append(f"D_{kind} {cname:8s} {sx}: CV AUC {auc:.3f}; too few PTDB dev frames ({m.sum()}) to apply")
                    if kind == "h" and sx == "f":
                        rule1[cname] = False
                    continue
                score = model.decision_function(design({k: v[m] for k, v in pt.items()}, kind))
                rho, lo, hi = spearman_boot(score, err_cp[m], pt["speaker"][m], rng)
                line = f"D_{kind} {cname:8s} {sx}: CV AUC {auc:.3f}; PTDB dev Spearman(real-ness, tv3 - cp F1) {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                if kind == "h":
                    nulls = []
                    for s in range(20):
                        mn, _ = fit_one(design(syn, kind), ws, syn["speaker"], design(r, kind), r["speaker"], seed=s, shuffle=True)
                        sc = mn.decision_function(design({k: v[m] for k, v in pt.items()}, kind))
                        ok = np.isfinite(err_cp[m])
                        nulls.append(spearmanr(sc[ok], err_cp[m][ok])[0])
                    line += f"; null max {max(nulls):+.3f} (median {np.median(nulls):+.3f})"
                    if sx == "f":
                        rule1[cname] = rho >= 0.2 and lo > 0 and rho > max(nulls)
                    rq = spearman_boot(score, err_q[m], pt["speaker"][m], rng)
                    line += f"\n    reported: with tv3 - QCP-FB F1 {rq[0]:+.3f} [{rq[1]:+.3f}, {rq[2]:+.3f}]"
                    for lab, v in (("closed quotient", pt["cq"][m]), ("f0", pt["f0"][m])):
                        rr = spearman_boot(score, v, pt["speaker"][m], rng)
                        line += f"; with {lab} {rr[0]:+.3f} [{rr[1]:+.3f}, {rr[2]:+.3f}]"
                    if cname == "so_adult":
                        sp = np.unique(r["speaker"])
                        sc_r = model.decision_function(design(r, kind))
                        mean_sc = np.array([sc_r[r["speaker"] == s_].mean() for s_ in sp])
                        ages = np.array([r["age"][r["speaker"] == s_][0] for s_ in sp], float)
                        line += f"\n    reported: real-ness against age over {len(sp)} speakers, Spearman {spearmanr(mean_sc, ages)[0]:+.3f}"
                lines.append(line)
    for sx in "fm":
        if ("h", "zeroth", sx) not in coefs or ("h", "so_adult", sx) not in coefs:
            continue
        a, b = coefs["h", "zeroth", sx], coefs["h", "so_adult", sx]
        lines.append(f"D_h direction overlap (cosine, zeroth vs so_adult) {sx}: {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):+.3f}")
    names = list(MEASURES) + ["B1", "B2", "B3", "log2 f0"]
    lines.append("\nD_a standardised coefficients (real = +):")
    for cname in corpora:
        for sx in "fm":
            if ("a", cname, sx) not in coefs:
                continue
            lines.append(f"  {cname:8s} {sx}: " + "  ".join(f"{n} {c:+.2f}" for n, c in zip(names, coefs["a", cname, sx])))
    lines.append("\nPer-measure f0-matched AUC (synthetic vs real) and PTDB dev Spearman with tv3 - cp F1:")
    cand = []
    for j, n in enumerate(names):
        aucs = {}
        for cname, d in corpora.items():
            for sx in "fm":
                r = {k: v[d["sex"] == sx] for k, v in d.items()}
                if len(r["f0"]) < 50:
                    aucs[cname, sx] = np.nan
                    continue
                xs, xr = design(syn, "a")[:, j], design(r, "a")[:, j]
                ws = f0_weights(syn["f0"], r["f0"])
                ok_s, ok_r = np.isfinite(xs), np.isfinite(xr)
                aucs[cname, sx] = roc_auc_score(np.r_[np.zeros(ok_s.sum()), np.ones(ok_r.sum())], np.r_[xs[ok_s], xr[ok_r]],
                                                sample_weight=np.r_[ws[ok_s], np.ones(ok_r.sum())])
        m = pt["sex"] == "f"
        rho = spearman_boot(design({k: v[m] for k, v in pt.items()}, "a")[:, j], err_cp[m], pt["speaker"][m], rng)
        both = all(abs(aucs[c, "f"] - 0.5) >= 0.2 for c in corpora)
        corr = abs(rho[0]) >= 0.15 and (rho[1] > 0 or rho[2] < 0)
        if both and corr:
            cand.append(n)
        lines.append(f"  {n:8s} " + " ".join(f"{c} {sx} {aucs[c, sx]:.2f}" for c in corpora for sx in "fm")
                     + f"; PTDB women {rho[0]:+.3f} [{rho[1]:+.3f}, {rho[2]:+.3f}]" + ("  CANDIDATE" if both and corr else ""))
    r1 = all(rule1.values())
    lines.append(f"\nRULE 1 (D_h locates the overread, both corpora): {'holds' if r1 else 'fails'} {rule1}")
    lines.append(f"RULE 2 candidates: {cand if r1 else 'not evaluated (rule 1 fails); reported above'}")
    lines.append("Children's arm (Hillenbrand dev children; speechocean762 children; Samromur after its licence check): separate step.")
    (out / "d045.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("features", "fit"):
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (features if sys.argv[1] == "features" else lambda o, s: fit(o))(out, "--smoke" in sys.argv)


if __name__ == "__main__":
    main()
