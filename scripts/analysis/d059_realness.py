"""D-059: how real do the KL clips look to D-045's discriminators, beside the interactive tube and cascade clips and
OPENGLOT Repositories II and III? Descriptive, one flag (docs/decisions.md).

    python scripts/analysis/d059_realness.py render <out>     (CPU, parallel: the three synthetic sets, clean audio)
    python scripts/analysis/d059_realness.py features <out>   (GPU in short bursts, then CPU: teacher v3 h and measures)
    python scripts/analysis/d059_realness.py score <out>      (CPU: D-045's discriminators refitted, applied)

Synthetic sets (train_v2's synth config, N_CLIPS each, seed 5059): kl (kl_set.generate), tube (interactive source),
cascade; each scored clean and through teacher v3's channel augmentation (fixed per clip). OPENGLOT II (channel 1) and
III (channel 2, the radiated pressure) as they are: evaluation only, fit nothing. The discriminators are refitted from
D-045's cached features (cache/d045) with D-045's own code, never on any set scored here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

N_CLIPS = int(os.environ.get("D059_N", 160))  # D059_N: a smaller run, to check the code path
SEED = 5059
PER_CLIP = 60
D045_CACHE = Path("cache/d045")
TRAIN_V2 = Path("data/synth/train_v2")
SETS = ("kl", "tube", "cascade")


def d045():
    spec = importlib.util.spec_from_file_location("d045", Path(__file__).with_name("d045_discriminator.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synth_config():
    from vocal_analysis.synth.sampler import SynthConfig

    ix = json.loads((TRAIN_V2 / "index.json").read_text(encoding="utf-8"))
    return SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ix["synth_config"].items()})


def _clip(args):
    name, i = args
    from vocal_analysis.synth import kl_set, sampler

    base = synth_config()
    if name == "kl":
        c = kl_set.generate(i, replace(kl_set.KLSetConfig(), synth=replace(base, p_tube=1.0)), SEED)
    elif name == "tube":
        c = sampler.generate(i, replace(base, p_tube=1.0, p_interactive=1.0), SEED)
    else:
        c = sampler.generate(i, replace(base, p_tube=0.0), SEED)
    return name, i, c.audio, c.labels["voiced"]


def render(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, i) for s in SETS for i in range(N_CLIPS) if not (out / "clips" / s / f"{i:04d}.npz").exists()]
    for s in SETS:
        (out / "clips" / s).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(10) as ex:
        for k, (s, i, audio, voiced) in enumerate(ex.map(_clip, jobs, chunksize=2), 1):
            np.savez(out / "clips" / s / f"{i:04d}.npz", audio=audio, voiced=voiced)
            if k % 40 == 0:
                print(f"rendered {k}/{len(jobs)}", flush=True)


def features(out: Path):
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from vocal_analysis.features.vqt import HOP
    from vocal_analysis.synth.channel import ChannelConfig, augment

    D = d045()
    T = D.Trunk(torch.device("cuda"))
    cfg3 = json.load(open(D.TEACHER_V3_CFG))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    rng = np.random.default_rng(SEED)

    def take(x, extra):
        r = T.file(x.astype(np.float32))
        idx = D.select(r, rng, PER_CLIP)
        return D.rows_from(r, idx, x.astype(np.float64), idx * HOP, extra) if len(idx) else None

    for s in SETS:
        for cond in ("clean", "channel"):
            path = out / f"{s}_{cond}.npz"
            if path.exists():
                continue
            rows = []
            for i in range(N_CLIPS):
                z = np.load(out / "clips" / s / f"{i:04d}.npz")
                x = z["audio"].astype(np.float64)
                if cond == "channel":
                    mask = np.pad(np.repeat(z["voiced"], 160), (0, max(0, len(x) - 160 * len(z["voiced"]))))[: len(x)]
                    x, _ = augment(x, np.random.default_rng([SEED, i]), channel, mask, codec_stream=False)
                r = take(x, {"speaker": f"{s}{i}", "sex": "?", "age": -1})
                if r is not None:
                    rows.append(r)
            np.savez(path, **D.stack(rows))
            D.log(f"{s} {cond}: {sum(len(r['f0']) for r in rows)} frames")
    if not (out / "openglot.npz").exists():
        rows = []
        files = [(f, 0, "II") for f in sorted(Path("data/openglot").glob("RepositoryII_*/*.wav"))]
        files += [(f, 1, "III") for f in sorted(Path("data/openglot/RepositoryIII").glob("*/*Hz.wav"))]
        for f, ch, rep in files:
            x, sr = sf.read(f)
            x = x[:, ch] if x.ndim == 2 else x
            x = resample_poly(x, 16000, sr)
            x = 0.5 * x / (np.abs(x).max() + 1e-12)
            sex = "f" if f.name.startswith("Female") or f.parent.name.startswith("Female") else "m"
            r = take(x, {"speaker": f"{rep}:{f.stem}", "sex": sex, "age": -1, "rep": rep})
            if r is not None:
                rows.append(r)
        np.savez(out / "openglot.npz", **D.stack(rows))
        D.log(f"openglot: {sum(len(r['f0']) for r in rows)} frames")


def score(out: Path):
    from sklearn.model_selection import GroupKFold

    D = d045()
    load = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}  # noqa: E731
    syn, zer, so = load(D045_CACHE / "synth.npz"), load(D045_CACHE / "zeroth.npz"), load(D045_CACHE / "so.npz")
    adult = so["group"] == "adult"
    corpora = {"zeroth": zer, "so_adult": {k: v[adult] for k, v in so.items()}}
    sets = {f"{s}_{c}": load(out / f"{s}_{c}.npz") for s in SETS for c in ("clean", "channel")}
    og = load(out / "openglot.npz")
    rng = np.random.default_rng(59)
    names = ["H1-H2", "H1-A1", "CPP", "HNR", "tilt", "HF", "jitter", "shimmer", "B1", "B2", "B3"]

    def wmed(v, w):
        o = np.argsort(v)
        c = np.cumsum(w[o])
        return float(v[o][np.searchsorted(c, 0.5 * c[-1])])

    def boot(v, w, groups, n=1000):
        us = np.unique(groups)
        by = {u: np.flatnonzero(groups == u) for u in us}
        b = []
        for _ in range(n):
            ii = np.concatenate([by[u] for u in rng.choice(us, len(us))])
            b.append(wmed(v[ii], w[ii]))
        return wmed(v, w), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), np.array(b)

    lines = ["D-059: realness (decision function of D-045's discriminators, refitted; log-odds real, > 0 = real side)",
             "per set: median over frames (synthetic sets weighted to the real sex's f0 distribution) [95 % bootstrap "
             "over clips / files]; train_v2 and real rows are out-of-fold (5-fold, grouped).", ""]
    boots = {}
    for kind in ("h", "a"):
        for cname, d in corpora.items():
            for sx in "fm":
                r = {k: v[d["sex"] == sx] for k, v in d.items()}
                ws = D.f0_weights(syn["f0"], r["f0"])
                model, auc = D.fit_one(D.design(syn, kind), ws, syn["speaker"], D.design(r, kind), r["speaker"])
                X = np.vstack([D.design(syn, kind), D.design(r, kind)])
                y = np.r_[np.zeros(len(syn["f0"])), np.ones(len(r["f0"]))]
                g = np.r_[syn["speaker"], r["speaker"]]
                # out-of-fold scores for the fitted classes, D-045's pipeline refitted per fold
                from sklearn.impute import SimpleImputer
                from sklearn.linear_model import LogisticRegressionCV
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                oof = np.zeros(len(y))
                w = np.r_[ws, np.ones(len(r["f0"]))]
                for tr, te in GroupKFold(5).split(X, y, g):
                    mm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       LogisticRegressionCV(Cs=8, cv=3, max_iter=2000, scoring="roc_auc"))
                    mm.fit(X[tr], y[tr], logisticregressioncv__sample_weight=w[tr])
                    oof[te] = mm.decision_function(X[te])
                lines.append(f"D_{kind} {cname} {sx} (CV AUC {auc:.3f})")
                rows = [("real (out-of-fold)", oof[y == 1], np.ones(int(y.sum())), g[y == 1]),
                        ("train_v2 channel (out-of-fold)", oof[y == 0], ws, g[y == 0])]
                for sname, sd in sets.items():
                    rows.append((sname, model.decision_function(D.design(sd, kind)), D.f0_weights(sd["f0"], r["f0"]),
                                 sd["speaker"]))
                for rep in ("II", "III"):
                    m = (og["rep"] == rep) & (og["sex"] == sx)
                    if m.any():
                        rows.append((f"OPENGLOT {rep}", model.decision_function(D.design({k: v[m] for k, v in og.items()}, kind)),
                                     np.ones(int(m.sum())), og["speaker"][m]))
                for sname, v, wv, gv in rows:
                    med, lo, hi, b = boot(v, wv, gv)
                    boots[kind, cname, sx, sname] = b
                    lines.append(f"  {sname:32s} {med:+7.2f} [{lo:+.2f}, {hi:+.2f}]  ({len(v)} frames)")
                lines.append("")
    flag = True
    for cname in corpora:
        kb = boots["h", cname, "f", "kl_channel"]
        for other in ("tube_channel", "cascade_channel"):
            d = kb - boots["h", cname, "f", other]
            ok = np.median(d) >= 1.0 and np.percentile(d, 2.5) > 0
            lines.append(f"flag check, women, D_h {cname}: kl_channel - {other} {np.median(d):+.2f} "
                         f"[{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}] -> {'yes' if ok else 'no'}")
            flag &= ok
    lines.append(f"Flag 'KL closer to real than the training synthesisers': {'RAISED' if flag else 'not raised'}")
    lines += ["", "Measures (median over frames, unweighted): " + "  ".join(names)]
    allsets = {"real zeroth": zer, "real so_adult": corpora["so_adult"], "train_v2 channel": syn, **sets}
    allsets.update({f"OPENGLOT {rep}": {k: v[og["rep"] == rep] for k, v in og.items()} for rep in ("II", "III")})
    for sname, d in allsets.items():
        m = np.column_stack([d["meas"], d["B"]])
        lines.append(f"  {sname:20s} " + "  ".join(f"{np.nanmedian(m[:, j]):7.2f}" for j in range(m.shape[1])))
    text = "\n".join(lines)
    print(text)
    (out / "d059.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"render": render, "features": features, "score": score}[sys.argv[1]](Path(sys.argv[2]))
