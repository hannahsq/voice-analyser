"""D-108: can v3.1's frozen trunk read the extra pole-zero sections? train_v2 clips regenerated (4000 from the training
split, the first 1000 of the validation split); per clip, the encoder output (D-082's hook) and the VQT input pooled
(mean, SD) over sonorant frames; logistic regression (presence) and ridge (frequencies), regularisation by 5-fold CV on
the training clips; held-out AUC / R^2 / MAE beside a 200-permutation shuffle null.

    python scripts/analysis/d108_extras_probe.py render      (CPU)
    python scripts/analysis/d108_extras_probe.py features    (GPU; VA_DEVICE=cuda)
    python scripts/analysis/d108_extras_probe.py probe > reports/analysis/d108/d108.txt
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CACHE = Path("cache/d108")
N_TRAIN, N_VAL = 4000, 1000
MODEL = "d040_5b_s2_b20k"
SR, HOP = 16000, 160


def cfg():
    from vocal_analysis.synth.sampler import SynthConfig

    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    return idx, SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in idx["synth_config"].items()
                               if k in SynthConfig.__dataclass_fields__})


def one(i):
    from vocal_analysis.synth.sampler import generate

    idx, c = cfg()
    clip = generate(int(i), c, idx["seed"])
    ex = clip.meta.get("extras") or {}
    pf, zf = np.asarray(ex.get("extra_pole_f", []), float), np.asarray(ex.get("extra_zero_f", []), float)
    nasal, sub = len(pf) in (1, 4), len(pf) >= 3
    t = [float(nasal), float(sub),
         np.log2(pf[0]) if nasal else np.nan, np.log2(zf[0]) if nasal else np.nan,
         np.log2(pf[1 if nasal else 0] / 600.0) if sub else np.nan]
    return clip.audio.astype(np.float32), clip.labels["sonorant"].astype(bool), np.array(t)


def render():
    from concurrent.futures import ProcessPoolExecutor

    from vocal_analysis.train.data import split_ids

    idx, _ = cfg()
    tr, va = split_ids(idx["n_clips"])
    CACHE.mkdir(parents=True, exist_ok=True)
    for split, ids in (("train", tr[:N_TRAIN]), ("val", va[:N_VAL])):
        out = CACHE / f"{split}.npz"
        if out.exists():
            continue
        with ProcessPoolExecutor(4) as ex:
            res = list(ex.map(one, [int(i) for i in ids], chunksize=16))
        a_off = np.cumsum([0] + [len(r[0]) for r in res])
        s_off = np.cumsum([0] + [len(r[1]) for r in res])
        np.savez(out, audio=np.concatenate([r[0] for r in res]), a_off=a_off, son=np.concatenate([r[1] for r in res]),
                 s_off=s_off, y=np.array([r[2] for r in res]))
        print(f"render {split}: {len(res)} clips", flush=True)


def features():
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(f"runs/{MODEL}/ckpt_last.pt", dev)
    vq = TorchVQT(dev)
    got = {}
    model.enc.register_forward_hook(lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    for split in ("train", "val"):
        out = CACHE / f"feat_{split}.npz"
        if out.exists():
            continue
        z = np.load(CACHE / f"{split}.npz")
        H, V = [], []
        with torch.inference_mode():
            for c in range(len(z["a_off"]) - 1):
                x = z["audio"][z["a_off"][c]:z["a_off"][c + 1]]
                son = z["son"][z["s_off"][c]:z["s_off"][c + 1]]
                pad = (-len(x)) % SR
                xx = torch.from_numpy(np.pad(x, (0, pad)))[None].to(dev)
                n_fr = (len(x) + pad) // HOP
                lp = vq.log_power_with_context(xx, n_fr)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    model(lp)
                h = got["h"][0].float().cpu().numpy()
                t = min(len(son), h.shape[0])
                m = son[:t] if son[:t].any() else np.ones(t, bool)
                H.append(np.r_[h[:t][m].mean(0), h[:t][m].std(0)])
                v = lp[0].float().cpu().numpy()  # input: pool every non-frame axis's values per frame
                v = v.reshape(-1, v.shape[-1]).T if v.shape[-1] >= t else v.reshape(v.shape[0], -1)
                v = v[:t][m]
                V.append(np.r_[v.mean(0), v.std(0)])
        np.savez(out, h=np.array(H, np.float32), v=np.array(V, np.float32))
        print(f"features {split}: {len(H)} clips, trunk {len(H[0])}, input {len(V[0])}", flush=True)


def probe():
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    ytr, yva = np.load(CACHE / "train.npz")["y"], np.load(CACHE / "val.npz")["y"]
    ftr, fva = np.load(CACHE / "feat_train.npz"), np.load(CACHE / "feat_val.npz")
    rng = np.random.default_rng(108)
    names = {0: "nasal present", 1: "subglottal present", 2: "nasal pole (log2 Hz)", 3: "nasal zero (log2 Hz)",
             4: "subglottal scale (log2)"}
    L = [f"D-108: frozen-feature probes for the extra sections; train {len(ytr)} clips, held-out {len(yva)}; "
         f"presence prevalence (train) nasal {ytr[:, 0].mean():.2f}, subglottal {ytr[:, 1].mean():.2f}", ""]
    res = {}
    for src in ("h", "v"):
        sc = StandardScaler().fit(ftr[src])
        Xtr, Xva = sc.transform(ftr[src]), sc.transform(fva[src])
        lab = "trunk (v3.1 encoder)" if src == "h" else "input (VQT)"
        L.append(f"{lab}, {Xtr.shape[1]} features")
        for k in (0, 1):
            m = LogisticRegressionCV(Cs=np.logspace(-4, 1, 8), cv=5, max_iter=3000).fit(Xtr, ytr[:, k])
            auc = roc_auc_score(yva[:, k], m.predict_proba(Xva)[:, 1])
            null = [roc_auc_score(yva[:, k], LogisticRegression(C=m.C_[0], max_iter=3000)
                                  .fit(Xtr, rng.permutation(ytr[:, k])).predict_proba(Xva)[:, 1]) for _ in range(200)]
            res[src, k] = auc
            L.append(f"  {names[k]:26s} AUC {auc:.3f}   (shuffle null 95th pct {np.percentile(null, 95):.3f})")
        for k in (2, 3, 4):
            a, b = np.isfinite(ytr[:, k]), np.isfinite(yva[:, k])
            m = RidgeCV(alphas=np.logspace(-2, 5, 15), cv=5).fit(Xtr[a], ytr[a, k])
            p = m.predict(Xva[b])
            r2 = r2_score(yva[b, k], p)
            mae = 1200 * np.median(np.abs(p - yva[b, k]))
            null = [r2_score(yva[b, k], Ridge(alpha=m.alpha_).fit(Xtr[a], rng.permutation(ytr[a, k])).predict(Xva[b])) for _ in range(200)]
            res[src, k] = r2
            L.append(f"  {names[k]:26s} R^2 {r2:+.3f}, median |error| {mae:.0f} c (n {a.sum()} / {b.sum()}; "
                     f"shuffle null 95th pct R^2 {np.percentile(null, 95):+.3f}; label SD {1200 * np.std(yva[b, k]):.0f} c)")
        L.append("")
    learn = res["h", 0] >= 0.85 and res["h", 1] >= 0.85 and res["h", 2] >= 0.5 and res["h", 4] >= 0.5
    not_learn = res["h", 0] < 0.70 and res["h", 1] < 0.70
    L.append("DECISION: " + ("LEARNABLE" if learn else "NOT LEARNABLE" if not_learn else "PARTLY")
             + f" (trunk: AUC nasal {res['h', 0]:.3f}, subglottal {res['h', 1]:.3f}; R^2 nasal pole {res['h', 2]:+.3f}, "
             f"subglottal scale {res['h', 4]:+.3f})")
    print("\n".join(L))


if __name__ == "__main__":
    {"render": render, "features": features, "probe": probe}[sys.argv[1]]()
