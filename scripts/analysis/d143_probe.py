"""D-143 stage A: is a wide F1 at low f0 linearly readable from 5b's frozen frame vector h (the formant heads' input)?
Ridge probes for log2 F1 from h, trained on synthetic clips with wide and narrow F1 bandwidths; against a control probe
on the model's own decoded formants and bandwidths, and a clip-shuffled null. CPU, inference only.

    python scripts/analysis/d143_probe.py smoke   -> prints the heads' failing-clip count on 20 test clips (feasibility)
    python scripts/analysis/d143_probe.py run     -> reports/analysis/d143/stageA.{json,txt}
    python scripts/analysis/d143_probe.py recon   -> reports/analysis/d143/stageC.{json,txt}  (stage C: M4's loss on the grid)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d143"
FS = 16000
MODELS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
N_TRAIN, N_TEST = 800, 400
GRID_SEEDS = (20, 21, 22)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
B = 2000


def _d142():
    spec = importlib.util.spec_from_file_location("d142", ROOT / "scripts/analysis/d142_flags.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def draw(rng, test: bool):
    """One clip's settings. Train: f0 100-230, B1 60-700; test: f0 100-150, B1 250-700 (where the heads fail)."""
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    F1 = rng.uniform(350, 650)
    F2 = max(F1 * rng.uniform(1.3, 2.2), 650.0)
    F3 = rng.uniform(2300, 3000)
    return dict(F=(F1, F2, F3, F3 + rng.uniform(500, 1000)), f0=lu(100, 150) if test else lu(100, 230),
                b1=lu(250, 700) if test else lu(60, 700), b=tuple(b * lu(0.7, 1.4) for b in (90, 120, 150)),
                rd=rng.uniform(0.6, 1.8))


def clip(p, seed):
    """3 s cascade vowel as D-142's grid (`_grid_audio`), with the drawn B2-B4 and Rd."""
    from vocal_analysis.synth import sampler, tract, voice

    C = int(3.0 * FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / FS
    pf = np.full((C, tract.N_POLES), np.nan)
    pb = pf.copy()
    pf[:, :4], pb[:, :4] = p["F"], (p["b1"], *p["b"])
    rng = np.random.default_rng(seed)
    tr = voice.Trajectories(f0=p["f0"] * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12), rd=np.full(C, p["rd"]),
                            voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1), whisper_amp=np.zeros(C),
                            fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
    return 0.5 * y / np.abs(y).max()


def clip_sets(n_train=N_TRAIN, n_test=N_TEST):
    d142 = _d142()
    rng = np.random.default_rng(1430)
    sets = {"train": [], "test": [], "grid": []}
    for name, n, test, base in (("train", n_train, False, 143000), ("test", n_test, True, 243000)):
        for i in range(n):
            p = draw(rng, test)
            sets[name].append((f"{name}{i}", clip(p, base + i), np.array(p["F"])))
    for sname, Fs in d142.SETS.items():
        for f0 in d142.F0S:
            for b1 in d142.B1S:
                for s in GRID_SEEDS:
                    sets["grid"].append((f"{sname}|f{f0:g}|b{b1:g}|s{s}", d142._grid_audio(Fs, f0, b1, s), np.array(Fs)))
    return sets


def features(model, dev, items):
    """Per voiced frame of the middle 60 % (D-142's selection): h, the decoded log2 F1-F4 and log2 B1-B4, the frame's
    failure (F2 nearer the true F3 than the true F2), the true log2 F1, and the clip index."""
    from vocal_analysis.models.infer import analyse

    store = {}
    hook = model.bw_head.register_forward_hook(lambda m, i, o: store.__setitem__("h", i[0][0].float().cpu().numpy()))
    H, C, fail, y1, cid = [], [], [], [], []
    try:
        for j, (_, y, true) in enumerate(items):
            r = analyse(model, y.astype(np.float32), dev)
            n = len(r["voiced_p"])
            for i in range(int(0.2 * n), int(0.8 * n)):
                if r["voiced_p"][i] <= 0.5 or not np.isfinite(r["f0"][i]) or i * 160 < 512 or i * 160 + 512 > len(y):
                    continue
                Fm, Bm = r["formants"][i, :4], r["bandwidths"][i, :4]
                if not (np.all(np.isfinite(Fm)) and np.all(np.isfinite(Bm))):
                    continue
                H.append(store["h"][i])
                C.append(np.r_[np.log2(Fm), np.log2(Bm)])
                fail.append(abs(np.log(Fm[1] / true[2])) < abs(np.log(Fm[1] / true[1])))
                y1.append(np.log2(true[0]))
                cid.append(j)
    finally:
        hook.remove()
    return dict(H=np.array(H), C=np.array(C), fail=np.array(fail), y=np.array(y1), cid=np.array(cid))


class Ridge:
    def __init__(self, X, y, alpha):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        self.ym = y.mean()
        self.w = np.linalg.solve(Z.T @ Z + alpha * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - self.ym))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.ym


def fit(X, y, cid):
    """Ridge with alpha by 5-fold cross-validation over clips."""
    folds = cid % 5
    err = [np.mean([np.mean((Ridge(X[folds != k], y[folds != k], a)(X[folds == k]) - y[folds == k]) ** 2) for k in range(5)])
           for a in ALPHAS]
    a = ALPHAS[int(np.argmin(err))]
    return Ridge(X, y, a), a


def _models():
    import torch

    from vocal_analysis.models.infer import load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    return dev, {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}


def cmd_smoke() -> None:
    sets = clip_sets(n_train=0, n_test=20)
    dev, models = _models()
    f = features(models[MODELS[0]], dev, sets["test"])
    share = np.array([f["fail"][f["cid"] == j].mean() for j in np.unique(f["cid"])])
    print(f"{MODELS[0]}: {len(share)} test clips, {int((share >= 0.5).sum())} failing (>= 0.5), "
          f"{int((share < 0.2).sum())} not (< 0.2)")


def _clip_stats(f, pred):
    """Per clip: failure share, median |error| (cents) of a prediction of log2 F1."""
    out = []
    for j in np.unique(f["cid"]):
        k = f["cid"] == j
        out.append((float(f["fail"][k].mean()), float(np.median(np.abs(pred[k] - f["y"][k])) * 1200)))
    return np.array(out)


def cmd_run() -> None:
    sets = clip_sets()
    dev, models = _models()
    rng = np.random.default_rng(1431)
    res = {}
    for m, mod in models.items():
        tr, te, gr = (features(mod, dev, sets[k]) for k in ("train", "test", "grid"))
        ph, ah = fit(tr["H"], tr["y"], tr["cid"])
        pc, ac = fit(tr["C"], tr["y"], tr["cid"])
        perm = rng.permutation(N_TRAIN)  # clip-shuffled null: each clip's frames get another clip's true F1
        ynull = np.array([np.log2(sets["train"][perm[c]][2][0]) for c in tr["cid"]])
        pn, an = fit(tr["H"], ynull, tr["cid"])
        r = {"alpha": {"h": ah, "control": ac, "null": an}, "n_frames": {"train": len(tr["y"]), "test": len(te["y"])}}
        for name, f in (("test", te), ("grid", gr)):
            heads = f["C"][:, 0]
            st = {k: _clip_stats(f, p) for k, p in (("h", ph(f["H"])), ("control", pc(f["C"])), ("null", pn(f["H"])),
                                                   ("heads", heads))}
            r[name] = {k: v.tolist() for k, v in st.items()}
        res[m] = r
        print(m, flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "stageA.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report(res)


def report(res=None) -> None:
    res = res or json.loads((REP / "stageA.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1432)
    lines = ["D-143 stage A: wide F1 from 5b's frozen h by a linear (ridge) probe; median |F1 error| in cents, median over clips", ""]
    verdict = {}
    for m, r in res.items():
        lines.append(f"{m} (alpha h {r['alpha']['h']}, control {r['alpha']['control']}, null {r['alpha']['null']}; "
                     f"frames train {r['n_frames']['train']}, test {r['n_frames']['test']})")
        ok = {}
        for name in ("test", "grid"):
            st = {k: np.array(v) for k, v in r[name].items()}
            fl, nf = st["h"][:, 0] >= 0.5, st["h"][:, 0] < 0.2
            for g, sel in (("failing", fl), ("not failing", nf)):
                if not sel.any():
                    lines.append(f"  {name} {g}: no clips")
                    continue
                e = {k: st[k][sel, 1] for k in st}
                bs = [np.median(e["h"][i]) for i in [rng.integers(0, sel.sum(), sel.sum()) for _ in range(B)]]
                lines.append(f"  {name} {g} (n {int(sel.sum())}): h {np.median(e['h']):.0f} "
                             f"[{np.percentile(bs, 2.5):.0f}, {np.percentile(bs, 97.5):.0f}], control "
                             f"{np.median(e['control']):.0f}, null {np.median(e['null']):.0f}, heads {np.median(e['heads']):.0f}")
                ok[(name, g)] = {k: float(np.median(v)) for k, v in e.items()}
        t = ok.get(("test", "failing"))
        n = ok.get(("test", "not failing"))
        c1 = bool(t and t["h"] <= 100 and t["h"] <= 0.5 * t["control"] and t["h"] <= 0.5 * t["null"])
        c2 = bool(n and n["h"] <= n["heads"] + 50)
        verdict[m] = (c1, c2)
        lines.append(f"  (1) failing clips read by h: {'met' if c1 else 'not met'}; (2) probe sanity on not-failing clips: "
                     f"{'met' if c2 else 'not met'}")
        lines.append("")
    if not all(v[1] for v in verdict.values()):
        out = "INCONCLUSIVE (the probe is a poor read-out on normal clips for at least one model)"
    elif all(v[0] for v in verdict.values()):
        out = "EVIDENCE IN h: stage B gets registered"
    elif not any(v[0] for v in verdict.values()):
        out = "NOT IN h (linearly): no stage B; the structural routes come first"
    else:
        out = "SPLIT between seeds: reported, Hannah decides"
    lines.append(f"Reading (registered rule): {out}")
    (REP / "stageA.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_recon() -> None:
    """Stage C: D-109's reconstruction loss (LF source by Rd search, level profiled, F5-F8 from F4, harmonics to 4000 Hz,
    D-104's 10 dB SNR gate) per frame, of the model's own F1-F4 / B1-B4 against the true ones, on the grid's clips."""
    import torch

    from vocal_analysis.models.infer import analyse
    from vocal_analysis.train.reconstruction import harmonic_targets, recon_loss_fb

    grid = clip_sets(n_train=0, n_test=0)["grid"]
    dev, models = _models()
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)[None]
    res = {}
    for m, mod in models.items():
        for key, y, true in grid:
            r = analyse(mod, y.astype(np.float32), dev)
            n = len(r["voiced_p"])
            f0 = r["f0"].astype(np.float64)
            sel = np.zeros(n, bool)
            for i in range(int(0.2 * n), int(0.8 * n)):
                sel[i] = (r["voiced_p"][i] > 0.5 and np.isfinite(f0[i]) and i * 160 >= 512 and i * 160 + 512 <= len(y)
                          and np.all(np.isfinite(r["formants"][i, :4])) and np.all(np.isfinite(r["bandwidths"][i, :4])))
            amp, ok = harmonic_targets(y, np.where(sel, f0, np.nan), sel, snr_min_db=10.0)
            lab = {"db": t(amp), "ok": t(ok, torch.bool), "f0": t(np.where(sel, f0, 100.0)), "mask": t(sel, torch.bool)}
            b1 = float(key.split("|")[2][1:])
            Fr = np.where(sel[:, None], r["formants"][:, :4], 1000.0)
            Br = np.where(sel[:, None], r["bandwidths"][:, :4], 100.0)
            Ft, Bt = np.tile(true, (n, 1)), np.tile([b1, 90.0, 120.0, 150.0], (n, 1))
            with torch.no_grad():
                lr, lt = (recon_loss_fb(t(F), t(Bw), lab, fmax=4000.0, hpc=True, per_frame=True)[0].numpy()
                          for F, Bw in ((Fr, Br), (Ft, Bt)))
            k = sel & np.isfinite(lr) & np.isfinite(lt)
            fail = np.abs(np.log(Fr[k, 1] / true[2])) < np.abs(np.log(Fr[k, 1] / true[1]))
            res.setdefault(key, {})[m] = {"fail": float(fail.mean()) if k.any() else float("nan"), "n": int(k.sum()),
                                          "read": float(np.median(lr[k])) if k.any() else float("nan"),
                                          "true": float(np.median(lt[k])) if k.any() else float("nan"),
                                          "truth_lower": float(np.mean(lt[k] < lr[k])) if k.any() else float("nan")}
        print(m, flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "stageC.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    report_c(res)


def report_c(res=None) -> None:
    res = res or json.loads((REP / "stageC.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1433)
    lines = ["D-143 stage C (reported, deciding nothing): D-109's reconstruction loss per frame (Huber, dB) of the model's "
             "reading against the true F1-F4 / B1-B4, on D-142's grid at seeds 20-22; medians over frames per clip", ""]
    for m in MODELS:
        rows = [v[m] for v in res.values() if m in v and v[m]["n"] > 0]
        for g, sel in (("failing (>= 0.5)", lambda x: x["fail"] >= 0.5), ("not failing (< 0.2)", lambda x: x["fail"] < 0.2)):
            rs = [x for x in rows if sel(x)]
            if not rs:
                lines.append(f"  {m} {g}: no clips")
                continue
            d = np.array([x["read"] - x["true"] for x in rs])
            bs = [np.median(d[i]) for i in [rng.integers(0, len(d), len(d)) for _ in range(B)]]
            lines.append(f"  {m} {g} (n {len(rs)}): loss reading {np.median([x['read'] for x in rs]):.2f}, truth "
                         f"{np.median([x['true'] for x in rs]):.2f}; reading - truth {np.median(d):+.2f} "
                         f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]; frames where the truth is lower "
                         f"{np.mean([x['truth_lower'] for x in rs]):.0%}")
    (REP / "stageC.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "recon":
        cmd_recon()
    elif cmd == "smoke":
        cmd_smoke()
    elif cmd == "run":
        cmd_run()
    elif cmd == "report":
        report()
