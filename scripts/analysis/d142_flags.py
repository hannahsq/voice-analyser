"""D-142: back-end flags for suspicious formant readings, tested on clips and takes that drop a formant.

A: spectrum consistency (the model's own F1-F4 and bandwidths as a cascade envelope against the harmonics, minus what an
LPC-14 envelope leaves); B: recording-level plausibility (kNN distance to train_v2's tube frames at the recording's
length); C: physical prior (the same at the frame's own length).

    python scripts/analysis/d142_flags.py synth               -> reports/analysis/d142/d142_synth.json
    python scripts/analysis/d142_flags.py private REC OUT FILES -> OUT/d142_private.json   (outside the repo)
    python scripts/analysis/d142_flags.py report [OUT]        -> reports/analysis/d142/d142.txt (aggregates; per-take to OUT)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d142"
FS = 16000
MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")  # deciding, reported
SETS = {"O-like": (450.0, 800.0, 2800.0, 3500.0), "6-like": (560.0, 1000.0, 2850.0, 3500.0)}
F0S = (110.0, 140.0, 180.0, 220.0)
B1S = (150.0, 250.0, 360.0, 500.0, 700.0)
SEEDS = (0, 1, 2)
K = 10
B = 2000


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Prior:
    """kNN over train_v2's tube frames: (log2 F1-F4, log2 anatomical length), standardised."""

    def __init__(self):
        from scipy.spatial import cKDTree

        d = ROOT / "data/synth/train_v2/flat"
        fo = np.load(d / "frame_offsets.npy")
        tube = np.repeat(np.load(d / "clip_tube.npy"), np.diff(fo))
        idx = np.arange(0, len(tube), 50)
        son = np.asarray(np.load(d / "label_sonorant.npy", mmap_mode="r")[idx]) > 0.5
        F = np.asarray(np.load(d / "label_formants.npy", mmap_mode="r")[idx, :4])
        L = np.asarray(np.load(d / "label_length_cm.npy", mmap_mode="r")[idx])
        ok = tube[idx] & son & np.all(np.isfinite(F), 1) & np.isfinite(L)
        X = np.log2(np.column_stack([F[ok], L[ok]]))
        self.mu, self.sd = X.mean(0), X.std(0)
        self.tree = cKDTree((X - self.mu) / self.sd)
        self.n = len(X)

    def score(self, F, L):
        X = np.log2(np.column_stack([F, L]))
        ok = np.all(np.isfinite(X), 1)
        out = np.full(len(X), np.nan)
        if ok.any():
            out[ok] = self.tree.query((X[ok] - self.mu) / self.sd, k=K)[0][:, -1]
        return out


def _lpc_env(fr, hf):
    from scipy.linalg import solve_toeplitz

    w = fr * np.hanning(len(fr))
    ac = np.array([w[:len(w) - k] @ w[k:] for k in range(15)])
    a = np.r_[1.0, -solve_toeplitz(ac[:14], ac[1:15])]
    z = np.exp(-2j * np.pi * np.outer(hf, np.arange(15)) / FS)
    return -20 * np.log10(np.abs(z @ a))


def _fit_rms(h, env, hf):
    X = np.column_stack([np.ones(len(hf)), np.log2(hf)])
    r = h - env
    c = np.linalg.lstsq(X, r, rcond=None)[0]
    return float(np.sqrt(np.mean((r - X @ c) ** 2)))


def frame_scores(y, res, prior, L_rec):
    """Per voiced frame of the middle 60 %: A, B, C; and the frame's F1-F4 (for the failure share)."""
    from vocal_analysis.synth import filters

    n = len(res["voiced_p"])
    rows = []
    for i in range(int(0.2 * n), int(0.8 * n)):
        f0 = res["f0"][i]
        if res["voiced_p"][i] <= 0.5 or not np.isfinite(f0) or i * 160 < 512 or i * 160 + 512 > len(y):
            continue
        fr = y[i * 160 - 512:i * 160 + 512]
        s = 20 * np.log10(np.abs(np.fft.rfft(fr * np.hanning(1024), 4096)) + 1e-9)
        f = np.fft.rfftfreq(4096, 1 / FS)
        hf = np.arange(1, int(4000 / f0) + 1) * f0
        h = np.array([s[(f >= x - f0 / 4) & (f <= x + f0 / 4)].max() for x in hf])
        Fm, Bm = res["formants"][i, :4], res["bandwidths"][i, :4]
        env = 20 * np.log10(np.abs(filters.response(hf, Fm[None, :], Bm[None, :], FS)))
        a = _fit_rms(h, env, hf) - _fit_rms(h, _lpc_env(fr, hf), hf)
        rows.append([a, *Fm, 2 ** res["vtl_log2cm"][i]])
    R = np.array(rows)
    if not len(R):
        return None
    Bs = prior.score(R[:, 1:5], np.full(len(R), L_rec))
    Cs = prior.score(R[:, 1:5], R[:, 5])
    return {"A": float(np.nanmedian(R[:, 0])), "B": float(np.nanmedian(Bs)), "C": float(np.nanmedian(Cs)), "F": R[:, 1:5]}


def _fail(Fr, true):
    return float(np.mean(np.abs(np.log(Fr[:, 1] / true[2])) < np.abs(np.log(Fr[:, 1] / true[1]))))


def _grid_audio(Fs, f0, b1, seed):
    from vocal_analysis.synth import sampler, tract, voice

    C = int(3.0 * FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / FS
    pf = np.full((C, tract.N_POLES), np.nan)
    pb = pf.copy()
    pf[:, :4], pb[:, :4] = Fs, (b1, 90, 120, 150)
    rng = np.random.default_rng(int(1420 + 1000 * seed + f0 + b1 + Fs[0]))
    tr = voice.Trajectories(f0=f0 * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12), rd=np.full(C, 1.0),
                            voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1), whisper_amp=np.zeros(C),
                            fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
    return 0.5 * y / np.abs(y).max()


def _score_group(items, models, dev, prior):
    """items: (key, audio, true F1-F4); one 'recording'. Leave-one-out recording length per model."""
    from vocal_analysis.models.infer import analyse

    out = {}
    for m, mod in models.items():
        res = [analyse(mod, y.astype(np.float32), dev) for _, y, _ in items]
        Lc = np.array([np.nanmedian(2 ** r["vtl_log2cm"][int(0.2 * len(r["voiced_p"])):int(0.8 * len(r["voiced_p"]))]) for r in res])
        for j, ((key, y, true), r) in enumerate(zip(items, res)):
            Lrec = float(np.median(np.delete(Lc, j)))
            s = frame_scores(y, r, prior, Lrec)
            if s is None:
                continue
            out.setdefault(key, {})[m] = {"A": s["A"], "B": s["B"], "C": s["C"], "fail": _fail(s["F"], true)}
    return out


def _models():
    import torch

    from vocal_analysis.models.infer import load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    return dev, {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}


def cmd_synth() -> None:
    from vocal_analysis.synth import tube

    d131 = _mod("d131", "scripts/analysis/d131_context.py")
    prior = Prior()
    dev, models = _models()
    res = {}
    for sname, Fs in SETS.items():
        items = [(f"{sname}|f{f0:g}|b{b1:g}|s{s}", _grid_audio(Fs, f0, b1, s), np.array(Fs))
                 for f0 in F0S for b1 in B1S for s in SEEDS]
        res.update(_score_group(items, models, dev, prior))
        print(sname, flush=True)
    items = []
    for v, q in tube.STORY_VOWELS.items():
        f, bw = tube.poles(tube.build_shape(*q, 15.6), f_max_hz=7600.0)
        k = min(d131.d130.N_P, len(f))
        F = np.full((2, d131.d130.N_P), np.nan)
        Bw = F.copy()
        F[:, :k], Bw[:, :k] = f[:k], bw[:k]
        for f0 in (110.0, 220.0):
            tag = "".join(c + "^" if c.isupper() else c for c in v)
            y, _, _ = d131.render_jit(F, Bw, np.array([0.0, 2.0]), f0, 1420 + int(f0) + len(items))
            items.append((f"story|{tag}|f{f0:g}", y.astype(np.float64), f[:4]))
    res.update(_score_group(items, models, dev, prior))
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d142_synth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("prior frames", prior.n)


def cmd_private(rec: Path, out: Path, files: list[str]) -> None:
    d138 = _mod("d138", "scripts/analysis/d138_real_balance.py")
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    prior = Prior()
    dev, models = _models()
    groups = {}
    for name, r, y in d138._load(rec, files):
        if name in lv:
            groups.setdefault(name.split("_")[0], []).append((name, y, np.array(lv[name]["F"])))
    res = {}
    for g, items in groups.items():
        res.update(_score_group(items, models, dev, prior))
        print("recording", g, flush=True)
    (out / "d142_private.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def cmd_report(out: Path | None) -> None:
    rng = np.random.default_rng(142)
    syn = json.loads((REP / "d142_synth.json").read_text(encoding="utf-8"))
    prv = json.loads((out / "d142_private.json").read_text(encoding="utf-8")) if out else None
    lines = ["D-142: flags A (spectrum consistency), B (recording-level plausibility), C (physical prior); AUC failing vs "
             "not failing [95 % bootstrap]; false flags = share above the 95th percentile of not-failing grid clips", ""]
    priv_lines = ["D-142 private per-take (not for the repo): take, failure share, A, B, C (deciding model)"]
    verdict = {}
    for m in MODELS:
        lines.append(f"{m}{' (deciding)' if m == MODELS[0] else ''}:")
        grid = {k: v[m] for k, v in syn.items() if not k.startswith("story") and m in v}
        story = {k: v[m] for k, v in syn.items() if k.startswith("story") and m in v}
        nf = [k for k, v in grid.items() if v["fail"] >= 0.5]
        ok = [k for k, v in grid.items() if v["fail"] < 0.2]
        lines.append(f"  synthetic grid: {len(nf)} failing, {len(ok)} not; Story's vowels: {sum(v['fail'] >= 0.5 for v in story.values())} of {len(story)} failing")
        pf = po = P = None
        if prv:
            P = {k: v[m] for k, v in prv.items() if m in v}
            pf = [k for k, v in P.items() if v["fail"] >= 0.5]
            po = [k for k, v in P.items() if v["fail"] < 0.2]
            lines.append(f"  private takes: {len(pf)} failing, {len(po)} not")
        for fl in "ABC":
            thr = np.percentile([grid[k][fl] for k in ok], 95)
            parts = []
            res_ = {}
            for dom, D, pos, neg in (("synthetic", grid, nf, ok), ("private", P if prv else None, pf, po)):
                if D is None or not pos or not neg:
                    continue
                a = _auc([D[k][fl] for k in pos], [D[k][fl] for k in neg])
                bs = []
                for _ in range(B):
                    bp = rng.choice(pos, len(pos))
                    bn = rng.choice(neg, len(neg))
                    bs.append(_auc([D[k][fl] for k in bp], [D[k][fl] for k in bn]))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                res_[dom] = (a, lo)
                parts.append(f"{dom} AUC {a:.2f} [{lo:.2f}, {hi:.2f}]")
            ff = f"false flags: Story {np.mean([v[fl] > thr for v in story.values()]):.0%}"
            if prv:
                ff += f", private not-failing takes {np.mean([P[k][fl] > thr for k in po]):.0%}"
            acc = ("synthetic" in res_ and "private" in res_ and res_["synthetic"][0] >= 0.8 and res_["private"][0] >= 0.8
                   and res_["private"][1] > 0.5)
            if m == MODELS[0]:
                verdict[fl] = acc
            lines.append(f"  {fl}: " + "; ".join(parts) + f"; {ff}" + (f" -> {'ACCEPTED' if acc else 'not accepted'}" if m == MODELS[0] else ""))
        if prv and m == MODELS[0]:
            for k in sorted(P, key=lambda k: -P[k]["fail"]):
                v = P[k]
                priv_lines.append(f"  {k:6s} fail {v['fail']:4.0%}  A {v['A']:+5.2f}  B {v['B']:.2f}  C {v['C']:.2f}")
        lines.append("")
    lines.append("Accepted (deciding model): " + (", ".join(f for f, a in verdict.items() if a) or "none"))
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d142.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if out:
        (out / "d142_private.txt").write_text("\n".join(priv_lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "synth":
        cmd_synth()
    elif cmd == "private":
        cmd_private(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    elif cmd == "report":
        cmd_report(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
