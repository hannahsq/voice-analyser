"""Bandwidth evidence (D-050 amendment 4, probe 12): do the bandwidth heads measure bandwidth or lean on a prior?

Arm A is rendered and scored by the runner (`run.py` probe `bwhold`: formants moved, bandwidths held); here its
bandwidth slopes are divided by the conditional prior's own slope in probe 1's length arm (0 = measures, 1 = reads
B off F). Arm B splits probe 7's per-item bandwidth slopes into tertiles of the item's f0. Arm C regresses the model's
log2 B on the label's and the prior's on new synthetic cascade clips, where the label is the prior times an
independent log-uniform factor.

    python -m vocal_analysis.eval.probes.bw_evidence <ckpt> [...]   ($VA_DEVICE=cuda)
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import counterfactual as cf
from . import measures as M

REPORTS = Path("reports/probes/bw_evidence")
SYNTH_SEED, SYNTH_CLIPS = 5054, 150  # seed used by no other set


def reliance(name: str) -> dict:
    """Arm A: model bandwidth slope with bandwidths held, over the prior's slope in probe 1's length arm."""
    held = json.loads(Path(f"reports/probes/bwhold/{name}.json").read_text(encoding="utf-8"))
    raw = np.load(f"reports/probes/counterfactual/{name}_raw.npz", allow_pickle=True)
    cols = [0] + [1 + i for i, a in enumerate(a for a in cf.ALPHAS if a != 1.0)]
    x = np.log2([1.0] + [a for a in cf.ALPHAS if a != 1.0])
    out = {}
    for k in range(4):
        prior = float(np.nanmedian(M.item_slopes(raw["LB"][:, cols, k], x)))
        model = held[f"length_bwhold/all/B{k + 1}"]
        out[f"B{k + 1}"] = {"model_slope": model["point"], "model_low": model["low"], "model_high": model["high"],
                            "prior_slope": prior, "reliance": model["point"] / prior if prior else float("nan")}
    return out


def by_f0(name: str) -> dict:
    """Arm B: probe 7's per-item bandwidth slopes (against the labels' own change) by tertile of the item's f0."""
    raw = np.load(f"reports/probes/bandwidth/{name}_raw.npz", allow_pickle=True)
    f0 = raw["Y_f0"][:, 0]
    edges = np.nanpercentile(f0, [100 / 3, 200 / 3])
    tert = np.digitize(f0, edges)
    groups = raw["groups"]
    out = {"f0_edges_hz": (2.0 ** edges).tolist()}
    for k in range(4):
        y, lb = raw[f"Y_B{k + 1}"], raw["LB"][:, :, k]
        s = M.item_slopes(y, lb - lb[:, [0]])
        for t in range(3):
            ok = (tert == t) & np.isfinite(s)
            out[f"B{k + 1}/tertile{t}"] = list(M.group_bootstrap(s[ok], groups[ok])) + [int(ok.sum())]
    return out


def synthetic(ckpt: str) -> dict:
    """Arm C: per formant, least squares of model log2 B on label log2 B and prior log2 B (sonorant voiced frames of
    new cascade clips), with a clip bootstrap; and var(model log2 B) / var(label log2 B)."""
    import torch

    from ...models.infer import analyse, load_teacher
    from ...synth import tract
    from ...synth.sampler import SynthConfig, generate

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    cfg = dataclasses.replace(SynthConfig(), p_tube=0.0)
    rows = []  # (clip, k, model, label, prior)
    for i in range(SYNTH_CLIPS):
        clip = generate(i, cfg, SYNTH_SEED)
        lab = clip.labels
        r = analyse(model, clip.audio, dev)
        n = min(len(lab["f0"]), len(r["f0"]))
        m = (lab["sonorant"][:n] & lab["voiced"][:n]).astype(bool)
        if m.sum() < 5:
            continue
        f0c = float(np.median(lab["f0"][:n][m]))
        for k in range(4):
            fl, bl, bm = lab["formants"][:n, k][m], lab["bandwidths"][:n, k][m], r["bandwidths"][:n, k][m]
            pr = tract.bandwidth_prior(fl, f0c)
            ok = np.isfinite(fl) & np.isfinite(bl) & np.isfinite(bm) & (bm > 0)
            for a, b, c in zip(np.log2(bm[ok]), np.log2(bl[ok]), np.log2(pr[ok])):
                rows.append((i, k, a, b, c))
    rows = np.array(rows)
    out = {}
    rng = np.random.default_rng(0)
    clips = np.unique(rows[:, 0])
    for k in range(4):
        rk = rows[rows[:, 1] == k]

        def fit(rr):
            X = np.c_[rr[:, 3], rr[:, 4], np.ones(len(rr))]
            return np.linalg.lstsq(X, rr[:, 2], rcond=None)[0][:2]

        w = fit(rk)
        boots = []
        for _ in range(500):
            pick = rng.choice(clips, len(clips))
            rr = np.concatenate([rk[rk[:, 0] == c] for c in pick])
            boots.append(fit(rr))
        lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
        out[f"B{k + 1}"] = {"w_label": float(w[0]), "w_label_ci": [float(lo[0]), float(hi[0])],
                            "w_prior": float(w[1]), "w_prior_ci": [float(lo[1]), float(hi[1])],
                            "var_ratio": float(np.var(rk[:, 2]) / np.var(rk[:, 3])), "n_frames": int(len(rk))}
    return out


def report(name: str, res: dict) -> str:
    lines = [f"Bandwidth evidence (D-050, probe 12): {name}",
             "arm A: bandwidths held while formants move; model B slope against log2 alpha (ideal 0), the prior's own "
             "slope in probe 1's length arm, and reliance = model / prior (0 measures, 1 reads B off F)"]
    for k, v in res["A"].items():
        lines.append(f"  {k}  model {v['model_slope']:+.3f} [{v['model_low']:+.3f}, {v['model_high']:+.3f}]  "
                     f"prior {v['prior_slope']:+.3f}  reliance {v['reliance']:+.2f}")
    e = res["B"]["f0_edges_hz"]
    lines.append(f"arm B: probe 7's bandwidth slopes (ideal 1) by f0 tertile (edges {e[0]:.0f} / {e[1]:.0f} Hz): low / mid / high")
    for k in range(1, 5):
        parts = [f"{res['B'][f'B{k}/tertile{t}'][0]:+.2f} [{res['B'][f'B{k}/tertile{t}'][1]:+.2f}, "
                 f"{res['B'][f'B{k}/tertile{t}'][2]:+.2f}]" for t in range(3)]
        lines.append(f"  B{k}  " + "  ".join(parts))
    lines.append("arm C: synthetic cascade clips; model log2 B on label log2 B and prior log2 B (clip bootstrap), "
                 "and var(model) / var(label)")
    for k, v in res["C"].items():
        lines.append(f"  {k}  w_label {v['w_label']:+.2f} [{v['w_label_ci'][0]:+.2f}, {v['w_label_ci'][1]:+.2f}]  "
                     f"w_prior {v['w_prior']:+.2f} [{v['w_prior_ci'][0]:+.2f}, {v['w_prior_ci'][1]:+.2f}]  "
                     f"var ratio {v['var_ratio']:.2f}  ({v['n_frames']} frames)")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not argv:
        print(__doc__)
        return 2
    REPORTS.mkdir(parents=True, exist_ok=True)
    for ckpt in argv:
        name = Path(ckpt).parent.name
        res = {"A": reliance(name), "B": by_f0(name), "C": synthetic(ckpt)}
        (REPORTS / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        text = report(name, res)
        (REPORTS / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
