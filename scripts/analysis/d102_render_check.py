"""D-102: where does the reconstruction loss put F1 when every other parameter is at its label?

Clips regenerated clean from train_v2's validation split (uncoupled) and train_v2ku's held-out split (coupled tube,
strength drawn). Targets: harmonic_targets with the labelled f0 and voicing. Every 5th voiced sonorant frame: F1 over
the closed label x 2^(c / 1200), c = -400..+400 cents in 5-cent steps, F2-F8 and bandwidths at their labels (acoustic
ones on coupled clips), no nuisance sections; the grid's minimum is the recovered F1. CPU.

    python scripts/analysis/d102_render_check.py [n_clips] > reports/analysis/d102/d102.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRID = np.arange(-400, 401, 5)
BANDS = ((0, 150), (150, 250), (250, 400), (400, 1e9))
PROFILES = ("tilt", "offset", "quadratic")


def load_cfg(root):
    from vocal_analysis.synth.sampler import SynthConfig

    idx = json.loads((Path(root) / "index.json").read_text())
    cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in idx["synth_config"].items()
                         if k in SynthConfig.__dataclass_fields__})
    return idx, cfg


def clip_rows(clip, coupled):
    import torch

    from vocal_analysis.train import reconstruction as R

    lab = clip.labels
    T = len(lab["f0"])
    db, ok = R.harmonic_targets(clip.audio, lab["f0"], lab["voiced"].astype(bool), k_max=64)
    fr = np.flatnonzero(lab["voiced"].astype(bool) & lab["sonorant"].astype(bool) & np.isfinite(lab["formants"][:, 0])
                        & (ok.sum(1) >= 4))[::5]
    fr = fr[fr < len(db)]
    if not len(fr):
        return []
    Fc = lab["formants"][fr]
    Fa = lab.get("formants_acoustic", lab["formants"])[fr] if coupled else Fc
    Ba = lab.get("bandwidths_acoustic", lab["bandwidths"])[fr] if coupled else lab["bandwidths"][fr]
    C, N = len(GRID), len(fr)
    poles = np.repeat(Fa[None], C, 0).astype(np.float64)
    poles[:, :, 0] = Fc[None, :, 0] * 2 ** (GRID[:, None] / 1200)
    bws = np.repeat(Ba[None], C, 0).astype(np.float64)
    tgt = torch.tensor(np.repeat(db[fr][None], C, 0), dtype=torch.float64)
    val = torch.tensor(np.repeat(ok[fr][None], C, 0))
    f0 = torch.tensor(np.repeat(lab["f0"][fr][None], C, 0), dtype=torch.float64)
    out = {}
    for prof in PROFILES:
        _, per = R.reconstruction_loss(tgt, val, f0, torch.tensor(poles), torch.tensor(bws), profile=prof)
        per = per.numpy()
        j = np.nanargmin(np.where(np.isfinite(per), per, np.inf), 0)
        out[prof] = (np.log2(poles[j, np.arange(N), 0]), (j == 0) | (j == C - 1))
    meta = clip.meta
    kind = "tube" if meta.get("tube") else ("cascade+extras" if (meta.get("n_extra_poles") or 0) > 0 else "cascade")
    rows = []
    for n in range(N):
        rows.append({"f0": float(lab["f0"][fr[n]]), "kind": kind, "coupled": coupled,
                     "closed": float(np.log2(Fc[n, 0])), "acoustic": float(np.log2(Fa[n, 0])),
                     **{f"rec_{p}": float(out[p][0][n]) for p in PROFILES}, "edge": bool(out["tilt"][1][n])})
    return rows


if __name__ == "__main__":
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.train.data import split_ids

    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = []
    for root, coupled in (("data/synth/train_v2", False), ("data/synth/train_v2ku", True)):
        idx, cfg = load_cfg(root)
        for i in split_ids(idx["n_clips"])[1][:n_clips]:
            rows += clip_rows(generate(int(i), cfg, idx["seed"]), coupled)
        print(f"{root}: {sum(r['coupled'] == coupled for r in rows)} frames", file=sys.stderr, flush=True)

    def med(v):
        return f"{np.median(v):+6.1f}" if len(v) else "   n/a"

    L = ["D-102: the reconstruction loss's F1 minimum with every other parameter at its label (cents; median over frames)", ""]
    for coupled in (False, True):
        sub = [r for r in rows if r["coupled"] == coupled]
        L.append(("coupled tube clips (train_v2ku held-out): recovered - closed | - acoustic; acoustic - closed"
                  if coupled else "uncoupled clips (train_v2 validation): recovered - label") + "; profile tilt unless named")
        kinds = sorted({r["kind"] for r in sub})
        for kd in ["all"] + kinds:
            for b, (lo, hi) in enumerate(BANDS):
                s = [r for r in sub if (kd == "all" or r["kind"] == kd) and lo <= r["f0"] < hi]
                if len(s) < 20:
                    continue
                rc = 1200 * np.array([r["rec_tilt"] - r["closed"] for r in s])
                ra = 1200 * np.array([r["rec_tilt"] - r["acoustic"] for r in s])
                ac = 1200 * np.array([r["acoustic"] - r["closed"] for r in s])
                ro = 1200 * np.array([r["rec_offset"] - r["closed"] for r in s])
                rq = 1200 * np.array([r["rec_quadratic"] - r["closed"] for r in s])
                edge = np.mean([r["edge"] for r in s])
                band = f"f0 {lo:.0f}-{hi if hi < 1e8 else 'up'}"
                if coupled:
                    L.append(f"  {kd:15s} {band:12s} n {len(s):5d}: - closed {med(rc)} | - acoustic {med(ra)}; acoustic - closed {med(ac)}; "
                             f"|.| closed {np.median(np.abs(rc)):5.1f} acoustic {np.median(np.abs(ra)):5.1f}; edge {100 * edge:4.1f} %")
                else:
                    L.append(f"  {kd:15s} {band:12s} n {len(s):5d}: {med(rc)} (|.| {np.median(np.abs(rc)):5.1f}); offset {med(ro)}, "
                             f"quadratic {med(rq)}; edge {100 * edge:4.1f} %")
        L.append("")
    unc = [r for r in rows if not r["coupled"]]
    ok_bands = []
    for lo, hi in BANDS:
        s = [r for r in unc if lo <= r["f0"] < hi]
        if len(s) >= 50:
            ok_bands.append((f"{lo:.0f}-{hi if hi < 1e8 else 'up'}", abs(np.median([1200 * (r["rec_tilt"] - r["closed"]) for r in s]))))
    unbiased = all(v <= 10 for _, v in ok_bands)
    cp = [r for r in rows if r["coupled"]]
    nearer = np.median([abs(r["rec_tilt"] - r["acoustic"]) for r in cp]) < np.median([abs(r["rec_tilt"] - r["closed"]) for r in cp]) if cp else None
    L.append("READING: uncoupled " + ("unbiased (every band within 10 c)" if unbiased else "biased: " + ", ".join(f"{b} {v:.1f} c" for b, v in ok_bands if v > 10))
             + f"; coupled: nearer the {'acoustic' if nearer else 'closed'} label")
    print("\n".join(L))
