"""D-081 B: oracle-input heads on v3.1's frozen features (D-079's cache, `Teacher.enc`, 128-d). D-079's head form
(LayerNorm -> 256 -> GELU -> 256 -> GELU -> out) on h, or on h with an oracle quantity (standardised) appended; L1 on
standardised targets, 4000 steps, head seeds 1 and 2; trained on train_v2ko's first 3000 clips' sonorant frames,
scored on val_v2ko. Labels from `d081_labels.py`, aligned by (clip, frame) and checked against the cache.

    python scripts/analysis/d081_oracle_heads.py v31_s1 v31_s2   (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

F, C, O = Path("cache/d079"), Path("cache/d081"), Path("reports/analysis/d081")
STEPS, BATCH, LR = 4000, 8192, 1e-3
SEEDS = (1, 2)
LOG = lambda x: np.log(np.maximum(x, 1e-6))  # noqa: E731
# test -> (target, [(variant, oracle inputs)])
TESTS = {
    "B1": (lambda z: LOG(z["formants"][:, :1]), [("h", []), ("h+<a>", ["glottal_opening"]), ("h+<a>+df1", ["glottal_opening", "df1"])]),
    "B2": (lambda z: LOG(z["formants"][:, :1]), [("h+B1ac", ["log_b1_acoustic"])]),
    "B3": (lambda z: LOG(z["formants"][:, :4]), [("h", []), ("h+aperiodicity", ["jitter", "shimmer_db", "hnr_db"])]),
    "B4": (lambda z: LOG(z["bandwidths"][:, :4]), [("h", []), ("h+source", ["rd", "open_quotient"])]),
    "B5": (lambda z: LOG(z["length_cm"][:, None]), [("h", []), ("h+iVTL", ["length_inst_cm"])]),
}


def aligned(trunk, split):
    z = np.load(F / f"{trunk}_{split}.npz")
    lab = dict(np.load(C / f"{split}_labels.npz"))
    clip = z["clip"]
    frame = np.arange(len(clip)) - np.searchsorted(clip, clip)  # the cache keeps each clip's frames in order from 0
    start = np.searchsorted(lab["clip"], np.arange(lab["clip"].max() + 2))
    rows = start[clip] + frame
    lab = {k: v[rows] for k, v in lab.items()}
    if not np.allclose(lab["formants"], z["formants"], rtol=1e-4, equal_nan=True):
        raise SystemExit(f"{trunk} {split}: labels don't line up with the feature cache")
    lab["df1"] = lab["coupling_df"][:, 0]
    lab["log_b1_acoustic"] = LOG(lab["bandwidths_acoustic"][:, 0])
    return z["h"], lab


def fit(xt, yt, xv, seed, dev):
    import torch
    from torch import nn

    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    ym, ys = yt.mean(0), yt.std(0) + 1e-6
    Xt = torch.from_numpy(((xt - mu) / sd).astype(np.float32)).to(dev)
    Yt = torch.from_numpy(((yt - ym) / ys).astype(np.float32)).to(dev)
    Xv = torch.from_numpy(((xv - mu) / sd).astype(np.float32)).to(dev)
    torch.manual_seed(seed)
    d = xt.shape[1]
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, yt.shape[1])).to(dev)
    opt = torch.optim.AdamW(head.parameters(), LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.075)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(STEPS):
        i = torch.randint(len(Yt), (BATCH,), generator=g).to(dev)
        loss = (head(Xt[i]) - Yt[i]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.inference_mode():
        p = torch.cat([head(Xv[j:j + 65536]) for j in range(0, len(Xv), 65536)]).float().cpu().numpy()
    return p * ys + ym


if __name__ == "__main__":
    import torch

    from d081_label_oracle import Slope

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    O.mkdir(parents=True, exist_ok=True)
    for trunk in sys.argv[1:]:
        out = O / f"heads_{trunk}.json"
        if out.exists():
            continue
        (ht, lt), (hv, lv) = aligned(trunk, "train"), aligned(trunk, "val")
        ht, hv = ht.astype(np.float32), hv.astype(np.float32)
        res = {}
        for test, (target, variants) in TESTS.items():
            yt, yv = target(lt), target(lv)
            # one frame set per test: sonorant, target and every variant's oracle inputs finite (all variants alike)
            need = sorted({k for _, ks in variants for k in ks} | ({"glottal_opening"} if test in ("B1", "B2") else set()))

            def ok(lab, y):
                m = lab["sonorant"].astype(bool) & np.all(np.isfinite(y), 1)
                for k in need:
                    m &= np.isfinite(lab[k])
                return m

            mt, mv = ok(lt, yt), ok(lv, yv)
            vsub = {k: lv[k][mv] for k in ("formants", "formants_acoustic", "clip")}
            S = Slope(vsub) if test in ("B1", "B2") else None
            res[test] = {}
            for name, keys in variants:
                xt = np.column_stack([ht[mt]] + [lt[k][mt] for k in keys])
                xv = np.column_stack([hv[mv]] + [lv[k][mv] for k in keys])
                per = {}
                for s in SEEDS:
                    p = fit(xt, yt[mt], xv, s, dev)
                    if S is not None:
                        per[s] = S(np.exp(p[:, 0]))
                    else:
                        per[s] = float(np.mean(np.abs(np.exp(p) - np.exp(yv[mv]))))
                res[test][name] = {"by_seed": per, "mean": float(np.mean(list(per.values()))), "frames": int(mv.sum())}
                print(f"{trunk} {test} {name}: {res[test][name]['mean']:.4f} (seeds {', '.join(f'{x:.4f}' for x in per.values())})",
                      flush=True)
        out.write_text(json.dumps(res, indent=1))
