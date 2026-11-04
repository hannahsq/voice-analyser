"""D-079: does a frozen trunk carry the frame's mean glottal opening <a>? Per trunk: the encoder's per-frame features
(`Teacher.enc`, 128-d) on whole clips of train_v2ko (the first N_TRAIN) and val_v2ko (all), cached; then an opening
head of the VTL head's form (LayerNorm -> 256 -> GELU -> 256 -> GELU -> 1, sigmoid) fitted on the training frames,
head seeds 1 and 2, and scored on held-out val_v2ko voiced frames: R² of <a>, per frame and per clip (clip means).
A ridge fit is reported beside it.

    python scripts/analysis/d079_probe.py features <name>=<ckpt> ...   (GPU; VA_DEVICE=cuda)
    python scripts/analysis/d079_probe.py fit <name> ...               (GPU or CPU)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

C = Path("cache/d079")
SETS = {"train": (Path("data/synth/train_v2ko/flat"), 3000), "val": (Path("data/synth/val_v2ko/flat"), None)}
SR, HOP = 16000, 160
STEPS, BATCH, LR = 4000, 8192, 1e-3
SEEDS = (1, 2)


def features(name, ck):
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ck, dev)
    vq = TorchVQT(dev)
    got = {}
    model.enc.register_forward_hook(lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    for split, (flat, n) in SETS.items():
        out = C / f"{name}_{split}.npz"
        if out.exists():
            continue
        a_off, f_off = np.load(flat / "audio_offsets.npy"), np.load(flat / "frame_offsets.npy")
        audio = np.load(flat / "audio.npy", mmap_mode="r")
        lab = {k: np.load(flat / f"label_{k}.npy", mmap_mode="r") for k in ("glottal_opening", "sonorant", "formants", "rd")}
        n_clips = (len(a_off) - 1) if n is None else n
        H, Y, clip = [], {k: [] for k in lab}, []
        with torch.inference_mode():
            for c in range(n_clips):
                x = audio[a_off[c]:a_off[c + 1]].astype(np.float32) / 32768.0
                t = int(f_off[c + 1] - f_off[c])
                pad = (-len(x)) % SR  # whole seconds (MIOpen fast find: one shape per length)
                xx = torch.from_numpy(np.pad(x, (0, pad)))[None].to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    model(vq.log_power_with_context(xx, (len(x) + pad) // HOP))
                t = min(t, got["h"].shape[1])  # the encoder can return a few frames fewer than the clip's labels
                H.append(got["h"][0, :t].float().cpu().numpy().astype(np.float16))
                for k in lab:
                    Y[k].append(np.asarray(lab[k][f_off[c]:f_off[c] + t]))
                clip.append(np.full(t, c))
        C.mkdir(parents=True, exist_ok=True)
        np.savez(out, h=np.concatenate(H), clip=np.concatenate(clip), **{k: np.concatenate(v) for k, v in Y.items()})
        print(f"{name} {split}: {n_clips} clips", flush=True)


def r2(y, p):
    return 1 - np.var(y - p) / np.var(y)


def fit(name):
    import torch
    from torch import nn

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    tr, va = (np.load(C / f"{name}_{s}.npz") for s in ("train", "val"))

    def frames(z):
        m = z["sonorant"].astype(bool) & np.isfinite(z["glottal_opening"])
        return z["h"][m].astype(np.float32), z["glottal_opening"][m].astype(np.float32), z["clip"][m]

    Xt, yt, _ = frames(tr)
    Xv, yv, cv = frames(va)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    # ridge on standardised features
    A = (Xt - mu) / sd
    w = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ (yt - yt.mean()))
    p_ridge = ((Xv - mu) / sd) @ w + yt.mean()
    res = {"ridge": r2(yv, p_ridge)}
    clips = np.unique(cv)
    cm = lambda v: np.array([v[cv == c].mean() for c in clips])  # noqa: E731
    Xt_t, yt_t = torch.from_numpy(Xt).to(dev), torch.from_numpy(yt).to(dev)
    Xv_t = torch.from_numpy(Xv).to(dev)
    for s in SEEDS:
        torch.manual_seed(s)
        head = nn.Sequential(nn.LayerNorm(Xt.shape[1]), nn.Linear(Xt.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                             nn.Linear(256, 1)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), LR)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.075)
        g = torch.Generator(device="cpu").manual_seed(s)
        for _ in range(STEPS):
            i = torch.randint(len(yt), (BATCH,), generator=g).to(dev)
            loss = (torch.sigmoid(head(Xt_t[i])[:, 0]) - yt_t[i]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.inference_mode():
            p = torch.cat([torch.sigmoid(head(Xv_t[j:j + 65536])[:, 0]) for j in range(0, len(Xv_t), 65536)]).cpu().numpy()
        res[f"mlp_s{s}"] = r2(yv, p)
        res[f"mlp_s{s}_clip"] = r2(cm(yv), cm(p))
    res["mlp"] = float(np.mean([res[f"mlp_s{s}"] for s in SEEDS]))
    res["mlp_clip"] = float(np.mean([res[f"mlp_s{s}_clip"] for s in SEEDS]))
    res["n_train_frames"], res["n_val_frames"] = len(yt), len(yv)
    (C / f"{name}_fit.json").write_text(json.dumps({k: float(v) for k, v in res.items()}, indent=1))
    print(f"{name}: R² frame {res['mlp']:.3f} (seeds {res['mlp_s1']:.3f}, {res['mlp_s2']:.3f}), clip {res['mlp_clip']:.3f}, "
          f"ridge {res['ridge']:.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "features":
        for a in sys.argv[2:]:
            features(*a.split("=", 1))
    else:
        for n in sys.argv[2:]:
            fit(n)
