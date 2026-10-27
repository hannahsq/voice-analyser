"""D-039 (exploratory, reported only): VTL heads over teacher (T, 128-d, 10 ms) and Parakeet layer-5 (P, 1024-d, 80 ms)
features together, on D-035's caches. Frames at 10 ms; each teacher frame t takes the Parakeet frame e covering it
(crops: e = round((t + 4) / 8), D-035's 8e - 4 convention, e >= 1).
Arms: E (early fusion: per-source LayerNorm, T || P || two presence indicators -> 256 -> GELU -> 256 -> GELU -> 2);
ED (E with source dropout per crop: 50 % both, 25 % T only, 25 % P only; a dropped source is 0 after its LayerNorm,
its indicator 0, no rescaling); F (D-035's T and P heads of the same seed, frozen, without their output layers: their
256-d activations || indicators -> 256 -> GELU -> 2); FD (F with ED's source dropout on the activations).
D-035's recipe: AdamW 1e-3, wd 0.01, 300 warm-up steps, cosine over 4,000 steps, batch 32 crops (their sonorant
frames), Gaussian NLL on log2 cm. Args: arm, seed, out dir. Env: VA_DEVICE (default cuda), VA_STEPS (code checks)."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parakeet"))
from vtl_head import Head, nll  # noqa: E402

STEPS, WARM, BATCH = int(os.environ.get("VA_STEPS", 4000)), 300, 32
P_LAYER = 2  # cache_heads.py stores layers 3-6; index 2 = layer 5
VTL, HEADS = Path("cache/vtl"), Path("cache/heads_l3_6")


def load_d035(arm, seed):
    ck = torch.load(VTL / "heads" / f"{arm}_s{seed}" / "head.pt", map_location="cpu", weights_only=False)
    h = Head(ck["in_dim"])
    h.load_state_dict(ck["state"])
    return h.eval()


class Early(nn.Module):
    def __init__(self):
        super().__init__()
        self.nt, self.np_ = nn.LayerNorm(128), nn.LayerNorm(1024)
        self.net = nn.Sequential(nn.Linear(128 + 1024 + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, t, p, keep):  # keep (n, 2) in {0, 1}: T present, P present
        o = self.net(torch.cat([self.nt(t) * keep[:, :1], self.np_(p) * keep[:, 1:], keep], -1))
        return o[..., 0], o[..., 1]


class Penultimate(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.bt, self.bp = load_d035("T", seed).net[:5], load_d035("P", seed).net[:5]  # LN -> Lin -> GELU -> Lin -> GELU
        for p in list(self.bt.parameters()) + list(self.bp.parameters()):
            p.requires_grad_(False)
        self.net = nn.Sequential(nn.Linear(256 + 256 + 2, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, t, p, keep):
        with torch.no_grad():
            a, b = self.bt(t), self.bp(p)
        o = self.net(torch.cat([a * keep[:, :1], b * keep[:, 1:], keep], -1))
        return o[..., 0], o[..., 1]


def training_crops():
    """-> list of (T frames (n, 128) fp16, P frames (n, 1024) fp16, log2 length), sonorant teacher frames, tube crops
    with length not drawn from f0 (D-035's T arm's crops)"""
    v = np.load(HEADS / "vtl.npz")
    son = np.load(HEADS / "sonorant.npy", mmap_mode="r")
    emb = np.load(HEADS / "emb.npy", mmap_mode="r")
    idx = np.load(VTL / "train_T" / "idx.npy")
    feats = np.load(VTL / "train_T" / "feats.npy", mmap_mode="r")
    pos = {int(i): k for k, i in enumerate(idx)}
    keep = np.isfinite(v["length_cm"]) & ~v["from_f0"] & v["tube"]
    n_enc = emb.shape[1]
    crops = []
    for i in np.flatnonzero(keep):
        if int(i) not in pos:
            continue
        t = np.flatnonzero(np.asarray(son[i]) > 0.5)
        e = np.clip(np.round((t + 4) / 8).astype(int), 1, n_enc - 1)
        if len(t):
            crops.append((np.asarray(feats[pos[int(i)]][t]), np.asarray(emb[i, :, P_LAYER])[e], float(np.log2(v["length_cm"][i]))))
    return crops


def build(arm, seed):
    return Early() if arm in ("E", "ED") else Penultimate(seed)


def main():
    arm, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    assert arm in ("E", "ED", "F", "FD"), arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done").exists():
        print(f"{out}: done")
        return
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    crops = training_crops()
    model = build(arm, seed).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARM) / max(1, STEPS - WARM)))))
    drop = arm in ("ED", "FD")
    t0, run = time.time(), []
    for step in range(STEPS):
        pick = rng.integers(len(crops), size=BATCH)
        ts, ps, ys, ks = [], [], [], []
        for k in pick:
            t, p, y = crops[k]
            u = rng.random() if drop else 0.0
            kp = (1.0, 1.0) if u < 0.5 else ((1.0, 0.0) if u < 0.75 else (0.0, 1.0))
            ts.append(t), ps.append(p), ys.append(np.full(len(t), y, np.float32)), ks.append(np.tile(kp, (len(t), 1)))
        tt = torch.from_numpy(np.concatenate(ts)).float().to(dev)
        pp = torch.from_numpy(np.concatenate(ps)).float().to(dev)
        yy = torch.from_numpy(np.concatenate(ys)).to(dev)
        kk = torch.from_numpy(np.concatenate(ks).astype(np.float32)).to(dev)
        loss = nll(*model(tt, pp, kk), yy)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"{time.strftime('%H:%M:%S')} {arm} s{seed} step {step + 1}: NLL {np.mean(run[-500:]):.4f} ({time.time() - t0:.0f} s)", flush=True)
    state = {k: v for k, v in model.state_dict().items()}
    torch.save({"state": state, "arm": arm, "seed": seed, "n_crops": len(crops)}, out / "model.pt")
    (out / "done").write_text(json.dumps({"arm": arm, "seed": seed, "steps": STEPS, "n_crops": len(crops),
                                          "final_nll": float(np.mean(run[-200:]))}))


if __name__ == "__main__":
    main()
