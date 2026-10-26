"""D-035: a per-frame anatomical vocal-tract-length head on frozen features (pre-registered). LayerNorm -> Linear(in, 256)
-> GELU -> Linear(256, 256) -> GELU -> Linear(256, 2): mean and log variance of log2 length (cm); Gaussian NLL on
sonorant frames. AdamW 1e-3, weight decay 0.01, 300 warm-up steps, cosine over 4000 steps, batch 32 crops (all their
sonorant frames). Crops: cache_heads.py's, labelled by vtl_cache_labels.py (length not drawn from f0).
Arms: T = teacher v3's encoder (vtl_cache.py train_T, 10 ms frames; tube crops); P = Parakeet layer 5 (cache_heads.py,
80 ms frames, encoder frame e read against label frame 8e - 4; tube crops); Tc = T on tube and cascade crops.
Args: arm, seed, out dir. Env: VA_HEADS (head cache), VA_VTL (vtl cache), VA_DEVICE (default cpu)."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HEADS = Path(os.environ.get("VA_HEADS", "cache/heads_l3_6"))
VTL = Path(os.environ.get("VA_VTL", "cache/vtl"))
STEPS, WARM, BATCH = int(os.environ.get("VA_STEPS", 4000)), 300, 32  # VA_STEPS: code checks only


class Head(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                 nn.Linear(256, 2))

    def forward(self, x):  # (..., in) -> mean, log variance of log2 cm
        o = self.net(x)
        return o[..., 0], o[..., 1]


def p_frames(n_enc):
    """Encoder frames e >= 1 and their label frames 8e - 4 (crop mel position j + 4)."""
    e = np.arange(1, n_enc)
    j = 8 * e - 4
    return e[j < 400], j[j < 400]


def load_arm(arm):
    """-> list of (frames (n, in) fp16, log2 length) per crop, sonorant frames only."""
    v = np.load(HEADS / "vtl.npz")
    son = np.load(HEADS / "sonorant.npy", mmap_mode="r")
    keep = np.isfinite(v["length_cm"]) & ~v["from_f0"] & (v["tube"] | (arm == "Tc"))
    crops = []
    if arm == "P":
        emb = np.load(HEADS / "emb.npy", mmap_mode="r")
        e, j = p_frames(emb.shape[1])
        for i in np.flatnonzero(keep):
            s = np.asarray(son[i])[j] > 0.5
            if s.sum():
                crops.append((np.asarray(emb[i, e[s], 2]), np.log2(v["length_cm"][i])))
    else:
        idx = np.load(VTL / "train_T" / "idx.npy")
        feats = np.load(VTL / "train_T" / "feats.npy", mmap_mode="r")
        pos = {int(i): k for k, i in enumerate(idx)}
        for i in np.flatnonzero(keep):
            s = np.asarray(son[i]) > 0.5
            if s.sum() and int(i) in pos:  # pos covers every labelled crop (only a code check's subset lacks some)
                crops.append((np.asarray(feats[pos[int(i)]][s]), np.log2(v["length_cm"][i])))
    return crops


def nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv)).mean()


if __name__ == "__main__":
    arm, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "head.pt").exists():
        print(f"{out}: done")
        sys.exit()
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    crops = load_arm(arm)
    in_dim = crops[0][0].shape[1]
    head = Head(in_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * (s - WARM) / (STEPS - WARM))))
    t0, run = time.time(), []
    for step in range(STEPS):
        pick = rng.integers(len(crops), size=BATCH)
        x = torch.from_numpy(np.concatenate([crops[k][0] for k in pick])).float().to(dev)
        y = torch.from_numpy(np.concatenate([np.full(len(crops[k][0]), crops[k][1], np.float32) for k in pick])).to(dev)
        mu, lv = head(x)
        loss = nll(mu, lv, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"{time.strftime('%H:%M:%S')} {arm} s{seed} step {step + 1}: NLL {np.mean(run[-500:]):.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    torch.save({"state": head.state_dict(), "in_dim": in_dim, "arm": arm, "seed": seed, "n_crops": len(crops)}, out / "head.pt")
    (out / "done").write_text(json.dumps({"arm": arm, "seed": seed, "steps": STEPS, "n_crops": len(crops),
                                          "final_nll": float(np.mean(run[-200:]))}))
