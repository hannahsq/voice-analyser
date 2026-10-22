"""D-033 part 4: train a ParakeetHead on cached Parakeet layers (cache_heads.py), so heads on different layer sets see
exactly the same crops. AdamW 1e-3, weight decay 0.01, 500 warm-up steps, cosine over STEPS, batch 32, seed 0.
Run from the main checkout: VA_DEVICE=cuda python <worktree>/scripts/parakeet/train_cached.py <cache dir> <out dir> <layers, e.g. 5 or 3,4,5,6> [steps]"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head import ParakeetHead, crop_mel_pos  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.train.losses import teacher_loss  # noqa: E402

CACHE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
want = tuple(int(x) for x in sys.argv[3].split(","))
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
BATCH, WARMUP, LR = 32, 500, 1e-3
OUT.mkdir(parents=True, exist_ok=True)
dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
torch.manual_seed(0)
rng = np.random.default_rng(0)
meta = json.loads((CACHE / "done").read_text())
cols = [meta["layers"].index(l) for l in want]
emb = np.load(CACHE / "emb.npy", mmap_mode="r")
N = meta["n"]
KEYS = ("f0", "voiced", "sonorant", "formants", "bandwidths", "w_f0", "w_formant")
labs = {k: np.load(CACHE / f"{k}.npy", mmap_mode="r") for k in KEYS}
head = ParakeetHead(1024 * len(want)).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
lr_at = lambda s: LR * (s + 1) / WARMUP if s < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP)))  # noqa: E731
pos = crop_mel_pos(meta["crop_frames"], dev)
log = open(OUT / "log.jsonl", "w")
print(f"head on layers {want}: {sum(p.numel() for p in head.parameters()):,} parameters; {STEPS} steps of {BATCH} from {N} cached crops", flush=True)
t0 = time.time()
for step in range(STEPS):
    idx = np.sort(rng.choice(N, BATCH, replace=False))
    feats = torch.from_numpy(np.asarray(emb[idx][:, :, cols], np.float16)).to(dev).float().flatten(2)  # (B, E, 1024 * k)
    lab = {k: torch.from_numpy(np.asarray(v[idx])).to(dev) for k, v in labs.items()}
    for k in ("voiced", "sonorant"):
        lab[k] = lab[k] > 0.5
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    loss, parts = teacher_loss(head(feats, pos), lab)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    if (step + 1) % 50 == 0:
        print(json.dumps({"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(gn),
                          **{k: float(v) for k, v in parts.items()}}), file=log, flush=True)
    if (step + 1) % 1000 == 0:
        print(f"step {step + 1}: loss {float(loss.detach()):.3f} ({time.time() - t0:.0f} s)", flush=True)
torch.save({"model": head.state_dict(), "step": STEPS, "layers": want, "in_dim": 1024 * len(want)}, OUT / "ckpt_last.pt")
print("done", flush=True)
