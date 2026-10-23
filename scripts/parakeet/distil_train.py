"""D-034 stage 1: distil Parakeet layer 5 into the student (pre-registered). Loss L1 + (1 - cosine) per Parakeet frame
on standardised targets, the student's 10 ms frames averaged in eights (frames 8e - 8 .. 8e - 1 for encoder frame e);
AdamW lr 1e-3, weight decay 0.01, warm-up 1000 then cosine to 0, 12,000 steps of 32 crops, gradient norm clipped at 1.0
(as our training loop), bf16 autocast, seed 0. The initial weights are saved first (ckpt_init.pt: control R).
Fidelity (mean per-dimension R^2 and cosine, standardised targets) on the held-out crops every 1,000 steps (EVERY); checkpoints
at the same steps; resumes from ckpt_last.pt. Args: cache dir (distil_cache.py), out dir. Run from the main checkout."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import Student  # noqa: E402

from vocal_analysis.features.vqt import TorchVQT  # noqa: E402

C, OUT = Path(sys.argv[1]), Path(sys.argv[2])
STEPS, BATCH, LR, WD, WARMUP, SEED, E = 12000, 32, 1e-3, 0.01, 1000, 0, 50
EVERY = 1000  # checkpoint and fidelity interval (was 2,000 to step 4,000; changed mid-run, no effect on training)
STEPS = int(os.environ.get("VA_STEPS", STEPS))  # stage 1b (student v3): 36,000
EVERY = int(os.environ.get("VA_EVERY", EVERY))
QUICK = os.environ.get("VA_QUICK") == "1"
if QUICK:
    STEPS, WARMUP = 40, 10
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
vq = TorchVQT(dev)
st = np.load(C / "stats.npz")
MU, SD = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
TRAIN = sorted(d.name for d in C.iterdir() if d.is_dir() and (d.name.startswith("real_") or d.name == "synth"))  # every training pool in the cache
HELD = ["held_zeroth", "held_vocalset", "held_synth_v2", "held_synth_v2s"]
load = lambda n: (np.load(C / n / "audio.npy", mmap_mode="r"), np.load(C / n / "target.npy", mmap_mode="r"))  # noqa: E731
data = {n: load(n) for n in TRAIN + HELD}
index = np.concatenate([np.stack([np.full(len(data[n][0]), k), np.arange(len(data[n][0]))], 1) for k, n in enumerate(TRAIN)])


def batch(names, rows):
    a = np.stack([np.asarray(data[names[k]][0][i], np.float32) for k, i in rows])
    t = np.stack([np.asarray(data[names[k]][1][i], np.float32) for k, i in rows])
    return torch.from_numpy(a).to(dev), (torch.from_numpy(t).to(dev) - MU) / SD


def forward(m, audio):
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        y = m(vq.log_power_with_context(audio, 400))  # (B, 400, 1024)
    return y.float().reshape(len(audio), E, 8, 1024).mean(2)  # encoder frames 1..50


def loss_fn(y, t):
    return (y - t).abs().mean() + (1 - torch.nn.functional.cosine_similarity(y, t, dim=-1)).mean()


def fidelity(m):
    m.eval()
    out = {}
    with torch.no_grad():
        for n in HELD:
            Y, T = [], []
            for s in range(0, len(data[n][0]), 32):
                a, t = batch([n], [(0, i) for i in range(s, min(s + 32, len(data[n][0])))])
                Y.append(forward(m, a).reshape(-1, 1024))
                T.append(t.reshape(-1, 1024))
            Y, T = torch.cat(Y), torch.cat(T)
            r2 = 1 - ((Y - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
            out[n] = (float(r2.mean()), float(torch.nn.functional.cosine_similarity(Y, T, dim=-1).mean()))
    m.train()
    return out


torch.manual_seed(SEED)
model = Student().to(dev)
if not (OUT / "ckpt_init.pt").exists():
    torch.save({"model": model.state_dict(), "step": 0, "seed": SEED}, OUT / "ckpt_init.pt")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
start = 0
if (OUT / "ckpt_last.pt").exists():
    ck = torch.load(OUT / "ckpt_last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start = ck["step"]
lr_at = lambda s: LR * (s + 1) / WARMUP if s < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP)))  # noqa: E731
log = open(OUT / "log.jsonl", "a")
print(f"student {sum(p.numel() for p in model.parameters()):,} parameters; {len(index)} training crops; from step {start}", flush=True)
if start == 0:
    f = fidelity(model)
    log.write(json.dumps({"step": 0, "fidelity": f}) + "\n")
    print("step 0 fidelity " + ", ".join(f"{n} R2 {r:.3f} cos {c:.3f}" for n, (r, c) in f.items()), flush=True)
t0, acc = time.time(), []
for step in range(start, STEPS):
    rng = np.random.default_rng([SEED, step])
    a, t = batch(TRAIN, index[rng.choice(len(index), BATCH, replace=False)])
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    loss = loss_fn(forward(model, a), t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc.append(float(loss))
    s = step + 1
    if s % 100 == 0 or QUICK:
        rate = BATCH * 4 * len(acc) / (time.time() - t0)
        log.write(json.dumps({"step": s, "loss": float(np.mean(acc)), "grad_norm": float(gn), "lr": lr_at(step)}) + "\n")
        log.flush()
        print(f"step {s} loss {np.mean(acc):.4f} grad {float(gn):.2f} lr {lr_at(step):.2e} {rate:.0f} audio s/s", flush=True)
        t0, acc = time.time(), []
    if s % EVERY == 0 or s == STEPS:
        f = fidelity(model)
        log.write(json.dumps({"step": s, "fidelity": f}) + "\n")
        log.flush()
        print(f"step {s} fidelity " + ", ".join(f"{n} R2 {r:.3f} cos {c:.3f}" for n, (r, c) in f.items()), flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "seed": SEED}
        torch.save(ck, OUT / f"ckpt_{s:06d}.pt")
        torch.save(ck, OUT / "ckpt_last.pt")
        t0 = time.time()
print("done", flush=True)
