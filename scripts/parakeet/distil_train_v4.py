"""D-034 student v4 (pre-registered): the stage-1 student trained on distil_cache_v4.py's targets.
  - Crops: per-frame targets at each crop's phase: student frame 8e + phi - 4 against Parakeet's frame e (e = 1..49).
  - Words: one batch of 16 isolated-word items of one length L per step (bucket drawn in proportion to its size),
    student frame 8e + phi against Parakeet's frame e where masked in; the crop and word losses are added.
  - Loss per frame: L1 + (1 - cosine) on standardised targets (as v2).
  - AdamW 1e-3, weight decay 0.01, gradient norm clipped at 1.0, bf16 autocast, seed 0 (v2's initial weights).
  - WSD: warm-up 1,000 steps, then constant. A decay branch (VA_BRANCH_FROM=<stable checkpoint>, VA_DECAY=D) starts from
    that checkpoint and decays linearly to 0 over D steps. Batches are seeded by step, so a branch sees the steps the
    stable run would have.
  - Fidelity every 1,000 steps: held-out crops at all eight phases (per-frame R^2 and cosine, mean over phases, and the
    spread across phases), v2's measure (phase 0, eights averaged), and held-out words (per-frame, all phases).
Args: cache dir, out dir; VA_STEPS (end step), VA_BRANCH_FROM, VA_DECAY. Run from the main checkout."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import Student  # noqa: E402

from vocal_analysis.features.vqt import HOP, N_MAX, TorchVQT  # noqa: E402

C, OUT = Path(sys.argv[1]), Path(sys.argv[2])
STEPS = int(os.environ.get("VA_STEPS", 16000))
BRANCH = os.environ.get("VA_BRANCH_FROM")
DECAY = int(os.environ.get("VA_DECAY", 0))
BATCH, WBATCH, LR, WD, WARMUP, SEED, EVERY = 32, 16, 1e-3, 0.01, 1000, 0, 1000
QUICK = os.environ.get("VA_QUICK") == "1"
if QUICK:
    STEPS, WARMUP, EVERY = min(STEPS, 40), 10, 20
OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
vq = TorchVQT(dev)
st = np.load(C / "stats.npz")
MU, SD = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
E4 = 49


def sets(prefix):
    return sorted(d.name for d in C.iterdir() if d.is_dir() and d.name.startswith(prefix))


def load(name):
    d = C / name
    out = {"audio": np.load(d / "audio.npy", mmap_mode="r"), "target": np.load(d / "target.npy", mmap_mode="r"),
           "phase": np.load(d / "phase.npy")}
    if (d / "mask.npy").exists():
        out["mask"], out["length"] = np.load(d / "mask.npy"), np.load(d / "length.npy")
    return out


CROPS = sets("real_") + ["synth"]
WORDS = sets("words_")
HELD = [n for n in sets("held_") if not n.startswith("held_words")]
HELDW = sets("held_words")
data = {n: load(n) for n in CROPS + WORDS + HELD + HELDW}
cindex = np.concatenate([np.stack([np.full(len(data[n]["phase"]), k), np.arange(len(data[n]["phase"]))], 1) for k, n in enumerate(CROPS)])
wrows = np.concatenate([np.stack([np.full(len(data[n]["phase"]), k), np.arange(len(data[n]["phase"])), data[n]["length"]], 1)
                        for k, n in enumerate(WORDS)])
BUCKETS = {int(L): wrows[wrows[:, 2] == L][:, :2] for L in np.unique(wrows[:, 2])}
BL = np.array(sorted(BUCKETS))
BP = np.array([len(BUCKETS[L]) for L in BL], float)
BP /= BP.sum()
ECROP = torch.arange(1, E4 + 1, device=dev) * 8 - 4  # student frame of encoder frame e at phase 0


def std(t):
    return (torch.from_numpy(np.asarray(t, np.float32)).to(dev) - MU) / SD


def run(m, audio, n_frames):
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        return m(vq.log_power_with_context(audio, n_frames)).float()


def frame_loss(y, t, w):
    """y, t (..., 1024); w (...) weights: mean over weighted frames of L1 + (1 - cosine)"""
    per = (y - t).abs().mean(-1) + 1 - torch.nn.functional.cosine_similarity(y, t, dim=-1)
    return (per * w).sum() / w.sum().clamp_min(1)


def crop_batch(rows, set_names, phase_col=0):
    a = np.stack([np.asarray(data[set_names[k]]["audio"][i], np.float32) for k, i in rows])
    t = np.stack([np.asarray(data[set_names[k]]["target"][i], np.float32) for k, i in rows])
    ph = np.array([data[set_names[k]]["phase"][i, phase_col] for k, i in rows])
    return torch.from_numpy(a).to(dev), std(t), torch.from_numpy(ph).to(dev)


def crop_loss(m, rows):
    a, t, ph = crop_batch(rows, CROPS)
    y = run(m, a, 400)
    j = ECROP[None] + ph[:, None]  # (B, 49)
    return frame_loss(y.gather(1, j[..., None].expand(-1, -1, 1024)), t, torch.ones_like(j, dtype=torch.float32))


def word_batch(rows, set_names, L, q=0):
    """word items of length L at phase column q (training sets have one column, held-out sets eight)"""
    a = np.stack([np.asarray(data[set_names[k]]["audio"][i, :(L - 1) * HOP + N_MAX], np.float32) for k, i in rows])
    t = np.stack([np.asarray(data[set_names[k]]["target"][i] if data[set_names[k]]["target"].ndim == 3
                             else data[set_names[k]]["target"][i, q], np.float32) for k, i in rows])
    msk = np.stack([data[set_names[k]]["mask"][i, q] for k, i in rows])
    ph = np.array([data[set_names[k]]["phase"][i, q] for k, i in rows])
    return torch.from_numpy(a).to(dev), std(t), torch.from_numpy(msk).to(dev).float(), torch.from_numpy(ph).to(dev)


def word_loss(m, rng):
    L = int(rng.choice(BL, p=BP))
    pool = BUCKETS[L]
    rows = pool[rng.choice(len(pool), WBATCH, replace=len(pool) < WBATCH)]
    a, t, w, ph = word_batch(rows, WORDS, L)
    y = run(m, a, L)
    j = (torch.arange(t.shape[1], device=dev) * 8)[None] + ph[:, None]
    w = w * (j < L)
    return frame_loss(y.gather(1, j.clamp(max=L - 1)[..., None].expand(-1, -1, 1024)), t, w)


def r2cos(Y, T):
    r2 = 1 - ((Y - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
    return float(r2.mean()), float(torch.nn.functional.cosine_similarity(Y, T, dim=-1).mean())


def fidelity(m):
    m.eval()
    out = {}
    with torch.no_grad():
        for n in HELD:
            d = data[n]
            N = len(d["phase"])
            Yq, Tq, Yp, Tp = [[] for _ in range(8)], [[] for _ in range(8)], [], []
            for s in range(0, N, 32):
                idx = range(s, min(s + 32, N))
                y = run(m, torch.from_numpy(np.stack([np.asarray(d["audio"][i], np.float32) for i in idx])).to(dev), 400)
                t = std(np.stack([np.asarray(d["target"][i], np.float32) for i in idx]))  # (b, 8, 49, 1024)
                for q in range(8):
                    Yq[q].append(y[:, ECROP + q].reshape(-1, 1024))
                    Tq[q].append(t[:, q].reshape(-1, 1024))
                Yp.append(y[:, :8 * E4].reshape(len(idx), E4, 8, 1024).mean(2).reshape(-1, 1024))  # v2's: frames 8e - 8 .. 8e - 1 (mels 8e - 4 .. 8e + 3), phase 0
                Tp.append(t[:, 0].reshape(-1, 1024))
            scores = [r2cos(torch.cat(Yq[q]), torch.cat(Tq[q])) for q in range(8)]
            out[n] = {"r2": float(np.mean([a for a, _ in scores])), "cos": float(np.mean([b for _, b in scores])),
                      "r2_by_phase": [round(a, 4) for a, _ in scores], "r2_pooled_phase0": r2cos(torch.cat(Yp), torch.cat(Tp))[0]}
        for n in HELDW:
            d = data[n]
            Y, T = [], []
            for L in np.unique(d["length"]):
                ids = np.flatnonzero(d["length"] == L)
                for q in range(8):
                    for s in range(0, len(ids), 32):
                        rows = [(0, i) for i in ids[s:s + 32]]
                        a, t, w, ph = word_batch(rows, [n], int(L), q)
                        y = run(m, a, int(L))
                        j = (torch.arange(t.shape[1], device=dev) * 8)[None] + ph[:, None]
                        keep = (w > 0) & (j < L)
                        Y.append(y.gather(1, j.clamp(max=int(L) - 1)[..., None].expand(-1, -1, 1024))[keep])
                        T.append(t[keep])
            r2, cos = r2cos(torch.cat(Y), torch.cat(T))
            out[n] = {"r2": r2, "cos": cos}
    m.train()
    return out


torch.manual_seed(SEED)
model = Student().to(dev)
if not (OUT / "ckpt_init.pt").exists():
    torch.save({"model": model.state_dict(), "step": 0, "seed": SEED}, OUT / "ckpt_init.pt")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
if BRANCH and not (OUT / "ckpt_last.pt").exists():
    shutil.copy(BRANCH, OUT / "ckpt_last.pt")
start, s0 = 0, None
if (OUT / "ckpt_last.pt").exists():
    ck = torch.load(OUT / "ckpt_last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start = ck["step"]
    s0 = ck.get("branch_start", start if BRANCH else None)
if BRANCH:
    s0 = s0 if s0 is not None else start


def lr_at(s):
    if s < WARMUP:
        return LR * (s + 1) / WARMUP
    if BRANCH:
        return LR * max(0.0, 1 - (s - s0) / DECAY)
    return LR


log = open(OUT / "log.jsonl", "a")
print(f"student v4 {sum(p.numel() for p in model.parameters()):,} parameters; {len(cindex)} crops, {len(wrows)} words in "
      f"{len(BL)} lengths; from step {start} to {STEPS}" + (f" (branch from {s0}, decay {DECAY})" if BRANCH else ""), flush=True)
if start == 0:
    f = fidelity(model)
    log.write(json.dumps({"step": 0, "fidelity": f}) + "\n")
t0, acc = time.time(), []
for step in range(start, STEPS):
    rng = np.random.default_rng([SEED, step])
    rows = cindex[rng.choice(len(cindex), BATCH, replace=False)]
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    lc = crop_loss(model, rows)
    lw = word_loss(model, rng)
    loss = lc + lw
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    acc.append((float(lc), float(lw)))
    s = step + 1
    if s % 100 == 0 or QUICK:
        lc_, lw_ = np.mean(acc, 0)
        log.write(json.dumps({"step": s, "loss_crops": lc_, "loss_words": lw_, "grad_norm": float(gn), "lr": lr_at(step)}) + "\n")
        log.flush()
        print(f"step {s} crops {lc_:.4f} words {lw_:.4f} grad {float(gn):.2f} lr {lr_at(step):.2e} ({time.time() - t0:.0f} s)", flush=True)
        t0, acc = time.time(), []
    if s % EVERY == 0 or s == STEPS:
        f = fidelity(model)
        log.write(json.dumps({"step": s, "fidelity": f}) + "\n")
        log.flush()
        print(f"step {s} fidelity " + ", ".join(f"{n} {v['r2']:.3f}" for n, v in f.items()), flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "seed": SEED, "branch_start": s0}
        torch.save(ck, OUT / f"ckpt_{s:06d}.pt")
        torch.save(ck, OUT / "ckpt_last.pt")
print("done", flush=True)
