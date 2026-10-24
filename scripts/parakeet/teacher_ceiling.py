"""D-034, reported only: Parakeet's own per-frame references on the held-out crop sets (targets only, CPU). With all eight
phases interleaved into a 10 ms trajectory (phase q of encoder frame e at time 8(e - 1) + q), per phase, distil_train_v4.py's
R^2 (per-dimension, set variance as reference) of Parakeet against itself after
  - box8: a centred 8-frame moving average of the trajectory (times t - 4 .. t + 3): the best a student whose output is
    flat within each 80 ms window could do;
  - no_comb: the crop's fixed 8-periodic pattern removed (for each crop and dimension, the mean over encoder frames of each
    phase, minus its mean over phases): the variance tied to where the audio sits on Parakeet's 8-frame grid, which a
    student cannot sensibly learn;
  - mean_e: each encoder frame's mean over its eight phases (Parakeet's own phase-averaged output).
Encoder frames 2..48 only, so every moving-average window is complete.
Args: v4 cache dir, out csv. Writes rows set,phase,reference,r2."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

C, OUT = Path(sys.argv[1]), sys.argv[2]
st = np.load(C / "stats.npz")
MU, SD = torch.tensor(st["mean"], dtype=torch.float32), torch.tensor(st["std"], dtype=torch.float32)
SETS = sorted(d.name for d in C.iterdir() if d.is_dir() and d.name.startswith("held_") and not d.name.startswith("held_words"))
E, T = 49, 8 * 49
KEEP = slice(1, 48)  # encoder frames 2..48 (0-based 1..47)
REFS = ("box8", "no_comb", "mean_e")
torch.set_num_threads(8)

with open(OUT, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["set", "phase", "reference", "r2"])
    for n in SETS:
        tgt = np.load(C / n / "target.npy", mmap_mode="r")  # (N, 8, 49, 1024)
        N = len(tgt)
        se = {r: torch.zeros(8, 1024, dtype=torch.float64) for r in REFS}
        s1 = torch.zeros(8, 1024, dtype=torch.float64)
        s2 = torch.zeros(8, 1024, dtype=torch.float64)
        cnt = 0
        for b in range(0, N, 16):
            t = (torch.from_numpy(np.asarray(tgt[b:b + 16], np.float32)) - MU) / SD  # (b, 8, 49, 1024)
            traj = t.permute(0, 2, 1, 3).reshape(len(t), T, 1024)  # time 8(e - 1) + q
            c = torch.cumsum(torch.nn.functional.pad(traj, (0, 0, 1, 0)), 1)
            box = torch.zeros_like(traj)
            box[:, 4:T - 4] = (c[:, 8:T] - c[:, 0:T - 8]) / 8  # times t - 4 .. t + 3
            box = box.reshape(len(t), E, 8, 1024).permute(0, 2, 1, 3)
            comb = t.mean(2, keepdim=True) - t.mean((1, 2), keepdim=True)  # (b, 8, 1, 1024)
            ref = {"box8": box, "no_comb": t - comb, "mean_e": t.mean(1, keepdim=True).expand_as(t)}
            tk = t[:, :, KEEP]
            for r in REFS:
                se[r] += ((ref[r][:, :, KEEP] - tk) ** 2).sum((0, 2)).double()
            s1 += tk.sum((0, 2)).double()
            s2 += (tk ** 2).sum((0, 2)).double()
            cnt += len(t) * tk.shape[2]
        var = s2 / cnt - (s1 / cnt) ** 2
        for r in REFS:
            r2 = (1 - se[r] / (cnt * var)).mean(1).numpy()
            for q in range(8):
                w.writerow([n, q, r, round(float(r2[q]), 4)])
            print(n, r, "mean over phases", round(float(r2.mean()), 4), "by phase", np.round(r2, 4).tolist(), flush=True)
