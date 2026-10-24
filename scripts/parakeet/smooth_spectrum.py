"""D-034, reported only: is v2's per-frame shortfall within-window wobble or misalignment? For each held-out crop set and
student, (1) per-phase R^2 (distil_train_v4.py's measure) of the raw 10 ms output and of the output after a centred
8-frame moving average (student frames j - 4 .. j + 3, the pooled measure's window at phase 0), and (2) the output's
temporal power spectrum: per crop, the mean removed, a Hann window over the 400 frames, |rfft|^2 summed over the 1024
dimensions, averaged over crops (0-50 Hz at 0.25 Hz). Students trained against another cache's statistics are
restandardised to v4's (as fid_v4_on.py).
Args: v4 cache dir, out prefix, then LABEL=CKPT=STATS_DIR per student. Writes <prefix>_r2.csv and <prefix>_spectrum.csv.
Run from the main checkout."""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from student import load_student  # noqa: E402

C4, PRE, SPECS = sys.argv[1], sys.argv[2], [a.split("=", 2) for a in sys.argv[3:]]
src = (HERE / "distil_train_v4.py").read_text(encoding="utf-8")
g = {"__name__": "smooth_spectrum", "__file__": str(HERE / "distil_train_v4.py")}
sys.argv = [sys.argv[0], C4, tempfile.mkdtemp()]
exec(compile(src[:src.index("torch.manual_seed(SEED)")], "distil_train_v4.py", "exec"), g)  # data, run(), std(), r2cos()
dev, MU4, SD4, data, run, std, ECROP, r2cos = (g[k] for k in ("dev", "MU", "SD", "data", "run", "std", "ECROP", "r2cos"))
T = 400
HANN = torch.hann_window(T, periodic=False, device=dev, dtype=torch.float64)[None, :, None]
FREQ = np.fft.rfftfreq(T, 0.01)

fr = open(f"{PRE}_r2.csv", "w", newline="")
fs = open(f"{PRE}_spectrum.csv", "w", newline="")
wr, ws = csv.writer(fr), csv.writer(fs)
wr.writerow(["student", "set", "phase", "variant", "r2"])
ws.writerow(["student", "set", "hz", "power_db"])
for label, ck, sdir in SPECS:
    s = np.load(Path(sdir) / "stats.npz")
    MU, SD = (torch.tensor(s[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    m = load_student(ck, dev)
    with torch.no_grad():
        for n in g["HELD"]:
            d = data[n]
            N = len(d["phase"])
            Y = {v: [[] for _ in range(8)] for v in ("raw", "box8")}
            Tq = [[] for _ in range(8)]
            P = torch.zeros(T // 2 + 1, device=dev, dtype=torch.float64)
            for b in range(0, N, 32):
                idx = range(b, min(b + 32, N))
                y = (run(m, torch.from_numpy(np.stack([np.asarray(d["audio"][i], np.float32) for i in idx])).to(dev), T) * SD + MU - MU4) / SD4
                t = std(np.stack([np.asarray(d["target"][i], np.float32) for i in idx]))
                c = torch.cumsum(torch.nn.functional.pad(y, (0, 0, 1, 0)), 1)  # c[:, k] = sum of y[:, :k]
                for q in range(8):
                    j = ECROP + q
                    Y["raw"][q].append(y[:, j].reshape(-1, 1024).cpu())
                    Y["box8"][q].append(((c[:, j + 4] - c[:, j - 4]) / 8).reshape(-1, 1024).cpu())
                    Tq[q].append(t[:, q].reshape(-1, 1024).cpu())
                z = (y.double() - y.double().mean(1, keepdim=True)) * HANN
                P += (torch.fft.rfft(z, dim=1).abs() ** 2).sum(-1).sum(0)
            for v in Y:
                for q in range(8):
                    wr.writerow([label, n, q, v, round(r2cos(torch.cat(Y[v][q]), torch.cat(Tq[q]))[0], 4)])
            P = (P / N).cpu().numpy()
            for f, p in zip(FREQ, 10 * np.log10(P / P[1:].max())):
                ws.writerow([label, n, f"{f:.2f}", round(float(p), 2)])
            print(label, n, "raw", round(float(np.mean([r2cos(torch.cat(Y["raw"][q]), torch.cat(Tq[q]))[0] for q in range(8)])), 4),
                  "box8", round(float(np.mean([r2cos(torch.cat(Y["box8"][q]), torch.cat(Tq[q]))[0] for q in range(8)])), 4), flush=True)
fr.close()
fs.close()
