"""D-034, reported only: the spread behind v4's per-phase held-out fidelity. For each held-out crop set, student and phase,
each crop's R^2 with the set's own per-dimension variance at that phase as the reference (1 - mean over dimensions of the
crop's squared error / (frames x set variance)), so the mean over crops is exactly distil_train_v4.py's per-phase R^2.
Students trained against another cache's statistics are restandardised to v4's (as fid_v4_on.py).
Args: v4 cache dir, out .csv, then LABEL=CKPT=STATS_DIR per student. Writes rows student,set,crop,phase,r2. Run from the main checkout."""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from student import load_student  # noqa: E402

C4, OUTF, SPECS = sys.argv[1], Path(sys.argv[2]), [a.split("=", 2) for a in sys.argv[3:]]
src = (HERE / "distil_train_v4.py").read_text(encoding="utf-8")
g = {"__name__": "phase_spread", "__file__": str(HERE / "distil_train_v4.py")}
sys.argv = [sys.argv[0], C4, tempfile.mkdtemp()]
exec(compile(src[:src.index("torch.manual_seed(SEED)")], "distil_train_v4.py", "exec"), g)  # data, run(), std()
dev, MU4, SD4, data, run, std, ECROP = (g[k] for k in ("dev", "MU", "SD", "data", "run", "std", "ECROP"))

with open(OUTF, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["student", "set", "crop", "phase", "r2"])
    for label, ck, sdir in SPECS:
        s = np.load(Path(sdir) / "stats.npz")
        MU, SD = (torch.tensor(s[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
        m = load_student(ck, dev)
        with torch.no_grad():
            for n in g["HELD"]:
                d = data[n]
                N = len(d["phase"])
                se = torch.zeros(N, 8, 1024, device=dev, dtype=torch.float64)
                ts = torch.zeros(8, 1024, device=dev, dtype=torch.float64)
                tq = torch.zeros(8, 1024, device=dev, dtype=torch.float64)
                for b in range(0, N, 32):
                    idx = range(b, min(b + 32, N))
                    y = (run(m, torch.from_numpy(np.stack([np.asarray(d["audio"][i], np.float32) for i in idx])).to(dev), 400) * SD + MU - MU4) / SD4
                    t = std(np.stack([np.asarray(d["target"][i], np.float32) for i in idx]))  # (b, 8, 49, 1024)
                    for q in range(8):
                        se[b:b + len(idx), q] = ((y[:, ECROP + q] - t[:, q]) ** 2).sum(1).double()
                        ts[q] += t[:, q].sum((0, 1)).double()
                        tq[q] += (t[:, q] ** 2).sum((0, 1)).double()
                F = 49
                var = tq / (N * F) - (ts / (N * F)) ** 2  # (8, 1024), the set's per-dimension variance at each phase
                r2 = (1 - (se / (F * var[None])).mean(-1)).cpu().numpy()  # (N, 8)
                for i in range(N):
                    for q in range(8):
                        w.writerow([label, n, i, q, round(float(r2[i, q]), 4)])
                print(label, n, "mean by phase", np.round(r2.mean(0), 4).tolist(), flush=True)
