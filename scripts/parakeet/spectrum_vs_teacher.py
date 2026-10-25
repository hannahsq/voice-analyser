"""D-034, reported only: the students' 10 ms output spectra against Parakeet's own. On the held-out crop sets every crop has
targets at all eight phases, so Parakeet's layer 5 interleaves into a 10 ms trajectory: phase q of encoder frame e sits
at student frame 8e + q - 4 (e = 1..49), frames 4..395. Students are taken on the same 392 frames. Per crop: mean removed,
Hann window, |rfft|^2 summed over the 1024 standardised dimensions; averaged over crops. Spectra in dB relative to
Parakeet's peak above 0 Hz in that set; band powers as student / Parakeet in dB. Students trained against another cache's
statistics are restandardised to v4's (as fid_v4_on.py).
Args: v4 cache dir, out prefix, then LABEL=CKPT=STATS_DIR per student. Writes <prefix>_spectrum.csv and <prefix>_bands.csv.
VA_CROPS=<csv with set,crop columns>: only those crops of those sets (e.g. one f0 range; crop_subset.py).
Run from the main checkout."""
import csv
import os
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
g = {"__name__": "spectrum_vs_teacher", "__file__": str(HERE / "distil_train_v4.py")}
sys.argv = [sys.argv[0], C4, tempfile.mkdtemp()]
exec(compile(src[:src.index("torch.manual_seed(SEED)")], "distil_train_v4.py", "exec"), g)  # data, run(), std()
dev, MU4, SD4, data, run, std, E4 = (g[k] for k in ("dev", "MU", "SD", "data", "run", "std", "E4"))
A, T = 4, 8 * g["E4"]  # frames 4 .. 395
HANN = torch.hann_window(T, periodic=False, device=dev, dtype=torch.float64)[None, :, None]
FREQ = np.fft.rfftfreq(T, 0.01)
BANDS = [(0, 6.25), (6.25, 12.5), (12.5, 25), (25, 50.01)]
IDX = {}  # set -> crop indices (all crops when VA_CROPS is unset)
if os.environ.get("VA_CROPS"):
    for r in csv.DictReader(open(os.environ["VA_CROPS"], encoding="utf-8")):
        IDX.setdefault(r["set"], []).append(int(r["crop"]))
HELD = list(IDX) or g["HELD"]
crops = lambda n: IDX.get(n) or list(range(len(data[n]["phase"])))  # noqa: E731


def power(y):
    """(b, T, 1024) -> (F,) summed over crops and dimensions"""
    z = (y.double() - y.double().mean(1, keepdim=True)) * HANN
    return (torch.fft.rfft(z, dim=1).abs() ** 2).sum(-1).sum(0)


P = {}
for n in HELD:
    d, ci = data[n], crops(n)
    N = len(ci)
    acc = torch.zeros(len(FREQ), device=dev, dtype=torch.float64)
    for b in range(0, N, 32):
        t = std(np.stack([np.asarray(d["target"][i], np.float32) for i in ci[b:b + 32]]))  # (b, 8, 49, 1024)
        acc += power(t.permute(0, 2, 1, 3).reshape(t.shape[0], T, 1024))  # time index 8(e-1) + q
    P[("parakeet", n)] = (acc / N).cpu().numpy()
for label, ck, sdir in SPECS:
    s = np.load(Path(sdir) / "stats.npz")
    MU, SD = (torch.tensor(s[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    m = load_student(ck, dev)
    with torch.no_grad():
        for n in HELD:
            d, ci = data[n], crops(n)
            N = len(ci)
            acc = torch.zeros(len(FREQ), device=dev, dtype=torch.float64)
            for b in range(0, N, 32):
                idx = ci[b:b + 32]
                y = (run(m, torch.from_numpy(np.stack([np.asarray(d["audio"][i], np.float32) for i in idx])).to(dev), 400) * SD + MU - MU4) / SD4
                acc += power(y[:, A:A + T])
            P[(label, n)] = (acc / N).cpu().numpy()

with open(f"{PRE}_spectrum.csv", "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["model", "set", "hz", "power_db"])
    for (lab, n), p in P.items():
        ref = P[("parakeet", n)][1:].max()
        for f, v in zip(FREQ, 10 * np.log10(p / ref)):
            w.writerow([lab, n, f"{f:.3f}", round(float(v), 2)])
with open(f"{PRE}_bands.csv", "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["model", "set", "band", "share_of_power", "excess_db"])
    for (lab, n), p in P.items():
        tp = P[("parakeet", n)]
        tot = p[1:].sum()
        for lo, hi in BANDS:
            k = (FREQ > lo) & (FREQ <= hi) if lo else (FREQ > 0) & (FREQ <= hi)
            w.writerow([lab, n, f"{lo:g}-{min(hi, 50):g} Hz", round(float(p[k].sum() / tot), 4), round(float(10 * np.log10(p[k].sum() / tp[k].sum())), 2)])
            print(lab, n, f"{lo:g}-{min(hi, 50):g} Hz share {p[k].sum() / tot:.3f} excess {10 * np.log10(p[k].sum() / tp[k].sum()):+.2f} dB", flush=True)
