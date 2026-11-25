"""Coverage of close F1-F2 pairs in the training sets (labels only; no audio, no model): per set, the share of
sonorant frames with F2 / F1 below given ratios, overall and with F1 above 600 Hz, and the frames in a low-back cell
(F1 650-800 Hz, F2 750-950 Hz). Every 10th frame.

    python scripts/analysis/close_f1f2_coverage.py
"""

from pathlib import Path

import numpy as np

SETS = ("data/synth/train_v2", "data/resynth/vocalset_train_d16k", "data/resynth/zeroth_korean_train_d16k")

if __name__ == "__main__":
    for s in SETS:
        flat = Path(s) / "flat"
        F = np.load(flat / "label_formants.npy", mmap_mode="r")
        son = np.load(flat / "label_sonorant.npy", mmap_mode="r")
        F = F.reshape(len(son), -1)[::10, :2].astype(np.float64)
        m = (np.asarray(son[::10]) > 0.5) & np.isfinite(F).all(1) & (F > 0).all(1)
        F = F[m]
        r = F[:, 1] / F[:, 0]
        op = F[:, 0] > 600
        cell = (F[:, 0] > 650) & (F[:, 0] < 800) & (F[:, 1] > 750) & (F[:, 1] < 950)
        print(f"{s}: {len(F):,} sonorant frames (every 10th); F2/F1 quantiles 0.1 %, 1 %, 5 %: "
              + ", ".join(f"{np.quantile(r, q):.2f}" for q in (0.001, 0.01, 0.05)))
        for t in (1.2, 1.3, 1.5):
            print(f"  F2/F1 < {t}: {np.mean(r < t):.3%} of all; {np.mean(r[op] < t):.3%} of frames with F1 > 600 Hz")
        print(f"  low-back cell (F1 650-800, F2 750-950 Hz): {cell.sum():,} frames ({cell.mean():.3%})")
