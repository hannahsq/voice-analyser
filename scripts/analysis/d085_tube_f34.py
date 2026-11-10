"""D-085 note (2026-11-11): how predictable are F3 / F4 from F1, F2 and the instantaneous length, by kind (cascade / tube)? Label-only, CPU.
Held-out validation clips of train_v2 (both kinds) and train_v2ku (tube, coupled); sonorant frames."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def frames(name, ids):
    f = Path(f"data/synth/{name}/flat")
    off = np.load(f / "frame_offsets.npy")
    F = np.load(f / "label_formants.npy", mmap_mode="r")
    son = np.load(f / "label_sonorant.npy", mmap_mode="r")
    li = np.load(f / "label_length_inst_cm.npy", mmap_mode="r")
    tube = np.load(f / "clip_tube.npy")
    out = {0: [], 1: []}
    for c in ids:
        s = slice(off[c], off[c + 1])
        m = np.asarray(son[s]).astype(bool)
        x = np.column_stack([np.asarray(F[s])[m, :4], np.asarray(li[s])[m]])
        out[int(tube[c])].append(x[::4])
    return {k: np.concatenate(v) for k, v in out.items() if v}


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingRegressor

    from vocal_analysis.train.data import split_ids

    for name in ("train_v2", "train_v2ku"):
        n = len(np.load(f"data/synth/{name}/flat/frame_offsets.npy")) - 1
        tr, va = split_ids(n)
        A, B = frames(name, tr[:3000]), frames(name, va[:512])
        for k in B:
            a, b = A[k], B[k]
            a, b = a[np.all(np.isfinite(a), 1)], b[np.all(np.isfinite(b), 1)]
            line = f"{name:11s} {'tube' if k else 'cascade':8s} val frames {len(b):6d}"
            for j in (2, 3):
                g = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(a[:, [0, 1, 4]], a[:, j])
                e = np.abs(g.predict(b[:, [0, 1, 4]]) - b[:, j])
                line += f" | F{j + 1} SD {b[:, j].std():5.0f} Hz, MAE from F1,F2,iVTL {e.mean():5.0f} Hz"
            print(line, flush=True)
