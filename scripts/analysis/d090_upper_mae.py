"""D-090, reported (decides nothing): the upper formant heads' own error on synthetic validation (val_v2ko, whole clips,
sonorant frames where the label is in band), by tract-length band. MAE in Hz and in %, per head, per run.

    python d090_upper_mae.py <out.txt> run1 run2 ...   (runs/<run>/ckpt_last.pt; VA_DEVICE=cuda)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np

VAL = Path("data/synth/val_v2ko/flat")
BANDS = {"all": (0, 99), "<14": (0, 14), "14-16.6": (14, 16.6), "16.6-18.5": (16.6, 18.5), ">18.5": (18.5, 99)}


def main():
    import torch

    from vocal_analysis.features.vqt import HOP
    from vocal_analysis.models.bins import UPPER_AXIS, decode
    from vocal_analysis.models.infer import _vqt, load_teacher

    out, runs = Path(sys.argv[1]), sys.argv[2:]
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    a_off, f_off = np.load(VAL / "audio_offsets.npy"), np.load(VAL / "frame_offsets.npy")
    audio = np.load(VAL / "audio.npy", mmap_mode="r")
    lab_f = np.load(VAL / "label_formants.npy", mmap_mode="r")
    son = np.load(VAL / "label_sonorant.npy", mmap_mode="r")
    shards = sorted(VAL.parent.glob("shard_*.npz"))
    lengths = np.concatenate([np.load(p)["label_length_cm"] for p in shards]) if "label_length_cm" in np.load(shards[0]).files else None
    vq, lines = _vqt(dev), []
    for name in runs:
        model = load_teacher(f"runs/{name}/ckpt_last.pt", dev)[0].float()
        K = model.cfg.n_upper
        pred, true, L = [], [], []
        for i in range(len(a_off) - 1):
            x = torch.from_numpy(np.asarray(audio[a_off[i]:a_off[i + 1]], np.float32))[None].to(dev)
            nf = f_off[i + 1] - f_off[i]
            with torch.inference_mode():
                o = model(vq.log_power(x, int(nf)))
            p = o["upper_logits"][0].float().softmax(-1).cpu().numpy()  # (T, K, bins)
            hz, _ = decode(p, UPPER_AXIS)
            sl = slice(f_off[i], f_off[i + 1])
            m = np.asarray(son[sl]).astype(bool)
            pred.append(hz[m]), true.append(np.asarray(lab_f[sl])[m, 4:4 + K])
            L.append(np.asarray(lengths[sl])[m] if lengths is not None else np.full(m.sum(), np.nan))
        pred, true, L = np.concatenate(pred), np.concatenate(true), np.concatenate(L)
        for k in range(K):
            for b, (lo, hi) in BANDS.items():
                ok = np.isfinite(true[:, k]) & (true[:, k] > 0) & ((L >= lo) & (L < hi) if b != "all" else True)
                e = np.abs(pred[ok, k] - true[ok, k])
                lines.append(f"{name} F{5 + k} [{b}]: MAE {e.mean():.0f} Hz, {100 * np.mean(e / true[ok, k]):.1f} % ({ok.sum()} frames)")
        print("\n".join(lines[-K * len(BANDS):]), flush=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
