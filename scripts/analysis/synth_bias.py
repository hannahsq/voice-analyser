"""Signed formant bias on the synthetic validation set (the 512 fixed val clips, 200 frames, as training's eval), fp32.
Args: data dir, then name=checkpoint ... Median signed error (Hz) on sonorant frames, by f0 band and by true F1."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vocal_analysis.device import setup_device
from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.bins import FORMANT_AXIS, decode
from vocal_analysis.models.teacher import Teacher, TeacherConfig
from vocal_analysis.train.data import SynthCrops, collate, consolidate, split_ids
from vocal_analysis.train.teacher import _saved_channel, cache_batches

F0B = ((0, 150), (150, 300), (300, 2000))
F1B = ((0, 450), (450, 650), (650, 5000))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data, models = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    dev = setup_device()
    flat = consolidate(data, log=lambda m: None)
    _, val_ids = split_ids(json.loads((flat / "done").read_text())["n_clips"])
    ch = _saved_channel(Path(models[0][1]).parent)
    ds = SynthCrops(flat, val_ids[:512], 200, ch, seed=12345, fixed=True)
    batches = cache_batches(DataLoader(ds, 32, num_workers=4, collate_fn=collate), TorchVQT(dev), dev)
    ref = np.concatenate([b[1]["formants"].cpu().numpy().reshape(-1, 4) for b in batches])
    son = np.concatenate([b[1]["sonorant"].cpu().numpy().reshape(-1) for b in batches]).astype(bool)
    f0 = np.concatenate([b[1]["f0"].cpu().numpy().reshape(-1) for b in batches])
    print(f"{data}: {son.sum()} sonorant frames. Median signed error Hz (mean |err|)")
    print(f"{'':10s} {'F1':>14s} {'F2':>14s} {'F3':>14s} {'F4':>14s} | F1 by f0 <150 / 150-300 / >300 | F1 by true F1 <450 / 450-650 / >650")
    for name, ck in models:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        m = Teacher(TeacherConfig(**c["teacher_config"])).to(dev)
        m.load_state_dict(c["model"])
        m.eval()
        pred = []
        with torch.inference_mode():
            for lp, _ in batches:
                p = m(lp.float())["formant_logits"].float().softmax(-1).cpu().numpy()
                pred.append(decode(p, FORMANT_AXIS)[0].reshape(-1, 4))
        e = np.concatenate(pred) - ref
        cols = []
        for n in range(4):
            k = son & np.isfinite(ref[:, n])
            cols.append(f"{np.median(e[k, n]):+6.1f} ({np.mean(np.abs(e[k, n])):5.1f})")
        k = son & np.isfinite(ref[:, 0])
        b0 = " / ".join(f"{np.median(e[k & (f0 >= lo) & (f0 < hi), 0]):+5.1f}" for lo, hi in F0B)
        b1 = " / ".join(f"{np.median(e[k & (ref[:, 0] >= lo) & (ref[:, 0] < hi), 0]):+5.1f}" for lo, hi in F1B)
        print(f"{name:10s} {' '.join(cols)} | {b0} | {b1}", flush=True)


if __name__ == "__main__":
    main()
