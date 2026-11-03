"""D-078 scoring (GPU): each run's F1-F4 on fixed crops, saved beside both labels. Sets: val_v2k (the held-out coupled
set, both labels), train_v2's validation clips (uncoupled: the two labels coincide) and the resynthesised validation
sets (their exact formants). Clean audio (no channel), so the comparison is between labels, not channels.

    python scripts/analysis/d078_score.py <run> ...   (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("cache/d078")
N_FRAMES, BATCH, VAL_CLIPS = 400, 16, 512


def sets():
    from vocal_analysis.train.data import SynthCrops, consolidate, split_ids

    out = {}
    flat = consolidate("data/synth/val_v2k")
    n = json.loads((flat / "done").read_text())["n_clips"]
    for lab in ("formants", "formants_acoustic"):
        out[f"v2k:{lab}"] = SynthCrops(flat, np.arange(n), N_FRAMES, None, seed=12345, fixed=True, formant_label=lab)
    flat = consolidate("data/synth/train_v2")
    n = json.loads((flat / "done").read_text())["n_clips"]
    out["v2:formants"] = SynthCrops(flat, split_ids(n)[1][:VAL_CLIPS], N_FRAMES, None, seed=12345, fixed=True)
    for d in ("data/resynth/vocalset_val_d16k", "data/resynth/zeroth_korean_val_d16k"):
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.sort(np.random.default_rng(0).permutation(n)[:VAL_CLIPS])
        out[f"{Path(d).name}:formants"] = SynthCrops(rflat, ids, N_FRAMES, None, seed=12345, fixed=True)
    return out


def score(run, data):
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.bins import FORMANT_AXIS, decode
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import collate

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(Path("runs") / run / "ckpt_last.pt", dev)
    vq = TorchVQT(dev)
    res = {}
    for name, ds in data.items():
        pred, ref, son = [], [], []
        with torch.inference_mode():
            for audio, lab in DataLoader(ds, BATCH, num_workers=4, collate_fn=collate):
                audio = audio.to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(vq.log_power_with_context(audio, lab["f0"].shape[1]))
                fm, _ = decode(out["formant_logits"].float().softmax(-1).cpu().numpy(), FORMANT_AXIS)
                pred.append(fm.reshape(-1, fm.shape[-1]))
                ref.append(lab["formants"].numpy().reshape(-1, fm.shape[-1]))
                son.append(lab["sonorant"].numpy().reshape(-1))
        res[f"{name}|pred"], res[f"{name}|ref"], res[f"{name}|son"] = map(np.concatenate, (pred, ref, son))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{run}.npz", **res)


if __name__ == "__main__":
    data = sets()
    for r in sys.argv[1:]:
        if not (OUT / f"{r}.npz").exists():
            score(r, data)
            print(f"scored {r}", flush=True)
