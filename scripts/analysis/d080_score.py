"""D-080 scoring (GPU): each run's F1-F4 and opening on fixed clean crops. val_v2ko at 400, 100 and 50 frames (both
formant labels, the opening label, the Rd label, the crop's clip), train_v2's validation clips (uncoupled) and the resynthesised
validation sets at 400 frames. Runs without an opening head score the formants only.

    python scripts/analysis/d080_score.py <run> ...   (VA_DEVICE=cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("cache/d080")
BATCH, VAL_CLIPS = 16, 512
LENGTHS = (400, 100, 50)


def sets():
    from vocal_analysis.train.data import SynthCrops, consolidate, split_ids

    out = {}
    flat = consolidate("data/synth/val_v2ko")
    n = json.loads((flat / "done").read_text())["n_clips"]
    for t in LENGTHS:
        for lab in ("formants", "formants_acoustic"):
            out[f"v2ko{t}:{lab}"] = SynthCrops(flat, np.arange(n), t, None, seed=12345, fixed=True, formant_label=lab,
                                               opening=lab == "formants")
    flat = consolidate("data/synth/train_v2")
    n = json.loads((flat / "done").read_text())["n_clips"]
    out["v2:formants"] = SynthCrops(flat, split_ids(n)[1][:VAL_CLIPS], 400, None, seed=12345, fixed=True)
    for d in ("data/resynth/vocalset_val_d16k", "data/resynth/zeroth_korean_val_d16k"):
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.sort(np.random.default_rng(0).permutation(n)[:VAL_CLIPS])
        out[f"{Path(d).name}:formants"] = SynthCrops(rflat, ids, 400, None, seed=12345, fixed=True)
    return out


def score(run, ck, data):
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.bins import FORMANT_AXIS, decode
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import collate

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ck, dev)
    vq = TorchVQT(dev)
    res = {}
    for name, ds in data.items():
        rows = {"pred": [], "ref": [], "son": [], "op": [], "op_ref": [], "rd": [], "item": []}
        k = 0
        with torch.inference_mode():
            for audio, lab in DataLoader(ds, BATCH, num_workers=4, collate_fn=collate):
                audio = audio.to(dev)
                t = lab["f0"].shape[1]
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    out = model(vq.log_power_with_context(audio, t))
                fm, _ = decode(out["formant_logits"].float().softmax(-1).cpu().numpy(), FORMANT_AXIS)
                rows["pred"].append(fm.reshape(-1, fm.shape[-1]))
                rows["ref"].append(lab["formants"].numpy().reshape(-1, fm.shape[-1]))
                rows["son"].append(lab["sonorant"].numpy().reshape(-1))
                rows["item"].append(np.repeat(np.arange(k, k + len(audio)), t))
                k += len(audio)
                if "glottal_opening" in lab:
                    rows["op_ref"].append(lab["glottal_opening"].numpy().reshape(-1))
                    rows["rd"].append(lab["rd"].numpy().reshape(-1))  # amendment 2026-11-04: the opening beyond Rd
                    rows["op"].append(out["opening"].float().cpu().numpy().reshape(-1) if "opening" in out
                                      else np.full(len(audio) * t, np.nan, np.float32))
        for key, v in rows.items():
            if v:
                res[f"{name}|{key}"] = np.concatenate(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{run}.npz", **res)


if __name__ == "__main__":
    data = sets()
    for a in sys.argv[1:]:
        run, ck = a.split("=", 1) if "=" in a else (a, f"runs/{a}/ckpt_last.pt")
        if not (OUT / f"{run}.npz").exists():
            score(run, ck, data)
            print(f"scored {run}", flush=True)
