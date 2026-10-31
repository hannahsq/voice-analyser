"""D-054: frequency-resolved read-out on teacher v3.1's frozen trunk, against flat and MLP heads on the same trunk.

Each arm trains only its formant and bandwidth heads on `d040_5b_s2_b20k`'s frozen f0 network, encoder, sonorant and
VTL heads, with 5b's data mix and channel, AdamW with a cosine schedule (the frozen-head recipe of D-035 / D-037).

    python scripts/analysis/d054_readout.py <arm> <seed> [steps]   (arm: flat, mlp, film; VA_DEVICE=cuda)
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

TRUNK = "runs/d040_5b_s2_b20k/ckpt_last.pt"
PARENT = Path("runs/d040_5b_s2")
STEPS = 8000
FREEZE = ("f0_net.", "f0_head.", "voicing_head.", "enc.", "sonorant_head.", "vtl_head.")
HEADS = ("formant_heads.", "bw_head.", "readout.")


def configs(arm: str, seed: int, steps: int):
    import torch

    from vocal_analysis.models.teacher import TeacherConfig
    from vocal_analysis.train.teacher import TrainConfig

    saved = json.loads((PARENT / "config.json").read_text())["train"]
    keep = ("data", "n_frames", "crop_frames", "crop_long_frac", "batch", "resynth", "resynth_val", "resynth_frac",
            "tract_left_weight", "val_clips", "workers", "log_every")
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items() if k in keep}
    cfg = TrainConfig(**fields, out=f"runs/d054_{arm}_s{seed}", steps=steps, optimizer="adamw", lr=1e-3, weight_decay=0.01,
                      warmup=300, schedule="cosine", comb_truth_steps=0, seed=seed, eval_every=2000, ckpt_every=2000,
                      init_from=TRUNK, freeze=FREEZE, reinit=HEADS)
    trunk_cfg = TeacherConfig(**torch.load(TRUNK, map_location="cpu", weights_only=False)["teacher_config"])
    return cfg, replace(trunk_cfg, readout=arm)


if __name__ == "__main__":
    from vocal_analysis.train.teacher import train

    arm, seed = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else STEPS
    train(*configs(arm, seed, steps))
