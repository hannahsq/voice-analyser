"""D-056 phase 1: the anchored per-bin read-out ("film_residual": flat heads plus a zero-initialised FreqReadout) trained
alone on a 5b seed's frozen branch point (ckpt_016000: trunk and flat heads frozen), D-054's recipe: AdamW 1e-3,
weight decay 0.01, 1,000 warm-up steps, then 3,000 at the full rate. Phase 2 is a `va train teacher --branch-from` decay
branch of the same checkpoint that adds this read-out (chains/d056.sh).

    python scripts/analysis/d056_anchored.py <seed> [steps]   (VA_DEVICE=cuda)
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

STEPS = 4000
FREEZE = ("f0_net.", "f0_head.", "voicing_head.", "enc.", "sonorant_head.", "vtl_head.", "formant_heads.", "bw_head.")


def configs(seed: int, steps: int):
    import torch

    from vocal_analysis.models.teacher import TeacherConfig
    from vocal_analysis.train.teacher import TrainConfig

    parent = Path(f"runs/d040_5b_s{seed}")
    trunk = str(parent / "ckpt_016000.pt")
    saved = json.loads((parent / "config.json").read_text())["train"]
    keep = ("data", "n_frames", "crop_frames", "crop_long_frac", "batch", "resynth", "resynth_val", "resynth_frac",
            "tract_left_weight", "val_clips", "workers", "log_every")
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items() if k in keep}
    cfg = TrainConfig(**fields, out=f"runs/d056_p1_s{seed}", steps=steps, optimizer="adamw", lr=1e-3, weight_decay=0.01,
                      warmup=1000, schedule="wsd", comb_truth_steps=0, seed=seed, eval_every=2000, ckpt_every=2000,
                      init_from=trunk, freeze=FREEZE)
    trunk_cfg = TeacherConfig(**torch.load(trunk, map_location="cpu", weights_only=False)["teacher_config"])
    return cfg, replace(trunk_cfg, readout="film_residual")


if __name__ == "__main__":
    from vocal_analysis.train.teacher import train

    seed = int(sys.argv[1])
    train(*configs(seed, int(sys.argv[2]) if len(sys.argv) > 2 else STEPS))
