"""D-078: an F1-F4 head (D-054's mlp) on teacher v3.1's frozen trunk, trained on train_v2k alone (no resynthesised
clips) against one label: "closed" (the closed-glottis poles) or "acoustic" (the cycle-mean pole). D-054's recipe.

    python scripts/analysis/d078_heads.py <label> <seed> [steps]   (label: closed, acoustic; VA_DEVICE=cuda)
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LABELS = {"closed": "formants", "acoustic": "formants_acoustic"}
DATA = "data/synth/train_v2k"

if __name__ == "__main__":
    import d054_readout as D

    from vocal_analysis.train.teacher import train

    label, seed = sys.argv[1], int(sys.argv[2])
    cfg, mcfg = D.configs("mlp", seed, int(sys.argv[3]) if len(sys.argv) > 3 else D.STEPS)
    cfg = replace(cfg, out=f"runs/d078_{label}_s{seed}", data=DATA, resynth=(), resynth_val=(), formant_label=LABELS[label])
    train(cfg, mcfg)
