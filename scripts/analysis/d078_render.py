"""D-078: render train_v2k (train_v2's settings, every clip a tube clip, coupled at scale 1, with the cycle-mean acoustic
label) and its held-out set, then consolidate both.

    python scripts/analysis/d078_render.py [n_train] [n_val] [workers]
"""

import json
import sys
from dataclasses import fields, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TRAIN, VAL = Path("data/synth/train_v2k"), Path("data/synth/val_v2k")
SEED_TRAIN, SEED_VAL = 78, 7801


def config():
    from vocal_analysis.synth.sampler import SynthConfig

    idx = json.loads(Path("docs/synth_sets/train_v2.index.json").read_text())
    names = {f.name for f in fields(SynthConfig)}
    cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in idx["synth_config"].items() if k in names})
    return replace(cfg, p_tube=1.0, p_coupling=1.0, coupling_scale=(1.0, 1.0), acoustic_labels=True)


if __name__ == "__main__":
    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import consolidate

    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    cfg = config()
    for out, n, seed in ((TRAIN, n_train, SEED_TRAIN), (VAL, n_val, SEED_VAL)):
        if not (out / "flat" / "done").exists():
            write_set(out, n, cfg, seed=seed, workers=workers, extra_index={"entry": "D-078"})
            consolidate(out)
        print(f"{out}: done", flush=True)
