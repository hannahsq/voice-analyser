"""D-079: re-render D-078's train_v2k / val_v2k (same settings and seeds) with the glottal-opening label (<a> per
frame), as train_v2ko / val_v2ko, consolidate, and check the audio and the existing labels equal D-078's render.

    python scripts/analysis/d079_render.py [workers]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SETS = ((Path("data/synth/train_v2ko"), Path("data/synth/train_v2k"), 6000, 78),
        (Path("data/synth/val_v2ko"), Path("data/synth/val_v2k"), 1000, 7801))

if __name__ == "__main__":
    import d078_render

    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import consolidate

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cfg = d078_render.config()
    for out, old, n, seed in SETS:
        if not (out / "flat" / "done").exists():
            write_set(out, n, cfg, seed=seed, workers=workers, extra_index={"entry": "D-079"})
            consolidate(out)
        new_f, old_f = out / "flat", old / "flat"
        for f in sorted(p.name for p in old_f.glob("*.npy")):
            a, b = np.load(new_f / f, mmap_mode="r"), np.load(old_f / f, mmap_mode="r")
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                raise SystemExit(f"{out}: {f} differs from {old}")
        op = np.load(new_f / "label_glottal_opening.npy")
        print(f"{out}: done; matches {old} on every file; opening label {op.shape}, NaN {np.isnan(op).mean():.1%}", flush=True)
