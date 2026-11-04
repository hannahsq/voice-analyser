"""D-080: render train_v2kb, train_v2ko's settings and seed (every clip a tube clip, coupling at strength 1, the
opening label) at 20000 clips, consolidate, and check its first clips equal train_v2ko's.

    python scripts/analysis/d080_render.py [workers]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT, REF, N, SEED = Path("data/synth/train_v2kb"), Path("data/synth/train_v2ko"), 20000, 78

if __name__ == "__main__":
    import d078_render

    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import consolidate

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if not (OUT / "flat" / "done").exists():
        write_set(OUT, N, d078_render.config(), seed=SEED, workers=workers, extra_index={"entry": "D-080"})
        consolidate(OUT)
    a, b = OUT / "flat", REF / "flat"
    for name, off in (("audio", "audio_offsets"), *((p.stem, "frame_offsets") for p in b.glob("label_*.npy"))):
        oa, ob = np.load(a / f"{off}.npy"), np.load(b / f"{off}.npy")
        k = len(ob) - 1
        if not np.array_equal(oa[: k + 1], ob):
            raise SystemExit(f"{name}: offsets differ from {REF}")
        x, y = np.load(a / f"{name}.npy", mmap_mode="r")[: ob[-1]], np.load(b / f"{name}.npy", mmap_mode="r")
        if not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            raise SystemExit(f"{name}: first {k} clips differ from {REF}")
    print(f"{OUT}: done; its first clips equal {REF} on audio and every label", flush=True)
