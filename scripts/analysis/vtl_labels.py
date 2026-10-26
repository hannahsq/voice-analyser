"""D-021 VTL labels for a synthetic set whose shards are gone (roadmap Next 1): per clip, the anatomical tract length,
whether it was drawn from the clip's f0, and tube or cascade, recovered with `sampler.clip_tract` (no rendering) and
checked against the stored audio's frame count. Writes beside `flat/`: clip_length_cm.npy (float32), clip_length_from_f0.npy
and clip_tube.npy (bool), one value per clip, and label_length_cm.npy (float32, per frame, the clip's value on every frame).
Args: set root (e.g. data/synth/train_v2) [workers, default 4]."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vocal_analysis.synth.sampler import SynthConfig, clip_tract

ROOT = Path(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
IX = json.loads((ROOT / "index.json").read_text())
CFG = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in IX["synth_config"].items()})


def chunk(ids):
    return [clip_tract(int(i), CFG, IX["seed"]) for i in ids]


if __name__ == "__main__":
    flat = ROOT / "flat"
    fo = np.load(flat / "frame_offsets.npy")
    n = len(fo) - 1
    assert n == IX["n_clips"], (n, IX["n_clips"])
    parts = np.array_split(np.arange(n), max(n // 500, 1))
    out = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, r in enumerate(ex.map(chunk, parts)):
            out.extend(r)
            if (k + 1) % 20 == 0:
                print(f"  {len(out)}/{n}", flush=True)
    length = np.array([o[0] for o in out], np.float32)
    from_f0 = np.array([o[1] for o in out], bool)
    tube = np.array([o[2] for o in out], bool)
    np.save(flat / "clip_length_cm.npy", length)
    np.save(flat / "clip_length_from_f0.npy", from_f0)
    np.save(flat / "clip_tube.npy", tube)
    np.save(flat / "label_length_cm.npy", np.repeat(length, np.diff(fo)))
    print(f"{n} clips: length {length.min():.2f}-{length.max():.2f} cm (median {np.median(length):.2f}); "
          f"from f0 {from_f0.mean():.3f}; tube {tube.mean():.3f}; tube and not from f0 {(tube & ~from_f0).mean():.3f}")
