"""D-051 instantaneous tract-length labels for a synthetic set: per frame, the tube's length at each target (anatomical
+ larynx + lips) with the formants' transitions, recovered with `sampler.clip_length_inst` (no rendering) and checked
against the stored frame count. Cascade clips get their anatomical length. Writes label_length_inst_cm.npy (float32)
beside `flat/`; train.data reads it on the clips that carry "vtl" (vtl_labels.py's files must exist too).
Args: set root (e.g. data/synth/train_v2) [workers, default 4]."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vocal_analysis.synth.sampler import SynthConfig, clip_length_inst

ROOT = Path(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
IX = json.loads((ROOT / "index.json").read_text())
CFG = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in IX["synth_config"].items()})


def chunk(ids):
    return [clip_length_inst(int(i), CFG, IX["seed"]).astype(np.float32) for i in ids]


if __name__ == "__main__":
    flat = ROOT / "flat"
    fo = np.load(flat / "frame_offsets.npy")
    n = len(fo) - 1
    assert n == IX["n_clips"], (n, IX["n_clips"])
    parts = np.array_split(np.arange(n), max(n // 200, 1))
    out = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, r in enumerate(ex.map(chunk, parts)):
            out.extend(r)
            if (k + 1) % 50 == 0:
                print(f"  {len(out)}/{n}", flush=True)
    bad = [i for i, (o, a, b) in enumerate(zip(out, fo[:-1], fo[1:])) if len(o) != b - a]
    assert not bad, f"frame counts differ on {len(bad)} clips, e.g. {bad[:5]}"
    lab = np.concatenate(out)
    anat = np.load(flat / "label_length_cm.npy", mmap_mode="r")
    np.save(flat / "label_length_inst_cm.npy", lab)
    d = lab - anat
    print(f"{n} clips: instantaneous - anatomical {np.percentile(d, 1):+.2f} to {np.percentile(d, 99):+.2f} cm (1-99 %); "
          f"frames moved > 0.1 cm {np.mean(np.abs(d) > 0.1):.3f}")
