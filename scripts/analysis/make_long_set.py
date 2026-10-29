"""D-051's long articulation set: train_v2's settings (the current SynthConfig defaults reproduce them) with 8-16 s
clips, every clip a tube (the movement is the tube's tier III), and sustained larynx-height and lip-protrusion
trajectories. Resumable (write_set skips finished shards).
Args: out dir [n clips, default 25000] [seed, default 5051] [workers, default 12]."""
import dataclasses
import sys
from pathlib import Path

from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set

if __name__ == "__main__":
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 25_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5051
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    cfg = dataclasses.replace(SynthConfig(), duration_s=(8.0, 16.0), p_tube=1.0, articulation="sustained")
    write_set(out, n, cfg, seed=seed, workers=workers, extra_index={"purpose": "D-051 long articulation clips"},
              log=lambda m: print(m, flush=True))
