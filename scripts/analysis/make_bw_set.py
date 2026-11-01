"""D-065 arm "bw": train_v2 re-rendered with bandwidths decoupled from the formants. Same seed and settings (the
current SynthConfig defaults reproduce train_v2's labels), so every clip is train_v2's with only its bandwidths changed:
a per-formant log-uniform factor of 0.4-2.5 on the cascade's Hawks & Miller prior (instead of 0.6-1.6) and on the
tube's own bandwidths (none before). Resumable (write_set skips finished shards).
Args: out dir [workers, default 6]."""
import dataclasses
import sys
from pathlib import Path

from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set

N_CLIPS, SEED, FACTOR = 100_000, 2, (0.4, 2.5)  # train_v2's size and seed

if __name__ == "__main__":
    out = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = dataclasses.replace(SynthConfig(), bw_factor=FACTOR, tube_bw_factor=FACTOR)
    write_set(out, N_CLIPS, cfg, seed=SEED, workers=workers,
              extra_index={"purpose": "D-065 arm bw: train_v2 with bandwidths decoupled from the formants"},
              log=lambda m: print(m, flush=True))
