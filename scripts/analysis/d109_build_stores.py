"""D-109: the reconstruction-loss stores (`train.recon_data.build_store`): 1500 recordings from each training pool's corpus
(VocalSet and Zeroth-Korean training files; never an evaluation set), first 10 s each, SNR-gated harmonic targets from
the labeller's f0 and voicing. Skips a store whose index exists.

    python scripts/analysis/d109_build_stores.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STORES = (("data/resynth/vocalset_train_d16k", "cache/recon/vocalset"), ("data/resynth/zeroth_korean_train_d16k", "cache/recon/zeroth"))

if __name__ == "__main__":
    from vocal_analysis.train.recon_data import build_store

    for pool, out in STORES:
        if (Path(out) / "index.json").exists():
            print(f"{out}: exists", flush=True)
            continue
        build_store(pool, out, 1500, seed=0, workers=8, log=lambda m: print(m, flush=True))
