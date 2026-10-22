"""Long-clip validation set for D-028: data/synth/val_long_v2, 150 clips of 8-16 s with train_v2's config (from its
index.json, only the duration changed), seed 2718 (used by no other set), consolidated to flat arrays for
long_test.py. CPU only (2 workers)."""
import json
from dataclasses import replace
from vocal_analysis.synth.obs_store import config_from_index
from vocal_analysis.synth.shards import write_set
from vocal_analysis.train.data import consolidate
if __name__ == "__main__":
    cfg = replace(config_from_index(json.load(open("data/synth/train_v2/index.json"))), duration_s=(8.0, 16.0))
    root = write_set("data/synth/val_long_v2", 150, cfg, seed=2718, shard_size=50, workers=2, log=print)
    consolidate(root, log=print)
