"""Data loader alone (no model): audio s/s the training loader can deliver, as in teacher.train (batch 32, 2 s crops,
12 workers, train_v2's channel), with and without codecs. Run from the main repo."""
import json, sys, time
from dataclasses import replace
from pathlib import Path
from torch.utils.data import DataLoader
from vocal_analysis.train.data import SynthCrops, collate, consolidate, split_ids
from vocal_analysis.train.teacher import _saved_channel

def main():
    flat = consolidate("data/synth/train_v2", log=lambda m: None)
    ids = split_ids(json.loads((flat / "done").read_text())["n_clips"])[0]
    ch = _saved_channel(Path("runs/teacher_v2"))
    for name, c, w in (("codecs, 12 workers", ch, 12), ("no codecs, 12 workers", replace(ch, p_codec=0.0), 12),
                       ("codecs, 16 workers", ch, 16)):
        ds = SynthCrops(flat, ids, 200, c, seed=0, epoch_size=32 * 1000)
        it = iter(DataLoader(ds, 32, num_workers=w, collate_fn=collate, persistent_workers=True, prefetch_factor=4, drop_last=True))
        for _ in range(30):
            next(it)
        t = time.perf_counter()
        for _ in range(150):
            next(it)
        dt = time.perf_counter() - t
        print(f"{name}: {150 * 32 * 2.0 / dt:.0f} audio s/s ({150 / dt:.2f} batches/s)", flush=True)

if __name__ == "__main__":
    main()
