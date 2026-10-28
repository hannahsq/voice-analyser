"""Backfill the VTL head's validation curve (teacher.evaluate's vtl_* metrics) for runs that trained before it was
logged (D-040 amendment, 2026-10-28). Rebuilds the run's "eval" set exactly as training does (same clips, crops and
channel; the vtl label adds no random draws) and scores every saved checkpoint and its moving average.

    python scripts/analysis/vtl_val_backfill.py runs/d040_5b_s1 [more runs...]

Appends to <run>/val_vtl.jsonl ({"step", "eval" / "eval_ema": metrics}); checkpoints already there are skipped, so it
can be rerun as a run goes on.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _step(p: Path) -> int:
    return int(re.findall(r"\d+", p.stem)[0])


def main():
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.device import setup_device
    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import SynthCrops, collate, consolidate, split_ids
    from vocal_analysis.train.teacher import _saved_channel, cache_batches, evaluate

    device = setup_device()
    for run in map(Path, sys.argv[1:]):
        dest = run / "val_vtl.jsonl"
        done = {json.loads(line)["step"] for line in dest.read_text().splitlines()} if dest.exists() else set()
        ckpts = [p for p in sorted(run.glob("ckpt_[0-9]*.pt"), key=_step) if _step(p) not in done]
        if not ckpts:
            print(f"{run}: up to date")
            continue
        cfg = json.loads((run / "config.json").read_text())["train"]
        flat = consolidate(cfg["data"])
        n_clips = json.loads((flat / "done").read_text())["n_clips"]
        _, val_ids = split_ids(n_clips)
        val_ids = val_ids[: cfg["val_clips"]]
        model, _ = load_teacher(ckpts[0], device)
        ds = SynthCrops(flat, val_ids, cfg["n_frames"], _saved_channel(run), seed=12345, fixed=True,
                        observability=model.cfg.obs_head, vtl=True)
        batches = cache_batches(DataLoader(ds, cfg["batch"], num_workers=4, collate_fn=collate), TorchVQT(device), device)
        for p in ckpts:
            rec = {"step": _step(p)}
            for key, suffix in (("eval", ""), ("eval_ema", "#ema")):
                try:
                    model, _ = load_teacher(str(p) + suffix, device)
                except ValueError:  # no moving average in this checkpoint
                    continue
                if not model.cfg.vtl_head:
                    continue
                rec[key] = {k: v for k, v in evaluate(model, batches, device).items() if k.startswith("vtl_")}
            with dest.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(run.name, json.dumps(rec), flush=True)
        del batches
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
