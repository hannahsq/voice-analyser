"""D-033 part 4: a larger cached training set for heads, Parakeet layers 3-6 only (fp16, 80 ms frames), with the labels
the trainer uses. Teacher v3's training mixture (train_v2 training clips + 30 % resynthesised VocalSet / Zeroth-Korean,
its saved channel model), N fixed 4 s crops (seed 999, so none repeats probe.py's / cache.py's fitting crops), loaded by
8 workers. Writes <out>/emb.npy (N, E, 4, 1024) and labels (N, 400, ...) as memmaps, then "done".
Run from the main checkout: VA_DEVICE=cuda python <worktree>/scripts/parakeet/cache_heads.py <out dir> [N]"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, collate, consolidate, split_ids  # noqa: E402

LAYERS = (3, 4, 5, 6)
CROP, BATCH = 400, 16

if __name__ == "__main__":
    OUT = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    flat = Path(cfg3["train"]["data"]) / "flat"
    train_ids, _ = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=999, epoch_size=N, fixed=True)], []
    for d in cfg3["train"]["resynth"]:
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=1000 + len(sizes), epoch_size=N, fixed=True,
                                tract_left_weight=tuple(cfg3["train"]["tract_left_weight"])))
        sizes.append(n)
    frac = cfg3["train"]["resynth_frac"]
    ds = MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], N, seed=999, fixed=True)
    loader = DataLoader(ds, BATCH, num_workers=8, collate_fn=collate, prefetch_factor=4)
    fe, enc = load.feature_extractor(), load.encoder(dev)
    enc.layers = enc.layers[:max(LAYERS)]
    emb = labs = None
    t0, i = time.time(), 0
    for audio, lab in loader:
        f = fe(list(audio.numpy()), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states
        h = torch.stack([h[l] for l in LAYERS], 2).half().cpu().numpy()  # (B, E, 4, 1024)
        if emb is None:
            emb = np.lib.format.open_memmap(OUT / "emb.npy", "w+", np.float16, (N, h.shape[1], len(LAYERS), 1024))
            labs = {k: np.lib.format.open_memmap(OUT / f"{k}.npy", "w+", np.float32 if v.dtype != torch.bool else np.bool_,
                                                 (N, *v.shape[1:])) for k, v in lab.items()}
        b = len(h)
        emb[i:i + b] = h
        for k, v in lab.items():
            labs[k][i:i + b] = v.numpy()
        i += b
        if (i // BATCH) % 250 == 0:
            print(f"{time.strftime('%H:%M:%S')} {i}/{N} ({i / (time.time() - t0):.1f} crops/s)", flush=True)
    emb.flush()
    for v in labs.values():
        v.flush()
    (OUT / "done").write_text(json.dumps({"n": i, "layers": LAYERS, "seed": 999, "crop_frames": CROP}))
    print(f"done: {i} crops in {time.time() - t0:.0f} s", flush=True)
