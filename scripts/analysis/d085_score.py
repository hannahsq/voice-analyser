"""D-085 scoring (GPU) for the measures the per-run scoring (score_run.sh) and the training log don't give, the same for
every model (5b's two seeds and both arms):
- *long*: synthetic validation on `train_long_v3u`'s held-out clips (the run's `eval_long`: split_ids' val part, the first
  512, fixed crops, seed 12345, the run's validation channel) at 1600 frames, and the same clips at 400 frames, so the
  two lengths differ in context only (registration: "evaluation at both lengths"). The trainer's `evaluate()`.
- *v2kn*: Q1's glottal slope inputs. Every frame of D-082's aspiration-free coupled validation clips (`cache/d082/
  val_render.npz`, 1000 clips, seed 7801) read by the model as D-082 read features (whole clips padded to whole
  seconds): the closed F1, and the acoustic F1 where the model has the acoustic heads.

    python scripts/analysis/d085_score.py name=ckpt ... [--out DIR] [--clips N]

Writes <out>/long_<name>.json and <out>/v2kn_<name>.npz (default out cache/d085); each skips if it exists. `--clips`
limits both sets (a code check only, with --out elsewhere)."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LONG_SET = "data/synth/train_long_v3u"
V2KN = Path("cache/d082/val_render.npz")
SR, HOP = 16000, 160
LENGTHS = {"long1600": 1600, "long400": 400}
BATCH_FRAMES = 32 * 400  # the trainer's batch x crop


def val_channel():
    """D-085's validation channel (both arms: the default channel, D-065's gain spread removed for validation); 5b's
    runs were started with the same one (checked 2026-11-11), so the crops are the same clips and conditions for all."""
    from vocal_analysis.synth.channel import ChannelConfig
    from vocal_analysis.train.teacher import _saved_channel

    return replace(_saved_channel(Path("runs/d085_base")), gain_db=ChannelConfig().gain_db)


def score_long(model, device, n_clips, out):
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.train.data import SynthCrops, collate, consolidate, split_ids
    from vocal_analysis.train.teacher import cache_batches, evaluate

    flat = consolidate(LONG_SET)
    n = json.loads((flat / "done").read_text())["n_clips"]
    ids = split_ids(n)[1][:n_clips]
    vq, ch, res = TorchVQT(device), val_channel(), {"clips": int(len(ids))}
    for key, t in LENGTHS.items():
        ds = SynthCrops(flat, ids, t, ch, seed=12345, fixed=True, vtl=True, vtl_inst=True, acoustic=True)
        batches = cache_batches(DataLoader(ds, max(1, BATCH_FRAMES // t), num_workers=0, collate_fn=collate), vq, device)
        res[key] = evaluate(model, batches, device)
        del batches
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"  {key}: F1-F4 MAE {[round(res[key][f'F{k}_mae_hz'], 1) for k in range(1, 5)]}, "
              f"aVTL {res[key].get('vtl_mae_pct', float('nan')):.2f} %", flush=True)
    out.write_text(json.dumps(res, indent=1))


def score_v2kn(model, device, n_clips, out):
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import decode_outputs

    z = np.load(V2KN)
    audio, a_off, f_off = z["audio"], z["a_off"], z["f_off"]
    n = min(len(a_off) - 1, n_clips)
    vq = TorchVQT(device)
    rows, f1, f1a = [], [], []
    with torch.inference_mode():
        for c in range(n):
            x = audio[a_off[c]:a_off[c + 1]]
            pad = (-len(x)) % SR
            xx = torch.from_numpy(np.pad(x, (0, pad)))[None].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                o = model(vq.log_power_with_context(xx, (len(x) + pad) // HOP))
            o = {k: v[0].float().cpu().numpy() for k, v in o.items()}
            t = min(int(f_off[c + 1] - f_off[c]), o["formant_logits"].shape[0])
            f1.append(decode_outputs(o)["formants"][:t, 0])
            if "formant_logits_acoustic" in o:
                f1a.append(decode_outputs({**o, "formant_logits": o["formant_logits_acoustic"]})["formants"][:t, 0])
            rows.append(np.arange(f_off[c], f_off[c] + t))
    np.savez(out, rows=np.concatenate(rows), f1=np.concatenate(f1).astype(np.float32),
             f1_acoustic=np.concatenate(f1a).astype(np.float32) if f1a else np.zeros(0, np.float32))
    print(f"  v2kn: {n} clips", flush=True)


def main():
    import torch

    from vocal_analysis.models.infer import load_teacher

    args = sys.argv[1:]
    out_dir, n_clips = Path("cache/d085"), 512
    if "--out" in args:
        i = args.index("--out")
        out_dir = Path(args[i + 1])
        del args[i:i + 2]
    lim = None
    if "--clips" in args:
        i = args.index("--clips")
        lim = int(args[i + 1])
        del args[i:i + 2]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    for a in args:
        name, ck = a.split("=", 1)
        lo, vo = out_dir / f"long_{name}.json", out_dir / f"v2kn_{name}.npz"
        if lo.exists() and vo.exists():
            continue
        print(name, flush=True)
        model, _ = load_teacher(ck, device)
        if not lo.exists():
            score_long(model, device, lim or n_clips, lo)
        if not vo.exists():
            score_v2kn(model, device, lim or 10**9, vo)
        del model


if __name__ == "__main__":
    main()
