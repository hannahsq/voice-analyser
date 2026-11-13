"""D-109 weight rule: at the branch point (d040_5b_s{1,2} step 16000), the encoder's gradient norm from the formant term
(teacher_loss's formant term, recomputed with heatmap_ce, synthetic train_v2 crops, batch 32 x 200, no comb truth) and from the reconstruction
loss (model_recon_loss on the recon stores, batch 16 x 200, weight 1); medians over 12 batches each. The registered
weight is 0.25 x their ratio (seed mean).

    python scripts/analysis/d109_grad_ratio.py [fmax] [--hpc] > reports/analysis/d109/grad_ratio.txt   (GPU; VA_DEVICE=cuda)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

STORES = os.environ.get("VA_RECON_STORES", "cache/recon/vocalset,cache/recon/zeroth").split(",")  # D-114: VocalSet alone
N_BATCH = 12


def enc_norm(model, loss):
    import torch

    model.zero_grad(set_to_none=True)
    loss.backward()
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.enc.parameters() if p.grad is not None)))


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import SynthCrops, collate, split_ids
    from vocal_analysis.models.bins import FORMANT_AXIS
    from vocal_analysis.train.losses import heatmap_ce
    from vocal_analysis.train.recon_data import ReconCrops, collate_recon
    from vocal_analysis.train.reconstruction import model_recon_loss
    import json

    fmax = float(sys.argv[1]) if len(sys.argv) > 1 else None
    hpc = "--hpc" in sys.argv
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    syn = SynthCrops("data/synth/train_v2/flat", split_ids(idx["n_clips"])[0], n_frames=200, seed=109, epoch_size=32 * N_BATCH)
    rec = ReconCrops(STORES, 200, seed=109, epoch_size=16 * N_BATCH)
    sl = list(DataLoader(syn, 32, num_workers=4, collate_fn=collate))
    rl = list(DataLoader(rec, 16, num_workers=2, collate_fn=collate_recon))
    print(f"D-109 gradient ratio: encoder gradient norms at the branch point; {N_BATCH} batches each; recon fmax {fmax or 5000:g} Hz, higher-pole correction {hpc}")
    ratios = []
    for seed in (1, 2):
        model, _ = load_teacher(f"runs/d040_5b_s{seed}/ckpt_016000.pt", dev)
        model.train()
        gf, gr, lr_ = [], [], []
        for a, lab in sl:
            a, lab = a.to(dev), {k: v.to(dev) for k, v in lab.items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(vq.log_power_with_context(a, lab["f0"].shape[1]))
            fl = out["formant_logits"]  # teacher_loss's formant term (its parts come back detached)
            term = sum(heatmap_ce(fl[:, :, n], lab["formants"][..., n], lab["sonorant"], FORMANT_AXIS, lab.get("w_formant"))
                       for n in range(fl.shape[2])) / fl.shape[2]
            gf.append(enc_norm(model, term))
        for a, lab in rl:
            a, lab = a.to(dev), {k: v.to(dev) for k, v in lab.items()}
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(vq.log_power_with_context(a, lab["f0"].shape[1]))
            loss = model_recon_loss(out, lab, fmax=fmax, hpc=hpc)
            lr_.append(float(loss))
            gr.append(enc_norm(model, loss))
        r = np.median(gf) / np.median(gr)
        ratios.append(r)
        print(f"  s{seed}: formant term {np.median(gf):.4f} (IQR {np.percentile(gf, 25):.4f}-{np.percentile(gf, 75):.4f}); "
              f"recon {np.median(gr):.4f} (IQR {np.percentile(gr, 25):.4f}-{np.percentile(gr, 75):.4f}); recon loss {np.median(lr_):.3f}; "
              f"ratio {r:.3f}")
        del model
    print(f"weight = 0.25 x mean ratio = {0.25 * np.mean(ratios):.4f}")
