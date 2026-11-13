"""D-114 pre-check, part 3: does the LF source run out of range on real audio? At the branch point (5b seed 2,
ckpt_016000), D-109's instrument (model F1-F4 and B1-B4, higher-pole correction, harmonics to 4000 Hz): each frame's
best Rd on the search grid (0.3-2.7), the share of frames at either end, and the residual at harmonics 1-3 after each
frame's best Rd and level, by f0 band; real recon-store crops against synthetic validation crops (targets from their
audio at the label f0). Descriptive.

    python scripts/analysis/d114_source_check.py > reports/analysis/d114/source_check.txt   (VA_DEVICE=cuda)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d114_stretch_scan import N_BATCH, SEED, synth_targets  # noqa: E402


def collect(model, vq, batches, dev, synth):
    import torch

    from vocal_analysis.train import reconstruction as R

    rds, res3, f0s = [], [], []
    for a, lab in batches:
        a = a.to(dev)
        with torch.no_grad():
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                o = model(vq.log_power_with_context(a, lab["f0"].shape[1]))
            L = synth_targets(a, lab) if synth else lab
            L = {k: v.to(dev) for k, v in L.items()}
            F = R.decode_formants(o["formant_logits"])[..., :4].float()
            B = torch.exp(o["logb_mean"].float())[..., :4]
            tgt, ok, f0, m = L["db"].float(), L["ok"].bool(), L["f0"].float(), L["mask"].bool()
            K = tgt.shape[-1]
            k = torch.arange(1, K + 1, device=dev, dtype=f0.dtype)
            ok = ok & (f0[..., None] * k <= 4000.0)
            m = m & (ok.sum(-1) >= 4)
            uf, ub = R.upper_poles(F[..., 3], f0)
            P, Bw = torch.cat([F, uf], -1), torch.cat([B, ub], -1)
            f = f0[..., None] * k
            w = (ok & m[..., None]).float()
            rend = R.render_db(torch.where(ok, f, torch.full_like(f, 1000.0)), P, Bw, None, None, False)
            best, best_rd, best_res = None, None, None
            for r in R.RD_SEARCH:
                res = R.profile_out((tgt - rend - R.lf_source_db(torch.full_like(f0, float(r)), K)) * w, f, w, "offset") * w
                lo = (torch.nn.functional.huber_loss(res, torch.zeros_like(res), reduction="none", delta=6.0) * w).sum(-1) / w.sum(-1).clamp_min(1)
                if best is None:
                    best, best_rd, best_res = lo, torch.full_like(lo, float(r)), res
                else:
                    better = lo < best
                    best = torch.where(better, lo, best)
                    best_rd = torch.where(better, torch.full_like(lo, float(r)), best_rd)
                    best_res = torch.where(better[..., None], res, best_res)
            sel = m & ok[..., :3].all(-1)
            rds.append(best_rd[sel].cpu().numpy()); res3.append(best_res[sel][:, :3].cpu().numpy()); f0s.append(f0[sel].cpu().numpy())
    return np.concatenate(rds), np.concatenate(res3), np.concatenate(f0s)


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train import reconstruction as R
    from vocal_analysis.train.data import SynthCrops, collate, split_ids
    from vocal_analysis.train.recon_data import ReconCrops, collate_recon

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    model, _ = load_teacher("runs/d040_5b_s2/ckpt_016000.pt", dev)
    lo_rd, hi_rd = float(R.RD_SEARCH[0]), float(R.RD_SEARCH[-1])
    print(f"D-114 pre-check, part 3: each frame's best Rd (grid {lo_rd}-{hi_rd}) and the residual (target - render - source, "
          "after the frame's level) at harmonics 1-3, dB; branch point 5b seed 2")
    sets = {name: (list(DataLoader(ReconCrops([store], 200, seed=SEED, epoch_size=16 * N_BATCH), 16, num_workers=2,
                                   collate_fn=collate_recon)), False)
            for name, store in (("VocalSet", "cache/recon/vocalset"), ("Zeroth-Korean", "cache/recon/zeroth"))}
    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    syn = SynthCrops("data/synth/train_v2/flat", split_ids(idx["n_clips"])[1], n_frames=200, seed=SEED, epoch_size=16 * N_BATCH)
    sets["train_v2 validation (synthetic)"] = (list(DataLoader(syn, 16, num_workers=4, collate_fn=collate)), True)
    for name, (batches, synth) in sets.items():
        rd, res, f0 = collect(model, vq, batches, dev, synth)
        print(f"\n  {name}: {len(rd)} frames")
        for lab, s in (("all", np.ones(len(rd), bool)), ("f0 < 150", f0 < 150), ("f0 150-250", (f0 >= 150) & (f0 < 250)), ("f0 >= 250", f0 >= 250)):
            if s.sum() < 100:
                continue
            q = np.percentile(rd[s], [10, 50, 90])
            print(f"    {lab:11s} n {int(s.sum()):6d}  Rd 10/50/90 {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}; at {hi_rd} {np.mean(rd[s] >= hi_rd - 1e-6):5.1%}, "
                  f"at {lo_rd} {np.mean(rd[s] <= lo_rd + 1e-6):5.1%}; residual H1 / H2 / H3 median "
                  + " / ".join(f"{np.median(res[s, j]):+.1f}" for j in range(3))
                  + f"; H1 at Rd {hi_rd} frames {np.median(res[s & (rd >= hi_rd - 1e-6), 0]) if (s & (rd >= hi_rd - 1e-6)).sum() > 20 else float('nan'):+.1f}")
