"""D-114 pre-check, part 2: where on real audio does D-109's loss want the F1 stretch? The stretch scan of
`d114_stretch_scan.py` (same models, crops, grid and loss) with per-frame losses, summed within frame strata:
F1 motion (the branch-point model's own F1, |ln F1(t+3) - ln F1(t-3)|: steady < 0.03, moving > 0.08), the model's F1
band (< 450, 450-600, >= 600 Hz) and f0 band (< 150, 150-250, >= 250 Hz). Per stratum: the minimising s, with a
bootstrap over crops; channel 0 (D-109) and 3 (plus a 3-term curve per crop). Descriptive.

    python scripts/analysis/d114_stretch_strata.py > reports/analysis/d114/stretch_strata.txt   (VA_DEVICE=cuda)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d114_stretch_scan import N_BATCH, S_GRID, SEED, argmin_parabola  # noqa: E402


def per_frame_scan(model, vq, batches, dev):
    import torch

    from vocal_analysis.train import reconstruction as R

    rows = {0: [], 3: []}
    meta = []
    crop0 = 0
    for a, lab in batches:
        a = a.to(dev)
        with torch.no_grad():
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                o = model(vq.log_power_with_context(a, lab["f0"].shape[1]))
            L = {k: v.to(dev) for k, v in lab.items()}
            F = R.decode_formants(o["formant_logits"])[..., :4].float()
            B = torch.exp(o["logb_mean"].float())[..., :4]
            lf1 = torch.log(F[..., 0])
            mot = torch.full_like(lf1, float("nan"))
            mot[:, 3:-3] = (lf1[:, 6:] - lf1[:, :-6]).abs()
            for ch in (0, 3):
                per = [R.recon_loss_fb(torch.cat([500.0 * (F[..., :1] / 500.0) ** (1.0 + float(s)), F[..., 1:]], -1), B, L,
                                       fmax=4000.0, hpc=True, channel=ch, per_frame=True) for s in S_GRID]
                rows[ch].append(torch.stack(per, -1).reshape(-1, len(S_GRID)).cpu().numpy())
            nb, T = F.shape[:2]
            meta.append(np.column_stack([np.repeat(np.arange(crop0, crop0 + nb), T), F[..., 0].reshape(-1).cpu().numpy(),
                                         mot.reshape(-1).cpu().numpy(), L["f0"].reshape(-1).cpu().numpy()]))
            crop0 += nb
    return {ch: np.concatenate(v) for ch, v in rows.items()}, np.concatenate(meta)


def report(rows, meta, rng):
    crop, f1, mot, f0 = meta.T
    strata = {"all": np.ones(len(f1), bool), "F1 steady (< 3 % over 60 ms)": mot < 0.03, "F1 moving (> 8 %)": mot > 0.08,
              "F1 < 450 Hz": f1 < 450, "F1 450-600 Hz": (f1 >= 450) & (f1 < 600), "F1 >= 600 Hz": f1 >= 600,
              "f0 < 150 Hz": f0 < 150, "f0 150-250 Hz": (f0 >= 150) & (f0 < 250), "f0 >= 250 Hz": f0 >= 250}
    out = []
    for name, sel in strata.items():
        cells = []
        for ch, arr in rows.items():
            ok = sel & np.all(np.isfinite(arr), 1)
            if ok.sum() < 200:
                cells.append(f"ch{ch}: n/a")
                continue
            a, c = arr[ok], crop[ok].astype(int)
            smin = argmin_parabola(S_GRID, a.mean(0))
            uc = np.unique(c)
            sums = np.stack([a[c == k].sum(0) for k in uc]); cnt = np.array([(c == k).sum() for k in uc])
            bs = []
            for _ in range(500):
                pick = rng.integers(0, len(uc), len(uc))
                bs.append(argmin_parabola(S_GRID, sums[pick].sum(0) / cnt[pick].sum()))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"ch{ch} {smin:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out.append(f"    {name:28s} n {int((sel & np.all(np.isfinite(rows[0]), 1)).sum()):6d}  " + "  ".join(cells))
    return out


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.recon_data import ReconCrops, collate_recon

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    rng = np.random.default_rng(SEED)
    print("D-114 pre-check, part 2: the minimising F1 stretch s by frame stratum (ch0 = D-109's loss, ch3 = plus a curve per crop); "
          "95 % bootstrap over crops")
    data = {name: list(DataLoader(ReconCrops([store], 200, seed=SEED, epoch_size=16 * N_BATCH), 16, num_workers=2,
                                  collate_fn=collate_recon))
            for name, store in (("VocalSet", "cache/recon/vocalset"), ("Zeroth-Korean", "cache/recon/zeroth"))}
    for seed in (1, 2):
        model, _ = load_teacher(f"runs/d040_5b_s{seed}/ckpt_016000.pt", dev)
        for name, batches in data.items():
            rows, meta = per_frame_scan(model, vq, batches, dev)
            print(f"\n  5b seed {seed} branch point, {name}")
            print("\n".join(report(rows, meta, rng)))
        del model
