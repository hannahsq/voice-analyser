"""D-114 pre-check: does D-109's reconstruction loss prefer a stretched F1 on real audio, and does profiling a smooth
curve per crop (`reconstruction.crop_channel_out`) take that preference away? Descriptive, no training.

The branch-point models (d040_5b_s{1,2}/ckpt_016000.pt, where D-109 started) run on recon-store crops (VocalSet,
Zeroth; 12 batches of 16 x 200 frames each) and, as the control, on train_v2 validation crops with targets from their
own audio at the label f0. Their F1 is stretched about 500 Hz, F1' = 500 (F1 / 500)^(1 + s), for s on a grid (F2-F4
and every bandwidth unchanged), and D-109's loss (`recon_loss_fb`, fmax 4000, higher-pole correction) is computed with
the level profiled per frame (channel 0, D-109 as run) and with a 3-term curve per crop as well (channel 3).
D-109's trained shift on PTDB-TUG women (+22 Hz above 600 Hz, -6 below 450, D-110) is about s = +0.08.
Reported: each variant's loss against s (relative to s = 0), the minimising s (parabola through the grid's lowest
three points) per corpus and seed, and a bootstrap over batches.

    python scripts/analysis/d114_stretch_scan.py > reports/analysis/d114/stretch_scan.txt   (VA_DEVICE=cuda)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

S_GRID = np.round(np.arange(-0.16, 0.16 + 1e-9, 0.02), 2)
N_BATCH, SEED = 12, 1114
VARIANTS = {"offset (D-109)": (0, "offset"), "crop curve": (3, "offset")}
if os.environ.get("D114_TILT"):  # part 4: the per-frame slope, beside the two above
    VARIANTS["per-frame slope"] = (0, "tilt")
if os.environ.get("D114_NOH1"):  # part 5: H1 left out of the loss, alone and with the crop curve
    VARIANTS = {"offset (D-109)": (0, "offset"), "no H1": (0, "offset", True), "no H1 + crop curve": (3, "offset", True)}


def argmin_parabola(s, y):
    i = int(np.clip(np.argmin(y), 1, len(s) - 2))
    a, b, _ = np.polyfit(s[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(np.clip(-b / (2 * a), s[0], s[-1])) if a > 0 else float(s[np.argmin(y)])


def synth_targets(a, lab):
    import torch

    from vocal_analysis.features.vqt import N_MAX
    from vocal_analysis.train import reconstruction as R

    tg, ok = [], []
    for b in range(a.shape[0]):
        x = a[b].float().cpu().numpy()
        f0 = lab["f0"][b].numpy()
        v = lab["voiced"][b].numpy().astype(bool) & np.isfinite(f0)
        pad = np.concatenate([x[N_MAX // 2:], np.zeros(N_MAX // 2, np.float32)])
        db, okk = R.harmonic_targets(pad, np.where(v, f0, np.nan), v, k_max=64, snr_min_db=10.0)
        T = len(f0)
        tg.append(np.nan_to_num(db[:T])), ok.append(okk[:T])
    m = lab["voiced"].bool() & lab["sonorant"].bool()
    L = {"db": torch.tensor(np.array(tg)), "ok": torch.tensor(np.array(ok)), "f0": torch.nan_to_num(lab["f0"]).float(), "mask": m}
    L["mask"] = L["mask"] & (L["ok"].sum(-1) >= 4)
    return L


def scan(model, vq, batches, dev, synth):
    import torch

    from vocal_analysis.train import reconstruction as R

    out = {v: [] for v in VARIANTS}  # variant -> (n_batch, n_s)
    for a, lab in batches:
        a = a.to(dev)
        with torch.no_grad():
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                o = model(vq.log_power_with_context(a, lab["f0"].shape[1]))
            L = synth_targets(a, lab) if synth else lab
            L = {k: v.to(dev) for k, v in L.items()}
            F = R.decode_formants(o["formant_logits"])[..., :4].float()
            B = torch.exp(o["logb_mean"].float())[..., :4]
            for v, spec in VARIANTS.items():
                ch, prof = spec[:2]
                LL = L
                if len(spec) > 2 and spec[2]:  # no H1: harmonic 1 out of every frame, frames re-gated on 4 valid harmonics
                    ok = L["ok"].clone()
                    ok[..., 0] = False
                    LL = {**L, "ok": ok, "mask": L["mask"] & (ok.sum(-1) >= 4)}
                row = []
                for s in S_GRID:
                    F1 = 500.0 * (F[..., :1] / 500.0) ** (1.0 + float(s))
                    row.append(float(R.recon_loss_fb(torch.cat([F1, F[..., 1:]], -1), B, LL, fmax=4000.0, hpc=True, channel=ch,
                                                     profile=prof)))
                out[v].append(row)
    return {k: np.array(v) for k, v in out.items()}


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import SynthCrops, collate, split_ids
    from vocal_analysis.train.recon_data import ReconCrops, collate_recon

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    rng = np.random.default_rng(SEED)
    print("D-114 pre-check: D-109's loss against an F1 stretch about 500 Hz, F1' = 500 (F1/500)^(1+s); "
          "loss relative to s = 0 (x1000); channel 0 = level per frame (D-109), 3 = plus a 3-term curve per crop")
    print("s grid: " + " ".join(f"{s:+.2f}" for s in S_GRID))
    data = {}
    for name, store in (("VocalSet", "cache/recon/vocalset"), ("Zeroth-Korean", "cache/recon/zeroth")):
        data[name] = (list(DataLoader(ReconCrops([store], 200, seed=SEED, epoch_size=16 * N_BATCH), 16, num_workers=2,
                                      collate_fn=collate_recon)), False)
    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    syn = SynthCrops("data/synth/train_v2/flat", split_ids(idx["n_clips"])[1], n_frames=200, seed=SEED, epoch_size=16 * N_BATCH)
    data["train_v2 validation (synthetic control)"] = (list(DataLoader(syn, 16, num_workers=4, collate_fn=collate)), True)
    for seed in (1, 2):
        model, _ = load_teacher(f"runs/d040_5b_s{seed}/ckpt_016000.pt", dev)
        for name, (batches, synth) in data.items():
            res = scan(model, vq, batches, dev, synth)
            print(f"\n5b seed {seed} branch point, {name}")
            for ch, arr in res.items():
                mean = arr.mean(0)
                i0 = list(S_GRID).index(0.0)
                rel = (mean - mean[i0]) * 1000
                smin = argmin_parabola(S_GRID, mean)
                bs = [argmin_parabola(S_GRID, arr[rng.integers(0, len(arr), len(arr))].mean(0)) for _ in range(1000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                curv = (mean[i0 + 2] + mean[i0 - 2] - 2 * mean[i0]) / (S_GRID[i0 + 2] - S_GRID[i0]) ** 2
                print(f"  {ch}: loss at s=0 {mean[i0]:.3f}; minimum at s = {smin:+.3f} [{lo:+.3f}, {hi:+.3f}]; "
                      f"curvature at 0 {curv:.1f} (relative {curv / mean[i0]:.2f})")
                print("    rel. loss x1000: " + " ".join(f"{v:+.0f}" for v in rel))
        del model
