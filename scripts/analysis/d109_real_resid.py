"""D-109 pre-check (descriptive): where the reconstruction loss's misfit sits on real audio at the branch point. v3.1's
seed-2 branch start (d040_5b_s2/ckpt_016000.pt), D-109's instrument (model F1-F4 and B1-B4, higher-pole correction,
harmonics to 4000 Hz, Rd searched per frame, level profiled). Per corpus (the recon stores) and, as the reference,
synthetic train_v2 crops with the training channel (targets from their audio at the label f0): the per-frame loss
(median, 90th percentile), the signed and absolute residual by harmonic frequency band (after each frame's best Rd and
level), and the per-frame loss with a level-and-slope profile instead.

    python scripts/analysis/d109_real_resid.py > reports/analysis/d109/real_resid.txt   (GPU; VA_DEVICE=cuda)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

N_BATCH, FMAX = 12, 4000.0
EDGES = np.array([0, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000.0])


def analyse(model, vq, batches, dev, synth):
    import torch

    from vocal_analysis.train import reconstruction as R

    per_all, per_tilt, rr, ff = [], [], [], []
    for a, lab in batches:
        a = a.to(dev)
        with torch.no_grad():
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out = model(vq.log_power_with_context(a, lab["f0"].shape[1]))
            if synth:  # targets from the crop's own audio at the label f0
                tg, ok = [], []
                for b in range(a.shape[0]):
                    x = a[b].float().cpu().numpy()
                    f0 = lab["f0"][b].numpy()
                    v = lab["voiced"][b].numpy().astype(bool) & np.isfinite(f0)
                    from vocal_analysis.features.vqt import N_MAX

                    # crop frame j is centred at N_MAX // 2 + j * HOP; harmonic_targets centres frame i at i * HOP
                    pad = np.concatenate([x[N_MAX // 2:], np.zeros(N_MAX // 2, np.float32)])
                    db, okk = R.harmonic_targets(pad, np.where(v, f0, np.nan), v, k_max=64, snr_min_db=10.0)
                    T = len(f0)
                    tg.append(np.nan_to_num(db[:T])), ok.append(okk[:T])
                m = lab["voiced"].bool() & lab["sonorant"].bool()
                L = {"db": torch.tensor(np.array(tg)), "ok": torch.tensor(np.array(ok)), "f0": torch.nan_to_num(lab["f0"]).float(), "mask": m}
                L["mask"] = L["mask"] & (L["ok"].sum(-1) >= 4)
            else:
                L = lab
            L = {k: v.to(dev) for k, v in L.items()}
            F = R.decode_formants(out["formant_logits"])[..., :4].float()
            B = torch.exp(out["logb_mean"].float())[..., :4]
            tgt, ok, f0 = L["db"].float(), L["ok"].bool(), L["f0"].float()
            K = tgt.shape[-1]
            k = torch.arange(1, K + 1, device=dev, dtype=f0.dtype)
            ok = ok & (f0[..., None] * k <= FMAX)
            m = L["mask"].bool() & (ok.sum(-1) >= 4)
            ok = ok | ((~m)[..., None] & (k <= 4))  # unscored frames: keep the slope profile's solve non-singular
            uf, ub = R.upper_poles(F[..., 3], f0)
            P, Bw = torch.cat([F, uf], -1), torch.cat([B, ub], -1)
            f = f0[..., None] * k
            w = ok.float()
            rend = R.render_db(torch.where(ok, f, torch.full_like(f, 1000.0)), P, Bw, radiation=False)
            best, best_res, best_t = None, None, None
            for r in R.RD_SEARCH:
                src = R.lf_source_db(torch.full_like(f0, float(r)), K)
                res = R.profile_out((tgt - rend - src) * w, f, w, "offset") * w
                lo = (torch.nn.functional.huber_loss(res, torch.zeros_like(res), reduction="none", delta=6.0) * w).sum(-1) / w.sum(-1).clamp_min(1)
                rt = R.profile_out((tgt - rend - src) * w, f, w, "tilt") * w
                lt = (torch.nn.functional.huber_loss(rt, torch.zeros_like(rt), reduction="none", delta=6.0) * w).sum(-1) / w.sum(-1).clamp_min(1)
                if best is None:
                    best, best_res, best_t = lo, res, lt
                else:
                    better = lo < best
                    best = torch.where(better, lo, best)
                    best_res = torch.where(better[..., None], res, best_res)
                    best_t = torch.minimum(best_t, lt)
            per_all.append(best[m].cpu().numpy())
            per_tilt.append(best_t[m].cpu().numpy())
            okm = ok & m[..., None]
            rr.append(best_res[okm].cpu().numpy())
            ff.append(f[okm].cpu().numpy())
    per, pt, rr, ff = map(np.concatenate, (per_all, per_tilt, rr, ff))
    lines = [f"    frames {len(per)}: per-frame loss median {np.median(per):.2f}, 90th pct {np.percentile(per, 90):.2f}; "
             f"with level and slope profiled: median {np.median(pt):.2f}"]
    cells = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (ff >= lo) & (ff < hi)
        if s.sum() >= 50:
            cells.append(f"{lo:.0f}-{hi:.0f} {np.median(rr[s]):+5.1f} / {np.median(np.abs(rr[s])):4.1f}")
    lines.append("    residual by harmonic frequency (Hz: median signed / median |.| dB): " + "; ".join(cells))
    return lines


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import SynthCrops, collate, split_ids
    from vocal_analysis.train.recon_data import ReconCrops, collate_recon

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    model, _ = load_teacher("runs/d040_5b_s2/ckpt_016000.pt", dev)
    print("D-109 pre-check: the reconstruction loss's misfit at the branch point (d040_5b_s2 step 16000); D-109's instrument "
          f"(F1-F4 + higher-pole correction, harmonics to {FMAX:g} Hz, Rd searched, level profiled); {N_BATCH} batches of 16 x 200 frames")
    for name, store in (("VocalSet", "cache/recon/vocalset"), ("Zeroth-Korean", "cache/recon/zeroth")):
        b = list(DataLoader(ReconCrops([store], 200, seed=1091, epoch_size=16 * N_BATCH), 16, num_workers=2, collate_fn=collate_recon))
        print(f"  {name} (real, labeller f0)")
        print("\n".join(analyse(model, vq, b, dev, False)))
    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    syn = SynthCrops("data/synth/train_v2/flat", split_ids(idx["n_clips"])[1], n_frames=200, seed=1091, epoch_size=16 * N_BATCH)
    b = list(DataLoader(syn, 16, num_workers=4, collate_fn=collate))
    print("  train_v2 validation crops (synthetic, training channel, label f0)")
    print("\n".join(analyse(model, vq, b, dev, True)))
