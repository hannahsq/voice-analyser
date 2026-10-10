"""D-020 guard-rail: each model on each training set's validation clips (all 2000 held-out clips per set, fixed crops
and channel), with clip-bootstrap intervals on the paired differences. Args: scratch dir. Run from the main repo."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.device import setup_device
from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.bins import F0_AXIS, FORMANT_AXIS, decode
from vocal_analysis.models.infer import load_teacher
from vocal_analysis.train.data import SynthCrops, collate, consolidate, split_ids
from vocal_analysis.train.teacher import _saved_channel

if __name__ == "__main__":
    S = Path(sys.argv[1])
    dev = setup_device()
    vq = TorchVQT(dev)
    MODELS = {"d06k": "runs/teacher_v2_d06k/ckpt_010000.pt", sys.argv[2]: f"runs/teacher_{sys.argv[2]}_d06k/ckpt_010000.pt"}
    SETS = {"v2 val": "data/synth/train_v2", f"{sys.argv[2]} val": f"data/synth/train_{sys.argv[2]}"}
    channel = _saved_channel(Path("runs/teacher_v2_d06k"))
    assert channel == _saved_channel(Path(f"runs/teacher_{sys.argv[2]}_d06k")), "runs used different channel settings"
    res = {}
    for sname, sdir in SETS.items():
        flat = consolidate(sdir, log=lambda *_: None)
        n = json.loads((flat / "done").read_text())["n_clips"]
        ds = SynthCrops(flat, split_ids(n)[1], 200, channel, seed=12345, fixed=True)
        for mname, ck in MODELS.items():
            model = load_teacher(ck, dev)[0]
            clip = {"F1": [], "F2": [], "F3": [], "F1dr": [], "f0": [], "f0_lo": []}
            with torch.inference_mode():
                for audio, lab in DataLoader(ds, 32, num_workers=6, collate_fn=collate):
                    audio = audio.to(dev)
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        out = model(vq.log_power_with_context(audio, lab["f0"].shape[1]))
                    fm, _ = decode(out["formant_logits"].float().softmax(-1).cpu().numpy(), FORMANT_AXIS)
                    f0, _ = decode(out["f0_logits"].float().softmax(-1).cpu().numpy(), F0_AXIS)
                    ref, son = lab["formants"].numpy(), lab["sonorant"].numpy().astype(bool)
                    rf0, vo = lab["f0"].numpy(), lab["voiced"].numpy().astype(bool)
                    for b in range(len(audio)):  # per-clip means, for the clip bootstrap
                        for k in range(3):
                            m = son[b] & np.isfinite(ref[b, :, k])
                            e = np.abs(fm[b, m, k] - ref[b, m, k])
                            clip[f"F{k + 1}"].append(e.mean() if m.any() else np.nan)
                            if k == 0:
                                clip["F1dr"].append(np.mean((e <= 0.2 * ref[b, m, 0]) & (e <= 200)) if m.any() else np.nan)
                        c = np.abs(1200 * np.log2(f0[b, vo[b]] / rf0[b, vo[b]]))
                        clip["f0"].append(np.mean(c <= 50) if vo[b].any() else np.nan)
                        lo = vo[b] & (rf0[b] < 150)
                        clip["f0_lo"].append(np.mean(np.abs(1200 * np.log2(f0[b, lo] / rf0[b, lo])) <= 50) if lo.any() else np.nan)
            res[(sname, mname)] = {k: np.array(v) for k, v in clip.items()}
            print(f"  {sname} / {mname} done", flush=True)
    np.savez(S / "cross_val.npz", **{f"{s}|{m}|{k}": v for (s, m), d in res.items() for k, v in d.items()})

    rng = np.random.default_rng(0)
    print("\nper-clip means over all 2000 validation clips of each set; [new - d06k] with 95 % clip-bootstrap interval")
    for sname in SETS:
        print(f"\n{sname}")
        for k, unit, scale in (("F1", "Hz MAE", 1), ("F2", "Hz MAE", 1), ("F3", "Hz MAE", 1), ("F1dr", "% F1 DR", 100),
                               ("f0", "% f0 RPA", 100), ("f0_lo", "% f0 RPA < 150 Hz", 100)):
            a, b = res[(sname, "d06k")][k], res[(sname, sys.argv[2])][k]
            ok = np.isfinite(a) & np.isfinite(b)
            d = (b - a)[ok] * scale
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {unit:18s} d06k {np.nanmean(a) * scale:6.1f}   new {np.nanmean(b) * scale:6.1f}   "
                  f"diff {d.mean():+5.1f} [{lo:+.1f}, {hi:+.1f}]")
