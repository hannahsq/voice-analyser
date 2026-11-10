"""D-085 smoke test, part 1: peak GPU memory and step time of the heads arm's model (1.5x trunk, paired acoustic heads,
the iVTL -> aVTL link, three upper heads) for one training step at each crop length, at the run's frame budget
(32 x 200 frames per batch), bf16 autocast as the trainer. Random log-power input and a stand-in loss (every output's
mean): memory and time do not depend on the values.

    python scripts/analysis/d085_probe_memory.py [frames ...]
"""

import sys
import time

if __name__ == "__main__":
    import torch

    from vocal_analysis.device import setup_device
    from vocal_analysis.features.vqt import N_BINS
    from vocal_analysis.models.teacher import Teacher, TeacherConfig, n_params

    device = setup_device()
    lengths = [int(a) for a in sys.argv[1:]] or [32, 200, 400, 800, 1600]
    cfg = TeacherConfig(temporal="conformer", conformer_dim=192, embed=576, vtl_head=True, vtl_inst_head=True, vtl_link=True,
                        acoustic_heads=True, n_upper=3)
    model = Teacher(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"heads arm: {n_params(model):,} parameters (v3.1: conformer_dim 128, embed 384)", flush=True)
    for t in lengths:
        b = max(1, round(32 * 200 / t))
        x = torch.randn(b, N_BINS, t, device=device)
        times = []
        torch.cuda.reset_peak_memory_stats(device)
        for k in range(4):  # the first step selects kernels; the rest are timed
            torch.cuda.synchronize(device)
            t0 = time.time()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(x)
            loss = sum(v.float().mean() for v in out.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize(device)
            if k:
                times.append(time.time() - t0)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        print(f"{t:5d} frames x {b:3d}: peak {peak:5.2f} GiB, step {1000 * sum(times) / len(times):6.0f} ms", flush=True)
        del x, out, loss
        torch.cuda.empty_cache()
