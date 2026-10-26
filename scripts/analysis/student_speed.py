"""D-034 (student size): speed of candidate student sizes. The teacher architecture (teacher v3's config) scaled up;
forward + backward on 32 x 400-frame crops from audio (VQT on the GPU, as in training), and inference throughput.
Parameter counts of the whole model and of the formant encoder. Random audio: speed only. Needs a GPU."""
import json
import time
from dataclasses import replace

import torch

from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.teacher import Teacher, TeacherConfig

VARIANTS = {
    "S (teacher v3)": {},
    "M": dict(conformer_dim=256, conformer_layers=4, conformer_heads=4, conformer_ff_mult=4),
    "L": dict(channels=(32, 64, 96), embed=512, conformer_dim=384, conformer_layers=8, conformer_heads=6, conformer_ff_mult=4),
}


def timed(fn, n=10, warm=3):
    for i in range(warm + n):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.time()
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    base = json.load(open("runs/conformer_v2r2_s1_d16k/config.json"))["model"]
    base = TeacherConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in base.items()})
    vq = TorchVQT(dev)
    audio = torch.randn(32, 400 * 160 + 16000, device=dev) * 0.1
    for name, kw in VARIANTS.items():
        m = Teacher(replace(base, **kw)).to(dev)
        n_all = sum(p.numel() for p in m.parameters()) / 1e6
        n_enc = sum(p.numel() for p in m.enc.parameters()) / 1e6
        opt = torch.optim.AdamW(m.parameters(), 1e-4)

        def train_step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(vq.log_power_with_context(audio, 400))
            loss = sum(v.float().pow(2).mean() for v in out.values() if torch.is_tensor(v))
            opt.zero_grad()
            loss.backward()
            opt.step()

        def infer():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                m(vq.log_power_with_context(audio, 400))

        tr = timed(train_step)
        m.eval()
        inf = timed(infer)
        print(f"{name:16s} {n_all:5.1f} M parameters (formant encoder {n_enc:5.1f} M); train step {tr * 1000:6.0f} ms"
              f" ({32 * 4 / tr:6.0f} audio s/s); inference {32 * 4 / inf:6.0f} audio s/s", flush=True)
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
