"""M5 CPU benchmark of streamed inference (plan §10.3): the streaming VQT and the causal model's `step`, per 10 ms hop,
in PyTorch eager mode on the CPU (ONNX Runtime comes later). Prints milliseconds per frame (median and 99th
percentile for one-frame chunks), the share of one core at 100 frames per second, and a whole-clip forward for
comparison.

    python -m vocal_analysis.rt.bench [--ckpt CKPT] [--seconds 10] [--window 400] [--lookahead 0] [--device cpu]

--device cuda runs the model on the GPU (the VQT stays on the CPU); not while a GPU chain is running.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np


def _time_stream(model, lp, chunk: int, torch) -> np.ndarray:
    """Seconds per call of model.step over lp (1, F, T) in chunks, the outputs read back to the CPU (as a display
    would need them); returns per-call times."""
    state, out = None, []
    with torch.inference_mode():
        for i in range(0, lp.shape[-1], chunk):
            t0 = time.perf_counter()
            o, state = model.step(lp[..., i:i + chunk], state)
            o["formant_logits"].cpu(), o["f0_logits"].cpu()  # waits for the device
            out.append(time.perf_counter() - t0)
    return np.array(out)


def main() -> None:
    import torch

    from ..features import vqt
    from ..models.teacher import Teacher, TeacherConfig, n_params
    from .streaming import StreamingVQT, log_power

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/d040_5b_s2_b20k/ckpt_last.pt")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--lookahead", type=int, default=0)
    ap.add_argument("--threads", default="1,0", help="comma list; 0 = PyTorch's default")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.device != "cpu":
        import os

        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = replace(TeacherConfig(**ck["teacher_config"]), causal=True, attn_window=a.window, lookahead=a.lookahead)
    dev = torch.device(a.device)
    model = Teacher(cfg).eval()
    model.load_state_dict(ck["model"])
    teacher = Teacher(TeacherConfig(**ck["teacher_config"])).eval()
    teacher.load_state_dict(ck["model"])
    model, teacher = model.to(dev), teacher.to(dev)
    print(f"causal copy of {a.ckpt}: {n_params(model):,} parameters, window {a.window}, lookahead {a.lookahead}")

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(int(a.seconds * vqt.FS))).astype(np.float32)
    n_frames = len(audio) // vqt.HOP

    s = StreamingVQT()
    times, frames = [], []
    for i in range(0, len(audio), vqt.HOP):
        t0 = time.perf_counter()
        fr = s.push(audio[i:i + vqt.HOP])
        if len(fr):
            frames.append(log_power(fr))
        times.append(time.perf_counter() - t0)
    times = np.array(times[50:])
    print(f"streaming VQT (numpy, default threads): {1e3 * np.median(times):.3f} ms per hop median, "
          f"{1e3 * np.percentile(times, 99):.3f} p99")
    lp = torch.tensor(np.concatenate(frames).T[None], dtype=torch.float32)[..., :n_frames].to(dev)

    default_threads = torch.get_num_threads()
    for th in (int(x) for x in a.threads.split(",")):
        torch.set_num_threads(th or default_threads)
        label = a.device if a.device != "cpu" else f"{torch.get_num_threads()} thread{'s' if torch.get_num_threads() > 1 else ''}"
        _time_stream(model, lp[..., :100], 1, torch)  # warm-up
        for chunk in (1, 2, 4, 10):
            t = _time_stream(model, lp, chunk, torch)[5:]
            per_frame = t / chunk
            core = 100 * np.median(per_frame) / 0.010 * (torch.get_num_threads() if th else 1)
            extra = (f", p99 {1e3 * np.percentile(t, 99):.2f} ms per call" if chunk == 1 else "")
            print(f"  {label:10s} chunk {chunk:2d}: {1e3 * np.median(per_frame):.2f} ms per frame median{extra}; "
                  f"real-time factor {np.median(per_frame) / 0.010:.3f}" + (f" (~{core:.0f} % of one core)" if th == 1 else ""))
        with torch.inference_mode():
            for net, name in ((model, "causal"), (teacher, "teacher")):
                net(lp)["formant_logits"].cpu()  # warm-up at this length (kernel choice)
                t0 = time.perf_counter()
                net(lp)["formant_logits"].cpu()
                dt = time.perf_counter() - t0
                print(f"  {label:10s} whole clip, {name:7s}: {1e3 * dt / n_frames:.3f} ms per frame "
                      f"(real-time factor {dt / a.seconds:.4f})")
    torch.set_num_threads(default_threads)


if __name__ == "__main__":
    main()
