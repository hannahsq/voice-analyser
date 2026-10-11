"""Inference real-time factor of a teacher checkpoint on 30 s of audio: GPU, and CPU at 4 threads. Args: ckpt."""
import sys, time
import numpy as np
import torch
from vocal_analysis.models.infer import analyse, load_teacher
x = np.random.default_rng(0).standard_normal(16000 * 30).astype(np.float32) * 0.1
for dev in (torch.device("cuda"), torch.device("cpu")):
    torch.set_num_threads(4)
    m = load_teacher(sys.argv[1], dev)[0]
    analyse(m, x, dev)
    t = time.perf_counter()
    for _ in range(3):
        analyse(m, x, dev)
    print(f"  infer {dev.type}: RTF {(time.perf_counter() - t) / 90:.4f}")
