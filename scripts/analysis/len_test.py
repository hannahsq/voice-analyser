"""F1 MAE vs input length on synthetic clips, GRU v2_s0 vs Conformer (whole-input inference as in analyse())."""
import sys, numpy as np
from vocal_analysis.synth.shards import ShardReader
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.device import setup_device
from vocal_analysis.features.vqt import HOP
dev = setup_device()
rd = ShardReader("data/synth/train_v2")
ids = np.random.default_rng(0).choice(len(rd), 150, replace=False)
clips = [rd[int(i)] for i in ids]
print("clip lengths (frames): median", np.median([len(c[1]["f0"]) for c in clips]))
for name in ("v2_s0", "conf_v2_s0"):
    m = load_teacher(f"runs/small/{name}_k35/ckpt_last.pt", dev)[0]
    row = []
    for L in (30, 50, 100, 200, 400, 0):
        e = []
        for audio, lab, _ in clips:
            n = len(lab["f0"]) if L == 0 else min(L, len(lab["f0"]))
            a = audio[: (n - 1) * HOP + 1] if L else audio
            res = analyse(m, a, dev)
            k = min(len(res["formants"]), len(lab["formants"]), n)
            d = np.abs(res["formants"][:k, 0] - lab["formants"][:k, 0])
            ok = np.isfinite(d) & (lab["voiced"][:k] > 0)
            e.append(d[ok])
        row.append(f"{'full' if L == 0 else L}: {np.mean(np.concatenate(e)):.1f}")
    print(name, "F1 MAE by input length  " + "  ".join(row), flush=True)
