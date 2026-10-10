"""Formant error against input length on synthetic clips (whole-input inference, as analyse() runs real files).
Args: name=checkpoint pairs. Run from the main checkout (data/synth/train_v2). 150 clips cut to 30-400 frames."""
import sys
import numpy as np
from vocal_analysis.synth.shards import ShardReader
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.device import setup_device
from vocal_analysis.features.vqt import HOP
dev = setup_device()
rd = ShardReader("data/synth/train_v2")
ids = np.random.default_rng(0).choice(len(rd), 150, replace=False)
clips = [rd[int(i)] for i in ids]
LENGTHS = (30, 50, 100, 200, 400)
print(f"{len(clips)} clips, median {np.median([len(c[1]['f0']) for c in clips]):.0f} frames; MAE on sonorant frames by input length (frames)")
print(f"{'model':16s} " + "".join(f"{'F1@' + str(L):>9s}" for L in LENGTHS) + "".join(f"{'F3@' + str(L):>9s}" for L in LENGTHS), flush=True)
for name, ck in (a.split("=", 1) for a in sys.argv[1:]):
    m = load_teacher(ck, dev)[0]
    err = {L: ([], []) for L in LENGTHS}
    for audio, lab, _ in clips:
        for L in LENGTHS:
            n = min(L, len(lab["f0"]))
            res = analyse(m, audio[: (n - 1) * HOP + 1], dev)
            k = min(len(res["formants"]), n)
            for j, e in ((0, err[L][0]), (2, err[L][1])):
                d = np.abs(res["formants"][:k, j] - lab["formants"][:k, j])
                e.append(d[np.isfinite(d) & lab["sonorant"][:k]])
    print(f"{name:16s} " + "".join(f"{np.mean(np.concatenate(err[L][0])):9.1f}" for L in LENGTHS)
          + "".join(f"{np.mean(np.concatenate(err[L][1])):9.1f}" for L in LENGTHS), flush=True)
