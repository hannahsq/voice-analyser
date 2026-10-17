"""Long inputs on synthetic clips (D-028): every training set draws 2-4 s clips, so no model has trained on more than
400 frames, while analyse() runs whole files. On data/synth/val_long_v2 (150 clips of 8-16 s, train_v2's config,
seed 2718), each model runs (a) on the whole clip and (b) on 4 s windows (400 frames, hop 200; each frame from the
window where it is most central), the training length. MAE (Hz) on sonorant frames by position in the clip: frames
0-400, 400-800, 800+. Lines "MAE <model> <mode> <band> <F1> <F2> <F3>".
Device from VA_DEVICE (default cpu). Args: name=checkpoint pairs."""
import os
import sys

import numpy as np

from vocal_analysis.features.vqt import HOP
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth.shards import INT16_SCALE

sys.stdout.reconfigure(encoding="utf-8")
import torch  # noqa: E402

dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
FLAT = "data/synth/val_long_v2/flat"
a_off, f_off = np.load(f"{FLAT}/audio_offsets.npy"), np.load(f"{FLAT}/frame_offsets.npy")
audio_all = np.load(f"{FLAT}/audio.npy", mmap_mode="r")
labels = {k: np.load(f"{FLAT}/label_{k}.npy", mmap_mode="r") for k in ("formants", "sonorant")}
N_CLIPS = int(os.environ.get("VA_N", "0")) or len(a_off) - 1  # a smoke test may use fewer
clips = [(audio_all[a_off[i]:a_off[i + 1]].astype(np.float32) / INT16_SCALE,
          {k: np.asarray(v[f_off[i]:f_off[i + 1]]) for k, v in labels.items()}) for i in range(N_CLIPS)]
W, H = 400, 200
BANDS = ((0, 400), (400, 800), (800, 10 ** 6))
print(f"{len(clips)} clips, {np.median([len(c[1]['sonorant']) for c in clips]):.0f} frames median "
      f"(max {max(len(c[1]['sonorant']) for c in clips)}); windows {W} frames, hop {H}")


def chunked(m, audio, n):
    out = np.full((n, 3), np.nan)
    best = np.full(n, -1.0)
    starts = list(range(0, max(n - W, 0) + 1, H))
    if starts[-1] + W < n:
        starts.append(n - W)
    for s in starts:
        seg = audio[s * HOP: (s + W - 1) * HOP + 1]
        f = analyse(m, seg, dev)["formants"][:, :3]
        k = min(len(f), n - s)
        centrality = -np.abs(np.arange(k) - (W - 1) / 2)
        take = centrality > best[s:s + k]
        out[s:s + k][take] = f[:k][take]
        best[s:s + k][take] = centrality[take]
    return out


for name, ck in (a.split("=", 1) for a in sys.argv[1:]):
    m = load_teacher(ck, dev)[0]
    err = {mode: {b: [] for b in BANDS} for mode in ("whole", "4s")}
    for audio, lab in clips:
        n = len(lab["sonorant"])
        whole = analyse(m, audio, dev)["formants"][:n, :3]
        ch = chunked(m, audio, n)
        for mode, est in (("whole", whole), ("4s", ch)):
            e = np.abs(est - lab["formants"][: len(est), :3])
            for lo, hi in BANDS:
                sl = slice(lo, min(hi, len(est)))
                son = lab["sonorant"][sl].astype(bool)
                if son.any():
                    err[mode][(lo, hi)].append(e[sl][son])
    for mode in err:
        for b, v in err[mode].items():
            if v:
                e = np.concatenate(v)
                mae = [np.nanmean(e[:, j]) for j in range(3)]
                print(f"MAE {name} {mode:5s} {b[0]}-{b[1] if b[1] < 10 ** 6 else 'end'} " + " ".join(f"{x:.1f}" for x in mae)
                      + f"  (n {len(e)})", flush=True)
