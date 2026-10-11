"""Add one teacher's F1-F3 to the saved closed-phase frames (arctic_egg.npz, ptdb_egg.npz), without redoing the
closed-phase analysis. Same audio path and resampling as arctic_egg.py / ptdb_egg.py.
Args: scratch dir, name, checkpoint, optional "_sub". Run from the voice-analyser-m3 worktree.
On the CPU by default (it can run alongside training); VA_DEVICE=cuda runs on the GPU in fp32 (no autocast), files
with the same frame count batched together: results identical to the CPU's (2026-10-11: v2_s0 on ARCTIC, 99th
percentile difference 0.00 Hz), about 2.5x faster. bf16, as analyse() uses on the GPU, is not: it moves the median
closed-phase F1 bias by about 1.7 Hz. MIOpen's fast find mode (heuristic kernel choice, set below unless already
set) matters on the GPU: by default it benchmarks kernels for every new batch shape, 966 s against 111 s."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")  # before torch loads MIOpen

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.features.vqt import HOP
from vocal_analysis.models.infer import _vqt, analyse, decode_outputs, load_teacher

S, NAME, CK = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SUF = sys.argv[4] if len(sys.argv) > 4 else ""  # "_sub": every 3rd utterance per speaker
torch.set_num_threads(int(os.environ.get("VA_THREADS", "8")))  # fewer alongside training
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
model = load_teacher(CK, dev)[0]
D = Path("../voice-analyser/data")


def arctic_wav(d, i):
    return D / "cmu_arctic" / f"cmu_us_{d['voice'][i]}_arctic" / "orig" / f"{d['utt'][i]}.wav"


def ptdb_wav(d, i):
    spk = d["speaker"][i]
    return D / "ptdb_tug" / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{d['utt'][i]}.wav"


for f, wav in (("arctic_egg", arctic_wav), ("ptdb_egg", ptdb_wav)):
    d = dict(np.load(S / f"{f}{SUF}.npz"))
    if NAME in d:
        print(f"{f}: {NAME} already there")
        continue
    out = np.full((len(d["t"]), 3), np.nan)
    keys = d["voice"] if "voice" in d else d["speaker"]
    groups = {}
    for i, (k, u) in enumerate(zip(keys, d["utt"])):
        groups.setdefault((k, u), []).append(i)
    groups = list(groups.values())
    audio = []
    for idx in groups:
        x, sr = sf.read(wav(d, idx[0]))
        audio.append(resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32))

    def put(idx, formants):
        j = np.clip(np.round(d["t"][idx] / 0.010).astype(int), 0, len(formants) - 1)
        out[idx] = formants[j, :3]

    if dev.type == "cpu":
        for n, (idx, x16) in enumerate(zip(groups, audio)):
            put(idx, analyse(model, x16, dev)["formants"])
            if (n + 1) % 250 == 0:
                print(f"  {f}: {n + 1}/{len(groups)}", flush=True)
    else:  # files with the same frame count in one batch: zero padding to a common length is what the VQT does anyway
        vq, by_frames = _vqt(dev), {}
        for k, x16 in enumerate(audio):
            by_frames.setdefault(len(x16) // HOP + 1, []).append(k)
        for nf, ks in by_frames.items():
            length = max(len(audio[k]) for k in ks)
            xb = torch.from_numpy(np.stack([np.pad(audio[k], (0, length - len(audio[k]))) for k in ks])).to(dev)
            with torch.inference_mode():
                o = {key: v.float().cpu().numpy() for key, v in model(vq.log_power(xb, nf)).items()}
            for r, k in enumerate(ks):
                put(groups[k], decode_outputs({key: v[r] for key, v in o.items()})["formants"])
    d[NAME] = out
    np.savez(S / f"{f}{SUF}.npz", **d)
    print(f"{f}: added {NAME}", flush=True)
