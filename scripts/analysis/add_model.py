"""Add one teacher's F1-F3 to the saved closed-phase frames (arctic_egg.npz, ptdb_egg.npz), on the CPU, without
redoing the closed-phase analysis. Same audio path and resampling as arctic_egg.py / ptdb_egg.py.
Args: scratch dir, name, checkpoint. Run from the voice-analyser-m3 worktree."""
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.models.infer import analyse, load_teacher

S, NAME, CK = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SUF = sys.argv[4] if len(sys.argv) > 4 else ""  # "_sub": every 3rd utterance per speaker
torch.set_num_threads(int(os.environ.get("VA_THREADS", "8")))  # fewer alongside training
dev = torch.device("cpu")
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
    for n, idx in enumerate(groups.values()):
        x, sr = sf.read(wav(d, idx[0]))
        x16 = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
        res = analyse(model, x16, dev)
        j = np.clip(np.round(d["t"][idx] / 0.010).astype(int), 0, len(res["t"]) - 1)
        out[idx] = res["formants"][j, :3]
        if (n + 1) % 250 == 0:
            print(f"  {f}: {n + 1}/{len(groups)}", flush=True)
    d[NAME] = out
    np.savez(S / f"{f}{SUF}.npz", **d)
    print(f"{f}: added {NAME}", flush=True)
