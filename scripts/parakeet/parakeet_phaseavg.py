"""D-034 (pre-registered 2026-10-26; reported only): test (a) for v4's F1 offset. Parakeet averaged over 10 ms phases
as the H5 diagnostic pools a student: for gold_ptdb_cp (the 479 dev utterances, in cache.py's order), layer 5 at encoder
frame e becomes the mean of Parakeet on the utterance shifted by 10q ms, q = 0..3 at frame e and q = 4..7 at frame e - 1
(mels 8e - 4 .. 8e + 3, the student's stored frames 8e - 8 .. 8e - 1 in student_as_parakeet.py's pooling). q = 0 alone
must reproduce emb_v1. Writes <out>/phaseavg mirroring emb_v1 for the four sets eval_heads.py reads (layer 5 only; the
other sets and every label column copied). Args: emb_v1 dir, dir holding ptdb_egg_sub.npz, out dir.
Run from the main checkout (VA_DEVICE=cuda)."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

from vocal_analysis.eval import tiers as TI  # noqa: E402

E1, S, OUT = map(Path, sys.argv[1:4])
SETS = ("synth_train_v2", "synth_train_v2s", "gold_hillenbrand", "gold_ptdb_cp")
ENC = 1280
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
fe, enc = load.feature_extractor(), load.encoder(dev)
enc.layers = enc.layers[:5]


def layer5(x):
    f = fe([x], sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        return enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[5][0].float().cpu().numpy()


# cache.py's closed_phase() order
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
tmap = TI.load()
devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == "dev" for s in d["speaker"]])
items = sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])})
audio = {}
for s, u in items:
    x, sr = sf.read(Path("data/ptdb_tug") / ("FEMALE" if s[0] == "F" else "MALE") / "MIC" / s / f"{u}.wav")
    audio[(s, u)] = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
by_spk = {}
for s, u in items:
    by_spk.setdefault(s, []).append(u)

pe, po = np.load(E1 / "gold_ptdb_cp" / "emb.npy", mmap_mode="r"), np.load(E1 / "gold_ptdb_cp" / "emb_off.npy")
assert len(po) - 1 == len(items), (len(po) - 1, len(items))
L5 = np.zeros((po[-1], 1024), np.float32)
worst = 0.0
for i, (s, u) in enumerate(items):
    x, n = audio[(s, u)], po[i + 1] - po[i]
    H = [layer5(x[160 * q:]) for q in range(8)]
    worst = max(worst, float(np.abs(H[0][:n] - np.asarray(pe[po[i]:po[i + 1], 5], np.float32)).max()))
    for e in range(n):
        fr = [H[q][min(e, len(H[q]) - 1)] for q in range(4)] + [H[q][min(max(e - 1, 0), len(H[q]) - 1)] for q in range(4, 8)]
        L5[po[i] + e] = np.mean(fr, 0)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(items)}", flush=True)
ref = np.asarray(pe[:, 5], np.float32)
print(f"q = 0 against emb_v1: max |diff| {worst:.3f}", flush=True)
print(f"phase-averaged R2 against emb_v1's layer 5: {1 - (((L5 - ref) ** 2).mean(0) / ref.var(0)).mean():.4f}", flush=True)
for name in SETS:
    dd = OUT / "phaseavg" / name
    dd.mkdir(parents=True, exist_ok=True)
    for f in (E1 / name).glob("*.npy"):
        if f.stem != "emb":
            shutil.copy(f, dd / f.name)
    src = np.load(E1 / name / "emb.npy", mmap_mode="r")
    emb = np.zeros((len(src), 6, 1024), np.float16)
    emb[:, 5] = L5 if name == "gold_ptdb_cp" else src[:, 5]
    np.save(dd / "emb.npy", emb)
print("done", flush=True)
