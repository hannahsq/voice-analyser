"""D-034 (pre-registered 2026-10-26; reported only): H5's F1 correction under different Parakeet input spans. Recomputes
Parakeet's layer 5 for gold_ptdb_cp (the 479 dev utterances with closed-phase reference frames, in cache.py's order) in
three regimes and writes scoring caches that mirror `emb_v1` for the four sets eval_heads.py reads (layer 5 only; labels,
teacher columns and the other sets copied):
  whole  each utterance alone (as emb_v1; checks the recomputation)
  short  consecutive 1.04 s windows (13 encoder frames) on the utterance's 80 ms grid, each run alone, frames put back
  long   the utterance inside its speaker's other dev utterances (in order, cyclic), >= 8 s each side, the preceding audio
         trimmed so the utterance starts on the 80 ms grid; its frames from the middle
Args: emb_v1 dir, dir holding ptdb_egg_sub.npz, out dir. Then eval_heads.py <out>/<regime> ... H5#k=...
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
ENC, TILE, SIDE = 1280, 13, 8 * 16000
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
L5 = {r: np.zeros((po[-1], 1024), np.float32) for r in ("whole", "short", "long")}
for i, (s, u) in enumerate(items):
    x, n = audio[(s, u)], po[i + 1] - po[i]
    h = layer5(x)
    L5["whole"][po[i]:po[i + 1]] = h[:n]
    out = np.zeros((n, 1024), np.float32)
    for a in range(0, n, TILE):  # tile a: frames a .. a + 12
        hw = layer5(x[a * ENC:(a + TILE) * ENC])
        k = min(TILE, n - a, len(hw))
        out[a:a + k] = hw[:k]
        if k < min(TILE, n - a):  # a short last window: its missing frames from the window's last
            out[a + k:a + min(TILE, n - a)] = hw[-1]
    L5["short"][po[i]:po[i + 1]] = out
    us = by_spk[s]
    j = us.index(u)
    pre, k = [], 1
    while sum(len(p) for p in pre) < SIDE and k < len(us):
        pre.insert(0, audio[(s, us[(j - k) % len(us)])])
        k += 1
    post, k = [], 1
    while sum(len(p) for p in post) < SIDE and k < len(us):
        post.append(audio[(s, us[(j + k) % len(us)])])
        k += 1
    pre = np.concatenate(pre) if pre else np.zeros(0, np.float32)
    pre = pre[len(pre) % ENC:]
    m = len(pre) // ENC
    hl = layer5(np.concatenate([pre, x] + post))
    L5["long"][po[i]:po[i + 1]] = hl[m:m + n]
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(items)}", flush=True)

ref = np.asarray(pe[:, 5], np.float32)
var = ref.var(0)
for r, v in L5.items():
    print(f"{r:6s} R2 against emb_v1's layer 5: {1 - (((v - ref) ** 2).mean(0) / var).mean():.4f}  max |diff| {np.abs(v - ref).max():.3f}", flush=True)
for r in L5:
    for name in SETS:
        dd = OUT / r / name
        dd.mkdir(parents=True, exist_ok=True)
        for f in (E1 / name).glob("*.npy"):
            if f.stem != "emb":
                shutil.copy(f, dd / f.name)
        src = np.load(E1 / name / "emb.npy", mmap_mode="r")
        emb = np.zeros((len(src), 6, 1024), np.float16)
        emb[:, 5] = L5[r] if name == "gold_ptdb_cp" else src[:, 5]
        np.save(dd / "emb.npy", emb)
print("done", flush=True)
