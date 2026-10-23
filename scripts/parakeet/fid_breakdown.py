"""D-034 reported measure (adopted 2026-10-23 after an exploratory look): student fidelity by language and by singing task. English: 512 LibriTTS-R
dev-clean crops (Parakeet layer-5 targets computed here, as distil_cache.py); Korean: the cached held_zeroth crops;
VocalSet: the cached held_vocalset crops split by task / technique (source file recovered from the crop's seed).
Args: distil cache dir, checkpoint(s)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\github\voice-analyser-parakeet\scripts\parakeet")
import distil_cache as dc  # noqa: E402
import load  # noqa: E402
from student import Student  # noqa: E402

from vocal_analysis.features.vqt import TorchVQT  # noqa: E402

C, CKS = Path(sys.argv[1]), sys.argv[2:]
dev = torch.device("cuda")
vq = TorchVQT(dev)
st = np.load(C / "stats.npz")
MU, SD = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))

# English held-out crops + targets (cached in the scratch dir next to this script)
EN = C / "fid_en.npz"
if not EN.exists():
    files = [(str(p), None) for p in sorted(Path(r"C:\github\datasets\libritts_r\LibriTTS_R\dev-clean").rglob("*.wav"))]
    ds = dc.RealCrops(files, 512, 12345)
    fe = load.feature_extractor()
    enc = load.encoder(dev)
    enc.layers = enc.layers[:5]
    A, T = [], []
    for s in range(0, 512, 16):
        b = [ds[i] for i in range(s, s + 16)]
        f = fe([c for _, c, _ in b], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[5]
        A += [a for a, _, _ in b]
        T += [h[i, p + 1:p + 51].half().cpu().numpy() for i, (_, _, p) in enumerate(b)]
    np.savez(EN, audio=np.stack(A).astype(np.float16), target=np.stack(T))
en = np.load(EN)


def vocalset_paths():
    """the source file of each cached held_vocalset crop (RealCrops' draw, replayed without reading audio twice)."""
    files = dc.real_files()["vocalset_val"]
    out = []
    for i in range(512):
        rng = np.random.default_rng([12345, i])
        while True:
            path, dur = files[rng.integers(len(files))]
            if dur is not None and dur * 16000 < dc.NS + 1:
                continue
            x = dc.read16k(path)
            if len(x) < dc.NS + 1:
                continue
            rng.integers(0, len(x) - dc.NS + 1)
            out.append(path)
            break
    return out


def per_crop(m, audio, target):
    """per-crop R^2 contributions are not additive, so return predictions and targets (standardised)."""
    Y, T = [], []
    with torch.no_grad():
        for s in range(0, len(audio), 32):
            a = torch.from_numpy(np.asarray(audio[s:s + 32], np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(vq.log_power_with_context(a, 400))
            Y.append(y.float().reshape(len(a), 50, 8, 1024).mean(2))
            T.append((torch.from_numpy(np.asarray(target[s:s + 32], np.float32)).to(dev) - MU) / SD)
    return torch.cat(Y), torch.cat(T)


def fid(Y, T):
    Y, T = Y.reshape(-1, 1024), T.reshape(-1, 1024)
    r2 = 1 - ((Y - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
    w = ((T - T.mean(0)) ** 2).sum(0)
    r2w = 1 - ((Y - T) ** 2).sum() / w.sum()  # variance-weighted
    return float(r2.mean()), float(r2w), float(torch.nn.functional.cosine_similarity(Y, T, dim=-1).mean())


vpaths = vocalset_paths()
task = np.array([Path(p).parts[-3] for p in vpaths])  # FULL/<singer>/<task>/<technique>/<file>
tech = np.array([Path(p).parts[-2] for p in vpaths])
za, zt = np.load(C / "held_zeroth" / "audio.npy", mmap_mode="r"), np.load(C / "held_zeroth" / "target.npy", mmap_mode="r")
va, vt = np.load(C / "held_vocalset" / "audio.npy", mmap_mode="r"), np.load(C / "held_vocalset" / "target.npy", mmap_mode="r")
for ck in CKS:
    m = Student().to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
    m.eval()
    print(f"\n{Path(ck).name}: R2 (mean per dim) / R2 (variance-weighted) / cosine")
    for name, (a, t) in (("English (LibriTTS-R dev-clean)", (en["audio"], en["target"])), ("Korean (Zeroth val)", (za, zt)),
                         ("Singing (VocalSet val), all", (va, vt))):
        r = fid(*per_crop(m, a, t))
        print(f"  {name:34s} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}  (n={len(a)})")
    Y, T = per_crop(m, va, vt)
    for lab, grp in (("task", task), ("technique", tech)):
        for g in sorted(set(grp)):
            ix = torch.from_numpy(np.flatnonzero(grp == g)).to(dev)
            if len(ix) >= 15:
                r = fid(Y[ix], T[ix])
                print(f"    VocalSet {lab} {g:22s} {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}  (n={len(ix)})")
