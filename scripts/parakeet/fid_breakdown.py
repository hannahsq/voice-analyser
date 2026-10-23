"""D-034 reported measure (adopted 2026-10-23 after an exploratory look): student fidelity to Parakeet layer 5 by
population. Held-out sets (targets computed here once, as distil_cache.py, cached as <cache>/fid_<set>.npz; 512 crops
each, seed 12345), none of them in any training set:
  English adults: LibriTTS-R dev-clean; English children and adults: speechocean762 test speakers (age <= 12 / >= 19),
  crops joined from each speaker's utterances; Icelandic children (<= 12) and teens (13-17): Samrómur Children test
  speakers (same joining);
  Italian, Portuguese, Polish: Multilingual LibriSpeech dev; Korean: the cached held_zeroth crops; singing: the cached
  held_vocalset crops, also split by VocalSet task and technique.
R^2 per dimension (mean), R^2 variance-weighted, mean cosine; standardised targets. Args: cache dir, checkpoint(s)."""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_cache as dc  # noqa: E402
import samromur  # noqa: E402
from student import Student  # noqa: E402

from vocal_analysis.features.vqt import TorchVQT  # noqa: E402

DS = Path(r"C:\github\datasets")
C, CKS = Path(sys.argv[1]), sys.argv[2:]
dev = torch.device("cuda")
vq = TorchVQT(dev)
st = np.load(C / "stats.npz")
MU, SD = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))


def so762(keep_age):
    root = DS / "speechocean762" / "speechocean762"
    age = dict(line.split() for line in open(root / "test" / "spk2age"))
    spk = {}
    for line in open(root / "test" / "utt2spk"):
        u, s = line.split()
        if keep_age(int(age[s])):
            spk.setdefault(s, []).append(str(root / "WAVE" / f"SPEAKER{int(s):04d}" / f"{u}.WAV"))
    return spk


def mls_dev(lang):
    return [(str(p), None) for p in sorted((DS / "mls" / f"mls_{lang}_opus" / "dev" / "audio").rglob("*.opus"))]


SETS = {
    "en_libritts": ("English adults (LibriTTS-R dev-clean)", lambda: dc.RealCrops(
        [(str(p), None) for p in sorted((DS / "libritts_r" / "LibriTTS_R" / "dev-clean").rglob("*.wav"))], 512, 12345)),
    "en_so762_child": ("English children <= 12 (speechocean762)", lambda: dc.SpeakerCrops(so762(lambda a: a <= 12), 512, 12345)),
    "en_so762_adult": ("English adults >= 19 (speechocean762)", lambda: dc.SpeakerCrops(so762(lambda a: a >= 19), 512, 12345)),
    "is_samromur_child": ("Icelandic children <= 12 (Samrómur)", lambda: dc.SpeakerCrops(samromur.speakers("test", lambda a: a <= 12), 512, 12345)),
    "is_samromur_teen": ("Icelandic teens 13-17 (Samrómur)", lambda: dc.SpeakerCrops(samromur.speakers("test", lambda a: a >= 13), 512, 12345)),
    "it_mls": ("Italian (MLS dev)", lambda: dc.RealCrops(mls_dev("italian"), 512, 12345)),
    "pt_mls": ("Portuguese (MLS dev)", lambda: dc.RealCrops(mls_dev("portuguese"), 512, 12345)),
    "pl_mls": ("Polish (MLS dev)", lambda: dc.RealCrops(mls_dev("polish"), 512, 12345)),
}


def build(key, make):
    f = C / f"fid_{key}.npz"
    if f.exists():
        return np.load(f)
    import load
    ds = make()
    fe = load.feature_extractor()
    enc = load.encoder(dev)
    enc.layers = enc.layers[:5]
    A, T = [], []
    for s in range(0, len(ds), 16):
        b = [ds[i] for i in range(s, min(s + 16, len(ds)))]
        feats = fe([c for _, c, _ in b], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = enc(feats["input_features"].to(dev), attention_mask=feats["attention_mask"].to(dev),
                    output_hidden_states=True).hidden_states[5]
        A += [a for a, _, _ in b]
        T += [h[i, p + 1:p + 51].half().cpu().numpy() for i, (_, _, p) in enumerate(b)]
    np.savez(f, audio=np.stack(A).astype(np.float16), target=np.stack(T))
    return np.load(f)


def vocalset_paths():
    """the source file of each cached held_vocalset crop (RealCrops' draw replayed)."""
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


def predict(m, audio, target):
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
    ss = ((T - T.mean(0)) ** 2).sum(0)
    r2 = 1 - ((Y - T) ** 2).sum(0) / ss
    return float(r2.mean()), float(1 - ((Y - T) ** 2).sum() / ss.sum()), float(torch.nn.functional.cosine_similarity(Y, T, dim=-1).mean())


if __name__ == "__main__":
    data = {}
    for key, (name, make) in SETS.items():
        try:
            d = build(key, make)
            data[name] = (d["audio"], d["target"])
        except (StopIteration, FileNotFoundError, ImportError) as e:
            print(f"  (skipped {name}: {e!r})", flush=True)
    data["Korean (Zeroth val)"] = tuple(np.load(C / "held_zeroth" / f"{k}.npy", mmap_mode="r") for k in ("audio", "target"))
    data["Singing (VocalSet val)"] = tuple(np.load(C / "held_vocalset" / f"{k}.npy", mmap_mode="r") for k in ("audio", "target"))
    vpaths = vocalset_paths()
    task = np.array([Path(p).parts[-3] for p in vpaths])  # FULL/<singer>/<task>/<technique>/<file>
    tech = np.array([Path(p).parts[-2] for p in vpaths])
    for ck in CKS:
        m = Student().to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["model"])
        m.eval()
        print(f"\n{Path(ck).name}: R2 (mean per dim) / R2 (variance-weighted) / cosine", flush=True)
        for name, (a, t) in data.items():
            Y, T = predict(m, a, t)
            r = fid(Y, T)
            print(f"  {name:42s} {r[0]:6.3f} / {r[1]:6.3f} / {r[2]:.3f}  (n={len(a)})", flush=True)
            if name.startswith("Singing"):
                for lab, grp in (("task", task), ("technique", tech)):
                    for g in sorted(set(grp)):
                        ix = torch.from_numpy(np.flatnonzero(grp == g)).to(dev)
                        if len(ix) >= 15:
                            r = fid(Y[ix], T[ix])
                            print(f"    VocalSet {lab} {g:22s} {r[0]:6.3f} / {r[1]:6.3f} / {r[2]:.3f}  (n={len(ix)})", flush=True)
