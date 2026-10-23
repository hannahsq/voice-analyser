"""D-034 student v4 targets: v2's 90,000 crops (the same files, offsets and seeds) with a 10 ms phase per crop, and
isolated-word items (words.py); held-out sets with all eight phases.
Phase phi (0..7, its own seeded generator, so v2's draws are unchanged): Parakeet sees the crop's audio shifted by phi * 10
ms (real crops: its context window starts phi * 160 samples later, the leading part still a whole number of encoder
frames; synthetic crops: the first phi * 160 samples dropped), so its encoder frame e sits at the crop's mel 8e + phi,
i.e. student crop frame j = 8e + phi - 4; targets are e = 1..49. Phase 0 reproduces v2's targets (checked).
Words: Parakeet sees the word file (its first phi * 160 samples dropped), frame e at the file's mel 8e + phi = student
frame 8e + phi (the student gets the file with N_MAX / 2 zeros either side, as log_power pads it at inference);
targets for e with 8e + phi < L and e inside Parakeet's output, masked.
-> <out>/<set>/audio.npy, target.npy, phase.npy (and length.npy, mask.npy for words); stats.npz from the training crops'
targets; done. Run from the main checkout: VA_DEVICE=cuda python <worktree>/scripts/parakeet/distil_cache_v4.py <out dir> <mfa dir>"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_cache as D  # noqa: E402
import load  # noqa: E402
import words as W  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.features.vqt import HOP, N_MAX  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, consolidate, split_ids  # noqa: E402

E4 = 49  # encoder frames 1..49 exist at every phase
EW = W.LMAX // 8  # word targets e = 0..23
AW = (W.LMAX - 1) * HOP + 1 + N_MAX  # word audio with the VQT's zero context
ALL = "all"


def phase_of(seed, i):
    return int(np.random.default_rng([seed, i, 99]).integers(8))


def shifted_ctx(x, s, phi):
    """Parakeet's input for a crop at s of x, shifted by phi frames: (context audio, leading encoder frames)"""
    s2 = s + phi * HOP
    pre = (min(s2, D.CTX) // 1280) * 1280
    return x[s2 - pre:s + D.NS + D.CTX], pre // 1280


class Phased(torch.utils.data.Dataset):
    """a D.RealCrops / D.SpeakerCrops / D.SynthWrap with phases: one per item (seeded) or all eight"""

    def __init__(self, ds, phases):
        self.ds, self.phases = ds, phases

    def __len__(self):
        return len(self.ds)

    def source(self, i):
        """(full audio, crop start) with exactly the base dataset's draws"""
        ds = self.ds
        if isinstance(ds, D.SynthWrap):
            return np.asarray(ds.ds[i][0], np.float32), None
        rng = np.random.default_rng([ds.seed, i])
        if isinstance(ds, D.RealCrops):
            while True:
                path, dur = ds.files[rng.integers(len(ds.files))]
                if dur is not None and dur * 16000 < D.NS + 1:
                    continue
                x = D.read16k(path)
                if len(x) < D.NS + 1:
                    continue
                return x, int(rng.integers(0, len(x) - D.NS + 1))
        _, paths = ds.spk[rng.integers(len(ds.spk))]  # SpeakerCrops
        order = rng.permutation(len(paths))
        x, k = np.zeros(0, np.float32), 0
        while len(x) < D.NS + 2 * D.CTX and k < 4 * len(paths):
            x = np.concatenate([x, D.read16k(paths[order[k % len(paths)]])])
            k += 1
        if len(x) < D.NS + 1:
            x = np.tile(x, int(np.ceil((D.NS + 1) / max(len(x), 1))))
        return x, int(rng.integers(0, len(x) - D.NS + 1))

    def __getitem__(self, i):
        x, s = self.source(i)
        phis = range(8) if self.phases == ALL else [phase_of(getattr(self.ds, "seed", 777), i)]
        if s is None:  # synthetic: the crop alone
            return x, [(x[p * HOP:], 0, p) for p in phis]
        return x[s:s + D.NS], [(*shifted_ctx(x, s, p), p) for p in phis]


class WordItems(torch.utils.data.Dataset):
    def __init__(self, cands, n, seed, phases):
        self.c, self.n, self.seed, self.phases = cands, n, seed, phases

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        while True:
            x = W.make_item(self.c[rng.integers(len(self.c))], rng)
            if x is not None:
                break
        L = (len(x) - 1) // HOP + 1
        phis = range(8) if self.phases == ALL else [int(rng.integers(8))]
        a = np.zeros(AW, np.float32)
        a[N_MAX // 2:N_MAX // 2 + len(x)] = x
        return a, [(x[p * HOP:], 0, p) for p in phis], L


def main():
    out, mfa = Path(sys.argv[1]), Path(sys.argv[2])
    part = os.environ.get("VA_PART", "all")  # "crops": everything but the word sets; "words": only them (then stats)
    quick = os.environ.get("VA_QUICK") == "1"
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    fe = load.feature_extractor()
    enc = load.encoder(dev)
    enc.layers = enc.layers[:D.LAYER]

    def parakeet(ctxs):
        """[audio] -> (hidden (B, frames, 1024) on the CPU, valid encoder frames per item)"""
        f = fe(list(ctxs), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[D.LAYER]
        return h.half().cpu().numpy(), (-(-f["attention_mask"].sum(1) // 8)).numpy()

    def build(name, ds, kind):
        d = out / name
        if (d / "done").exists():
            return D.log(f"[{name}] cached")
        d.mkdir(parents=True, exist_ok=True)
        n = min(len(ds), 64) if quick else len(ds)
        P = 8 if ds.phases == ALL else 1
        words = kind == "words"
        ef = EW if words else E4
        A = np.lib.format.open_memmap(d / "audio.npy", "w+", np.float16, (n, AW if words else D.NS))
        T = np.lib.format.open_memmap(d / "target.npy", "w+", np.float16, (n, P, ef, 1024) if P > 1 else (n, ef, 1024))
        PH = np.zeros((n, P), np.int8)
        M = np.zeros((n, P, ef), bool) if words else None
        LEN = np.zeros(n, np.int16) if words else None
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=16, num_workers=6,
                                         collate_fn=D.as_list, persistent_workers=False)
        k = 0
        for b in dl:
            A[k:k + len(b)] = np.stack([it[0] for it in b]).astype(np.float16)
            for q in range(P):
                ctxs = [it[1][q] for it in b]
                h, valid = parakeet([c for c, _, _ in ctxs])
                for r, (c, pre, phi) in enumerate(ctxs):
                    PH[k + r, q] = phi
                    if words:
                        L = b[r][2]
                        LEN[k + r] = L
                        m = np.array([e < valid[r] and 8 * e + phi < L for e in range(ef)])
                        t = np.zeros((ef, 1024), np.float16)
                        t[m] = h[r, np.flatnonzero(m)]
                        M[k + r, q] = m
                    else:
                        t = h[r, pre + 1:pre + 1 + ef]
                    if P > 1:
                        T[k + r, q] = t
                    else:
                        T[k + r] = t
            k += len(b)
            if (k // 16) % 100 == 0:
                D.log(f"  [{name}] {k}/{n}")
        A.flush()
        T.flush()
        del A, T
        np.save(d / "phase.npy", PH)
        if words:
            np.save(d / "mask.npy", M)
            np.save(d / "length.npy", LEN)
        (d / "done").write_text(json.dumps({"n": n, "phases": P, "kind": kind}))
        D.log(f"[{name}] {n} items")

    import samromur

    pools = D.real_files()
    R = lambda ds: Phased(ds, "random")  # noqa: E731
    if part == "words":
        cl = W.candidates(mfa / "libritts_train", mfa / "libritts_train_aligned") + W.candidates(mfa / "libritts_train360", mfa / "libritts_train360_aligned")
        cs = W.candidates(mfa / "so762_train", mfa / "so762_train_aligned")
        D.log(f"word candidates (pauses both sides): LibriTTS-R train {len(cl)}, speechocean762 train {len(cs)}")
        build("words_libritts", WordItems(cl, 24000, 5101, "random"), "words")
        build("words_so762", WordItems(cs, 12000, 5102, "random"), "words")
        build("held_words_libritts", WordItems(W.candidates(mfa / "libritts_dev", mfa / "libritts_dev_aligned"), 512, 12345, ALL), "words")
        build("held_words_so762", WordItems(W.candidates(mfa / "so762_test", mfa / "so762_test_aligned"), 512, 12345, ALL), "words")
        (out / "done").write_text(json.dumps({"mix": "v4", "crop": D.CROP, "frames": E4, "word_frames": EW, "layer": D.LAYER}))
        return D.log("done")
    build("real_libritts", R(D.SpeakerCrops(D.libritts_train_speakers(), 15000, 3411)), "crops")
    build("real_so762", R(D.SpeakerCrops(D.so762_speakers("train"), 3000, 3412)), "crops")
    build("real_samromur", R(D.SpeakerCrops(samromur.speakers("train"), 6000, 3413)), "crops")
    build("real_zeroth", R(D.RealCrops(pools["zeroth_korean_train"], 7500, 3401)), "crops")
    build("real_cv_ko", R(D.RealCrops(pools["cv_ko_train"], 2500, 3403)), "crops")
    build("real_vocalset", R(D.RealCrops(pools["vocalset_train"], 8000, 3402)), "crops")
    for k, lang in enumerate(("italian", "portuguese", "polish")):
        build(f"real_mls_{lang}", R(D.RealCrops(D.mls_files(lang, "train"), 1000, 3414 + k)), "crops")

    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    tlw = tuple(cfg3["train"]["tract_left_weight"])
    flat = Path(cfg3["train"]["data"]) / "flat"
    train_ids, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    parts, sizes = [SynthCrops(flat, train_ids, D.CROP, channel, seed=777, epoch_size=45000, fixed=True)], []
    for dd in cfg3["train"]["resynth"]:
        rflat = consolidate(dd)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        parts.append(SynthCrops(rflat, np.arange(n), D.CROP, channel, seed=778 + len(sizes), epoch_size=45000, fixed=True,
                                tract_left_weight=tlw))
        sizes.append(n)
    frac = cfg3["train"]["resynth_frac"]
    build("synth", R(D.SynthWrap(MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], 45000, seed=777, fixed=True), 45000)), "crops")

    # held out, all eight phases
    A8 = lambda ds: Phased(ds, ALL)  # noqa: E731
    build("held_zeroth", A8(D.RealCrops(pools["zeroth_korean_val"], 512, 12345)), "crops")
    build("held_vocalset", A8(D.RealCrops(pools["vocalset_val"], 512, 12345)), "crops")
    build("held_synth_v2", A8(D.SynthWrap(SynthCrops(flat, val_ids[:512], D.CROP, channel, seed=12345, fixed=True), 512)), "crops")
    v2s = Path("data/synth/train_v2s")
    build("held_synth_v2s", A8(D.SynthWrap(SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                                      D.CROP, channel, seed=12345, fixed=True), 512)), "crops")

    if not (out / "stats.npz").exists():  # per-dimension statistics of the training crops' targets (as v2)
        s1, s2, cnt = np.zeros(1024), np.zeros(1024), 0
        for name in sorted(d.name for d in out.iterdir() if d.is_dir() and (d.name.startswith("real_") or d.name == "synth")):
            T = np.load(out / name / "target.npy", mmap_mode="r")
            for i in range(0, len(T), 2000):
                x = np.asarray(T[i:i + 2000], np.float64).reshape(-1, 1024)
                s1 += x.sum(0)
                s2 += (x * x).sum(0)
                cnt += len(x)
        mu = s1 / cnt
        np.savez(out / "stats.npz", mean=mu, std=np.sqrt(np.maximum(s2 / cnt - mu * mu, 1e-8)))
    D.log("crops done; the word sets follow (VA_PART=words)")


if __name__ == "__main__":
    main()
