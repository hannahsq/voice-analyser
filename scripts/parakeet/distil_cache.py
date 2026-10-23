"""D-034 stage 1: fixed 4 s crops and Parakeet layer-5 targets for distilling a student.
Crops are 400 of our 10 ms frames with the VQT's context (as SynthCrops makes them: (400 - 1) * HOP + N_MAX samples);
our frame j is Parakeet mel frame j + 4, encoder frame e is mel 8e, so the targets are encoder frames e = 1..50 (each the
mean of our frames 8e - 8 .. 8e - 1 in the loss). Parakeet sees each real crop with up to 0.96 s of its source file on
either side (the leading part a whole number of encoder frames, 1280 samples each, so frames stay aligned), and synthetic
crops alone: never zero padding, which Parakeet's per-utterance feature normalisation would count (checked: it moves the
crop's features by about their own scale); items of different lengths are batched with the attention mask (checked:
identical features).
Sets (training pools only; no evaluation corpus):
  train: 30,000 real (Zeroth-Korean train 15,000, VocalSet train 7,500, Common Voice 27.0 Korean 7,500) and 30,000
         synthetic (teacher v3's training mixture, as cache.py's fit set; seed 777)
  held out: 512 each from Zeroth-Korean val, VocalSet val, train_v2 and train_v2s synthetic validation (seed 12345)
-> <out>/<set>/audio.npy (n, samples) fp16, target.npy (n, 50, 1024) fp16; <out>/stats.npz (per-dimension mean, std of
the training targets); <out>/done. Run from the main checkout:
VA_DEVICE=cuda python <worktree>/scripts/parakeet/distil_cache.py <out dir>"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.features.vqt import HOP, N_MAX  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, consolidate, split_ids  # noqa: E402

CROP = 400
NS = (CROP - 1) * HOP + N_MAX  # crop samples, as SynthCrops
CTX = 96 * HOP  # 0.96 s: 96 mel frames = 12 encoder frames
E = 50  # encoder frames 1..50 of the crop
LAYER = 5


def as_list(b):
    return b


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    return x


def real_files():
    pools = {}
    for corpus, root in (("zeroth_korean", "data/zeroth_korean"), ("vocalset", "data/vocalset")):
        rows = list(csv.DictReader(open(f"data/manifests/{corpus}.csv", encoding="utf-8")))
        for split in ("train", "val"):
            pools[f"{corpus}_{split}"] = [(str(Path(root) / r["path"]), float(r["duration"])) for r in rows if r["split"] == split]
    pools["cv_ko_train"] = [(str(p), None) for p in sorted(Path("data/cv_ko/clips").glob("*.mp3"))]
    return pools


class RealCrops(torch.utils.data.Dataset):
    """n fixed crops from (path, duration) files: file and offset drawn per index from a seeded generator; files
    shorter than the crop are redrawn. Returns (crop audio, Parakeet input with context)."""

    def __init__(self, files, n, seed):
        self.files, self.n, self.seed = files, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        while True:
            path, dur = self.files[rng.integers(len(self.files))]
            if dur is not None and dur * 16000 < NS + 1:
                continue
            x = read16k(path)
            if len(x) < NS + 1:
                continue
            s = int(rng.integers(0, len(x) - NS + 1))
            pre = (min(s, CTX) // 1280) * 1280
            return x[s:s + NS], x[s - pre:s + NS + CTX], pre // 1280


class SpeakerCrops(torch.utils.data.Dataset):
    """n fixed crops for pools of short utterances (children's corpora): per index a speaker is drawn, then that
    speaker's utterances in a seeded random order are joined until the crop and up to CTX either side fit, so short
    utterances are not dropped (which would favour older, more fluent speakers). Returns as RealCrops."""

    def __init__(self, speakers: dict, n, seed):
        self.spk = sorted(speakers.items())  # [(speaker, [paths])]
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        _, paths = self.spk[rng.integers(len(self.spk))]
        order = rng.permutation(len(paths))
        x, k = np.zeros(0, np.float32), 0
        while len(x) < NS + 2 * CTX and k < 4 * len(paths):
            x = np.concatenate([x, read16k(paths[order[k % len(paths)]])])
            k += 1
        if len(x) < NS + 1:
            x = np.tile(x, int(np.ceil((NS + 1) / max(len(x), 1))))
        s = int(rng.integers(0, len(x) - NS + 1))
        pre = (min(s, CTX) // 1280) * 1280
        return x[s:s + NS], x[s - pre:s + NS + CTX], pre // 1280


class SynthWrap(torch.utils.data.Dataset):
    def __init__(self, ds, n):
        self.ds, self.n = ds, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a = np.asarray(self.ds[i][0], np.float32)
        return a, a, 0


def main():
    out = Path(sys.argv[1])
    quick = os.environ.get("VA_QUICK") == "1"
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    fe = load.feature_extractor()
    enc = load.encoder(dev)
    enc.layers = enc.layers[:LAYER]

    def targets(ctx_batch, pres):
        f = fe(list(ctx_batch), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[LAYER]
        # `pre` leading encoder frames of context; the crop's encoder frames 1..50 follow
        return np.stack([h[i, p + 1:p + 1 + E].half().cpu().numpy() for i, p in enumerate(pres)])

    def build(name, ds):
        d = out / name
        if (d / "done").exists():
            log(f"[{name}] cached")
            return
        d.mkdir(parents=True, exist_ok=True)
        n = min(len(ds), 64) if quick else len(ds)
        A = np.lib.format.open_memmap(d / "audio.npy", "w+", np.float16, (n, NS))
        T = np.lib.format.open_memmap(d / "target.npy", "w+", np.float16, (n, E, 1024))
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(n)), batch_size=16, num_workers=6,
                                         collate_fn=as_list, persistent_workers=False)
        k = 0
        for b in dl:
            A[k:k + len(b)] = np.stack([a for a, _, _ in b]).astype(np.float16)
            T[k:k + len(b)] = targets([c for _, c, _ in b], [p for _, _, p in b])
            k += len(b)
            if (k // 16) % 100 == 0:
                log(f"  [{name}] {k}/{n}")
        A.flush()
        T.flush()
        del A, T
        (d / "done").write_text(json.dumps({"n": n}))
        log(f"[{name}] {n} crops")

    pools = real_files()
    build("real_zeroth", RealCrops(pools["zeroth_korean_train"], 15000, 3401))
    build("real_vocalset", RealCrops(pools["vocalset_train"], 7500, 3402))
    build("real_cv_ko", RealCrops(pools["cv_ko_train"], 7500, 3403))
    build("held_zeroth", RealCrops(pools["zeroth_korean_val"], 512, 12345))
    build("held_vocalset", RealCrops(pools["vocalset_val"], 512, 12345))

    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    tlw = tuple(cfg3["train"]["tract_left_weight"])
    flat = Path(cfg3["train"]["data"]) / "flat"
    train_ids, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=777, epoch_size=30000, fixed=True)], []
    for dd in cfg3["train"]["resynth"]:
        rflat = consolidate(dd)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=778 + len(sizes), epoch_size=30000, fixed=True,
                                tract_left_weight=tlw))
        sizes.append(n)
    frac = cfg3["train"]["resynth_frac"]
    build("synth", SynthWrap(MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], 30000, seed=777, fixed=True), 30000))
    build("held_synth_v2", SynthWrap(SynthCrops(flat, val_ids[:512], CROP, channel, seed=12345, fixed=True), 512))
    v2s = Path("data/synth/train_v2s")
    build("held_synth_v2s", SynthWrap(SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                                 CROP, channel, seed=12345, fixed=True), 512))

    if not (out / "stats.npz").exists():  # per-dimension statistics of the training targets
        s1, s2, cnt = np.zeros(1024), np.zeros(1024), 0
        for name in ("real_zeroth", "real_vocalset", "real_cv_ko", "synth"):
            T = np.load(out / name / "target.npy", mmap_mode="r")
            for i in range(0, len(T), 2000):
                x = np.asarray(T[i:i + 2000], np.float64).reshape(-1, 1024)
                s1 += x.sum(0)
                s2 += (x * x).sum(0)
                cnt += len(x)
        mu = s1 / cnt
        np.savez(out / "stats.npz", mean=mu, std=np.sqrt(np.maximum(s2 / cnt - mu * mu, 1e-8)))
    (out / "done").write_text(json.dumps({"crop": CROP, "samples": NS, "frames": E, "layer": LAYER}))
    log("done")


if __name__ == "__main__":
    main()
