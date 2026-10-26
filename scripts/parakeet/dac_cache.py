"""D-033 part 5: cache DAC (16 kHz) encoder taps for every item in the Parakeet scoring cache (`emb_v1`), in the same
item order, so labels, reference frames and teacher v3's columns are reused from there. Taps (all on DAC's 50 Hz grid,
frame f centred at sample 320 f + 160, i.e. our 10 ms frame 2 f + 1; click-checked): T1-T3 the first three strided
blocks' outputs (128 / 256 / 512 channels at 8 kHz / 2 kHz / 400 Hz, averaged over each 320-sample frame), T4 the fourth
block (1024), T5 the encoder's output latent (1024, before quantisation), T6 the quantised latent (1024).
Writes <out>/<set>/emb.npy (frames, 3968) fp16 and emb_off.npy, and <out>/meta.json. Also a 40,000-crop head cache of
chosen taps with `--heads TAPS N` (part 4's crops: cache_heads.py's seed 999 mixture).
Run from the main checkout with .venv-dac: python scripts/parakeet/dac_cache.py <scratch dir> <out dir> [--heads 4,5 40000]"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

import dac
from vocal_analysis.config import DeviceConfig
from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.device import setup_device
from vocal_analysis.eval import tiers as TI
from vocal_analysis.eval.datasets import load_audio, registry
from vocal_analysis.synth.channel import ChannelConfig
from vocal_analysis.train.data import MixedCrops, SynthCrops, collate, consolidate, split_ids

# all work under the main guard: DataLoader workers re-import this file on Windows (spawn)
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    S, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    WIDTHS = [128, 256, 512, 1024, 1024, 1024]
    HOP = 320
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    W = Path(__file__).resolve().parents[2] / "external_models" / "dac_16khz" / "weights_16khz.pth"
    model = dac.DAC.load(str(W)).eval().to(dev)
    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    QUICK = os.environ.get("VA_QUICK") == "1"


    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)


    def taps(audio: np.ndarray, keep=None) -> np.ndarray:
        """(B, n) audio -> (B, frames, sum of kept widths) fp16 on DAC's 50 Hz grid."""
        with torch.inference_mode():
            a = model.preprocess(torch.from_numpy(np.ascontiguousarray(audio, np.float32))[:, None].to(dev), 16000)
            n_f = a.shape[-1] // HOP
            h, outs = a, []
            for i, b in enumerate(model.encoder.block):
                h = b(h)
                if i in (1, 2, 3):  # strided blocks 1-3: average each 320-sample frame
                    r = h.shape[-1] // n_f
                    outs.append(h[..., : n_f * r].reshape(h.shape[0], h.shape[1], n_f, r).mean(-1))
                elif i == 4:
                    outs.append(h)
            outs.append(h)  # encoder output latent (after the final conv)
            zq = model.quantizer(h, None)[0]
            outs.append(zq)
            sel = [o for k, o in enumerate(outs) if keep is None or k + 1 in keep]
            return torch.cat(sel, 1).transpose(1, 2).half().cpu().numpy()


    def taps_long(x: np.ndarray, chunk_s: float = 20.0, ctx_s: float = 1.0) -> np.ndarray:
        """One file -> (frames, width); files over chunk_s run in chunk_s pieces with ctx_s of context each side (frames on
        the 320-sample grid, context frames dropped), so no GPU burst is long (a long one hung the machine, 2026-10-22)."""
        n_f = -(-len(x) // HOP)
        if len(x) <= chunk_s * 16000:
            return taps(x[None])[0][:n_f]
        cf, xf = int(chunk_s * 16000) // HOP, int(ctx_s * 16000) // HOP
        xp = np.concatenate([x, np.zeros(n_f * HOP - len(x), np.float32)])
        out = []
        for f0 in range(0, n_f, cf):
            a, b = max(f0 - xf, 0), min(f0 + cf + xf, n_f)
            h = taps(xp[a * HOP: b * HOP][None])[0]
            out.append(h[f0 - a: f0 - a + min(cf, n_f - f0)])
        return np.concatenate(out)


    def save(name, embs):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", np.concatenate(embs))
        np.save(d / "emb_off.npy", np.cumsum([0] + [len(e) for e in embs]))
        log(f"[{name}] {len(embs)} items, {sum(len(e) for e in embs)} frames")


    def crop_set(name, ds, n):
        if (OUT / name / "emb.npy").exists():
            return log(f"[{name}] cached")
        n = 32 if QUICK else n
        embs = []
        for s in range(0, n, 16):
            audio = np.stack([ds[i][0] for i in range(s, min(s + 16, n))])
            embs += list(taps(audio))
        save(name, embs)


    def gold_set(name, items):
        if (OUT / name / "emb.npy").exists():
            return log(f"[{name}] cached")
        embs = []
        for k, x in enumerate(items):
            if QUICK and k >= 8:
                break
            embs.append(taps_long(x))
        save(name, embs)


    if "--heads" in sys.argv:
        keep = tuple(int(t) for t in sys.argv[sys.argv.index("--heads") + 1].split(","))
        N = int(sys.argv[sys.argv.index("--heads") + 2])
        flat = Path(cfg3["train"]["data"]) / "flat"
        train_ids, _ = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
        parts, sizes = [SynthCrops(flat, train_ids, 400, channel, seed=999, epoch_size=N, fixed=True)], []
        for d in cfg3["train"]["resynth"]:
            rflat = consolidate(d)
            n = json.loads((rflat / "done").read_text())["n_clips"]
            parts.append(SynthCrops(rflat, np.arange(n), 400, channel, seed=1000 + len(sizes), epoch_size=N, fixed=True,
                                    tract_left_weight=tuple(cfg3["train"]["tract_left_weight"])))
            sizes.append(n)
        frac = cfg3["train"]["resynth_frac"]
        ds = MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], N, seed=999, fixed=True)
        loader = torch.utils.data.DataLoader(ds, 16, num_workers=8, collate_fn=collate, prefetch_factor=4)
        emb = labs = None
        i, t0 = 0, time.time()
        for audio, lab in loader:
            h = taps(audio.numpy(), keep)
            if emb is None:
                emb = np.lib.format.open_memmap(OUT / "emb.npy", "w+", np.float16, (N, *h.shape[1:]))
                labs = {k: np.lib.format.open_memmap(OUT / f"{k}.npy", "w+", np.float32 if v.dtype != torch.bool else np.bool_,
                                                     (N, *v.shape[1:])) for k, v in lab.items()}
            emb[i:i + len(h)] = h
            for k, v in lab.items():
                labs[k][i:i + len(h)] = v.numpy()
            i += len(h)
            if (i // 16) % 250 == 0:
                log(f"{i}/{N} ({i / (time.time() - t0):.1f} crops/s)")
        emb.flush()
        [v.flush() for v in labs.values()]
        (OUT / "done").write_text(json.dumps({"n": i, "taps": keep, "widths": [WIDTHS[t - 1] for t in keep], "seed": 999,
                                              "crop_frames": 400, "mel_per_frame": 2, "mel_offset": 1}))
        log(f"done: {i} crops")
        raise SystemExit(0)

    # ---- scoring and fitting sets, the same items and order as cache.py
    flat = Path(cfg3["train"]["data"]) / "flat"
    train_ids, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    tlw = tuple(cfg3["train"]["tract_left_weight"])
    parts, sizes = [SynthCrops(flat, train_ids, 400, channel, seed=777, epoch_size=3000, fixed=True)], []
    for d in cfg3["train"]["resynth"]:
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        parts.append(SynthCrops(rflat, np.arange(n), 400, channel, seed=778 + len(sizes), epoch_size=3000, fixed=True, tract_left_weight=tlw))
        sizes.append(n)
    frac = cfg3["train"]["resynth_frac"]
    crop_set("fit", MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], 3000, seed=777, fixed=True), 3000)
    crop_set("synth_train_v2", SynthCrops(flat, val_ids[:512], 400, channel, seed=12345, fixed=True), 512)
    v2s = Path("data/synth/train_v2s")
    crop_set("synth_train_v2s", SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                           400, channel, seed=12345, fixed=True), 512)
    for d in cfg3["train"]["resynth_val"]:
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.sort(np.random.default_rng(0).permutation(n)[:512])
        crop_set(f"resynth_{Path(d).name}", SynthCrops(rflat, ids, 400, channel, seed=12345, fixed=True, tract_left_weight=tlw), 512)
    mods = registry()


    def hillenbrand():
        for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
            item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
            if np.isfinite(item.times).any():
                yield load_audio(item, 16000)


    def f0_corpus(corpus):
        for row in TI.filter_rows(read_manifest(f"data/manifests/{corpus}.csv"), ("dev",)):
            yield load_audio(mods[corpus].load(Path("data") / corpus, row), 16000)


    def closed_phase():
        d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
        tmap = TI.load()
        devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == "dev" for s in d["speaker"]])
        for s, u in sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])}):
            x, sr = sf.read(Path("data/ptdb_tug") / ("FEMALE" if s[0] == "F" else "MALE") / "MIC" / s / f"{u}.wav")
            yield resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)


    gold_set("gold_hillenbrand", hillenbrand())
    gold_set("gold_ptdb_cp", closed_phase())
    for c in ("ptdb_tug", "mir1k", "mdb_stem_synth"):
        gold_set(f"gold_{c}", f0_corpus(c))
    (OUT / "meta.json").write_text(json.dumps({"widths": WIDTHS, "names": ["T1", "T2", "T3", "T4", "T5", "T6"], "mel_per_frame": 2,
                                               "mel_offset": 1, "labels_from": "emb_v1"}))
    log("done")
