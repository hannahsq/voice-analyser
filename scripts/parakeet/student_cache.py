"""D-034 head stage: cache a frozen student's 1024-dimensional output (student.py) at our 10 ms frames for every item in
the Parakeet scoring cache (`emb_v1`) that eval_heads.py reads, in the same item order, so labels, reference frames and
teacher v3's columns are reused from there; and a head cache of N fixed crops (cache_heads.py's seed 999 mixture).
Frames: on a crop, student frame j sits at mel position j + 4 (the VQT's context, as the crops' labels), so heads use
ratio 1, offset 4. Whole files run in one pass (frames centred at i * 10 ms, the file zero-padded by 40 ms at its end) and
frames 4.. are stored, so stored frame k is file frame k + 4 and the same head offset reads them (the first 40 ms of a
file clamp to its frame 4). bf16 autocast, as the student trained; fp16 storage.
Writes <out>/<set>/emb.npy (frames, 1024), emb_off.npy and <out>/meta.json; with --heads N, <out>/heads/emb.npy
(N, 400, 1024) and labels, then heads/done. NAME labels the encoder in head checkpoints (eval_heads.py --feat=NAME=<out>).
Run from the main checkout: VA_DEVICE=cuda python <worktree>/scripts/parakeet/student_cache.py <scratch dir> <student ckpt> <out dir> NAME [--heads N]"""
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
from student import load_student  # noqa: E402

from vocal_analysis.data.manifest import read_manifest  # noqa: E402
from vocal_analysis.eval import tiers as TI  # noqa: E402
from vocal_analysis.eval.datasets import load_audio, registry  # noqa: E402
from vocal_analysis.features.vqt import HOP, TorchVQT  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, collate, consolidate, split_ids  # noqa: E402

CROP, OFFSET = 400, 4
REF = Path(r"C:\github\voice-analyser-parakeet\cache\emb_v1")  # the scoring cache whose items are mirrored

# all work under the main guard: DataLoader workers re-import this file on Windows (spawn)
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    S, CK, OUT, NAME = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    model = load_student(CK, dev).requires_grad_(False)
    vq = TorchVQT(dev)
    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    tlw = tuple(cfg3["train"]["tract_left_weight"])
    QUICK = os.environ.get("VA_QUICK") == "1"

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    def feats(logpow: torch.Tensor) -> np.ndarray:
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16):
            return model(logpow).float().half().cpu().numpy()

    def crop_feats(audio: np.ndarray) -> np.ndarray:
        """(B, crop samples with context) -> (B, 400, 1024)"""
        return feats(vq.log_power_with_context(torch.from_numpy(np.ascontiguousarray(audio, np.float32)).to(dev), CROP))

    def file_feats(x: np.ndarray) -> np.ndarray:
        """one file -> (frames, 1024), stored frame k = file frame k + OFFSET"""
        n = len(x) // HOP + 1
        assert n <= 6000, f"{n} frames: too long for one GPU pass"
        xp = np.concatenate([x, np.zeros(OFFSET * HOP, np.float32)]).astype(np.float32)
        return feats(vq.log_power(torch.from_numpy(xp)[None].to(dev), n + OFFSET))[0][OFFSET:]

    def save(name, embs):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", np.concatenate(embs))
        np.save(d / "emb_off.npy", np.cumsum([0] + [len(e) for e in embs]))
        ref = len(np.load(REF / name / "ref_off.npy")) - 1
        assert QUICK or len(embs) == ref, f"[{name}] {len(embs)} items against the scoring cache's {ref}"
        log(f"[{name}] {len(embs)} items, {sum(len(e) for e in embs)} frames")

    def crop_set(name, ds, n):
        if (OUT / name / "emb.npy").exists():
            return log(f"[{name}] cached")
        n = 32 if QUICK else n
        embs = []
        for s in range(0, n, 16):
            embs += list(crop_feats(np.stack([ds[i][0] for i in range(s, min(s + 16, n))])))
        save(name, embs)

    def gold_set(name, items):
        if (OUT / name / "emb.npy").exists():
            return log(f"[{name}] cached")
        embs = []
        for k, x in enumerate(items):
            if QUICK and k >= 8:
                break
            embs.append(file_feats(x))
        save(name, embs)

    if "--heads" in sys.argv:  # cache_heads.py's crops and labels (seed 999)
        N = int(sys.argv[sys.argv.index("--heads") + 1])
        N = 64 if QUICK else N
        H = OUT / "heads"
        if (H / "done").exists():
            log("[heads] cached")
            raise SystemExit(0)
        H.mkdir(parents=True, exist_ok=True)
        flat = Path(cfg3["train"]["data"]) / "flat"
        train_ids, _ = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
        parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=999, epoch_size=N, fixed=True)], []
        for d in cfg3["train"]["resynth"]:
            rflat = consolidate(d)
            n = json.loads((rflat / "done").read_text())["n_clips"]
            parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=1000 + len(sizes), epoch_size=N, fixed=True,
                                    tract_left_weight=tlw))
            sizes.append(n)
        frac = cfg3["train"]["resynth_frac"]
        ds = MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], N, seed=999, fixed=True)
        loader = torch.utils.data.DataLoader(ds, 16, num_workers=8, collate_fn=collate, prefetch_factor=4)
        emb = labs = None
        i, t0 = 0, time.time()
        for audio, lab in loader:
            h = crop_feats(audio.numpy())
            if emb is None:
                emb = np.lib.format.open_memmap(H / "emb.npy", "w+", np.float16, (N, *h.shape[1:]))
                labs = {k: np.lib.format.open_memmap(H / f"{k}.npy", "w+", np.float32 if v.dtype != torch.bool else np.bool_,
                                                     (N, *v.shape[1:])) for k, v in lab.items()}
            emb[i:i + len(h)] = h
            for k, v in lab.items():
                labs[k][i:i + len(h)] = v.numpy()
            i += len(h)
            if (i // 16) % 250 == 0:
                log(f"{i}/{N} ({i / (time.time() - t0):.1f} crops/s)")
        emb.flush()
        [v.flush() for v in labs.values()]
        (H / "done").write_text(json.dumps({"n": i, "encoder": NAME, "checkpoint": str(CK), "widths": [1024], "seed": 999,
                                            "crop_frames": CROP, "mel_per_frame": 1, "mel_offset": OFFSET}))
        log(f"done: {i} crops")
        raise SystemExit(0)

    # ---- the scoring sets eval_heads.py reads, the same items and order as cache.py
    flat = Path(cfg3["train"]["data"]) / "flat"
    train_ids, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    crop_set("synth_train_v2", SynthCrops(flat, val_ids[:512], CROP, channel, seed=12345, fixed=True), 512)
    v2s = Path("data/synth/train_v2s")
    crop_set("synth_train_v2s", SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                           CROP, channel, seed=12345, fixed=True), 512)
    for d in cfg3["train"]["resynth_val"]:
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.sort(np.random.default_rng(0).permutation(n)[:512])
        crop_set(f"resynth_{Path(d).name}", SynthCrops(rflat, ids, CROP, channel, seed=12345, fixed=True, tract_left_weight=tlw), 512)
    mods = registry()

    def hillenbrand():
        for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
            item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
            if np.isfinite(item.times).any():
                yield load_audio(item, 16000)

    def closed_phase():
        d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
        tmap = TI.load()
        devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == "dev" for s in d["speaker"]])
        for s, u in sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])}):
            x, sr = sf.read(Path("data/ptdb_tug") / ("FEMALE" if s[0] == "F" else "MALE") / "MIC" / s / f"{u}.wav")
            yield resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)

    gold_set("gold_hillenbrand", hillenbrand())
    gold_set("gold_ptdb_cp", closed_phase())
    (OUT / "meta.json").write_text(json.dumps({"encoder": NAME, "checkpoint": str(CK), "widths": [1024], "mel_per_frame": 1,
                                               "mel_offset": OFFSET, "labels_from": "emb_v1"}))
    log("done")
