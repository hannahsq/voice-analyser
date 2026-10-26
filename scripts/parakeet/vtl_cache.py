"""D-035: frozen features for the VTL probe that the existing caches lack. Teacher v3 (conformer_v2r2_s1_d16k): the
formant encoder's output (Teacher.enc, 128-d, 10 ms frames), fp32 as analyse(). Parakeet layer 5 where none is cached.
Parts (argv[2:], default all; each skipped if done):
- train_T: teacher features on cache_heads.py's labelled crops (length not from f0; tube and cascade), regenerated from
  its seeds and checked against its formant labels;
- val: teacher features on cache.py's synth_train_v2 crops (train_v2's first 512 held-out clips, seed 12345), checked
  against that cache's labels; clip ids for the length labels;
- probe: the f0-invariance probe's 144 clips (eval.invariance), whole clips, teacher and Parakeet layer 5;
- hb: Hillenbrand dev files, whole files, teacher features at the reference frames (cache.py's order, checked).
Run from the main checkout: VA_DEVICE=cuda python scripts/parakeet/vtl_cache.py <out dir> [parts]"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.data.manifest import read_manifest  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.eval import invariance as inv  # noqa: E402
from vocal_analysis.eval import tiers as TI  # noqa: E402
from vocal_analysis.eval.datasets import load_audio, registry  # noqa: E402
from vocal_analysis.features.vqt import HOP, TorchVQT  # noqa: E402
from vocal_analysis.models.infer import load_teacher  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, collate, consolidate, split_ids  # noqa: E402

OUT = Path(sys.argv[1])
PARTS = sys.argv[2:] or ["train_T", "val", "probe", "hb"]
WT = Path(__file__).resolve().parents[2]
HEADS, EMB = WT / "cache" / "heads_l3_6", WT / "cache" / "emb_v1"
CROP = 400
dev = teacher = vq = None
_h = {}
cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def enc_crops(audio):  # (B, n) crops with context -> (B, 400, 128)
    with torch.inference_mode():
        teacher(vq.log_power_with_context(audio.to(dev), CROP))
    return _h["h"].float().cpu().numpy()


def enc_file(x):  # whole file -> (n_frames, 128), frame i at i * 10 ms
    n = len(x) // HOP + 1
    with torch.inference_mode():
        teacher(vq.log_power(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(dev), n))
    return _h["h"][0].float().cpu().numpy()


def done(part):
    return (OUT / part / "done").exists()


def finish(part, **info):
    (OUT / part / "done").write_text(json.dumps(info))
    log(f"[{part}] done {info}")


def main():
    global dev, teacher, vq
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    teacher = load_teacher("runs/conformer_v2r2_s1_d16k/ckpt_last.pt", dev)[0]
    vq = TorchVQT(dev)
    teacher.enc.register_forward_hook(lambda m, i, o: _h.__setitem__("h", o))
    if "train_T" in PARTS and not done("train_T"):
        (OUT / "train_T").mkdir(parents=True, exist_ok=True)
        v = np.load(HEADS / "vtl.npz")
        idx = np.flatnonzero(np.isfinite(v["length_cm"]) & ~v["from_f0"])
        flat = Path(cfg3["train"]["data"]) / "flat"
        train_ids, _ = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
        N = json.loads((HEADS / "done").read_text())["n"]
        parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=999, epoch_size=N, fixed=True)], []
        for d in cfg3["train"]["resynth"]:  # as cache_heads.py, so MixedCrops draws the same parts
            rflat = consolidate(d)
            n = json.loads((rflat / "done").read_text())["n_clips"]
            parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=1000 + len(sizes), epoch_size=N, fixed=True,
                                    tract_left_weight=tuple(cfg3["train"]["tract_left_weight"])))
            sizes.append(n)
        frac = cfg3["train"]["resynth_frac"]
        ds = MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], N, seed=999, fixed=True)
        cf = np.load(HEADS / "formants.npy", mmap_mode="r")
        feats = np.lib.format.open_memmap(OUT / "train_T" / "feats.npy", "w+", np.float16, (len(idx), CROP, 128))
        k, t0 = 0, time.time()
        for audio, lab in DataLoader(Subset(ds, idx), 16, num_workers=8, collate_fn=collate, prefetch_factor=4):
            b = len(audio)
            assert np.array_equal(lab["formants"].numpy(), np.asarray(cf[idx[k:k + b]]), equal_nan=True), k
            feats[k:k + b] = enc_crops(audio)
            k += b
            if (k // 16) % 100 == 0:
                log(f"  [train_T] {k}/{len(idx)} ({k / (time.time() - t0):.1f} crops/s)")
        feats.flush()
        np.save(OUT / "train_T" / "idx.npy", idx)
        finish("train_T", n=int(k))

    if "val" in PARTS and not done("val"):
        (OUT / "val").mkdir(parents=True, exist_ok=True)
        flat = Path(cfg3["train"]["data"]) / "flat"
        _, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
        ids = val_ids[:512]
        ds = SynthCrops(flat, ids, CROP, channel, seed=12345, fixed=True)  # cache.py's synth_train_v2
        ef = np.load(EMB / "synth_train_v2" / "formants.npy", mmap_mode="r").reshape(512, CROP, 4)
        out, k = [], 0
        for audio, lab in DataLoader(ds, 16, num_workers=8, collate_fn=collate):
            b = len(audio)
            assert np.array_equal(lab["formants"].numpy(), np.asarray(ef[k:k + b]), equal_nan=True), k
            out.append(enc_crops(audio).astype(np.float16))
            k += b
        np.save(OUT / "val" / "feats.npy", np.concatenate(out))
        np.save(OUT / "val" / "clip.npy", ids)
        finish("val", n=int(k))

    if "probe" in PARTS and not done("probe"):
        (OUT / "probe").mkdir(parents=True, exist_ok=True)
        fe, penc = load.feature_extractor(), load.encoder(dev)
        penc.layers = penc.layers[:5]
        clips = inv.probe_clips()
        T, P = [], []
        for c in clips:
            T.append(enc_file(c.audio).astype(np.float16))
            f = fe([c.audio], sampling_rate=16000, return_tensors="pt")
            with torch.inference_mode():
                h = penc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states
            P.append(h[5][0].half().cpu().numpy())
        np.save(OUT / "probe" / "T.npy", np.stack(T))
        np.save(OUT / "probe" / "P.npy", np.stack(P))
        np.savez(OUT / "probe" / "meta.npz", vowel=np.array([c.vowel for c in clips]), length_cm=np.array([c.length_cm for c in clips]),
                 f0=np.array([c.f0 for c in clips]))
        finish("probe", n=len(clips), T_frames=int(T[0].shape[0]), P_frames=int(P[0].shape[0]))

    if "hb" in PARTS and not done("hb"):
        (OUT / "hb").mkdir(parents=True, exist_ok=True)
        mods = registry()
        ref = np.load(EMB / "gold_hillenbrand" / "ref_formants.npy")
        feats, k = [], 0
        for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
            item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
            ok = np.isfinite(item.times)
            if not ok.any():
                continue
            t = item.times[ok]
            assert np.allclose(item.formants[ok][:, :4], ref[k:k + len(t)], equal_nan=True), k
            h = enc_file(load_audio(item, 16000))
            feats.append(h[np.clip(np.round(t / 0.01).astype(int), 0, len(h) - 1)].astype(np.float16))
            k += len(t)
        assert k == len(ref), (k, len(ref))
        np.save(OUT / "hb" / "feats.npy", np.concatenate(feats))
        finish("hb", n_ref=int(k), n_files=len(feats))


if __name__ == "__main__":
    main()
