"""D-037 / D-038 (drafts): frozen-trunk features for the VTL head refits and the cascades. For one teacher checkpoint, per
part, every frame's formant-encoder output `h` (128-d, fp16), the trunk's own F1-F4 (expected position, log2 Hz, fp32)
and, for a trunk with a VTL head (D-036), that head's mean (log2 cm) and log variance.
Parts (argv[3:], default all; each skipped if done):
- train: every train_v2 training clip that is a tube clip with length not drawn from f0 (29,356), each under two
  independent channel draws (fixed seeds), 400-frame crops with context; also the frame labels (formants, sonorant) and
  each crop's clip id and log2 length;
- val: train_v2's first 512 held-out clips (seed 12345, as vtl_cache.py / cache.py's synth_train_v2), clip ids;
- probe: the f0-invariance probe's 144 clips (eval.invariance), whole clips;
- hb: Hillenbrand dev files, whole files, at the reference frames (vtl_cache.py's order, checked).
Whole files and fp32 as D-035's caches, so teacher v3's rebuilt parts can be checked against them.
Run from the main checkout: VA_DEVICE=cuda python scripts/analysis/d037_cache.py <checkpoint> <out dir> [parts]
Env VA_LIMIT=n: only the first n train crops and val clips (smoke tests)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vocal_analysis.config import DeviceConfig
from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.device import setup_device
from vocal_analysis.eval import invariance as inv
from vocal_analysis.eval import tiers as TI
from vocal_analysis.eval.datasets import load_audio, registry
from vocal_analysis.features.vqt import HOP, TorchVQT
from vocal_analysis.models.infer import decode_outputs, load_teacher
from vocal_analysis.synth.channel import ChannelConfig
from vocal_analysis.train.data import SynthCrops, collate, split_ids

CROP = 400
DRAWS = 2
EMB = Path("cache/emb_v1")
LIMIT = int(os.environ.get("VA_LIMIT", 0))


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Trunk:
    """the teacher with a hook on its formant encoder: forward -> (h, log2 F1-F4, vtl mean, vtl log variance) per frame"""

    def __init__(self, ckpt, dev):
        self.dev = dev
        self.model = load_teacher(ckpt, dev)[0]
        self.vq = TorchVQT(dev)
        self._h = {}
        self.model.enc.register_forward_hook(lambda m, i, o: self._h.__setitem__("h", o))
        self.has_vtl = bool(self.model.cfg.vtl_head)

    def _decode(self, out):
        h = self._h["h"].float().cpu().numpy()
        o = {k: v.float().cpu().numpy() for k, v in out.items()}
        fm, mu, lv = [], [], []
        for b in range(h.shape[0]):
            d = decode_outputs({k: v[b] for k, v in o.items()})
            fm.append(np.log2(d["formants"][:, :4]))
            if self.has_vtl:
                mu.append(o["vtl_mu"][b])
                lv.append(o["vtl_logvar"][b])
        vtl = (np.stack(mu), np.stack(lv)) if self.has_vtl else (None, None)
        return h, np.stack(fm), *vtl

    def crops(self, audio):  # (B, n) crops with context
        with torch.inference_mode():
            return self._decode(self.model(self.vq.log_power_with_context(audio.to(self.dev), CROP)))

    def file(self, x):  # whole file, frame i at i * 10 ms
        n = len(x) // HOP + 1
        with torch.inference_mode():
            r = self._decode(self.model(self.vq.log_power(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(self.dev), n)))
        return tuple(None if a is None else a[0] for a in r)


def save_cols(d, fm, mu, lv):
    np.save(d / "log2F.npy", fm.astype(np.float32))
    if mu is not None:
        np.save(d / "vtl_mu.npy", mu.astype(np.float32))
        np.save(d / "vtl_logvar.npy", lv.astype(np.float32))


def main():
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    parts = sys.argv[3:] or ["train", "val", "probe", "hb"]
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    T = Trunk(ckpt, dev)
    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))  # teacher v3's data and channel, as D-035
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    data = Path(cfg3["train"]["data"])
    flat = data / "flat"
    n_clips = json.loads((data / "index.json").read_text())["n_clips"]
    done = lambda p: (out / p / "done").exists()  # noqa: E731
    finish = lambda p, **i: ((out / p / "done").write_text(json.dumps({"ckpt": ckpt, **i})), log(f"[{p}] done {i}"))  # noqa: E731
    out.mkdir(parents=True, exist_ok=True)
    log(f"{ckpt}: VTL head {T.has_vtl}; parts {parts}")

    if "train" in parts and not done("train"):
        d = out / "train"
        d.mkdir(exist_ok=True)
        train_ids, _ = split_ids(n_clips)
        tube, f0c = np.load(flat / "clip_tube.npy"), np.load(flat / "clip_length_from_f0.npy")
        ids = train_ids[tube[train_ids] & ~f0c[train_ids]]
        ids = np.concatenate([ids] * DRAWS)  # item i -> clip ids[i], channel seeded by (seed, i): a fresh draw per copy
        ids = ids[:LIMIT] if LIMIT else ids
        N = len(ids)
        ds = SynthCrops(flat, ids, CROP, channel, seed=37037, fixed=True)
        feats = np.lib.format.open_memmap(d / "feats.npy", "w+", np.float16, (N, CROP, 128))
        fm = np.lib.format.open_memmap(d / "log2F.npy", "w+", np.float32, (N, CROP, 4))
        lab_f = np.lib.format.open_memmap(d / "label_formants.npy", "w+", np.float32, (N, CROP, 4))
        lab_s = np.lib.format.open_memmap(d / "label_sonorant.npy", "w+", np.float32, (N, CROP))
        mu = lv = None
        if T.has_vtl:
            mu = np.lib.format.open_memmap(d / "vtl_mu.npy", "w+", np.float32, (N, CROP))
            lv = np.lib.format.open_memmap(d / "vtl_logvar.npy", "w+", np.float32, (N, CROP))
        k, t0 = 0, time.time()
        for audio, lab in DataLoader(ds, 16, num_workers=8, collate_fn=collate, prefetch_factor=4):
            b = len(audio)
            h, f, m, v = T.crops(audio)
            feats[k:k + b], fm[k:k + b] = h, f
            lab_f[k:k + b] = lab["formants"].numpy()[..., :4]
            lab_s[k:k + b] = lab["sonorant"].numpy()
            if mu is not None:
                mu[k:k + b], lv[k:k + b] = m, v
            k += b
            if (k // 16) % 200 == 0:
                log(f"  [train] {k}/{N} ({k / (time.time() - t0):.1f} crops/s)")
        for a in (feats, fm, lab_f, lab_s, mu, lv):
            if a is not None:
                a.flush()
        np.save(d / "clip.npy", ids)
        np.save(d / "log2_length.npy", np.log2(np.load(flat / "clip_length_cm.npy")[ids]).astype(np.float32))
        finish("train", n=int(k), clips=int(N // DRAWS), draws=DRAWS)

    if "val" in parts and not done("val"):
        d = out / "val"
        d.mkdir(exist_ok=True)
        _, val_ids = split_ids(n_clips)
        ids = val_ids[:LIMIT or 512]
        ds = SynthCrops(flat, ids, CROP, channel, seed=12345, fixed=True)  # cache.py's synth_train_v2
        ef = np.load(EMB / "synth_train_v2" / "formants.npy", mmap_mode="r").reshape(512, CROP, 4)[:len(ids)]
        H, F, M, V, k = [], [], [], [], 0
        for audio, lab in DataLoader(ds, 16, num_workers=8, collate_fn=collate):
            b = len(audio)
            assert np.array_equal(lab["formants"].numpy(), np.asarray(ef[k:k + b]), equal_nan=True), k
            h, f, m, v = T.crops(audio)
            H.append(h.astype(np.float16)), F.append(f), M.append(m), V.append(v)
            k += b
        np.save(d / "feats.npy", np.concatenate(H))
        save_cols(d, np.concatenate(F), *((np.concatenate(M), np.concatenate(V)) if T.has_vtl else (None, None)))
        np.save(d / "clip.npy", ids)
        finish("val", n=int(k))

    if "probe" in parts and not done("probe"):
        d = out / "probe"
        d.mkdir(exist_ok=True)
        clips = inv.probe_clips()
        r = [T.file(c.audio) for c in clips]
        np.save(d / "feats.npy", np.stack([a[0] for a in r]).astype(np.float16))
        save_cols(d, np.stack([a[1] for a in r]), *((np.stack([a[2] for a in r]), np.stack([a[3] for a in r])) if T.has_vtl else (None, None)))
        np.savez(d / "meta.npz", vowel=np.array([c.vowel for c in clips]), length_cm=np.array([c.length_cm for c in clips]),
                 f0=np.array([c.f0 for c in clips]))
        finish("probe", n=len(clips))

    if "hb" in parts and not done("hb"):
        d = out / "hb"
        d.mkdir(exist_ok=True)
        mods = registry()
        ref = np.load(EMB / "gold_hillenbrand" / "ref_formants.npy")
        H, F, M, V, files, k = [], [], [], [], [], 0
        for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
            item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
            ok = np.isfinite(item.times)
            if not ok.any():
                continue
            t = item.times[ok]
            assert np.allclose(item.formants[ok][:, :4], ref[k:k + len(t)], equal_nan=True), k
            h, f, m, v = T.file(load_audio(item, 16000))
            j = np.clip(np.round(t / 0.01).astype(int), 0, len(h) - 1)
            H.append(h[j].astype(np.float16)), F.append(f[j])
            if T.has_vtl:
                M.append(m[j]), V.append(v[j])
            files += [row["file"] if "file" in row else str(k)] * len(t)
            k += len(t)
        assert k == len(ref), (k, len(ref))
        np.save(d / "feats.npy", np.concatenate(H))
        save_cols(d, np.concatenate(F), *((np.concatenate(M), np.concatenate(V)) if T.has_vtl else (None, None)))
        np.save(d / "file.npy", np.array(files))
        finish("hb", n_ref=int(k), n_files=len(H))


if __name__ == "__main__":
    main()
