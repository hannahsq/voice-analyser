"""D-033 part 3: cache Parakeet TDT 0.6B v3's hidden states 0-6 (subsampling output and layers 1-6; fp16, at the
encoder's 80 ms frames) for every item the probes fit or score, with the labels and teacher v3's outputs on the same
frames, so every layer combination can be fitted and scored from the cache alone.
Sets (as probe.py / bias.py): fit: teacher v3's training mixture, 3000 fixed 4 s crops (seed 777); synthetic validation
(train_v2, train_v2s, resynthesised VocalSet / Zeroth-Korean val; 512 fixed crops each, seed 12345); gold, dev speakers:
Hillenbrand (formants at the reference times), PTDB-TUG / MIR-1K / MDB-stem-synth (f0 at the reference frames), and the
PTDB-TUG dev closed-phase frames (ptdb_egg_sub.npz).
Each set -> <out>/<set>/: emb.npy (frames, 7, 1024) fp16, emb_off.npy (item offsets into emb), and per-frame arrays
(10 ms label frames or reference frames) with ref_off.npy offsets and mel positions (mel.npy). Run from the main
checkout: VA_DEVICE=cuda python <worktree>/scripts/parakeet/cache.py <scratch dir> <out dir>"""
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
from head import crop_mel_pos  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.data.manifest import read_manifest  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.eval import tiers as TI  # noqa: E402
from vocal_analysis.eval.datasets import load_audio, registry  # noqa: E402
from vocal_analysis.features.vqt import TorchVQT  # noqa: E402
from vocal_analysis.models.infer import analyse, decode_outputs, load_teacher  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, consolidate, split_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
S, OUT = Path(sys.argv[1]), Path(sys.argv[2])
DEPTH = 6
dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
fe = load.feature_extractor()
enc = load.encoder(dev)
enc.layers = enc.layers[:DEPTH]
teacher = load_teacher("runs/conformer_v2r2_s1_d16k/ckpt_last.pt", dev)[0]
vq = TorchVQT(dev)
cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
tlw = tuple(cfg3["train"]["tract_left_weight"])
CROP = 400
QUICK = os.environ.get("VA_QUICK") == "1"  # code check: a few items per set


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def hidden(audio_batch) -> np.ndarray:
    """list of (n,) audio -> (B, E, 7, 1024) fp16 (equal lengths in a batch)."""
    f = fe(list(audio_batch), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states
    return torch.stack(h, 2).half().cpu().numpy()


def teacher_frames(r, idx):
    """analyse() output, frame indices -> teacher v3's per-frame columns."""
    return {"t_log2F": np.log2(r["formants"][idx, :4]), "t_sigma": r["formant_sigma_cents"][idx, :4],
            "t_log2f0": np.log2(r["f0"][idx]), "t_voiced": r["voiced_p"][idx], "t_sonorant": r["sonorant_p"][idx],
            "t_log2B": np.log2(r["bandwidths"][idx, :4])}


def save(name, embs, cols):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "emb.npy", np.concatenate(embs))
    np.save(d / "emb_off.npy", np.cumsum([0] + [len(e) for e in embs]))
    for k, v in cols.items():
        np.save(d / f"{k}.npy", np.concatenate(v) if isinstance(v, list) else v)
    log(f"[{name}] {len(embs)} items, {sum(len(e) for e in embs)} encoder frames")


def crop_set(name, ds, n):
    """fixed crops: labels and teacher v3 at all 400 frames (mel position j + 4)."""
    if (OUT / name / "emb.npy").exists():
        log(f"[{name}] cached")
        return
    n = 32 if QUICK else n
    embs, cols = [], {k: [] for k in ("formants", "bandwidths", "f0", "voiced", "sonorant", "w_f0", "w_formant",
                                     "t_log2F", "t_sigma", "t_log2f0", "t_voiced", "t_sonorant", "t_log2B")}
    for s in range(0, n, 16):
        items = [ds[i] for i in range(s, min(s + 16, n))]
        audio = np.stack([a for a, _ in items])
        h = hidden(audio)
        with torch.inference_mode():
            out = teacher(vq.log_power_with_context(torch.from_numpy(audio).to(dev), CROP))
        for i, (_, lab) in enumerate(items):
            embs.append(h[i])
            r = decode_outputs({k: v[i].float().cpu().numpy() for k, v in out.items()})
            for k, v in teacher_frames(r, np.arange(CROP)).items():
                cols[k].append(v)
            for k in ("formants", "bandwidths"):
                cols[k].append(lab[k][:, :4])
            for k in ("f0", "voiced", "sonorant"):
                cols[k].append(lab[k])
            cols["w_f0"].append(lab["w_f0"])
            cols["w_formant"].append(lab["w_formant"])
        if (s // 16) % 20 == 0:
            log(f"  [{name}] {s + len(items)}/{n}")
    cols["mel"] = np.tile(crop_mel_pos(CROP, "cpu").numpy(), n)
    cols["ref_off"] = np.arange(n + 1) * CROP
    save(name, embs, cols)


# ---- fit and synthetic validation crops
flat = Path(cfg3["train"]["data"]) / "flat"
train_ids, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=777, epoch_size=3000, fixed=True)], []
for d in cfg3["train"]["resynth"]:
    rflat = consolidate(d)
    n = json.loads((rflat / "done").read_text())["n_clips"]
    parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=778 + len(sizes), epoch_size=3000, fixed=True, tract_left_weight=tlw))
    sizes.append(n)
frac = cfg3["train"]["resynth_frac"]
crop_set("fit", MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], 3000, seed=777, fixed=True), 3000)
crop_set("synth_train_v2", SynthCrops(flat, val_ids[:512], CROP, channel, seed=12345, fixed=True), 512)
v2s = Path("data/synth/train_v2s")
crop_set("synth_train_v2s", SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                       CROP, channel, seed=12345, fixed=True), 512)
for d in cfg3["train"]["resynth_val"]:
    rflat = consolidate(d)
    n = json.loads((rflat / "done").read_text())["n_clips"]
    ids = np.sort(np.random.default_rng(0).permutation(n)[:512])
    crop_set(f"resynth_{Path(d).name}", SynthCrops(rflat, ids, CROP, channel, seed=12345, fixed=True, tract_left_weight=tlw), 512)


# ---- gold sets, dev speakers: whole files, reference frames only
def gold_set(name, items, window=None):
    """items: iterable of (audio, times (s), dict of reference columns) -> cache."""
    if (OUT / name / "emb.npy").exists():
        log(f"[{name}] cached")
        return
    embs, cols, n = [], {}, 0
    for x, times, refs in items:
        if QUICK and n >= 8:
            break
        embs.append(hidden([x])[0])
        r = analyse(teacher, x, dev, window_frames=window)
        idx = np.clip(np.round(times / 0.01).astype(int), 0, len(r["f0"]) - 1)
        for k, v in {**refs, **teacher_frames(r, idx), "mel": times / 0.01}.items():
            cols.setdefault(k, []).append(v)
        cols.setdefault("ref_len", []).append(np.array([len(times)]))
        n += 1
        if n % 500 == 0:
            log(f"  [{name}] {n}")
    cols["ref_off"] = np.cumsum([0] + [int(v[0]) for v in cols.pop("ref_len")])
    save(name, embs, cols)


mods = registry()


def hillenbrand():
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        ok = np.isfinite(item.times)
        if ok.any():
            yield load_audio(item, 16000), item.times[ok], {"ref_formants": item.formants[ok][:, :4],
                                                             "group": np.array([item.meta.get("group", "?")] * int(ok.sum()))}


def f0_corpus(corpus):
    def gen():
        for row in TI.filter_rows(read_manifest(f"data/manifests/{corpus}.csv"), ("dev",)):
            item = mods[corpus].load(Path("data") / corpus, row)
            yield load_audio(item, 16000), np.asarray(item.times, float), {"ref_f0": item.f0.astype(float)}
    return gen()


def closed_phase():
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    tmap = TI.load()
    devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == "dev" for s in d["speaker"]])
    for s, u in sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])}):
        x, sr = sf.read(Path("data/ptdb_tug") / ("FEMALE" if s[0] == "F" else "MALE") / "MIC" / s / f"{u}.wav")
        x = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
        i = np.flatnonzero((d["speaker"] == s) & (d["utt"] == u))
        yield x, d["t"][i].astype(float), {"ref_cp": d["cp"][i], "sex": d["sex"][i]}


gold_set("gold_hillenbrand", hillenbrand())
gold_set("gold_ptdb_cp", closed_phase())
for c in ("ptdb_tug", "mir1k", "mdb_stem_synth"):
    # MDB-stem-synth's tracks run for minutes: teacher v3's whole-file attention needed > 10 GB per matrix and took the
    # machine to 7 GB free, so its teacher columns come from D-028's windowed analysis (400 frames, hop 200)
    gold_set(f"gold_{c}", f0_corpus(c), window=400 if c == "mdb_stem_synth" else None)
log("done")
