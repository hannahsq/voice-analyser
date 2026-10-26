"""D-033 part 4b: S2 models scored on exactly the scoring cache's frames: the fixed synthetic crops regenerated from the
same seeds as cache.py (train_v2 / train_v2s validation, 512 crops, seed 12345), Hillenbrand dev and the PTDB-TUG dev
closed-phase files whole, in cache.py's item order. Writes <cache>/<set>/m_<name>_log2F.npy and m_<name>_sigma.npy
(per reference frame), which eval_heads.py reads with `col:<name>`. Checks the regenerated crops' labels against the cache.
Run from the main checkout: VA_DEVICE=cuda python scripts/parakeet/s2_score.py <scratch dir> <cache dir> name=ckpt ..."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.config import DeviceConfig
from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.device import setup_device
from vocal_analysis.eval import tiers as TI
from vocal_analysis.eval.datasets import load_audio, registry
from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.infer import analyse, decode_outputs, load_teacher
from vocal_analysis.synth.channel import ChannelConfig
from vocal_analysis.train.data import SynthCrops, consolidate, split_ids

sys.stdout.reconfigure(encoding="utf-8")
S, C = Path(sys.argv[1]), Path(sys.argv[2])
TIER = os.environ.get("VA_TIER", "dev")  # as cache.py: "semi" scores the semi-private gold sets only
dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
vq = TorchVQT(dev)
cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
flat = Path(cfg3["train"]["data"]) / "flat"
v2s = Path("data/synth/train_v2s")
synth = {"synth_train_v2": SynthCrops(flat, split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])[1][:512],
                                      400, channel, seed=12345, fixed=True),
         "synth_train_v2s": SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                       400, channel, seed=12345, fixed=True)}
tlw = tuple(cfg3["train"]["tract_left_weight"])
for d in cfg3["train"]["resynth_val"]:  # as cache.py: 512 clips (permutation seed 0), crop seed 12345
    rflat = consolidate(d)
    n = json.loads((rflat / "done").read_text())["n_clips"]
    ids = np.sort(np.random.default_rng(0).permutation(n)[:512])
    synth[f"resynth_{Path(d).name}"] = SynthCrops(rflat, ids, 400, channel, seed=12345, fixed=True, tract_left_weight=tlw)
mods = registry()


def hillenbrand():
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), (TIER,)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        ok = np.isfinite(item.times)
        if ok.any():
            yield load_audio(item, 16000), item.times[ok]


def closed_phase():
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    tmap = TI.load()
    devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == TIER for s in d["speaker"]])
    for s, u in sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])}):
        x, sr = sf.read(Path("data/ptdb_tug") / ("FEMALE" if s[0] == "F" else "MALE") / "MIC" / s / f"{u}.wav")
        x = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
        i = np.flatnonzero((d["speaker"] == s) & (d["utt"] == u))
        yield x, d["t"][i].astype(float)


for name, ck in (a.split("=", 1) for a in sys.argv[3:]):
    m = load_teacher(ck, dev)[0]
    for sname, ds in (synth.items() if TIER == "dev" else ()):
        if (C / sname / f"m_{name}_log2F.npy").exists():
            continue
        F, Sg, chk = [], [], []
        for s0 in range(0, 512, 16):
            items = [ds[i] for i in range(s0, min(s0 + 16, 512))]
            audio = np.stack([a for a, _ in items])
            with torch.inference_mode():
                out = m(vq.log_power_with_context(torch.from_numpy(audio).to(dev), 400))
            for i, (_, lab) in enumerate(items):
                d = decode_outputs({k: v[i].float().cpu().numpy() for k, v in out.items()})
                F.append(np.log2(d["formants"][:, :4]))
                Sg.append(d["formant_sigma_cents"][:, :4])
                chk.append(lab["formants"][:, :4])
        cached = np.load(C / sname / "formants.npy")
        assert np.array_equal(np.concatenate(chk), cached, equal_nan=True), f"{sname}: regenerated crops differ from the cache"
        np.save(C / sname / f"m_{name}_log2F.npy", np.concatenate(F))
        np.save(C / sname / f"m_{name}_sigma.npy", np.concatenate(Sg))
        print(f"{name} {sname}: done (labels match the cache)", flush=True)
    for sname, gen in (("gold_hillenbrand", hillenbrand()), ("gold_ptdb_cp", closed_phase())):
        if (C / sname / f"m_{name}_log2F.npy").exists():
            continue
        F, Sg = [], []
        for x, times in gen:
            r = analyse(m, x, dev)
            idx = np.clip(np.round(times / 0.01).astype(int), 0, len(r["f0"]) - 1)
            F.append(np.log2(r["formants"][idx, :4]))
            Sg.append(r["formant_sigma_cents"][idx, :4])
        F = np.concatenate(F)
        assert len(F) == len(np.load(C / sname / "mel.npy")), f"{sname}: frame count differs from the cache"
        np.save(C / sname / f"m_{name}_log2F.npy", F)
        np.save(C / sname / f"m_{name}_sigma.npy", np.concatenate(Sg))
        print(f"{name} {sname}: done", flush=True)
