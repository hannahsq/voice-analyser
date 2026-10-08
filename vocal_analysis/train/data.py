"""Training data from synthetic or resynthesised shards: memory-mapped, randomly cropped, channel-augmented
on the fly.

`consolidate()` turns a shard set into flat .npy arrays once (audio int16, labels, offsets),
so DataLoader workers share the OS page cache instead of each holding copies of shards.

Every item carries per-frame loss weights. Synthetic clips weigh 1 everywhere. Resynthesised clips
(plan §5.2, decisions D-016) have exact formants but tracker f0, voicing and sonorant labels, so
`w_f0` (which also covers voicing and sonorant) is 0, and `w_formant` falls with the clip's tract_left
QC value (`tract_left_weight`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..features.vqt import HOP, N_MAX
from ..synth.channel import ChannelConfig, augment
from ..synth.shards import INT16_SCALE, ShardReader

LABEL_KEYS = ("f0", "voiced", "sonorant", "whisper", "formants", "bandwidths", "rd", "source_h_db")
N_TRAIN_FORMANTS = 4
TRACT_LEFT_WEIGHT = (0.15, 0.5)  # formant loss weight 1 up to this |tract_left|, falling linearly to 0 at that


def formant_weight(tract_left: np.ndarray | float, ramp: tuple[float, float] = TRACT_LEFT_WEIGHT) -> np.ndarray:
    """Per-clip formant/bandwidth loss weight from the clip's median tract_left; NaN (unmeasured) weighs 1."""
    b = np.abs(np.asarray(tract_left, float))
    w = np.clip((ramp[1] - b) / (ramp[1] - ramp[0]), 0.0, 1.0)
    return np.where(np.isfinite(b), w, 1.0)


def consolidate(root: str | Path, log=print) -> Path:
    root = Path(root)
    out = root / "flat"
    if (out / "done").exists():
        return out
    out.mkdir(exist_ok=True)
    reader = ShardReader(root, cache_shards=False)
    shards = reader.shards
    n_audio = n_frames = 0
    sizes = []
    for s in shards:
        with np.load(s) as z:
            sizes.append((int(z["audio_offsets"][-1]), int(z["frame_offsets"][-1])))
    n_audio, n_frames = map(sum, zip(*sizes))
    audio = np.lib.format.open_memmap(out / "audio.npy", "w+", np.int16, (n_audio,))
    labels: dict[str, np.memmap] = {}
    a_off, f_off = [0], [0]
    a_pos = f_pos = 0
    resynth = reader.index.get("kind") == "resynth"
    clip_tl: list[float] = []
    for k, s in enumerate(shards):
        with np.load(s) as z:
            a, ao, fo = z["audio"], z["audio_offsets"], z["frame_offsets"]
            if resynth:
                clip_tl.extend(m["qc"]["tract_left_median"] if m["qc"]["tract_left_median"] is not None else np.nan
                               for m in json.loads(str(z["meta"])))
            audio[a_pos : a_pos + len(a)] = a
            for key in LABEL_KEYS:
                arr = z[f"label_{key}"]
                if key not in labels:
                    labels[key] = np.lib.format.open_memmap(out / f"label_{key}.npy", "w+", arr.dtype, (n_frames,) + arr.shape[1:])
                labels[key][f_pos : f_pos + len(arr)] = arr
            a_off.extend((a_pos + ao[1:]).tolist())
            f_off.extend((f_pos + fo[1:]).tolist())
            a_pos += len(a)
            f_pos += int(fo[-1])
        if (k + 1) % 50 == 0:
            log(f"  consolidated {k + 1}/{len(shards)} shards")
    np.save(out / "audio_offsets.npy", np.array(a_off, np.int64))
    np.save(out / "frame_offsets.npy", np.array(f_off, np.int64))
    if resynth:  # per clip; synthetic sets have no such file
        np.save(out / "clip_tract_left.npy", np.array(clip_tl, np.float64))
    for m in [audio, *labels.values()]:
        m.flush()
    (out / "done").write_text(json.dumps({"n_clips": len(a_off) - 1, "n_audio": n_audio, "n_frames": n_frames}))
    return out


class SynthCrops:
    """torch-style dataset of fixed-length crops. Item = (audio with VQT context, labels dict).

    Audio length is (n_frames - 1) * HOP + N_MAX, so frame j of the crop is centred at sample
    N_MAX // 2 + j * HOP, with real signal (or zeros at clip edges) on both sides.
    """

    def __init__(self, flat_dir: str | Path, clip_ids: np.ndarray, n_frames: int = 200,
                 channel: ChannelConfig | None = ChannelConfig(), seed: int = 0, epoch_size: int | None = None,
                 fixed: bool = False, tract_left_weight: tuple[float, float] = TRACT_LEFT_WEIGHT):
        self.dir = Path(flat_dir)
        self.clip_ids = np.asarray(clip_ids)
        self.n_frames = n_frames
        self.channel = channel
        self.seed = seed
        self.epoch_size = epoch_size or len(self.clip_ids)
        self.fixed = fixed  # same crop and channel for an item every time (validation)
        self.tract_left_weight = tuple(tract_left_weight)
        self._arrays = None

    def _open(self):
        if self._arrays is None:
            load = lambda name: np.load(self.dir / name, mmap_mode="r")  # noqa: E731
            self._arrays = {"audio": load("audio.npy"), "a_off": np.load(self.dir / "audio_offsets.npy"),
                            "f_off": np.load(self.dir / "frame_offsets.npy"),
                            **{k: load(f"label_{k}.npy") for k in LABEL_KEYS}}
            tl = self.dir / "clip_tract_left.npy"
            self._arrays["clip_weight"] = formant_weight(np.load(tl), self.tract_left_weight) if tl.exists() else None
        return self._arrays

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, item: int):
        import torch

        arr = self._open()
        # torch's per-worker RNG is reseeded every epoch by the DataLoader, so crops and channels differ per epoch,
        # unless the set is fixed (validation: every eval must see the same crops, or checkpoints can't be compared).
        rng = np.random.default_rng([self.seed, item] if self.fixed else [self.seed, item, int(torch.randint(1 << 30, (1,)))])
        cid = self.clip_ids[rng.integers(len(self.clip_ids))] if self.epoch_size != len(self.clip_ids) else self.clip_ids[item]
        a0, a1 = arr["a_off"][cid], arr["a_off"][cid + 1]
        f0_, f1_ = arr["f_off"][cid], arr["f_off"][cid + 1]
        x = arr["audio"][a0:a1].astype(np.float32) / INT16_SCALE
        n_clip_frames = int(f1_ - f0_)
        lab = {k: np.asarray(arr[k][f0_:f1_]) for k in LABEL_KEYS}
        if self.channel is not None:
            mask = np.repeat(lab["voiced"], HOP)[: len(x)]
            x, _ = augment(x, rng, self.channel, np.pad(mask, (0, len(x) - len(mask))))
        t = self.n_frames
        start = int(rng.integers(0, max(n_clip_frames - t, 0) + 1))
        lab = {k: v[start : start + t] for k, v in lab.items()}
        if len(lab["f0"]) < t:  # short clip: pad labels as unvoiced/absent
            pad = t - len(lab["f0"])
            lab = {k: np.concatenate([v, np.zeros((pad,) + v.shape[1:], v.dtype) if v.dtype == bool else np.full((pad,) + v.shape[1:], np.nan, v.dtype)]) for k, v in lab.items()}
        resynth = arr["clip_weight"] is not None
        lab["w_f0"] = np.full(t, 0.0 if resynth else 1.0, np.float32)
        lab["w_formant"] = np.full(t, arr["clip_weight"][cid] if resynth else 1.0, np.float32)
        lab["formants"] = lab["formants"][:, :N_TRAIN_FORMANTS]
        lab["bandwidths"] = lab["bandwidths"][:, :N_TRAIN_FORMANTS]
        c0 = start * HOP - N_MAX // 2
        n_audio = (t - 1) * HOP + N_MAX
        seg = np.zeros(n_audio, np.float32)
        lo, hi = max(c0, 0), min(c0 + n_audio, len(x))
        if hi > lo:
            seg[lo - c0 : hi - c0] = x[lo:hi]
        return seg, {k: v.astype(np.float32) if v.dtype != bool else v for k, v in lab.items()}


class MixedCrops:
    """Items drawn from several datasets (each with an epoch_size of its own random clips) in given proportions.
    The choice is seeded like SynthCrops' (per item, plus torch's per-worker RNG unless fixed)."""

    def __init__(self, parts: list, weights: list[float], epoch_size: int, seed: int = 0, fixed: bool = False):
        self.parts = parts
        self.p = np.asarray(weights, float) / np.sum(weights)
        self.epoch_size = epoch_size
        self.seed = seed
        self.fixed = fixed

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, item: int):
        import torch

        rng = np.random.default_rng([self.seed, item, 17] if self.fixed else [self.seed, item, 17, int(torch.randint(1 << 30, (1,)))])
        k = int(rng.choice(len(self.parts), p=self.p))
        return self.parts[k][int(rng.integers(len(self.parts[k])))]


def collate(batch):
    import torch

    audio = torch.from_numpy(np.stack([b[0] for b in batch]))
    labels = {k: torch.from_numpy(np.stack([b[1][k] for b in batch])) for k in batch[0][1]}
    return audio, labels


def split_ids(n_clips: int, val_frac: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Last val_frac of clip indices are validation (clips are i.i.d., so a contiguous split is fine)."""
    n_val = max(1, int(n_clips * val_frac))
    return np.arange(n_clips - n_val), np.arange(n_clips - n_val, n_clips)
