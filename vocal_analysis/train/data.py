"""Training data from synthetic shards: memory-mapped, randomly cropped, channel-augmented on the fly.

`consolidate()` turns a shard set into flat .npy arrays once (audio int16, labels, offsets),
so DataLoader workers share the OS page cache instead of each holding copies of shards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..features.vqt import HOP, N_MAX
from ..synth.channel import ChannelConfig, augment
from ..synth.shards import INT16_SCALE, ShardReader

LABEL_KEYS = ("f0", "voiced", "sonorant", "whisper", "formants", "bandwidths", "rd", "source_h_db")
N_TRAIN_FORMANTS = 4


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
    for k, s in enumerate(shards):
        with np.load(s) as z:
            a, ao, fo = z["audio"], z["audio_offsets"], z["frame_offsets"]
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
    for m in [audio, *labels.values()]:
        m.flush()
    (out / "done").write_text(json.dumps({"n_clips": len(a_off) - 1, "n_audio": n_audio, "n_frames": n_frames}))
    return out


def _pad_labels(key: str, v: np.ndarray, n: int) -> np.ndarray:
    """Flags pad False; formants and bandwidths NaN (masked in the loss); f0, rd and source_h_db, which the data
    keep finite on every frame (f0 carries its contour through unvoiced frames; the comb reads it), repeat the last."""
    if v.dtype == bool:
        return np.zeros((n,) + v.shape[1:], bool)
    if key in ("formants", "bandwidths"):
        return np.full((n,) + v.shape[1:], np.nan, v.dtype)
    return np.repeat(v[-1:], n, axis=0)


class SynthCrops:
    """torch-style dataset of fixed-length crops. Item = (audio with VQT context, labels dict).

    Audio length is (n_frames - 1) * HOP + N_MAX, so frame j of the crop is centred at sample
    N_MAX // 2 + j * HOP, with real signal (or zeros at clip edges) on both sides.
    """

    def __init__(self, flat_dir: str | Path, clip_ids: np.ndarray, n_frames: int = 200,
                 channel: ChannelConfig | None = ChannelConfig(), seed: int = 0, epoch_size: int | None = None,
                 fixed: bool = False):
        self.dir = Path(flat_dir)
        self.clip_ids = np.asarray(clip_ids)
        self.n_frames = n_frames
        self.channel = channel
        self.seed = seed
        self.epoch_size = epoch_size or len(self.clip_ids)
        self.fixed = fixed  # same crop and channel for an item every time (validation)
        self._arrays = None

    def _open(self):
        if self._arrays is None:
            load = lambda name: np.load(self.dir / name, mmap_mode="r")  # noqa: E731
            self._arrays = {"audio": load("audio.npy"), "a_off": np.load(self.dir / "audio_offsets.npy"),
                            "f_off": np.load(self.dir / "frame_offsets.npy"),
                            **{k: load(f"label_{k}.npy") for k in LABEL_KEYS}}
        return self._arrays

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, item):
        """item: an index, or (index, n_frames, n_crops) from LengthBatches: a crop length per batch, and for short
        crops several crops from one augmented clip (a list), so a batch augments about as many clips as a fixed one."""
        import torch

        item, t, n_crops = item if isinstance(item, tuple) else (item, self.n_frames, None)
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
            # fixed (validation) sets code each clip in a fresh process: the same audio every time and in every run
            x, _ = augment(x, rng, self.channel, np.pad(mask, (0, len(x) - len(mask))), codec_stream=False if self.fixed else None)
        if n_crops is not None:
            return [self._crop(x, lab, n_clip_frames, t, rng) for _ in range(n_crops)]
        return self._crop(x, lab, n_clip_frames, t, rng)

    def _crop(self, x, lab, n_clip_frames, t, rng):
        start = int(rng.integers(0, max(n_clip_frames - t, 0) + 1))
        lab = {k: v[start : start + t] for k, v in lab.items()}
        if len(lab["f0"]) < t:  # crop runs past the clip's end: pad labels as unvoiced/absent
            pad = t - len(lab["f0"])
            lab = {k: np.concatenate([v, _pad_labels(k, v, pad)]) for k, v in lab.items()}
        lab["formants"] = lab["formants"][:, :N_TRAIN_FORMANTS]
        lab["bandwidths"] = lab["bandwidths"][:, :N_TRAIN_FORMANTS]
        c0 = start * HOP - N_MAX // 2
        n_audio = (t - 1) * HOP + N_MAX
        seg = np.zeros(n_audio, np.float32)
        lo, hi = max(c0, 0), min(c0 + n_audio, len(x))
        if hi > lo:
            seg[lo - c0 : hi - c0] = x[lo:hi]
        return seg, {k: v.astype(np.float32) if v.dtype != bool else v for k, v in lab.items()}


class LengthBatches:
    """Batch sampler with one crop length per batch, drawn uniformly from n_lengths values spaced geometrically over
    [lo, hi] frames (log-uniform, discretised: every new tensor shape costs kernel selection and compilation on ROCm,
    so the shapes are kept few). The batch size scales so every batch holds about frames_per_batch frames: the same
    audio per step, and the same weight per frame in the loss. A batch of short crops takes several from each clip
    (about crop_span frames' worth), so it augments about as many clips as a batch of fixed crops: augmenting costs per
    clip (AMR launches ffmpeg), not per frame. Yields lists of (index, n_frames, n_crops) for SynthCrops."""

    def __init__(self, n_batches: int, epoch_size: int, frames: tuple[int, int], frames_per_batch: int, seed: int = 0,
                 n_lengths: int = 8, crop_span: int = 200):
        self.n_batches, self.epoch_size, self.fpb, self.crop_span = n_batches, epoch_size, frames_per_batch, crop_span
        self.lengths = np.unique(np.round(np.geomspace(frames[0], frames[1], n_lengths)).astype(int))
        self.rng = np.random.default_rng(seed)  # main process; carries on across epochs, so each epoch differs

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            t = int(self.rng.choice(self.lengths))
            k = max(1, math.ceil(self.crop_span / t - 0.1))  # crops per clip: at most about crop_span frames per clip
            b = max(1, round(self.fpb / (t * k)))
            yield [(int(i), t, k) for i in self.rng.integers(self.epoch_size, size=b)]


def collate(batch):
    import torch

    batch = [c for b in batch for c in (b if isinstance(b, list) else [b])]  # several crops per clip (LengthBatches)
    audio = torch.from_numpy(np.stack([b[0] for b in batch]))
    labels = {k: torch.from_numpy(np.stack([b[1][k] for b in batch])) for k in batch[0][1]}
    return audio, labels


def split_ids(n_clips: int, val_frac: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Last val_frac of clip indices are validation (clips are i.i.d., so a contiguous split is fine)."""
    n_val = max(1, int(n_clips * val_frac))
    return np.arange(n_clips - n_val), np.arange(n_clips - n_val, n_clips)
