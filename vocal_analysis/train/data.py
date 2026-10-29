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
import math
from pathlib import Path

import numpy as np

from ..features.vqt import HOP, N_MAX
from ..synth.channel import ChannelConfig, apply_channel, augment, draw_channel
from ..synth.shards import INT16_SCALE, ShardReader

LABEL_KEYS = ("f0", "voiced", "sonorant", "whisper", "formants", "bandwidths", "rd", "source_h_db")
N_TRAIN_FORMANTS = 4
TRACT_LEFT_WEIGHT = (0.15, 0.5)
OBS_MARGIN = 20  # frames each side of a crop for observability's temporal pooling (4 x 50 ms reaches 0.02 weight)  # formant loss weight 1 up to this |tract_left|, falling linearly to 0 at that


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


def _pad_labels(key: str, v: np.ndarray, n: int) -> np.ndarray:
    """Flags pad False; formants and bandwidths NaN (masked in the loss); f0, rd and source_h_db, which the data
    keep finite on every frame (f0 carries its contour through unvoiced frames; the comb reads it), repeat the last."""
    if v.dtype == bool:
        return np.zeros((n,) + v.shape[1:], bool)
    if key in ("formants", "bandwidths", "vtl", "vtl_inst"):
        return np.full((n,) + v.shape[1:], np.nan, v.dtype)
    return np.repeat(v[-1:], n, axis=0)


class SynthCrops:
    """torch-style dataset of fixed-length crops. Item = (audio with VQT context, labels dict).

    Audio length is (n_frames - 1) * HOP + N_MAX, so frame j of the crop is centred at sample
    N_MAX // 2 + j * HOP, with real signal (or zeros at clip edges) on both sides.
    """

    def __init__(self, flat_dir: str | Path, clip_ids: np.ndarray, n_frames: int = 200,
                 channel: ChannelConfig | None = ChannelConfig(), seed: int = 0, epoch_size: int | None = None,
                 fixed: bool = False, tract_left_weight: tuple[float, float] = TRACT_LEFT_WEIGHT,
                 observability: bool = False, obs_config=None, obs_floor: str = "recording",
                 obs_rec_snr_db: float | None = None, vtl: bool = False, vtl_inst: bool = False):
        self.dir = Path(flat_dir)
        self.clip_ids = np.asarray(clip_ids)
        self.n_frames = n_frames
        self.channel = channel
        self.seed = seed
        self.epoch_size = epoch_size or len(self.clip_ids)
        self.fixed = fixed  # same crop and channel for an item every time (validation)
        self.tract_left_weight = tuple(tract_left_weight)
        # D-027: per-frame observability labels ("obs", (t, 4)) for the channelled audio, from the set's store
        # (obs_store.write_store); NaN where there is none (resynthesised sets, absent formants, non-sonorant frames)
        self.observability = observability
        # the label's definition (None: the current defaults, `ObservabilityConfig` and `obs_store.REC_SNR_DB`)
        self.obs_config, self.obs_floor, self.obs_rec_snr_db = obs_config, obs_floor, obs_rec_snr_db
        # D-036: per-frame anatomical length label ("vtl", log2 cm) on tube clips whose length was not drawn from f0
        # (scripts/analysis/vtl_labels.py's files beside flat/); NaN elsewhere and for sets without those files
        self.vtl = vtl
        # D-051: per-frame instantaneous length ("vtl_inst", log2 cm; scripts/analysis/inst_labels.py's
        # label_length_inst_cm.npy), on the same clips as "vtl"; NaN elsewhere
        self.vtl_inst = vtl_inst
        self._arrays = None

    def _open(self):
        if self._arrays is None:
            load = lambda name: np.load(self.dir / name, mmap_mode="r")  # noqa: E731
            self._arrays = {"audio": load("audio.npy"), "a_off": np.load(self.dir / "audio_offsets.npy"),
                            "f_off": np.load(self.dir / "frame_offsets.npy"),
                            **{k: load(f"label_{k}.npy") for k in LABEL_KEYS}}
            if self.observability and (self.dir / "obs_sig.npy").exists():
                self._arrays.update(obs_sig=load("obs_sig.npy"), obs_noise=load("obs_noise.npy"), obs_peak=load("obs_peak.npy"))
            if self.vtl and (self.dir / "label_length_cm.npy").exists():
                self._arrays["vtl_frames"] = load("label_length_cm.npy")
                self._arrays["vtl_ok"] = np.load(self.dir / "clip_tube.npy") & ~np.load(self.dir / "clip_length_from_f0.npy")
                if self.vtl_inst and (self.dir / "label_length_inst_cm.npy").exists():
                    self._arrays["vtl_inst_frames"] = load("label_length_inst_cm.npy")
            tl = self.dir / "clip_tract_left.npy"
            self._arrays["clip_weight"] = formant_weight(np.load(tl), self.tract_left_weight) if tl.exists() else None
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
        if self.vtl:
            ok = "vtl_ok" in arr and bool(arr["vtl_ok"][cid])
            lab["vtl"] = np.log2(np.asarray(arr["vtl_frames"][f0_:f1_], np.float32)) if ok else np.full(n_clip_frames, np.nan, np.float32)
            if self.vtl_inst:
                ok_i = ok and "vtl_inst_frames" in arr
                lab["vtl_inst"] = (np.log2(np.asarray(arr["vtl_inst_frames"][f0_:f1_], np.float32)) if ok_i
                                   else np.full(n_clip_frames, np.nan, np.float32))
        params = {"clean": True}
        if self.channel is not None:
            mask = np.repeat(lab["voiced"], HOP)[: len(x)]
            mask = np.pad(mask, (0, len(x) - len(mask)))
            # fixed (validation) sets code each clip in a fresh process: the same audio every time and in every run
            stream = False if self.fixed else None
            if self.observability:  # augment's draws and audio, keeping what the store's channel needs
                ch = draw_channel(rng, len(x), self.channel)
                x, used = apply_channel(x, ch, mask, codec_stream=stream)
                params = {k: v for k, v in ch.items() if not k.startswith("_")} | ({"clip_loss": used["clip_loss"]} if "clip_loss" in used else {})
            else:
                x, _ = augment(x, rng, self.channel, mask, codec_stream=stream)
        obs = self._observability(arr, f0_, f1_, lab, params) if self.observability else None
        resynth = arr["clip_weight"] is not None
        w = (0.0 if resynth else 1.0, float(arr["clip_weight"][cid]) if resynth else 1.0)  # w_f0, w_formant
        if n_crops is not None:
            return [self._crop(x, lab, n_clip_frames, t, rng, *w, obs=obs) for _ in range(n_crops)]
        return self._crop(x, lab, n_clip_frames, t, rng, *w, obs=obs)

    def _observability(self, arr, f0_, f1_, lab, params):
        """A callable (start, t) -> (t, 4) observability of the clip's frames under the drawn channel, computed for
        the crop plus OBS_MARGIN frames each side (the temporal pooling's reach)."""
        from ..synth import obs_store
        from ..synth.observability import observability

        n = int(f1_ - f0_)
        if "obs_sig" not in arr:
            return lambda start, t: np.full((t, N_TRAIN_FORMANTS), np.nan, np.float32)
        cs = obs_store.ClipSpectra(arr["obs_sig"][f0_:f1_], arr["obs_noise"][f0_:f1_], arr["obs_peak"][f0_:f1_],
                                   lab["voiced"], lab["f0"], floor=self.obs_floor,
                                   rec_snr_db=obs_store.REC_SNR_DB if self.obs_rec_snr_db is None else self.obs_rec_snr_db).channel(params)

        def at(start, t):
            lo, hi = max(start - OBS_MARGIN, 0), min(start + t + OBS_MARGIN, n)
            out = np.full((t, N_TRAIN_FORMANTS), np.nan, np.float32)
            if hi <= lo:
                return out
            sl = slice(lo, hi)
            o, _ = observability(lab["f0"][sl], lab["voiced"][sl], lab["sonorant"][sl], lab["formants"][sl, :N_TRAIN_FORMANTS],
                                 lab["bandwidths"][sl, :N_TRAIN_FORMANTS], lambda j, f: cs.snr_db_at(j + lo, f), self.obs_config,
                                 snr_db_rows=lambda js, f: cs.snr_db_rows(js + lo, f))
            k0, k1 = start - lo, min(start + t, n) - lo
            out[: k1 - k0] = o[k0:k1]
            return out

        return at

    def _crop(self, x, lab, n_clip_frames, t, rng, w_f0=1.0, w_formant=1.0, obs=None):
        start = int(rng.integers(0, max(n_clip_frames - t, 0) + 1))
        lab = {k: v[start : start + t] for k, v in lab.items()}
        if len(lab["f0"]) < t:  # crop runs past the clip's end: pad labels as unvoiced/absent
            pad = t - len(lab["f0"])
            lab = {k: np.concatenate([v, _pad_labels(k, v, pad)]) for k, v in lab.items()}
        lab["w_f0"] = np.full(t, w_f0, np.float32)
        lab["w_formant"] = np.full(t, w_formant, np.float32)
        if obs is not None:  # (t, 4), NaN past the clip's end
            lab["obs"] = obs(start, t)
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
                 n_lengths: int = 8, crop_span: int = 200, long_frac: float | None = None):
        self.n_batches, self.epoch_size, self.fpb, self.crop_span = n_batches, epoch_size, frames_per_batch, crop_span
        self.lengths = np.unique(np.round(np.geomspace(frames[0], frames[1], n_lengths)).astype(int))
        # long_frac: the share of batches (so of frames) with lengths of at least 0.97 * crop_span, split equally
        # among those lengths; the rest split equally among the shorter ones. None: all lengths equally likely.
        self.p = None
        if long_frac is not None:
            long = self.lengths >= 0.97 * crop_span
            self.p = np.where(long, long_frac / long.sum(), (1.0 - long_frac) / (~long).sum())
        self.rng = np.random.default_rng(seed)  # main process; carries on across epochs, so each epoch differs

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            t = int(self.rng.choice(self.lengths, p=self.p))
            k = max(1, math.ceil(self.crop_span / t - 0.1))  # crops per clip: at most about crop_span frames per clip
            b = max(1, round(self.fpb / (t * k)))
            yield [(int(i), t, k) for i in self.rng.integers(self.epoch_size, size=b)]


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

    def __getitem__(self, item):
        """item: an index, or (index, n_frames, n_crops) from LengthBatches, passed on with the part's own index."""
        import torch

        idx, rest = (item[0], item[1:]) if isinstance(item, tuple) else (item, None)
        rng = np.random.default_rng([self.seed, idx, 17] if self.fixed else [self.seed, idx, 17, int(torch.randint(1 << 30, (1,)))])
        k = int(rng.choice(len(self.parts), p=self.p))
        j = int(rng.integers(len(self.parts[k])))
        return self.parts[k][(j, *rest) if rest is not None else j]


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
