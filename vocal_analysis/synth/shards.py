"""Synthetic data on disk: training shards (clean audio; channel applied on the fly) and the
fixed-seed hard-case suite (plan §5.1, §11).

Layout: <out>/index.json plus shard_00000.npz, ... Each shard holds int16 audio for a run
of consecutive clip indices, concatenated, with per-frame labels concatenated alongside.
Clip i of a set is generate(i, cfg, seed), so any clip can be regenerated or checked.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .channel import ChannelConfig, augment
from .sampler import SynthConfig, generate

FORMAT_VERSION = 1
INT16_SCALE = 32767.0


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _pack(clips: list[tuple]) -> dict:
    """clips: (audio float32, labels, meta) -> arrays for one npz."""
    audio = np.concatenate([np.clip(np.round(a * INT16_SCALE), -32768, 32767).astype(np.int16) for a, _, _ in clips])
    a_off = np.cumsum([0] + [len(a) for a, _, _ in clips])
    f_off = np.cumsum([0] + [len(lab["t"]) for _, lab, _ in clips])
    out = {"audio": audio, "audio_offsets": a_off, "frame_offsets": f_off,
           "meta": np.array(json.dumps([m for _, _, m in clips]))}
    for key in clips[0][1]:
        out[f"label_{key}"] = np.concatenate([lab[key] for _, lab, _ in clips])
    return out


def _make_shard(args) -> str:
    path, indices, cfg, seed, channel, channel_seed = args
    clips = []
    for i in indices:
        c = generate(i, cfg, seed)
        audio = c.audio
        if channel is not None:  # suite: bake a fixed channel into the stored audio
            mask = np.repeat(c.labels["voiced"], int(c.meta["fs"] * cfg.hop_s))[: len(audio)]
            mask = np.pad(mask, (0, len(audio) - len(mask)))
            audio, params = augment(audio, np.random.default_rng([channel_seed, i]), channel, mask)
            c.meta["channel"] = params
        c.meta.pop("config", None)  # stored once in index.json
        clips.append((audio, c.labels, c.meta))
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez(tmp, **_pack(clips))
    os.replace(tmp, path)
    return path


# Values that config fields had before they existed, for comparing against older index.json files.
LEGACY_DEFAULTS = {"synth_config": {"p_tube": 0.0, "p_tube_constrict": 0.15},
                   "channel_config": {"p_codec": 0.0, "codecs": ["opus", "aac", "mp3", "amr_wb", "amr_nb"]}}


def write_set(out_dir: str | Path, n_clips: int, cfg: SynthConfig, seed: int, shard_size: int = 256,
              workers: int | None = None, channel: ChannelConfig | None = None, channel_seed: int = 0,
              extra_index: dict | None = None, log=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"format_version": FORMAT_VERSION, "n_clips": n_clips, "shard_size": shard_size, "seed": seed,
             "synth_config": asdict(cfg), "channel_config": asdict(channel) if channel else None,
             "channel_seed": channel_seed, **(extra_index or {})}
    index_path = out / "index.json"
    if index_path.exists():  # resuming: the shards already there must come from the same settings
        old = json.loads(index_path.read_text(encoding="utf-8"))
        for key, legacy in LEGACY_DEFAULTS.items():  # fields added after a set was written had these values then
            if old.get(key) is not None:
                old[key] = legacy | old[key]
        new = json.loads(json.dumps(index))
        diff = sorted(k for k in new if old.get(k) != new[k])
        if diff:
            raise ValueError(f"{out} was written with different settings ({', '.join(diff)}); use a new directory")
    else:
        index_path.write_text(json.dumps(index | {"complete": False}, indent=1), encoding="utf-8")
    jobs = []
    for s, start in enumerate(range(0, n_clips, shard_size)):
        path = out / f"shard_{s:05d}.npz"
        if not path.exists():  # resumable
            jobs.append((str(path), list(range(start, min(start + shard_size, n_clips))), cfg, seed, channel, channel_seed))
    log(f"{out}: {n_clips} clips, {len(jobs)} shards to write")
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(_make_shard, jobs), 1):
            if k % 10 == 0 or k == len(jobs):
                log(f"  {k}/{len(jobs)} shards")
    index_path.write_text(json.dumps(index | {"git_commit": _git_commit()}, indent=1), encoding="utf-8")
    return out


class ShardReader:
    """Random access to clips in a written set (numpy only; wrap in a torch Dataset for training)."""

    def __init__(self, root: str | Path, cache_shards: bool = True):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.shards = sorted(self.root.glob("shard_*.npz"))
        self.shard_size = self.index["shard_size"]
        self._cache: dict[int, dict] = {}
        self.cache_shards = cache_shards

    def __len__(self) -> int:
        return self.index["n_clips"]

    def _shard(self, s: int) -> dict:
        if s not in self._cache:
            with np.load(self.shards[s]) as z:
                d = {k: z[k] for k in z.files}
            d["meta"] = json.loads(str(d["meta"]))
            if not self.cache_shards:
                return d
            self._cache[s] = d
        return self._cache[s]

    def __getitem__(self, i: int) -> tuple[np.ndarray, dict, dict]:
        s, j = divmod(i, self.shard_size)
        d = self._shard(s)
        a0, a1 = d["audio_offsets"][j], d["audio_offsets"][j + 1]
        f0, f1 = d["frame_offsets"][j], d["frame_offsets"][j + 1]
        labels = {k[6:]: d[k][f0:f1] for k in d if k.startswith("label_")}
        return d["audio"][a0:a1].astype(np.float32) / INT16_SCALE, labels, d["meta"][j]


# ----------------------------------------------------------------------------- hard-case suite

SUITE_SEED = 20261007
SUITE_CHANNEL_SEED = 1
_CLEAN = ChannelConfig(p_clean=1.0, p_codec=0.0)
# The original ten cases are pinned to the settings suite_v1 was written with, so their audio stays
# bit-identical as the training defaults move on; later cases are appended (each case's seed is its position).
_V1_SYNTH = {"p_tube": 0.0}
_V1_CHANNEL = {"p_codec": 0.0}


def suite_cases() -> dict[str, tuple[SynthConfig, ChannelConfig]]:
    """Plan §5.1 hard cases. Each case changes one thing from the training defaults; the channel
    is clean except where the channel is the hard case."""
    base = replace(SynthConfig(), **_V1_SYNTH)
    ch = lambda **kw: ChannelConfig(**(_V1_CHANNEL | kw))  # noqa: E731
    return {
        "reference": (base, _CLEAN),
        "high_f0": (replace(base, p_high_f0=1.0, high_f0_hz=(400.0, 1300.0)), _CLEAN),
        "f1_near_h1": (replace(base, formant_constraint="f1_near_h1"), _CLEAN),
        "f1_near_h2": (replace(base, formant_constraint="f1_near_h2"), _CLEAN),
        "f2_f3_close": (replace(base, formant_constraint="f2_f3_close"), _CLEAN),
        "nasal": (replace(base, p_nasal=1.0), _CLEAN),
        "breathy": (replace(base, rd=(2.0, 2.7)), _CLEAN),
        "wide_vibrato": (replace(base, p_high_f0=0.7, contour_probs_low={"singing": 1}, contour_probs_high={"singing": 1},
                                 p_vibrato=1.0, p_vibrato_wide=0.0, vibrato_extent_cents=(100.0, 150.0)), _CLEAN),
        "low_snr": (base, ch(p_clean=0.0, p_noise=1.0, snr_db=(0.0, 10.0))),
        "phone": (base, ch(p_clean=0.0, p_phone=1.0, p_noise=0.5, snr_db=(15.0, 35.0))),
        # added with the tube tract and codecs (D-011)
        "tube": (replace(base, p_tube=1.0), _CLEAN),
        "codec": (base, ChannelConfig(p_clean=0.0, p_rir=0.0, p_noise=0.0, p_highpass=0.0, p_lowpass=0.0, p_eq=0.0,
                                      p_phone=0.0, gain_db=(0.0, 0.0), p_clip=0.0, p_codec=1.0)),
    }


def write_suite(out_dir: str | Path, n_per_case: int = 64, workers: int | None = None, log=print) -> Path:
    out = Path(out_dir)
    for k, (name, (cfg, ch)) in enumerate(suite_cases().items()):
        write_set(out / name, n_per_case, cfg, seed=SUITE_SEED + k, shard_size=16, workers=workers,
                  channel=ch, channel_seed=SUITE_CHANNEL_SEED, extra_index={"case": name}, log=log)
    return out
