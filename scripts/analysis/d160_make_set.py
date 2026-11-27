"""D-160: train_v2w, train_v2 re-rendered at its own seed and stored settings with two changes (`p_wide_bw` 0.35, caps
700 / 450 / 650 / 800 Hz; `p_ra_offline` 0.5, Ra x log-uniform 0.5-3), each from a generator of its own, so every other
draw is train_v2's. `check` first: train_v2's stored settings reproduce its clips exactly, and the new set differs only in
bandwidths and source. Resumable (write_set skips finished shards).

    python scripts/analysis/d160_make_set.py check
    python scripts/analysis/d160_make_set.py render OUT [WORKERS]
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/synth/train_v2"
N_CLIPS, SEED = 100_000, 2  # train_v2's size and seed
D160 = {"p_wide_bw": 0.35, "wide_bw_caps": (700.0, 450.0, 650.0, 800.0), "p_ra_offline": 0.5, "ra_factor": (0.5, 3.0)}


def v2_config():
    """train_v2's stored synth_config; fields added since take V1_SETTINGS' legacy value, or today's default."""
    from vocal_analysis.synth.sampler import V1_SETTINGS, SynthConfig

    ix = json.loads((V2 / "index.json").read_text(encoding="utf-8"))["synth_config"]
    kw = {}
    for f in dataclasses.fields(SynthConfig):
        v = ix[f.name] if f.name in ix else V1_SETTINGS.get(f.name, getattr(SynthConfig(), f.name))
        kw[f.name] = tuple(v) if isinstance(v, list) else v
    return SynthConfig(**kw)


def cmd_check():
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.synth.shards import INT16_SCALE

    base = v2_config()
    new = dataclasses.replace(base, **D160)
    d = V2 / "flat"
    a = np.load(d / "audio.npy", mmap_mode="r")
    ao, fo = np.load(d / "audio_offsets.npy"), np.load(d / "frame_offsets.npy")
    tube = np.load(d / "clip_tube.npy")
    B = np.load(d / "label_bandwidths.npy", mmap_mode="r")
    F = np.load(d / "label_formants.npy", mmap_mode="r")
    for i in (0, 1, 2, 3, 5, 8, 13, 21):
        c = generate(i, base, seed=SEED)
        q = np.clip(np.round(c.audio * INT16_SCALE), -32768, 32767).astype(np.int16)
        assert np.array_equal(q, np.asarray(a[ao[i]:ao[i + 1]])), f"clip {i}: train_v2's settings don't reproduce its audio"
        assert np.allclose(np.asarray(B[fo[i]:fo[i + 1]]), c.labels["bandwidths"], equal_nan=True), f"clip {i}: bandwidths"
        w = generate(i, new, seed=SEED)
        assert np.array_equal(w.labels["formants"], np.asarray(F[fo[i]:fo[i + 1]], w.labels["formants"].dtype)) or \
            np.allclose(w.labels["formants"], np.asarray(F[fo[i]:fo[i + 1]]), equal_nan=True), f"clip {i}: formants moved"
        for k in ("f0", "voiced", "rd", "length_cm"):
            assert np.array_equal(w.labels[k], c.labels[k]), f"clip {i}: {k} moved"
        print(i, "tube" if tube[i] else "cascade", "reproduces;", w.meta["d160"], flush=True)
    print("check passed")


def cmd_render(out, workers):
    from vocal_analysis.synth.shards import write_set

    cfg = dataclasses.replace(v2_config(), **D160)
    write_set(out, N_CLIPS, cfg, seed=SEED, workers=workers,
              extra_index={"purpose": "D-160: train_v2 with wide low-f0 bandwidths and an off-line LF return phase"},
              log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        cmd_check()
    else:
        cmd_render(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 24)
