"""M4: real-audio crops with harmonic targets for the reconstruction loss (`reconstruction.py`).

`build_store` precomputes, per recording of a resynthesis pool's corpus (its training files, never an evaluation set),
the SNR-gated harmonic targets (`harmonic_targets`, D-104) from the labeller's own f0 and voicing (the pool's analysis
files: a fixed estimate, not the model's), and stores them beside the audio. `ReconCrops` serves fixed-length crops in
the trainer's layout (audio with VQT context, frame j centred at sample N_MAX // 2 + j * HOP), with the targets, the
valid mask, f0 and a frame mask (voiced and sonorant by the labeller, >= 4 valid harmonics)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..features.vqt import N_MAX
from .reconstruction import FS, HOP, harmonic_targets

K_MAX = 64
SNR_DB = 10.0


def _file_job(job):
    corpus_root, adir, f, max_s = job
    from ..resynth.corpus import _analysis_name, load_audio

    x = load_audio(Path(corpus_root) / f)[: int(max_s * FS)].astype(np.float32)
    a = np.load(Path(adir) / _analysis_name(f))
    n = min(len(a["f0"]), len(x) // HOP + 1)
    f0 = a["f0"][:n]
    voiced = (a["voiced_p"][:n] > 0.5) & np.isfinite(f0) & (f0 > 0)
    son = a["sonorant_p"][:n] > 0.5
    db, ok = harmonic_targets(x, np.where(voiced, f0, np.nan), voiced, k_max=K_MAX, snr_min_db=SNR_DB)
    mask = voiced & son & (ok.sum(1) >= 4)
    return f, x, db.astype(np.float16), ok, np.nan_to_num(f0).astype(np.float32), mask


def build_store(pool: str, out: str, n_files: int, seed: int = 0, max_s: float = 10.0, workers: int = 8, log=print) -> Path:
    """pool: a resynthesis set (its index names the corpus, the corpus root and the labeller's analysis); n_files of its
    training files drawn with `seed`; each recording's first max_s seconds."""
    from concurrent.futures import ProcessPoolExecutor

    idx = json.loads((Path(pool) / "index.json").read_text(encoding="utf-8"))
    corpus, root = idx["corpus"], idx.get("corpus_root") or f"data/{idx['corpus']}"
    adir = Path("data/resynth/analysis") / corpus / idx["teacher"]["sha256"]
    files = list(idx["files"])
    pick = sorted(np.random.default_rng(seed).choice(len(files), min(n_files, len(files)), replace=False))
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    jobs = [(root, str(adir), files[i], max_s) for i in pick]
    names, lens = [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (f, x, db, ok, f0, mask) in enumerate(ex.map(_file_job, jobs, chunksize=4)):
            np.savez(outd / f"{k:05d}.npz", audio=x, db=db, ok=ok, f0=f0, mask=mask)
            names.append(f)
            lens.append(len(mask))
    (outd / "index.json").write_text(json.dumps({"pool": pool, "corpus": corpus, "files": names, "frames": lens, "seed": seed,
                                                 "max_s": max_s, "snr_db": SNR_DB, "k_max": K_MAX}, indent=1), encoding="utf-8")
    log(f"recon store {outd}: {len(names)} recordings, {sum(lens):,} frames")
    return outd


class ReconCrops:
    """Crops of n_frames from the stores' recordings (uniform over recordings, then over start frames), as
    (audio (L,), dict(db (T, K), ok (T, K), f0 (T,), mask (T,)))."""

    def __init__(self, stores: list[str], n_frames: int, seed: int = 0, epoch_size: int = 10000):
        self.files = [p for s in stores for p in sorted(Path(s).glob("*.npz"))]
        self.n_frames, self.seed, self.epoch_size = n_frames, seed, epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        z = np.load(self.files[rng.integers(len(self.files))])
        T = len(z["mask"])
        s = int(rng.integers(0, max(1, T - self.n_frames)))
        t = slice(s, s + self.n_frames)
        a0 = s * HOP - N_MAX // 2
        L = (self.n_frames - 1) * HOP + N_MAX
        x = z["audio"]
        audio = np.zeros(L, np.float32)
        lo, hi = max(0, a0), min(len(x), a0 + L)
        if hi > lo:
            audio[lo - a0:hi - a0] = x[lo:hi]

        def take(v, fill):
            w = v[t]
            pad = self.n_frames - len(w)
            return np.concatenate([w, np.full((pad, *w.shape[1:]), fill, w.dtype)]) if pad > 0 else w

        return audio, {"db": take(z["db"], 0).astype(np.float32), "ok": take(z["ok"], False), "f0": take(z["f0"], 0.0),
                       "mask": take(z["mask"], False)}


def collate_recon(items):
    import torch

    audio = torch.from_numpy(np.stack([a for a, _ in items]))
    lab = {k: torch.from_numpy(np.stack([d[k] for _, d in items])) for k in items[0][1]}
    return audio, lab
