"""Resynthesised training sets from a corpus manifest (plan §5.2): `va resynth <corpus>`.

Stage 1, analysis: a trained teacher (the current best tracker) runs once per recording; its
per-frame outputs are cached in <out>/analysis/, keyed by the recording's manifest path.
Stage 2, render: per recording, inverse filtering once, then `variants` perturbations, each cut into
clips and QC'd (`render.py`). Recordings are grouped into shards of `files_per_shard`, written in the
synthetic sets' format with the clip count of each shard in index.json, so `synth.shards.ShardReader`
and `train.data.consolidate` read them unchanged. Clip RNG is seeded by (seed, recording, variant),
so a shard can be regenerated on its own and an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..data.manifest import ManifestRow
from ..synth.shards import _git_commit, _pack
from .render import FS, ResynthConfig, render_variant, source_from_audio

FORMAT_VERSION = 1
ANALYSIS_KEYS = ("f0", "voiced_p", "sonorant_p", "formants", "bandwidths", "formant_sigma_cents")


def load_audio(path: str | Path) -> np.ndarray:
    """Mono float64 at 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != FS:
        r = Fraction(FS, sr)
        x = resample_poly(x, r.numerator, r.denominator)
    return x


def _analysis_name(path: str) -> str:
    return path.replace("/", "__").rsplit(".", 1)[0] + ".npz"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def analyse(rows: list[ManifestRow], root: Path, teacher: str | Path, out_dir: Path, device=None, log=print) -> Path:
    """Run the teacher on every recording not yet analysed; outputs to out_dir/<path>.npz."""
    import torch

    from ..device import setup_device
    from ..models.infer import analyse as run, load_teacher

    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out_dir / _analysis_name(r.path)).exists()]
    log(f"analysis: {len(rows) - len(todo)} cached, {len(todo)} to run")
    if not todo:
        return out_dir
    device = device or setup_device()
    model, _ = load_teacher(teacher, device)
    for k, r in enumerate(todo, 1):
        res = run(model, load_audio(root / r.path).astype(np.float32), device)
        tmp = out_dir / (_analysis_name(r.path) + ".tmp.npz")
        np.savez(tmp, **{key: res[key] for key in ANALYSIS_KEYS})
        os.replace(tmp, out_dir / _analysis_name(r.path))
        if k % 50 == 0 or k == len(todo):
            log(f"  analysed {k}/{len(todo)}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out_dir


def _render_shard(args) -> dict:
    path, items, root, analysis_dir, cfg, seed = args
    clips, summary = [], {"files": [], "rejected": []}
    for file_idx, row in items:
        with np.load(analysis_dir / _analysis_name(row.path)) as z:
            analysis = {k: z[k] for k in z.files}
        audio = load_audio(root / row.path)
        src = source_from_audio(audio, analysis, cfg)
        for v in range(cfg.variants):
            rng = np.random.default_rng([seed, file_idx, v])
            got, info = render_variant(audio, src, rng, cfg)
            for _, _, meta in got:
                meta.update(corpus=row.corpus, speaker_id=row.speaker_id, source_path=row.path, file_index=file_idx, variant=v)
            clips.extend(got)
            summary["rejected"].extend({"path": row.path, "variant": v, **r} for r in info["rejected"])
        summary["files"].append(row.path)
    summary["n_clips"] = len(clips)
    if clips:
        tmp = Path(path).with_suffix(".tmp.npz")
        np.savez(tmp, **_pack(clips))
        os.replace(tmp, path)
    Path(path).with_suffix(".json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def render(rows: list[ManifestRow], root: Path, analysis_dir: Path, out_dir: Path, cfg: ResynthConfig, seed: int,
           files_per_shard: int = 8, workers: int | None = None, extra_index: dict | None = None, log=print) -> Path:
    """Write <out_dir>/shard_*.npz and index.json; resumable (a shard is done when its .json exists)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r.path)
    index = {"format_version": FORMAT_VERSION, "kind": "resynth", "seed": seed, "files_per_shard": files_per_shard,
             "files": [r.path for r in rows], "resynth_config": asdict(cfg), **(extra_index or {})}
    index_path = out_dir / "index.json"
    if index_path.exists():
        old = json.loads(index_path.read_text(encoding="utf-8"))
        new = json.loads(json.dumps(index))
        diff = sorted(k for k in new if old.get(k) != new[k])
        if diff:
            raise ValueError(f"{out_dir} was written with different settings ({', '.join(diff)}); use a new directory")
    else:
        index_path.write_text(json.dumps(index | {"complete": False}, indent=1), encoding="utf-8")
    groups = [list(enumerate(rows))[i : i + files_per_shard] for i in range(0, len(rows), files_per_shard)]
    jobs = [(str(out_dir / f"shard_{s:05d}.npz"), g, root, analysis_dir, cfg, seed)
            for s, g in enumerate(groups) if not (out_dir / f"shard_{s:05d}.json").exists()]
    log(f"{out_dir}: {len(rows)} recordings, {len(groups)} shards, {len(jobs)} to write")
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(_render_shard, jobs), 1):
            if k % 10 == 0 or k == len(jobs):
                log(f"  {k}/{len(jobs)} shards")
    summaries = [json.loads((out_dir / f"shard_{s:05d}.json").read_text(encoding="utf-8")) for s in range(len(groups))]
    # empty shards (every clip rejected) have no .npz; ShardReader lists the files it finds, so they get no entry
    shard_clips = [s["n_clips"] for s in summaries if s["n_clips"]]
    rejected = [r for s in summaries for r in s["rejected"]]
    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    qc = {"n_clips": sum(shard_clips), "n_rejected": len(rejected), "rejected_by_reason": reasons}
    index_path.write_text(json.dumps(index | {"n_clips": sum(shard_clips), "shard_clips": shard_clips, "qc": qc,
                                              "git_commit": _git_commit()}, indent=1), encoding="utf-8")
    log(f"  {qc['n_clips']} clips, {qc['n_rejected']} rejected {reasons}")
    return out_dir
