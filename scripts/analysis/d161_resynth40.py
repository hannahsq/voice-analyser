"""D-161: the four resynthesised sets re-rendered with 40 stored source harmonics (`ResynthConfig.n_source_harmonics`).

Each set is re-rendered from its own index.json: its stored config and seed, its file list, and the original whole-file
analyses it was made from (data/resynth/analysis/<corpus>/<teacher sha>), not `va resynth`'s windowed ones (D-028),
which would give different f0 and voicing. Only the source_h_db label should change, from 8 to 40 harmonics.

    python scripts/analysis/d161_resynth40.py check SET [N_SHARDS] [WORKERS]   re-render N_SHARDS (8) evenly spaced shards
        to cache/d161/check/SET/ and compare them with SET's flat arrays -> reports/analysis/d161/resynth_check_SET.txt
    python scripts/analysis/d161_resynth40.py render SET [WORKERS]             -> data/resynth/SET_h40/ (shards + flat)
    python scripts/analysis/d161_resynth40.py verify SET                       the whole re-rendered set against SET's flat
        arrays -> reports/analysis/d161/resynth_verify_SET.txt

SET is one of vocalset_train_d16k, zeroth_korean_train_d16k, vocalset_val_d16k, zeroth_korean_val_d16k.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = ("vocalset_train_d16k", "zeroth_korean_train_d16k", "vocalset_val_d16k", "zeroth_korean_val_d16k")
N_H = 40
REP = ROOT / "reports/analysis/d161"


def _tuples(v):
    return tuple(v) if isinstance(v, list) else v


def _config(stored: dict):
    """The stored ResynthConfig with n_source_harmonics = 40, and the fields the stored one lacks (they take defaults)."""
    from vocal_analysis.resynth.render import ResynthConfig
    from vocal_analysis.resynth.tracks import PerturbConfig

    pnames = {f.name for f in fields(PerturbConfig)}
    rnames = {f.name for f in fields(ResynthConfig)}
    extra = sorted((set(stored) - rnames) | {f"perturb.{k}" for k in set(stored["perturb"]) - pnames})
    if extra:
        raise ValueError(f"stored config has fields the code no longer knows: {extra}")
    pc = PerturbConfig(**{k: _tuples(v) for k, v in stored["perturb"].items()})
    cfg = ResynthConfig(**{k: _tuples(v) for k, v in stored.items() if k != "perturb"}, perturb=pc)
    missing = sorted((rnames - set(stored) - {"n_source_harmonics"}) | {f"perturb.{k}" for k in pnames - set(stored["perturb"])})
    cfg.n_source_harmonics = N_H
    return cfg, missing


def _setup(name: str):
    from vocal_analysis.data.manifest import manifest_hash, read_manifest

    if name not in SETS:
        raise ValueError(f"{name}: not one of {SETS}")
    d = json.loads((ROOT / f"data/resynth/{name}/index.json").read_text(encoding="utf-8"))
    man = ROOT / f"data/manifests/{d['corpus']}.csv"
    if manifest_hash(man) != d["manifest_hash"]:
        raise ValueError(f"{man} changed since {name} was rendered")
    files = d["files"]
    keep = set(files)
    rows = sorted((r for r in read_manifest(man) if r.split == d["split"] and r.path in keep), key=lambda r: r.path)
    if [r.path for r in rows] != files:
        raise ValueError(f"{name}: the manifest's rows don't reproduce the stored file list")
    adir = ROOT / f"data/resynth/analysis/{d['corpus']}/{d['teacher']['sha256']}"
    from vocal_analysis.resynth.corpus import _analysis_name

    gone = [r.path for r in rows if not (adir / _analysis_name(r.path)).exists()]
    if gone:
        raise ValueError(f"{len(gone)} analyses missing from {adir}, e.g. {gone[0]}")
    cfg, missing = _config(d["resynth_config"])
    return d, cfg, rows, adir, ROOT / f"data/{d['corpus']}", missing


def _timed(args):
    from vocal_analysis.resynth.corpus import _render_shard

    t = time.perf_counter()
    s = _render_shard(args)
    return s, time.perf_counter() - t


def _flat(name: str) -> dict:
    f = ROOT / f"data/resynth/{name}/flat"
    out = {p.stem: np.load(p, mmap_mode="r") for p in f.glob("*.npy")}
    return out


def _compare(new: dict, flat: dict, c0: int, n: int) -> list[str]:
    """Differences between a re-rendered shard's packed arrays (n clips) and flat clips c0..c0+n."""
    bad = []
    ao, fo = np.asarray(flat["audio_offsets"]), np.asarray(flat["frame_offsets"])
    na, nf = new["audio_offsets"], new["frame_offsets"]
    if len(na) - 1 != n:
        return [f"clips {len(na) - 1} != {n}"]
    if not np.array_equal(np.diff(na), np.diff(ao[c0 : c0 + n + 1])):
        bad.append("audio lengths")
    elif not np.array_equal(new["audio"], flat["audio"][ao[c0] : ao[c0 + n]]):
        bad.append("audio")
    if not np.array_equal(np.diff(nf), np.diff(fo[c0 : c0 + n + 1])):
        return bad + ["frame counts"]
    sl = slice(fo[c0], fo[c0 + n])
    for key in sorted(k[6:] for k in flat if k.startswith("label_")):
        a, b = new[f"label_{key}"], np.asarray(flat[f"label_{key}"][sl])
        if key == "source_h_db":
            if a.shape[1] != N_H:
                bad.append(f"source_h_db has {a.shape[1]} harmonics")
            a = a[:, : b.shape[1]]
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            bad.append(f"label {key}")
    meta = json.loads(str(new["meta"]))
    tl = np.array([m["qc"]["tract_left_median"] if m["qc"]["tract_left_median"] is not None else np.nan for m in meta])
    if "clip_tract_left" in flat and not np.array_equal(tl, np.asarray(flat["clip_tract_left"][c0 : c0 + n]), equal_nan=True):
        bad.append("clip tract_left")
    return bad


def cmd_check(name: str, n_shards: int = 8, workers: int = 8) -> None:
    d, cfg, rows, adir, root, missing = _setup(name)
    fps = d["files_per_shard"]
    groups = [list(enumerate(rows))[i : i + fps] for i in range(0, len(rows), fps)]
    pick = [int(i) for i in np.unique(np.linspace(0, len(groups) - 1, n_shards).round())]
    out = ROOT / f"cache/d161/check/{name}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(out / f"shard_{s:05d}.npz"), groups[s], root, adir, cfg, d["seed"]) for s in pick]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        res = list(ex.map(_timed, jobs))
    wall = time.perf_counter() - t0
    old_n = [json.loads((ROOT / f"data/resynth/{name}/shard_{s:05d}.json").read_text(encoding="utf-8"))["n_clips"]
             for s in range(len(groups))]
    flat = _flat(name)
    lines = [f"D-161 re-render check: {name}, {len(pick)} of {len(groups)} shards, n_source_harmonics {N_H}",
             f"  from {adir.relative_to(ROOT).as_posix()}, seed {d['seed']}, stored config (fields defaulted: {missing or 'none'})"]
    ok = True
    for s, (summ, sec) in zip(pick, res):
        c0 = sum(old_n[:s])
        if summ["n_clips"] == 0:
            bad = [] if old_n[s] == 0 else [f"clips 0 != {old_n[s]}"]
        else:
            with np.load(out / f"shard_{s:05d}.npz") as z:
                bad = _compare({k: z[k] for k in z.files}, flat, c0, old_n[s])
        ok &= not bad
        lines.append(f"  shard {s:5d}: {summ['n_clips']:3d} clips, {sec:6.1f} s  " + ("identical" if not bad else "DIFFERS: " + ", ".join(bad)))
    per = float(np.mean([sec for _, sec in res]))
    lines += [f"  mean {per:.1f} s per shard of {fps} recordings ({wall:.0f} s wall on {min(workers, len(jobs))} workers);"
              f" the whole set's {len(groups)} shards at that rate: {len(groups) * per / 3600:.2f} worker-hours",
              f"Result: {'REPRODUCES' if ok else 'DIFFERS'}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / f"resynth_check_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_render(name: str, workers: int = 16) -> None:
    from vocal_analysis.resynth import corpus
    from vocal_analysis.train.data import consolidate

    d, cfg, rows, adir, root, missing = _setup(name)
    out = ROOT / f"data/resynth/{name}_h40"
    extra = {k: d[k] for k in ("corpus", "split", "manifest_hash", "teacher")}
    extra["d161"] = {"from": f"data/resynth/{name}", "n_source_harmonics": N_H, "analysis": adir.relative_to(ROOT).as_posix(),
                     "defaulted_fields": missing}
    corpus.render(rows, root, adir, out, cfg, seed=d["seed"], files_per_shard=d["files_per_shard"], workers=workers,
                  extra_index=extra, log=lambda m: print(m, flush=True))
    consolidate(out, log=lambda m: print(m, flush=True))
    print("rendered", out, flush=True)


def cmd_verify(name: str) -> None:
    old, new = _flat(name), _flat(f"{name}_h40")
    n_old, n_new = len(old["audio_offsets"]) - 1, len(new["audio_offsets"]) - 1
    lines = [f"D-161 re-render verify: data/resynth/{name}_h40 against data/resynth/{name} (flat arrays, every clip)"]
    if n_old != n_new:
        bad = [f"clips {n_new} != {n_old}"]
    else:
        packed = {k: np.asarray(v) for k, v in new.items()}
        bad = []
        for k in ("audio_offsets", "frame_offsets"):
            if not np.array_equal(packed[k], np.asarray(old[k])):
                bad.append(k)
        if not bad:
            if not np.array_equal(packed["audio"], np.asarray(old["audio"])):
                bad.append("audio")
            for key in sorted(k[6:] for k in old if k.startswith("label_")):
                a, b = packed[f"label_{key}"], np.asarray(old[f"label_{key}"])
                if key == "source_h_db":
                    lines.append(f"  source_h_db: {b.shape[1]} -> {a.shape[1]} harmonics")
                    a = a[:, : b.shape[1]]
                if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                    bad.append(f"label {key}")
            if not np.array_equal(packed["clip_tract_left"], np.asarray(old["clip_tract_left"]), equal_nan=True):
                bad.append("clip tract_left")
    lines += [f"  {n_new} clips", f"Result: {'IDENTICAL apart from the added harmonics' if not bad else 'DIFFERS: ' + ', '.join(bad)}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / f"resynth_verify_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "check":
        cmd_check(name, *(int(a) for a in sys.argv[3:5]))
    elif cmd == "render":
        cmd_render(name, *(int(a) for a in sys.argv[3:4]))
    elif cmd == "verify":
        cmd_verify(name)
    else:
        raise SystemExit(__doc__)
