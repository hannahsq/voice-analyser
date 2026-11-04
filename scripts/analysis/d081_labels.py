"""D-081: regenerate train_v2ko's first 3000 clips and val_v2ko (D-078's config and seeds) with the oracle labels
(acoustic bandwidths, the clip's coupling, gap offset, open quotient), plus the clip's jitter, shimmer and HNR per
frame. Checks every clip's formants and opening equal the flat set's, so D-079's cached features line up.

    python scripts/analysis/d081_labels.py [workers]
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

C = Path("cache/d081")
SETS = {"train": (Path("data/synth/train_v2ko/flat"), 3000, 78), "val": (Path("data/synth/val_v2ko/flat"), None, 7801)}
KEYS = ("formants", "formants_acoustic", "bandwidths", "bandwidths_acoustic", "glottal_opening", "coupling_df",
        "coupling_bm", "gap_offset", "open_quotient", "rd", "f0", "sonorant", "length_cm", "length_inst_cm")
CLIP = ("jitter", "shimmer_db", "hnr_db")


def one(job):
    import d078_render

    from vocal_analysis.synth.sampler import generate

    i, seed = job
    c = generate(i, d078_render.config(), seed)
    n = len(c.labels["f0"])
    out = {}
    for k in KEYS:
        v = c.labels.get(k)
        if v is None:  # an uncoupled (interactive) clip: no coupling labels
            v = np.full((n, 4) if k in ("coupling_df", "coupling_bm", "bandwidths_acoustic") else n, np.nan)
        out[k] = np.asarray(v, np.float64)
    for k in CLIP:
        out[k] = np.full(n, np.nan if c.meta[k] is None else float(c.meta[k]))
    return i, out


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    C.mkdir(parents=True, exist_ok=True)
    for split, (flat, n, seed) in SETS.items():
        out = C / f"{split}_labels.npz"
        if out.exists():
            continue
        f_off = np.load(flat / "frame_offsets.npy")
        n = (len(f_off) - 1) if n is None else n
        ref = {k: np.load(flat / f"label_{k}.npy", mmap_mode="r") for k in ("formants", "glottal_opening")}
        res = {}
        with ProcessPoolExecutor(workers) as ex:
            for i, lab in ex.map(one, [(i, seed) for i in range(n)], chunksize=8):
                res[i] = lab
        for i in range(n):
            t = int(f_off[i + 1] - f_off[i])
            for k in ref:
                a, b = res[i][k][:t], np.asarray(ref[k][f_off[i]:f_off[i + 1]], np.float64)
                if len(res[i][k]) != t or not np.allclose(a, b, rtol=1e-4, equal_nan=True):
                    raise SystemExit(f"{split} clip {i}: {k} differs from {flat}")
        arr = {k: np.concatenate([res[i][k] for i in range(n)]).astype(np.float32) for k in (*KEYS, *CLIP)}
        arr["clip"] = np.concatenate([np.full(len(res[i]["f0"]), i) for i in range(n)])
        arr["frame"] = np.concatenate([np.arange(len(res[i]["f0"])) for i in range(n)])
        np.savez(out, **arr)
        print(f"{split}: {n} clips, labels match {flat}", flush=True)
