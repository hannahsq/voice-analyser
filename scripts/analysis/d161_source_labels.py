"""D-161: the synthetic sets' 40-harmonic source labels, `source_h_db(rd, 40, ra_factor)` from each frame's Rd label
and the clip's Ra factor (D-160's meta; None on Fant's line), written as flat/label_source_h_db40.npy (float32).

Only the two Rd grid rows around each frame are computed, with the arithmetic of `glottal.harmonic_table_ra` and
`source_coefficients` (log magnitude interpolated linearly in Rd), so the result is the table's, without building a
full table per clip's factor. The first 8 harmonics are checked against the stored label_source_h_db on every frame.

    python scripts/analysis/d161_source_labels.py SET [WORKERS]   -> data/synth/SET/flat/label_source_h_db40.npy,
                                                                     reports/analysis/d161/source_labels_SET.txt
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N_H = 40
REP = ROOT / "reports/analysis/d161"


def _rows(idx: np.ndarray, ra_factor: float | None) -> dict[int, np.ndarray]:
    """log |c_k|, k = 1..40, at the Rd grid rows idx."""
    from vocal_analysis.synth import glottal as g

    if ra_factor is None:
        lm = g.harmonic_table()[0]
        return {int(i): lm[i, :N_H] for i in idx}
    n = 8192
    out = {}
    for i in idx:
        c = np.fft.rfft(g._waveform(g.lf_constants(*g.ra_scaled(float(g.RD_GRID[i]), float(ra_factor))), n))[1 : N_H + 1] / n
        out[int(i)] = np.log(np.abs(c))
    return out


def _levels(rd: np.ndarray, ra_factor: float | None) -> np.ndarray:
    from vocal_analysis.synth import glottal as g

    r = np.clip(np.asarray(rd, float), g.RD_MIN, g.RD_MAX)
    pos = (r - g.RD_MIN) / 0.01
    i0 = np.clip(np.floor(pos).astype(int), 0, len(g.RD_GRID) - 2)
    w = (pos - i0)[:, None]
    rows = _rows(np.unique(np.r_[i0, i0 + 1]), ra_factor)
    a = np.stack([rows[int(i)] for i in i0])
    b = np.stack([rows[int(i)] for i in i0 + 1])
    lm = (1 - w) * a + w * b
    return 20.0 * np.log10(np.abs(np.exp(lm)))


def _chunk(args) -> tuple[int, float, int]:
    """Clips lo..hi -> writes their frames; returns (frames, max |first 8 - stored|, frames with a non-finite stored H1)."""
    flat, lo, hi, factors = args
    fo = np.load(flat / "frame_offsets.npy")
    rd = np.load(flat / "label_rd.npy", mmap_mode="r")
    old = np.load(flat / "label_source_h_db.npy", mmap_mode="r")
    out = np.load(flat / "label_source_h_db40.npy", mmap_mode="r+")
    worst, nonfinite = 0.0, 0
    for c in range(lo, hi):
        a, b = int(fo[c]), int(fo[c + 1])
        h = _levels(np.asarray(rd[a:b]), factors[c - lo])
        st = np.asarray(old[a:b])
        fin = np.all(np.isfinite(st), 1)
        nonfinite += int((~fin).sum())
        h[~fin] = np.nan  # interactive clips: no LF source (none in train_v2 / train_v2w)
        if fin.any():
            worst = max(worst, float(np.max(np.abs(h[fin, : st.shape[1]] - st[fin]))))
        out[a:b] = h.astype(np.float32)
    out.flush()
    return int(fo[hi] - fo[lo]), worst, nonfinite


def _factors(set_dir: Path, n_clips: int) -> list[float | None]:
    shards = sorted(set_dir.glob("shard_*.npz"))
    if not shards:  # sets before D-160 have no Ra factors, and their shards were removed after consolidation
        idx = json.loads((set_dir / "index.json").read_text(encoding="utf-8"))
        cfg = idx.get("synth_config", {})
        if cfg.get("p_ra_offline", 0.0):
            raise ValueError(f"{set_dir}: off-line Ra in its config but no shards to read the factors from")
        return [None] * n_clips
    out: list[float | None] = []
    for s in shards:
        with np.load(s) as z:
            out.extend((m.get("d160") or {}).get("ra_factor") for m in json.loads(str(z["meta"])))
    if len(out) != n_clips:
        raise ValueError(f"{set_dir}: {len(out)} clips in the shards' meta, {n_clips} in flat")
    return out


def main(name: str, workers: int = 16) -> None:
    set_dir = ROOT / f"data/synth/{name}"
    flat = set_dir / "flat"
    done = json.loads((flat / "done").read_text(encoding="utf-8"))
    n_clips, n_frames = done["n_clips"], done["n_frames"]
    factors = _factors(set_dir, n_clips)
    target = flat / "label_source_h_db40.npy"
    np.lib.format.open_memmap(target, "w+", np.float32, (n_frames, N_H)).flush()
    edges = np.linspace(0, n_clips, 4 * workers + 1).round().astype(int)
    jobs = [(flat, int(a), int(b), factors[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_chunk, jobs))
    frames = sum(r[0] for r in res)
    worst = max(r[1] for r in res)
    nonfinite = sum(r[2] for r in res)
    off = sum(f is not None for f in factors)
    ok = frames == n_frames and worst < 1e-3
    lines = [f"D-161 synthetic source labels: data/synth/{name}, {n_clips} clips, {n_frames} frames, {N_H} harmonics",
             f"  clips with an off-line Ra factor: {off} ({off / n_clips:.1%})",
             f"  frames with a non-finite stored source label (left NaN): {nonfinite}",
             f"  first 8 against the stored label_source_h_db: max |difference| {worst:.2e} dB",
             f"Result: {'MATCHES' if ok else 'DIFFERS'}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / f"source_labels_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not ok:
        target.unlink()
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
