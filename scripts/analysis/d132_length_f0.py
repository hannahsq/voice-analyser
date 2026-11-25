"""D-132: does the tract-length head read pitch as length? Story's ten vowels at five fixed tract lengths, rendered at
seven f0s (90-350 Hz) with D-131's speech-like source; each model's length heads and F1-F3 per clip.

    python scripts/analysis/d132_length_f0.py render             -> data/probes/length_f0_v2/ (v1: before the 2026-11-25 file-name fix)
    python scripts/analysis/d132_length_f0.py score NAME CKPT    -> reports/probes/length_f0_v2/NAME.npz ($VA_DEVICE)
    python scripts/analysis/d132_length_f0.py analyse            -> reports/analysis/d132/d132_v2.txt (d132.txt: the v1 run)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/probes/length_f0_v2"  # v2: names safe on case-insensitive disks
RAW = ROOT / "reports/probes/length_f0_v2"
OUT = ROOT / "reports/analysis/d132"
LENGTHS = (13.0, 14.5, 15.6, 17.0, 18.5)
F0S = (90.0, 110.0, 140.0, 180.0, 220.0, 280.0, 350.0)
SEEDS = (0, 1)
DUR = 1.5
EDGE = 20  # frames left out at each end (fades, onsets)
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")
FIELDS = ("vtl", "vtl_inst", "F1", "F2", "F3")  # log2 cm, log2 cm, log2 Hz
B, SHUF = 1000, 200


def _d131():
    spec = importlib.util.spec_from_file_location("d131_context", ROOT / "scripts/analysis/d131_context.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_render() -> None:
    from vocal_analysis.synth import tube

    d131 = _d131()
    CACHE.mkdir(parents=True, exist_ok=True)
    items = []
    for L in LENGTHS:
        for v, (q1, q2) in tube.STORY_VOWELS.items():
            f, bw = tube.poles(tube.build_shape(q1, q2, L), f_max_hz=7600.0)
            k = min(d131.d130.N_P, len(f))
            F = np.full((2, d131.d130.N_P), np.nan)
            Bw = np.full((2, d131.d130.N_P), np.nan)
            F[:, :k], Bw[:, :k] = f[:k], bw[:k]
            for f0 in F0S:
                for seed in SEEDS:
                    tag = "".join(c + "^" if c.isupper() else c for c in v)  # Windows names ignore case: I and i would share
                    name = f"L{L:g}_{tag}_f{f0:g}_s{seed}"
                    y, tf, lab = d131.render_jit(F, Bw, np.array([0.0, DUR]), f0, zlib.crc32(name.encode()))
                    np.savez_compressed(CACHE / f"{name}.npz", times=tf, formants=lab,
                                        audio=np.round(y / (np.abs(y).max() + 1e-12) * 16000).astype(np.int16))
                    items.append({"name": name, "length": L, "vowel": v, "f0": f0, "seed": seed})
        print(f"rendered length {L}", flush=True)
    (CACHE / "meta.json").write_text(json.dumps({"items": items}, indent=1))
    print(f"{len(items)} items")


def cmd_score(name: str, ckpt: str) -> None:
    import torch

    from vocal_analysis.models.infer import DEFAULT_WINDOW, analyse, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ckpt, dev)
    causal = bool(model.cfg.causal)
    items = json.loads((CACHE / "meta.json").read_text())["items"]
    out = np.full((len(items), len(FIELDS)), np.nan)
    for i, it in enumerate(items):
        d = np.load(CACHE / f"{it['name']}.npz")
        r = analyse(model, d["audio"].astype(np.float32) / 32000, dev, window_frames=None if causal else DEFAULT_WINDOW)
        s = slice(EDGE, len(d["times"]) - EDGE)
        cols = [r.get("vtl_log2cm"), r.get("vtl_inst_log2cm")] + [np.log2(r["formants"][:, k]) for k in range(3)]
        out[i] = [np.nanmedian(np.asarray(c)[s]) if c is not None else np.nan for c in cols]
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / f"{name}.npz", est=out, fields=np.array(FIELDS))
    print(f"{name}: {len(items)} clips ({'causal' if causal else 'windowed'})")


def _slope(x, y, cell):
    """Fixed-effects slope of y on x within cells."""
    dx, dy = x.copy(), y.copy()
    for c in np.unique(cell):
        m = cell == c
        dx[m] -= dx[m].mean()
        dy[m] -= dy[m].mean()
    ok = np.isfinite(dy)
    return float((dx[ok] * dy[ok]).sum() / (dx[ok] ** 2).sum())


def cmd_analyse() -> None:
    items = json.loads((CACHE / "meta.json").read_text())["items"]
    L = np.array([it["length"] for it in items])
    x = np.log2([it["f0"] for it in items])
    cell = np.array([f"{it['length']}_{it['vowel']}" for it in items])
    cells = np.unique(cell)
    idx = {c: np.flatnonzero(cell == c) for c in cells}
    est = {n: np.load(RAW / f"{n}.npz")["est"] for n in TEACHERS + STUDENTS}
    rng = np.random.default_rng(132)
    lines = ["D-132: pitch slope within (true length, vowel) cells; length heads in cm per octave of f0 (estimate in cm), "
             "formants in cents per octave; [95 % bootstrap over cells]; shuffle null = 97.5th percentile of |slope| "
             "with f0 permuted within cells", ""]

    def value(n, j, rows):
        e = est[n][rows, j]
        return 2 ** e if j < 2 else 1200 * e  # cm, or cents

    def group_slope(names, j, rows):
        return np.mean([_slope(x[rows], value(n, j, rows), cell[rows]) for n in names])

    groups = [(n, [n]) for n in TEACHERS] + [("teacher (seed mean)", list(TEACHERS)), ("students (mean of four)", list(STUDENTS))]
    for j, fld in enumerate(FIELDS):
        if all(np.isnan(est[n][:, j]).all() for n in TEACHERS):
            lines.append(f"{fld}: not output by these models")
            continue
        unit = "cm/oct" if j < 2 else "c/oct"
        lines.append(f"{fld}:")
        for gname, names in groups:
            if all(np.isnan(est[n][:, j]).all() for n in names):
                continue
            allr = np.arange(len(items))
            s = group_slope(names, j, allr)
            boots = []
            for _ in range(B):
                rows = np.concatenate([idx[c] for c in rng.choice(cells, len(cells))])
                boots.append(group_slope(names, j, rows))
            null = []
            for _ in range(SHUF):
                xs = x.copy()
                for c in cells:
                    xs[idx[c]] = rng.permutation(xs[idx[c]])
                null.append(abs(np.mean([_slope(xs, value(n, j, allr), cell) for n in names])))
            per_len = []
            for Lv in LENGTHS:
                r = np.flatnonzero(L == Lv)
                per_len.append(f"{Lv:g}: {group_slope(names, j, r):+.2f}")
            lo, hi = np.percentile(boots, [2.5, 97.5])
            line = (f"  {gname:24s} {s:+.3f} {unit} [{lo:+.3f}, {hi:+.3f}]; null {np.percentile(null, 97.5):.3f}; "
                    f"by length {', '.join(per_len)}")
            if j < 2:
                bias = [np.mean([np.median(value(n, j, np.flatnonzero(L == Lv))) for n in names]) - Lv for Lv in LENGTHS]
                line += "; median estimate - truth (cm) " + ", ".join(f"{b:+.2f}" for b in bias)
            lines.append(line)
        lines.append("")
    if any(np.isfinite(est[n][:, 0]).any() for n in TEACHERS):
        lines.append("Teacher (seed mean) length estimate (cm) by f0, averaged over vowels, per true length:")
        for Lv in LENGTHS:
            row = []
            for f0 in F0S:
                r = np.flatnonzero((L == Lv) & (np.array([it["f0"] for it in items]) == f0))
                row.append(np.mean([np.median(2 ** est[n][r, 0]) for n in TEACHERS]))
            lines.append(f"  {Lv:5.1f}: " + "  ".join(f"{f0:g} Hz {v:5.2f}" for f0, v in zip(F0S, row)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d132_v2.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        cmd_render()
    elif cmd == "score":
        cmd_score(sys.argv[2], sys.argv[3])
    elif cmd == "analyse":
        cmd_analyse()
