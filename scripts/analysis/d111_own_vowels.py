"""D-111: does D-109's F1 shift appear on sustained vowels in one voice and one recording chain? Inference only.

Reads a folder of sustained vowels laid out <vowel IPA>/<mode>/<file> (mode e.g. Sung_M1, Sung_M1_Inhale), runs each
model on every file (resampled to 16 kHz by `load_audio`'s method), and per file takes the median over its scored frames
of each model's F1-F3. Scored frames: every model voiced and sonorant (p > 0.5), inside the file's voiced span less
5 frames at each end; files with fewer than 10 such frames are dropped.
Per file: S = d109 seed mean - 5b seed mean, and each seed alone. Nulls: the branch null S0 = D-065's gain arm - 5b
(seed means; decay branches from the same checkpoints that moved nothing on PTDB-TUG, D-110's dry run) and the seed null
(5b s1 - s2). Sets by the 5b seed
mean's median f0: M1 egressive (main), M1 inhaled, and files with f0 250-350 Hz; f0 above 350 Hz is left out (F1 of the
close vowels at or below f0).

The recordings are private: the per-file table goes to <out dir> (outside the repo); stdout carries only summaries.

    python scripts/analysis/d111_own_vowels.py <recordings root> <out dir> [--only-5b (5b and the null arm only)]   (VA_DEVICE=cpu|cuda)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

B = {"b1": "runs/d040_5b_s1_b20k/ckpt_last.pt", "b2": "runs/d040_5b_s2_b20k/ckpt_last.pt"}
R = {"r1": "runs/d109_s1/ckpt_last.pt", "r2": "runs/d109_s2/ckpt_last.pt"}
N = {"n1": "runs/d065_gain_s1_b20k/ckpt_last.pt", "n2": "runs/d065_gain_s2_b20k/ckpt_last.pt"}  # branch null: an arm that moved nothing
CLOSE, MID, OPEN = ("iː", "ʉː"), ("eː", "ɔ"), ("æ", "ɐː")


def load16(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    return x


def score(root, models, dev):
    from vocal_analysis.models.infer import analyse, load_teacher

    files = sorted(f for f in Path(root).rglob("*") if f.is_file() and len(f.relative_to(root).parts) == 3)
    outs = {}
    for name, ck in models.items():
        model, _ = load_teacher(ck, dev)
        outs[name] = [analyse(model, load16(f), dev) for f in files]
        del model
    rows = []
    for i, f in enumerate(files):
        vowel, mode = f.relative_to(root).parts[:2]
        o = [outs[m][i] for m in models]
        ok = np.all([(r["voiced_p"] > 0.5) & (r["sonorant_p"] > 0.5) for r in o], 0)
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            continue
        span = np.zeros_like(ok)
        span[idx[0] + 5: idx[-1] - 4] = True
        ok &= span
        if ok.sum() < 10:
            continue
        row = {"file": str(f.relative_to(root)), "vowel": vowel, "mode": mode, "n": int(ok.sum()),
               "f0": float(np.median(np.mean([outs[m][i]["f0"][ok] for m in B if m in outs], 0)))}
        for m in models:
            row[m] = np.median(outs[m][i]["formants"][ok, :3], 0).tolist()
        rows.append(row)
    return rows


def summarise(rows, have_r):
    def arr(r, k):
        return np.array(r[k])

    lines = []
    sets = {
        "M1 egressive (main)": [r for r in rows if "M1" in r["mode"] and "nhale" not in r["mode"] and r["f0"] <= 350],
        "M1 inhaled": [r for r in rows if "M1" in r["mode"] and "nhale" in r["mode"] and r["f0"] <= 350],
        "f0 250-350 Hz (M2)": [r for r in rows if 250 <= r["f0"] <= 350],
    }
    left = [r for r in rows if r["f0"] > 350]
    lines.append(f"files scored {len(rows)}; left out with f0 > 350 Hz: {len(left)}")
    for name, rs in sets.items():
        if not rs:
            lines.append(f"{name}: no files")
            continue
        null = np.array([arr(r, "b1") - arr(r, "b2") for r in rs])
        lines.append(f"{name}: {len(rs)} files, f0 {min(r['f0'] for r in rs):.0f}-{max(r['f0'] for r in rs):.0f} Hz, "
                     f"{sum(r['n'] for r in rs)} frames")
        lines.append(f"  seed null (5b s1 - s2), per-file F1: median {np.median(null[:, 0]):+.1f}, median |.| "
                     f"{np.median(np.abs(null[:, 0])):.1f} Hz, max |.| {np.max(np.abs(null[:, 0])):.1f}")
        S0 = np.array([(arr(r, "n1") + arr(r, "n2") - arr(r, "b1") - arr(r, "b2")) / 2 for r in rs])
        lines.append(f"  branch null S0 (D-065 - 5b) F1: median {np.median(S0[:, 0]):+.1f}, median |.| {np.median(np.abs(S0[:, 0])):.1f}, "
                     f"max |.| {np.max(np.abs(S0[:, 0])):.1f} Hz; positive in {int((S0[:, 0] > 0).sum())} of {len(rs)}")
        for gname, vs in (("close", CLOSE), ("mid", MID), ("open", OPEN)):
            k = np.array([r["vowel"] in vs for r in rs])
            if k.any():
                lines.append(f"    S0 {gname:5s}: {int(k.sum())} files, median {np.median(S0[k, 0]):+.1f} Hz")
        if not have_r:
            continue
        S = np.array([(arr(r, "r1") + arr(r, "r2") - arr(r, "b1") - arr(r, "b2")) / 2 for r in rs])
        s1 = np.array([arr(r, "r1")[0] - arr(r, "b1")[0] for r in rs])
        s2 = np.array([arr(r, "r2")[0] - arr(r, "b2")[0] for r in rs])
        lines.append(f"  S F1: median {np.median(S[:, 0]):+.1f} Hz, positive in {int((S[:, 0] > 0).sum())} of {len(rs)} "
                     f"(seed 1 {np.median(s1):+.1f}, seed 2 {np.median(s2):+.1f}); F2 {np.median(S[:, 1]):+.1f}, F3 {np.median(S[:, 2]):+.1f}")
        groups = {}
        for gname, vs in (("close", CLOSE), ("mid", MID), ("open", OPEN)):
            k = np.array([r["vowel"] in vs for r in rs])
            if k.any():
                groups[gname] = float(np.median(S[k, 0]))
                lines.append(f"    {gname:5s} ({', '.join(vs)}): {int(k.sum())} files, S F1 median {groups[gname]:+.1f} Hz")
        if "open" in groups and "close" in groups:
            lines.append(f"    open - close: {groups['open'] - groups['close']:+.1f} Hz")
        f1 = np.array([(arr(r, "b1")[0] + arr(r, "b2")[0]) / 2 for r in rs])
        if len(rs) >= 4 and np.ptp(f1) > 100:
            sl = np.polyfit(f1, S[:, 0], 1)[0] * 100
            lines.append(f"    S F1 on 5b's F1 across files: {sl:+.1f} Hz per 100 Hz (5b's F1 range {np.ptp(f1):.0f} Hz)")
    return lines


if __name__ == "__main__":
    import torch

    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    only_b = "--only-5b" in sys.argv
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    models = {**B, **N} if only_b else {**B, **N, **R}
    rows = score(root, models, dev)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("d111_files_5b.json" if only_b else "d111_files.json")).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(summarise(rows, not only_b)))
