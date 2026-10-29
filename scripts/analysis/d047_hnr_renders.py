"""D-047: does aspiration level move teacher v3's F1 bias on synthetic renders? (D-045 addendum's HNR lead)

    python scripts/analysis/d047_hnr_renders.py check <out dir>   (code path: two arms of two clips differ only in aspiration)
    python scripts/analysis/d047_hnr_renders.py pilot <out dir>   (20 dB arm only: per-clip error spread, for the noise floor)
    python scripts/analysis/d047_hnr_renders.py run <out dir>     (all arms; writes d047.txt and d047.tsv)

Fresh train_v2 renders (teacher v3's synthetic training config, a seed it never saw, no channel), each regenerated with
hnr_db fixed per arm; nothing else in the clip changes (the generator still draws hnr_db, from a one-point range).
Clips are kept by their labels alone: not whispered, median label-voiced f0 165-300 Hz (women's range) or 90-160 Hz
(men's), so selection does not depend on the arm. Teacher v3 (whole-file analyse) against the label F1 on label-voiced,
non-whispered frames with finite F1, 5 frames trimmed at each end; per clip the median signed error (Hz).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TEACHER_V3 = Path("runs/conformer_v2r2_s1_d16k/ckpt_020000.pt")
INDEX = Path("docs/synth_sets/train_v2.index.json")
SEED = 4547  # train_v2 is seed 2
ARMS = (5.0, 10.0, 20.0, 30.0, 40.0, 60.0)  # dB; training draws 5-40, 60 is cleaner than any training clip
RANGES = {"women": (165.0, 300.0, 150), "men": (90.0, 160.0, 100)}
TRIM = 5
N_BOOT = 2000


def config(hnr):
    from vocal_analysis.synth.sampler import SynthConfig

    raw = json.loads(INDEX.read_text(encoding="utf-8"))["synth_config"]
    names = {f.name for f in dataclasses.fields(SynthConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in names}
    kw["hnr_db"] = (hnr, hnr)
    return SynthConfig(**kw)


def select(n_scan=4000):
    """clip indices per range, chosen from labels (arm-independent; labels don't depend on hnr_db)."""
    from vocal_analysis.synth.sampler import generate

    cfg = config(20.0)
    picked = {k: [] for k in RANGES}
    for i in range(n_scan):
        if all(len(picked[k]) >= RANGES[k][2] for k in RANGES):
            break
        c = generate(i, cfg, seed=SEED)
        lab = c.labels
        v = lab["voiced"] & ~lab["whisper"]
        if c.meta["config"]["p_whisper_clip"] and v.sum() < 50:
            continue
        f = np.nanmedian(lab["f0"][v]) if v.any() else np.nan
        for k, (lo, hi, n) in RANGES.items():
            if lo <= f < hi and len(picked[k]) < n:
                picked[k].append(i)
    return picked


def clip_error(teacher, dev, i, hnr):
    from vocal_analysis.models.infer import analyse
    from vocal_analysis.synth.sampler import generate

    c = generate(i, config(hnr), seed=SEED)
    r = analyse(teacher, c.audio, dev)
    lab = c.labels
    n = min(len(lab["t"]), len(r["formants"]))
    keep = lab["voiced"][:n] & ~lab["whisper"][:n] & np.isfinite(lab["formants"][:n, 0])
    keep[:TRIM] = False
    keep[n - TRIM:] = False
    e = r["formants"][:n, 0] - lab["formants"][:n, 0]
    e = e[keep & np.isfinite(e)]
    hnr_praat = praat_hnr(c.audio, lab["t"][:n][keep])
    return (float(np.median(e)) if len(e) else np.nan, float(np.median(np.abs(e))) if len(e) else np.nan,
            int(len(e)), hnr_praat)


def praat_hnr(x, times):
    import parselmouth

    try:
        h = parselmouth.Sound(np.asarray(x, np.float64), 16000).to_harmonicity_cc(time_step=0.01, minimum_pitch=75.0)
    except Exception:
        return np.nan
    v = np.array([h.get_value(t) for t in times])
    v = v[np.isfinite(v) & (v > -100)]
    return float(np.median(v)) if len(v) else np.nan


def boot_mean(d, rng):
    d = d[np.isfinite(d)]
    bs = [rng.choice(d, len(d)).mean() for _ in range(N_BOOT)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    import torch

    from vocal_analysis.models.infer import load_teacher

    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    teacher = load_teacher(TEACHER_V3, dev)[0]

    if stage == "check":
        from vocal_analysis.synth.sampler import generate

        for i in (0, 1):
            a = generate(i, config(5.0), seed=SEED, keep_components=True)
            b = generate(i, config(40.0), seed=SEED, keep_components=True)
            same_v = np.array_equal(a.components["voiced"], b.components["voiced"])
            same_l = all(np.array_equal(a.labels[k], b.labels[k], equal_nan=True) for k in a.labels)
            ra = np.sqrt(np.mean(a.components["aspiration"] ** 2) / np.mean(b.components["aspiration"] ** 2))
            print(f"clip {i}: voiced component identical {same_v}; labels identical {same_l}; "
                  f"aspiration rms ratio 5/40 dB {ra:.1f} (expected {10 ** (35 / 20):.1f})")
        return

    sel_path = out / "selection.json"
    if not sel_path.exists():
        sel_path.write_text(json.dumps(select()), encoding="utf-8")
    picked = json.loads(sel_path.read_text(encoding="utf-8"))
    print({k: len(v) for k, v in picked.items()}, flush=True)

    arms = (20.0,) if stage == "pilot" else ARMS
    tsv = out / ("pilot.tsv" if stage == "pilot" else "d047.tsv")
    done = set()
    if tsv.exists():
        for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            p = line.split("\t")
            done.add((int(p[1]), float(p[2])))
    else:
        tsv.write_text("range\tclip\thnr_db\tmed_err\tmed_abs\tframes\tpraat_hnr\n", encoding="utf-8")
    t0 = time.time()
    with tsv.open("a", encoding="utf-8") as f:
        for k, idx in picked.items():
            for i in idx:
                for h in arms:
                    if (i, h) in done:
                        continue
                    me, ma, nf, ph = clip_error(teacher, dev, i, h)
                    f.write(f"{k}\t{i}\t{h}\t{me:.3f}\t{ma:.3f}\t{nf}\t{ph:.2f}\n")
                    f.flush()
            print(f"{k} done ({time.time() - t0:.0f} s)", flush=True)

    rows = [line.split("\t") for line in tsv.read_text(encoding="utf-8").splitlines()[1:]]
    E = {}
    for k, i, h, me, ma, nf, ph in rows:
        E[k, int(i), float(h)] = (float(me), float(ma), int(nf), float(ph))
    rng = np.random.default_rng(47)
    lines = []
    if stage == "pilot":
        for k, idx in picked.items():
            d = np.array([E[k, i, 20.0][0] for i in idx])
            lines.append(f"{k}: {len(idx)} clips at 20 dB; per-clip median error SD {np.nanstd(d):.1f} Hz; "
                         f"interval half-width for a paired mean, bounded by this SD: {1.96 * np.nanstd(d) / np.sqrt(len(idx)):.1f} Hz")
        (out / "pilot.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return

    lines.append(f"D-047 ({time.strftime('%Y-%m-%d %H:%M')}): teacher v3 on fresh train_v2 renders (seed {SEED}), "
                 "hnr_db fixed per arm, no channel; per clip the median signed F1 error (Hz) on label-voiced frames")
    for k, idx in picked.items():
        lines.append(f"\n{k} ({len(idx)} clips)")
        for h in ARMS:
            v = np.array([E[k, i, h] for i in idx])
            lines.append(f"  {h:4.0f} dB: mean signed {np.nanmean(v[:, 0]):+6.1f}  median signed {np.nanmedian(v[:, 0]):+6.1f}  "
                         f"mean |err| {np.nanmean(v[:, 1]):6.1f}  Praat HNR median {np.nanmedian(v[:, 3]):5.1f} dB")
        for a, b in ((40.0, 5.0), (20.0, 5.0), (40.0, 20.0), (60.0, 40.0)):
            d = np.array([E[k, i, a][0] - E[k, i, b][0] for i in idx])
            m, lo, hi = boot_mean(d, rng)
            lines.append(f"  paired {a:.0f} - {b:.0f} dB: {m:+.1f} [{lo:+.1f}, {hi:+.1f}] Hz")
            if k == "women" and (a, b) == (40.0, 5.0):
                if lo >= 4:
                    rd = "cleaner excitation raises F1 on renders"
                elif hi <= -4:
                    rd = "cleaner excitation lowers F1 on renders (opposite)"
                elif lo > -4 and hi < 4:
                    rd = "no HNR effect on renders"
                else:
                    rd = "inconclusive"
                reading = rd
    lines.append(f"\nREADING (women, paired 40 - 5 dB, +-4 Hz): {reading}")
    (out / "d047.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
