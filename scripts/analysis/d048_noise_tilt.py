"""D-048: does an aspiration shape that matches real voices' band periodic fraction move teacher v3's F1 bias on renders?

    python scripts/analysis/d048_noise_tilt.py calib <out dir>   (band profile only, on calibration clips: tilt x HNR grid)
    python scripts/analysis/d048_noise_tilt.py run <out dir>     (registered arms on D-047's clips; writes d048.txt)

As D-047 (fresh train_v2 renders, seed 4547, paired, no channel, teacher v3 against label F1), but the arms fix both the
aspiration level and its tilt before the tract (`SynthConfig.asp_tilt_db_oct`; None = the +6 dB/octave difference of
every earlier set). Calibration clips are drawn separately (seed 4550), so the arms are chosen without the test clips.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d045_probes as P  # noqa: E402
import d047_hnr_renders as R  # noqa: E402

TILTS = (None, 0.0, -6.0, -12.0)
HNRS = (5.0, 10.0, 15.0, 20.0, 30.0)
CALIB_SEED = 4550
N_CALIB = 30
# real women's band periodic fraction (D-045 follow-up A): mean of Zeroth-Korean test and speechocean762 adults
REAL_WOMEN = np.array([(0.911 + 0.904) / 2, (0.527 + 0.527) / 2, (0.277 + 0.295) / 2, (0.119 + 0.134) / 2, (0.035 + 0.048) / 2])
ARMS_FILE = "arms.json"


def config(hnr, tilt):
    return dataclasses.replace(R.config(hnr), asp_tilt_db_oct=tilt)


def profile(i, hnr, tilt, seed):
    from vocal_analysis.synth.sampler import generate

    c = generate(i, config(hnr, tilt), seed=seed)
    cs, _ = P.praat_frames(c.audio, 60, np.random.default_rng(i))
    return np.nanmean(P.bpf(c.audio.astype(np.float64), cs), 0) if len(cs) else np.full(5, np.nan)


def calib(out):
    from vocal_analysis.synth.sampler import generate

    cfg = config(20.0, None)
    clips = []
    for i in range(2000):
        c = generate(i, cfg, seed=CALIB_SEED)
        v = c.labels["voiced"] & ~c.labels["whisper"]
        if v.sum() >= 50 and 165 <= np.nanmedian(c.labels["f0"][v]) < 300:
            clips.append(i)
        if len(clips) >= N_CALIB:
            break
    lines = [f"D-048 calibration: {len(clips)} women's-range clips (seed {CALIB_SEED}), clean; band periodic fraction "
             "0-1 / 1-2 / 2-4 / 4-6 / 6-8 kHz; distance = root mean square over bands to real women's mean profile",
             "real women " + " ".join(f"{v:.3f}" for v in REAL_WOMEN), ""]
    best = {}
    for tilt in TILTS:
        for h in HNRS:
            prof = np.nanmean([profile(i, h, tilt, CALIB_SEED) for i in clips], 0)
            d = float(np.sqrt(np.mean((prof - REAL_WOMEN) ** 2)))
            best[tilt, h] = d
            lines.append(f"tilt {'diff' if tilt is None else f'{tilt:+.0f}':>4s} hnr {h:4.0f}: " + " ".join(f"{v:.3f}" for v in prof)
                         + f"  distance {d:.3f}")
            print(lines[-1], flush=True)
    shape = min(best, key=best.get)
    level = min((k for k in best if k[0] is None), key=best.get)
    arms = {"reference": [None, 20.0], "level_matched": list(level), "shape_matched": list(shape)}
    lines.append(f"\nchosen: reference = diff at 20 dB (D-047's arm); level_matched = diff at {level[1]:.0f} dB "
                 f"(distance {best[level]:.3f}); shape_matched = tilt {shape[0]} at {shape[1]:.0f} dB (distance {best[shape]:.3f})")
    (out / "calib.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / ARMS_FILE).write_text(json.dumps(arms), encoding="utf-8")
    print("\n".join(lines[-1:]))


def clip_row(teacher, dev, i, hnr, tilt):
    from vocal_analysis.models.infer import analyse
    from vocal_analysis.synth.sampler import generate

    c = generate(i, config(hnr, tilt), seed=R.SEED)
    r = analyse(teacher, c.audio, dev)
    lab = c.labels
    n = min(len(lab["t"]), len(r["formants"]))
    keep = lab["voiced"][:n] & ~lab["whisper"][:n] & np.isfinite(lab["formants"][:n, 0])
    keep[:R.TRIM] = False
    keep[n - R.TRIM:] = False
    e = r["formants"][:n, 0] - lab["formants"][:n, 0]
    e = e[keep & np.isfinite(e)]
    cs, _ = P.praat_frames(c.audio, 60, np.random.default_rng(i))
    prof = np.nanmean(P.bpf(c.audio.astype(np.float64), cs), 0) if len(cs) else np.full(5, np.nan)
    return (float(np.median(e)) if len(e) else np.nan, float(np.median(np.abs(e))) if len(e) else np.nan, prof)


def run(out):
    import torch

    from vocal_analysis.models.infer import load_teacher

    arms = json.loads((out / ARMS_FILE).read_text())
    picked = json.loads(Path("runs/d047/selection.json").read_text())
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    teacher = load_teacher(R.TEACHER_V3, dev)[0]
    tsv = out / "d048.tsv"
    done = set()
    if tsv.exists():
        done = {(p[0], int(p[1]), p[2]) for p in (ln.split("\t") for ln in tsv.read_text().splitlines()[1:])}
    else:
        tsv.write_text("range\tclip\tarm\tmed_err\tmed_abs\t" + "\t".join(f"bpf{b}" for b in range(5)) + "\n")
    t0 = time.time()
    with tsv.open("a") as f:
        for k, idx in picked.items():
            for i in idx:
                for arm, (tilt, h) in arms.items():
                    if (k, i, arm) in done:
                        continue
                    me, ma, prof = clip_row(teacher, dev, i, h, tilt)
                    f.write(f"{k}\t{i}\t{arm}\t{me:.3f}\t{ma:.3f}\t" + "\t".join(f"{v:.4f}" for v in prof) + "\n")
                    f.flush()
            print(f"{k} done ({time.time() - t0:.0f} s)", flush=True)

    E = {}
    for p in (ln.split("\t") for ln in tsv.read_text().splitlines()[1:]):
        E[p[0], int(p[1]), p[2]] = (float(p[3]), float(p[4]), np.array(p[5:10], float))
    rng = np.random.default_rng(48)
    lines = [f"D-048 ({time.strftime('%Y-%m-%d %H:%M')}): arms " + "; ".join(f"{a} = tilt {t} at {h:.0f} dB" for a, (t, h) in arms.items()),
             "per clip the median signed F1 error (Hz); band periodic fraction 0-1 / 1-2 / 2-4 / 4-6 / 6-8 kHz",
             "real women " + " ".join(f"{v:.3f}" for v in REAL_WOMEN)]
    reading = None
    for k, idx in picked.items():
        lines.append(f"\n{k} ({len(idx)} clips)")
        for arm in arms:
            v = [E[k, i, arm] for i in idx]
            lines.append(f"  {arm:14s} mean signed {np.nanmean([x[0] for x in v]):+6.1f}  mean |err| {np.nanmean([x[1] for x in v]):5.1f}  "
                         f"bpf " + " ".join(f"{x:.3f}" for x in np.nanmean([x[2] for x in v], 0)))
        for a, b in (("shape_matched", "reference"), ("level_matched", "reference"), ("shape_matched", "level_matched")):
            d = np.array([E[k, i, a][0] - E[k, i, b][0] for i in idx])
            d = d[np.isfinite(d)]
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"  paired {a} - {b}: {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] Hz")
            if k == "women" and a == "shape_matched" and b == "reference":
                reading = ("matched noise shape raises F1 on renders" if lo >= 4 else
                           "matched noise shape lowers F1 on renders" if hi <= -4 else
                           "no noise-shape effect on renders" if lo > -4 and hi < 4 else "inconclusive")
    lines.append(f"\nREADING (women, shape_matched - reference, +-4 Hz): {reading}")
    (out / "d048.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    {"calib": calib, "run": run}[stage](out)


if __name__ == "__main__":
    main()
