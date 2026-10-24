"""D-034, reported only (pre-registered 2026-10-24): harmonic locking of teacher v3's F1 on real singing, by VocalSet
technique (straight, vibrato, breathy), val split, arpeggios and scales matched on (singer, exercise, vowel).
Part 1, original recordings: locking excess (share of voiced frames with F1 / f0 within 0.1 of a whole number >= 1, minus
the same with F1 shuffled against f0 within the file, 20 shuffles) and the following slope (per voiced run >= 50 frames,
log2 f0 and log2 F1 detrended by a centred 31-frame moving average; slope of F1's residual on f0's where the f0
residual's SD >= 15 cents). Part 2, a fresh resynthesis of each file (source_from_audio + one render_variant, default
ResynthConfig, seed per file): teacher v3's |F1 error| in cents against the exact labels, median per file; and the same
on clips with median |tract_left| < 0.2. Readings R1-R3 as registered in decisions.md.
Args: teacher checkpoint, out prefix. Writes <prefix>.txt and <prefix>_files.csv. $VA_DEVICE."""
import csv
import os
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.resynth.corpus import load_audio
from vocal_analysis.resynth.render import ResynthConfig, render_variant, source_from_audio

sys.stdout.reconfigure(encoding="utf-8")
TCK, PRE = sys.argv[1], sys.argv[2]
ROOT = Path("data/vocalset")
TECH = (("vibrato", re.compile(r"vibrat|vibrad")), ("straight", re.compile("straight")), ("breathy", re.compile("breathy")))
EXERCISES = ("arpeggios", "scales")


def technique(path):
    return next((t for t, rx in TECH if rx.search(path)), None)


def locking(f0, f1, rng):
    r = f1 / f0
    hit = lambda x: np.mean((np.abs(x - np.round(x)) < 0.1) & (np.round(x) >= 1))  # noqa: E731
    base = np.mean([hit(rng.permutation(f1) / f0) for _ in range(20)])
    return hit(r) - base


def following(f0, f1, ok):
    slopes = []
    idx = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(idx[::2], idx[1::2]):
        if b - a < 50:
            continue
        lf, l1 = np.log2(f0[a:b]), np.log2(f1[a:b])
        rf, r1 = lf - uniform_filter1d(lf, 31, mode="nearest"), l1 - uniform_filter1d(l1, 31, mode="nearest")
        if 1200 * rf.std() >= 15:
            slopes.append(float((rf * r1).sum() / (rf * rf).sum()))
    return slopes


def main():
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(TCK, dev)
    cfg = ResynthConfig()
    rows = [r for r in read_manifest("data/manifests/vocalset.csv") if r.split == "val"]
    files = []
    for r in rows:
        p = r.path.split("/")
        t, vowel = technique(r.path), Path(r.path).stem.split("_")[-1]
        if t and p[2] in EXERCISES and vowel in "aeiou" and len(vowel) == 1:
            files.append((r.path, t, p[1], p[2], vowel))
    print(f"{len(files)} files", flush=True)
    out_rows = []
    for k, (path, t, singer, ex, vowel) in enumerate(files, 1):
        x = load_audio(ROOT / path).astype(np.float32)
        res = analyse(model, x, dev)
        f0, f1, vp = res["f0"], res["formants"][:, 0], res["voiced_p"]
        ok = (vp > 0.5) & np.isfinite(f0) & np.isfinite(f1) & (f0 > 0)
        rng = np.random.default_rng(zlib.crc32(path.encode()))
        lock = locking(f0[ok], f1[ok], rng) if ok.sum() >= 20 else np.nan
        slopes = following(np.where(ok, f0, np.nan), np.where(ok, f1, np.nan), ok)
        # part 2: fresh resynthesis with exact labels
        an = {key: res[key] for key in ("f0", "voiced_p", "sonorant_p", "formants", "bandwidths")}
        err_all, err_low, tls = [], [], []
        try:
            src = source_from_audio(x, an, cfg)
            clips, _ = render_variant(x, src, rng, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  resynthesis failed for {path}: {e}", flush=True)
            clips = []
        for audio, lab, meta in clips:
            est = analyse(model, audio, dev)["formants"][:len(lab["f0"]), 0]
            true = lab["formants"][:, 0]
            m = lab["voiced"] & lab["sonorant"] & np.isfinite(true) & np.isfinite(est) & (true > 0)
            if not m.any():
                continue
            e = np.abs(1200 * np.log2(est[m] / true[m]))
            tl = meta["qc"]["tract_left_median"]
            tls.append(abs(tl) if tl is not None else np.nan)
            err_all.append(e)
            if tl is not None and abs(tl) < 0.2:
                err_low.append(e)
        out_rows.append({"path": path, "technique": t, "singer": singer, "exercise": ex, "vowel": vowel,
                         "median_f0": float(np.median(f0[ok])) if ok.any() else np.nan, "voiced_frames": int(ok.sum()),
                         "lock_excess": lock, "follow_slope": float(np.median(slopes)) if slopes else np.nan, "n_runs": len(slopes),
                         "resynth_abs_err_cents": float(np.median(np.concatenate(err_all))) if err_all else np.nan,
                         "resynth_abs_err_lowtl_cents": float(np.median(np.concatenate(err_low))) if err_low else np.nan,
                         "tract_left_median": float(np.nanmedian(tls)) if tls else np.nan, "clips": len(err_all)})
        o = out_rows[-1]
        print(f"[{k}/{len(files)}] {path}: f0 {o['median_f0']:.0f}, lock {o['lock_excess']:+.3f}, follow {o['follow_slope']:+.2f} ({o['n_runs']} runs), "
              f"resynth |err| {o['resynth_abs_err_cents']:.0f} c, |tract_left| {o['tract_left_median']:.2f}", flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    with open(f"{PRE}_files.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    fo = open(f"{PRE}.txt", "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        fo.write(s + "\n")

    say("D-034 harmonic locking on real singing, VocalSet val (reported only; pre-registered 2026-10-24); teacher v3")
    by = defaultdict(lambda: defaultdict(list))
    for o in out_rows:
        by[(o["singer"], o["exercise"], o["vowel"])][o["technique"]].append(o)
    triples = [k for k, v in by.items() if all(t in v for t in ("straight", "vibrato", "breathy"))]
    say(f"{len(out_rows)} files; {len(triples)} matched (singer, exercise, vowel) triples")
    mean = lambda os_, key: float(np.nanmean([o[key] for o in os_])) if os_ else np.nan  # noqa: E731
    say("\nby technique (all files): median f0 / locking excess / following slope (median of file medians) / resynth |F1 error| / with |tract_left| < 0.2 / |tract_left|")
    for t in ("straight", "vibrato", "breathy"):
        os_ = [o for o in out_rows if o["technique"] == t]
        say(f"  {t:9s} n {len(os_):3d}  f0 {np.nanmedian([o['median_f0'] for o in os_]):5.0f} Hz  lock {np.nanmean([o['lock_excess'] for o in os_]):+.3f}  "
            f"follow {np.nanmedian([o['follow_slope'] for o in os_]):+.2f}  |err| {np.nanmedian([o['resynth_abs_err_cents'] for o in os_]):5.0f} c  "
            f"low-tl {np.nanmedian([o['resynth_abs_err_lowtl_cents'] for o in os_]):5.0f} c  |tl| {np.nanmedian([o['tract_left_median'] for o in os_]):.2f}")

    rng = np.random.default_rng(0)

    def paired(key, a, b):
        d = np.array([mean(by[k][a], key) - mean(by[k][b], key) for k in triples])
        d = d[np.isfinite(d)]
        if not len(d):
            return np.nan, np.nan, np.nan, 0
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        return d.mean(), *np.percentile(bs, [2.5, 97.5]), len(d)

    vib = [o["follow_slope"] for o in out_rows if o["technique"] == "vibrato" and np.isfinite(o["follow_slope"])]
    m = float(np.median(vib)) if vib else np.nan
    say(f"\nR1: vibrato files' following slope, median {m:+.2f} over {len(vib)} files -> "
        f"{'F1 follows the vibrato' if m >= 0.5 else 'F1 holds' if m <= 0.2 else 'partly'}")
    for other in ("vibrato", "breathy"):
        d, lo, hi, n = paired("lock_excess", "straight", other)
        say(f"R2: locking excess straight - {other}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] over {n} triples -> "
            f"{other + ' locks less' if lo > 0 else other + ' locks more' if hi < 0 else 'no difference'}")
    for key, label in (("resynth_abs_err_cents", "R3"), ("resynth_abs_err_lowtl_cents", "R3 (|tract_left| < 0.2, check)")):
        for other in ("vibrato", "breathy"):
            d, lo, hi, n = paired(key, "straight", other)
            say(f"{label}: resynth |F1 error| straight - {other}: {d:+.0f} cents [{lo:+.0f}, {hi:+.0f}] over {n} triples -> "
                f"{other + ' reads F1 better' if lo > 0 else other + ' reads F1 worse' if hi < 0 else 'no difference'}")
    say("\nby singer (locking excess straight / vibrato / breathy; resynth |error| straight / vibrato / breathy)")
    for s in sorted({o["singer"] for o in out_rows}):
        g = {t: [o for o in out_rows if o["singer"] == s and o["technique"] == t] for t in ("straight", "vibrato", "breathy")}
        say(f"  {s:9s} " + " / ".join(f"{mean(g[t], 'lock_excess'):+.3f}" for t in g) + "   " + " / ".join(f"{mean(g[t], 'resynth_abs_err_cents'):4.0f}" for t in g))
    fo.close()


if __name__ == "__main__":
    main()
