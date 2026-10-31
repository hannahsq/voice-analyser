"""D-057 (descriptive, one flag): is the region just above F1 fuller in real voices than in our synthesis?

Per voiced frame, a spectral envelope from the harmonic amplitudes (`backend.harmonics`, interpolated linearly in log
frequency between harmonics); F1's level is the envelope's own peak within +-15 % of the reference F1 (so a slightly
misplaced reference does not lower it); the valley is measured relative to that peak at the two places D-055 found
F1 sensitive to: a third of the way from F1 to F2 in log frequency ("low"), and the mean over D-055's fill region
(F1 + 0.25 d to F2 - 0.25 d; "fill"). Sets, each with its own references: Hillenbrand dev (steady-state frames, its
F1 / F2), synthetic validation (train_v2's held-out clips, exact labels, clean audio), resynthesised Zeroth val
(exact labels; recorded sex from AUDIO_INFO, numeric ids only). Frames are matched on f0 band (women 150-300 Hz,
men 80-150 Hz), F1 300-1000 Hz and F2 / F1 >= 1.6.

    python scripts/analysis/d057_valley_level.py   (CPU)
"""

import json
import sys
from pathlib import Path

import numpy as np

FS, HOP = 16000, 160
OUT = Path("reports/analysis/valley_level")
K = 40
SEED = 5057  # used by no other set
BANDS = {"women": (150.0, 300.0), "men": (80.0, 150.0)}
SYNTH_CLIPS, ZEROTH_CLIPS, FRAMES_PER_CLIP = 600, 600, 20
N_BOOT = 2000
SENS_C_PER_DB = {"low": 9.25 / 6.0, "fill": 11.45 / 6.0}  # D-055, v3.1 seed means at +6 dB (peak33, fill); linear, rough


def envelope_measures(h_db: np.ndarray, f0: float, F1: float, F2: float) -> tuple[float, float]:
    """(low, fill) in dB relative to F1's envelope peak; NaN if the envelope does not reach the points."""
    k = np.arange(1, len(h_db) + 1)
    ok = np.isfinite(h_db)
    if ok.sum() < 4:
        return np.nan, np.nan
    u, e = np.log2(f0 * k[ok]), h_db[ok]
    env = lambda x: np.interp(x, u, e, left=np.nan, right=np.nan)  # noqa: E731
    grid = np.linspace(np.log2(0.85 * F1), np.log2(1.15 * F1), 31)
    ge = env(grid)
    if not np.isfinite(ge).any():
        return np.nan, np.nan
    i = int(np.nanargmax(ge))
    a, peak = grid[i], ge[i]
    b = np.log2(F2)
    d = b - a
    if d < np.log2(1.6) * 0.9:
        return np.nan, np.nan
    low = env(a + d / 3) - peak
    fill = np.nanmean(env(np.linspace(a + 0.25 * d, b - 0.25 * d, 21))) - peak
    return float(low), float(fill)


def _keep(f0, F1, F2, band):
    lo, hi = BANDS[band]
    return lo <= f0 < hi and 300 <= F1 <= 1000 and F2 / F1 >= 1.6


def hillenbrand() -> list[dict]:
    from vocal_analysis.backend.harmonics import harmonic_amplitudes
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import hillenbrand as HB
    from vocal_analysis.eval.datasets import load_audio

    rows = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        it = HB.load(Path("data/hillenbrand"), row)
        g = it.meta["group"]
        if g not in ("w", "m"):
            continue
        t, f0, F1, F2 = it.times[0], it.f0[0], it.formants[0, 0], it.formants[0, 1]
        if not (np.isfinite(t) and f0 > 0 and F1 > 0 and F2 > 0):
            continue
        band = "women" if g == "w" else "men"
        if not _keep(f0, F1, F2, band):
            continue
        x = load_audio(it, FS)
        h = harmonic_amplitudes(x, FS, np.array([t]), np.array([f0]), K)[0]  # times in seconds
        low, fill = envelope_measures(h, f0, F1, F2)
        rows.append({"set": "hillenbrand", "band": band, "group": it.speaker_id, "low": low, "fill": fill, "f0": f0, "F1": F1})
    return rows


def _flat_frames(name: str, flat: str, ids: np.ndarray, groups, rng, sex=None) -> list[dict]:
    from vocal_analysis.backend.harmonics import harmonic_amplitudes

    d = Path(flat)
    a = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("audio", "label_f0", "label_voiced", "label_sonorant", "label_formants")}
    a_off, f_off = np.load(d / "audio_offsets.npy"), np.load(d / "frame_offsets.npy")
    rows = []
    for c in ids:
        sl = slice(int(f_off[c]), int(f_off[c + 1]))
        f0 = np.asarray(a["label_f0"][sl], float)
        F = np.asarray(a["label_formants"][sl], float)[:, :2]
        ok = np.asarray(a["label_voiced"][sl], bool) & np.asarray(a["label_sonorant"][sl], bool) & np.isfinite(F).all(1) & (f0 > 0)
        ok[:3] = ok[-3:] = False
        fr = np.flatnonzero(ok)
        if len(fr) > FRAMES_PER_CLIP:
            fr = np.sort(rng.choice(fr, FRAMES_PER_CLIP, replace=False))
        if not len(fr):
            continue
        x = np.asarray(a["audio"][a_off[c]: a_off[c + 1]], np.float32) / 32768.0
        h = harmonic_amplitudes(x, FS, fr * HOP / FS, f0[fr], K)
        for j, i in enumerate(fr):
            s = sex[c] if sex is not None else None
            for band in BANDS:
                if s is not None and s != ("f" if band == "women" else "m"):
                    continue
                if _keep(f0[i], F[i, 0], F[i, 1], band):
                    low, fill = envelope_measures(h[j], f0[i], F[i, 0], F[i, 1])
                    rows.append({"set": name, "band": band, "group": str(groups[c]), "low": low, "fill": fill, "f0": float(f0[i]),
                                 "F1": float(F[i, 0])})
    return rows


def synthetic(rng) -> list[dict]:
    from vocal_analysis.train.data import split_ids

    n = len(np.load("data/synth/train_v2/flat/audio_offsets.npy")) - 1
    ids = np.sort(rng.choice(split_ids(n)[1], SYNTH_CLIPS, replace=False))
    return _flat_frames("synthetic", "data/synth/train_v2/flat", ids, {int(c): f"clip{c}" for c in ids}, rng)


def zeroth(rng) -> list[dict]:
    sys.path.insert(0, "scripts/analysis")
    import d052_source_coverage as D52
    import d053_iseli_alwan as D53

    root = "data/resynth/zeroth_korean_val_d16k/flat"
    n = len(np.load(f"{root}/audio_offsets.npy")) - 1
    spk = D53._groups("zeroth", n)
    sexmap = D52.zeroth_sex()
    sex = np.array([sexmap.get(s) if s else None for s in spk], object)
    ids = np.flatnonzero(spk != "")
    ids = np.sort(rng.choice(ids, min(ZEROTH_CLIPS, len(ids)), replace=False))
    return _flat_frames("zeroth", root, ids, spk, rng, sex)


def boot(v: np.ndarray, g: np.ndarray, rng) -> tuple[float, float, float]:
    u, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(u))]
    bs = [np.median(v[np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])]) for _ in range(N_BOOT)]
    return float(np.median(v)), *map(float, np.percentile(bs, [2.5, 97.5]))


def diff_boot(a, ga, b, gb, rng) -> tuple[float, float, float]:
    """median(a) - median(b), each resampled by its own groups."""
    def res(v, g):
        u, inv = np.unique(g, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(u))]
        return lambda: np.median(v[np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])])
    ra, rb = res(a, ga), res(b, gb)
    bs = [ra() - rb() for _ in range(N_BOOT)]
    return float(np.median(a) - np.median(b)), *map(float, np.percentile(bs, [2.5, 97.5]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = hillenbrand() + synthetic(rng) + zeroth(rng)
    (OUT / "frames.json").write_text(json.dumps(rows), encoding="utf-8")
    lines = ["D-057: level just above F1, relative to F1's envelope peak (dB; median [95 % bootstrap over speakers or clips])",
             "low: a third of the way from F1 to F2 (log frequency); fill: mean over D-055's fill region. Frames matched on",
             "f0 band, F1 300-1000 Hz, F2 / F1 >= 1.6. Implied F1 shift: the difference times D-055's sensitivity at +6 dB",
             f"({SENS_C_PER_DB['low']:.2f} c/dB low, {SENS_C_PER_DB['fill']:.2f} c/dB fill; linear, rough).", ""]
    doc = {}
    for band in BANDS:
        lines.append(f"== {band} (f0 {BANDS[band][0]:.0f}-{BANDS[band][1]:.0f} Hz)")
        sel = {s: [r for r in rows if r["set"] == s and r["band"] == band] for s in ("hillenbrand", "synthetic", "zeroth")}
        for m in ("low", "fill"):
            vals = {s: (np.array([r[m] for r in v], float), np.array([r["group"] for r in v])) for s, v in sel.items()}
            vals = {s: (v[np.isfinite(v)], g[np.isfinite(v)]) for s, (v, g) in vals.items()}
            for s, (v, g) in vals.items():
                if len(v) >= 20:
                    pt, lo, hi = boot(v, g, rng)
                    doc[f"{band}/{m}/{s}"] = [pt, lo, hi, int(len(v)), int(len(np.unique(g)))]
                    lines.append(f"  {m:4s} {s:12s} {pt:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] dB  ({len(v)} frames, {len(np.unique(g))} groups)")
            for ref in ("synthetic", "zeroth"):
                (a, ga), (b, gb) = vals["hillenbrand"], vals[ref]
                if len(a) >= 20 and len(b) >= 20:
                    pt, lo, hi = diff_boot(a, ga, b, gb, rng)
                    doc[f"{band}/{m}/hillenbrand-{ref}"] = [pt, lo, hi]
                    lines.append(f"  {m:4s} real - {ref:9s} {pt:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] dB  -> implied F1 shift {pt * SENS_C_PER_DB[m]:+5.1f} c")
        lines.append("")
    w = doc
    flag = all(f"women/low/hillenbrand-{r}" in w and w[f"women/low/hillenbrand-{r}"][0] >= 2 and w[f"women/low/hillenbrand-{r}"][1] > 0
               for r in ("synthetic", "zeroth"))
    lines.append(f"FLAG (women, low: real - synthetic and real - resynthesised both >= 2 dB with intervals above 0): "
                 f"{'RAISED: real lower valley fuller' if flag else 'not raised'}")
    (OUT / "valley_level.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "valley_level.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("\n".join(lines))
