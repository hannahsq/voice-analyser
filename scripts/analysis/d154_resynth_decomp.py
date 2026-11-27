"""D-154: is the real-voice dropped formant (D-152) in the source, the tract, or the formant pattern? D-152's primary
VocalSet files on D-153's fixed frames, scored as the original (O), the real IAIF source through our resonators at
Praat's smoothed tract (R), and the synthesiser's source through the same tract (S). CPU, inference only.

    python scripts/analysis/d154_resynth_decomp.py run [SINGER,...]   -> cache/d154/<singer>.json (one process per list)
    python scripts/analysis/d154_resynth_decomp.py report             -> reports/analysis/d154/d154.{txt,json}
"""

from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d154"
CACHE = ROOT / "cache/d154"
D152 = ROOT / "cache/d152"
MODELS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
REF = "d040_5b_s2_b20k"
FS, HOP = 16000, 160
MIN_FRAMES = 10


def _files(singer):
    """D-152's primary files for one singer (D-153's selection), with each file's full f0 and voicing tracks."""
    z = np.load(D152 / f"{singer}.npz")
    meta = json.loads(str(z["meta"]))
    P, f0, fid = z["P"], z["f0"], z["fid"]
    v = z[f"{REF}|v"] > 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        cell = v & np.all(np.isfinite(P), 1) & (P[:, 0] >= 350) & (P[:, 0] <= 650) & (P[:, 1] / P[:, 0] < 2.2) & (f0 <= 150)
    for u, mt in enumerate(meta):
        k = fid == u
        if cell[k].sum() >= MIN_FRAMES:
            yield {"meta": mt, "idx": np.where(cell[k])[0], "P": P[k], "f0": f0[k], "v": v[k],
                   "O": {m: z[f"{m}|F"][k] for m in MODELS}}


def _fill(x):
    t = np.arange(len(x))
    ok = np.isfinite(x)
    return np.interp(t, t[ok], x[ok]) if ok.any() else np.full(len(x), np.nan)


def _tract(y, n, voiced):
    """Praat F1-F4 and B1-B4 per 10 ms frame, gaps filled; formants smoothed as singing resynthesis, bandwidths
    median-filtered and clipped to 20-1000 Hz."""
    import parselmouth
    from scipy.ndimage import median_filter

    from vocal_analysis.resynth.tracks import smooth_tracks

    fm = parselmouth.Sound(y.astype(np.float64), FS).to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                                                    maximum_formant=5000.0)
    ts = np.arange(n) * 0.01
    F = np.array([[fm.get_value_at_time(k, t) for k in range(1, 5)] for t in ts], float)
    B = np.array([[fm.get_bandwidth_at_time(k, t) for k in range(1, 5)] for t in ts], float)
    F = np.stack([_fill(F[:, k]) for k in range(4)], 1)
    B = np.stack([_fill(B[:, k]) for k in range(4)], 1)
    F = smooth_tracks(F, voiced, 9, 6.0)
    B = np.clip(median_filter(B, size=(9, 1), mode="nearest"), 20.0, 1000.0)
    return F, B


def _synth_source(n_samples, f0, voiced, F, B, seed):
    """The grid's source (D-142: Rd 1.0, jitter 0.01, shimmer 0.5 dB) along the recording's f0 track and voicing."""
    from scipy.ndimage import uniform_filter1d

    from vocal_analysis.synth import tract, voice

    C = n_samples // voice.CTRL + 2
    tc = np.arange(C) * voice.CTRL / FS
    tf = np.arange(len(f0)) * HOP / FS
    f0f = f0.astype(float).copy()
    f0f[~voiced] = np.nan
    f0c = np.interp(tc, tf, _fill(f0f))
    amp = uniform_filter1d(np.interp(tc, tf, voiced.astype(float)), 20)
    pf = np.full((C, tract.N_POLES), np.nan)
    pb = pf.copy()
    for k in range(4):
        pf[:, k] = np.interp(tc, tf, F[:, k])
        pb[:, k] = np.interp(tc, tf, B[:, k])
    tr = voice.Trajectories(f0=f0c, rd=np.full(C, 1.0), voice_amp=amp, whisper_amp=np.zeros(C), fric_amp=np.zeros(C),
                            poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(tr, np.random.default_rng(seed), jitter=0.01, shimmer_db=0.5)["total"]
    return y[:n_samples] if len(y) >= n_samples else np.r_[y, np.zeros(n_samples - len(y))]


def _share(F, ref):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.mean(np.abs(np.log(F[:, 1] / ref[:, 2])) < np.abs(np.log(F[:, 1] / ref[:, 1]))))


def cmd_run(singers: list[str]) -> None:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import analyse, load_teacher
    from vocal_analysis.resynth.inverse import iaif
    from vocal_analysis.resynth.render import refilter, source_qc

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    models = {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}
    CACHE.mkdir(parents=True, exist_ok=True)
    for singer in singers:
        out = CACHE / f"{singer}.json"
        if out.exists():
            continue
        rows = []
        for j, f in enumerate(_files(singer)):
            y, sr = sf.read(ROOT / "data/vocalset" / f["meta"]["path"], dtype="float64", always_2d=True)
            g = gcd(sr, FS)
            y = resample_poly(y.mean(1), FS // g, sr // g)
            n = len(f["v"])
            idx = f["idx"]
            F, B = _tract(y, n, f["v"])
            src, _ = iaif(y, FS)
            yr = refilter(src, F, B)
            ys = _synth_source(len(y), f["f0"], f["v"], F, B, 1540 + j)
            tl = source_qc(src, f["f0"], f["v"], F)["tract_left"]
            row = {"path": f["meta"]["path"], "vowel": f["meta"]["vowel"], "n": int(len(idx)),
                   "tract_left": float(np.nanmedian(tl[:n][f["v"][:len(tl)]])) if f["v"].any() else float("nan")}
            for m in MODELS:
                row[f"O|{m}"] = _share(f["O"][m][idx], f["P"][idx])
                row[f"Os|{m}"] = _share(f["O"][m][idx], F[idx])
            for c, a in (("R", yr), ("S", ys)):
                a = (0.5 * a / (np.abs(a).max() + 1e-12)).astype(np.float32)
                for m, md in models.items():
                    r = analyse(md, a, dev)["formants"]
                    row[f"{c}|{m}"] = _share(r[idx, :3], F[idx])
            rows.append(row)
        out.write_text(json.dumps(rows), encoding="utf-8")
        print(singer, len(rows), flush=True)


def cmd_report() -> None:
    rows = [r for p in sorted(CACHE.glob("*.json")) for r in json.loads(p.read_text(encoding="utf-8"))]
    sm = lambda r, c: float(np.mean([r[f"{c}|{m}"] for m in MODELS]))  # noqa: E731
    for r in rows:
        o = sm(r, "O")
        r["cls"] = "failing" if o >= 0.5 else "ok" if o < 0.2 else "between"
    lines = ["D-154: the VocalSet failure decomposed (D-152's primary files, D-153's fixed frames; 5b seed mean)",
             "  O original vs Praat; Os original vs the smoothed tract; R the IAIF source through our resonators at that tract; "
             "S the synthesiser's source through the same tract", ""]
    res = {}
    for cls in ("failing", "ok", "between"):
        R = [r for r in rows if r["cls"] == cls]
        if not R:
            continue
        row = {c: float(np.mean([sm(r, c) for r in R])) for c in ("O", "Os", "R", "S")}
        row.update({f"{c}_fail_files": int(sum(sm(r, c) >= 0.5 for r in R)) for c in ("O", "R", "S")})
        row["files"] = len(R)
        row["tract_left"] = float(np.nanmedian([r["tract_left"] for r in R]))
        res[cls] = row
        lines.append(f"  {cls:8s} {len(R):3d} files: mean failure O {row['O']:.3f} (Os {row['Os']:.3f}), R {row['R']:.3f}, "
                     f"S {row['S']:.3f}; files failing O {row['O_fail_files']}, R {row['R_fail_files']}, S {row['S_fail_files']}; "
                     f"IAIF tract left median {row['tract_left']:.2f}")
        for m in MODELS:
            lines.append(f"      {m}: O {np.mean([r[f'O|{m}'] for r in R]):.3f}, R {np.mean([r[f'R|{m}'] for r in R]):.3f}, "
                         f"S {np.mean([r[f'S|{m}'] for r in R]):.3f}")
    F = [r for r in rows if r["cls"] == "failing"]
    lines.append("  failing files by vowel (mean R / S): " + ", ".join(
        f"{v} {len(G)} files {np.mean([sm(r, 'R') for r in G]):.2f} / {np.mean([sm(r, 'S') for r in G]):.2f}"
        for v in sorted({r['vowel'] for r in F}) for G in [[r for r in F if r["vowel"] == v]]))
    ok, fl = res.get("ok"), res.get("failing")
    if ok["R"] >= 0.2 or ok["S"] >= 0.2:
        out = "RESYNTHESIS ARTEFACT: the conversion itself creates failures on OK files"
    elif fl["R"] < 0.2:
        out = "REAL TRACT: the failure goes once our resonators replace the real tract's response"
    elif fl["R"] >= 0.5 and fl["S"] < 0.2:
        out = "REAL SOURCE: the failure survives our tract with the real source, and goes with ours"
    elif fl["R"] >= 0.5 and fl["S"] >= 0.5:
        out = "FORMANT PATTERN: our source and tract at the recording's tract values reproduce it"
    else:
        out = "PARTLY: reported, Hannah decides"
    res["reading"] = out
    lines += ["", f"Reading (registered): {out}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d154.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d154.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2].split(",") if len(sys.argv) > 2 else ["male10", "male3", "male5"])
    else:
        cmd_report()
