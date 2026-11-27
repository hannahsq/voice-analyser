"""D-156: do training's B3 and B4 (and B1, B2) reach as wide as real voices', measured by the same instrument (Praat), and
does 5b care (D-155's grid with B3 or B4 stepped)? CPU.

    python scripts/analysis/d156_upper_bw.py praat [SET,...]   -> cache/d156/<set>.npz (ptdb, vocalset, hillenbrand, train)
    python scripts/analysis/d156_upper_bw.py grid              -> cache/d156/grid.json
    python scripts/analysis/d156_upper_bw.py report            -> reports/analysis/d156/d156.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d156"
CACHE = ROOT / "cache/d156"
FS = 16000
REAL = ("ptdb", "vocalset", "hillenbrand")
BANDS = {1: ((300, 450), (450, 650), (650, 900)), 2: ((700, 1100), (1100, 1700), (1700, 2500)),
         3: ((2000, 2500), (2500, 3000), (3000, 3500)), 4: ((3000, 3500), (3500, 4000), (4000, 4500))}
FIVEB = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STEPS = {3: (120.0, 250.0, 400.0, 600.0), 4: (150.0, 300.0, 500.0, 800.0)}
GRID_B1 = (150.0, 360.0)
SEEDS = (20, 21, 22)


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _praat(y, ceiling):
    """Praat Burg F1-F4, B1-B4 and Praat's f0 every 10 ms; voiced frames (Praat's pitch) only."""
    import parselmouth

    snd = parselmouth.Sound(np.asarray(y, np.float64), FS)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling)
    pitch = snd.to_pitch(time_step=0.01)
    ts = np.arange(len(y) // 160) * 0.01
    f0 = np.array([pitch.get_value_at_time(t) for t in ts], float)
    v = np.isfinite(f0)
    ts, f0 = ts[v], f0[v]
    F = np.array([[fm.get_value_at_time(k, t) for k in range(1, 5)] for t in ts], float).reshape(-1, 4)
    B = np.array([[fm.get_bandwidth_at_time(k, t) for k in range(1, 5)] for t in ts], float).reshape(-1, 4)
    return F, B, f0


def _resample(y, sr):
    from scipy.signal import resample_poly

    g = gcd(sr, FS)
    return resample_poly(y, FS // g, sr // g) if sr != FS else y


def _items(name):
    """(audio at 16 kHz, Praat ceiling) per item of a set."""
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry

    if name == "ptdb":
        mod = registry()["ptdb_tug"]
        for r in TI.filter_rows(read_manifest(str(ROOT / "data/manifests/ptdb_tug.csv")), ("dev",)):
            yield load_audio(mod.load(ROOT / "data/ptdb_tug", r)), 5000.0 if r.speaker_id.split(":")[-1][0] == "M" else 5500.0
    elif name == "hillenbrand":
        mod = registry()["hillenbrand"]
        for r in TI.filter_rows(read_manifest(str(ROOT / "data/manifests/hillenbrand.csv")), ("dev",)):
            it = mod.load(ROOT / "data/hillenbrand", r)
            yield load_audio(it), 5000.0 if it.meta["group"] == "m" else 5500.0
    elif name == "vocalset":
        import soundfile as sf

        for f in _mod("d152_vocalset_flag")._files():
            y, sr = sf.read(ROOT / "data/vocalset" / f["path"], dtype="float64", always_2d=True)
            yield _resample(y.mean(1), sr), 5000.0 if f["singer"].startswith("male") else 5500.0
    elif name == "train":
        d = ROOT / "data/synth/train_v2/flat"
        a = np.load(d / "audio.npy", mmap_mode="r")
        ao = np.load(d / "audio_offsets.npy")
        L = np.load(d / "clip_length_cm.npy")
        for j in np.unique(np.linspace(0, len(ao) - 2, 2000).round().astype(int)):
            yield np.asarray(a[ao[j]:ao[j + 1]], np.float64) / 32768.0, 5000.0 if L[j] >= 16 else 5500.0


def cmd_praat(sets: list[str]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in sets:
        out = CACHE / f"{name}.npz"
        if out.exists():
            continue
        acc = {"F": [], "B": [], "f0": []}
        for y, ceiling in _items(name):
            F, B, f0 = _praat(y, ceiling)
            acc["F"].append(F), acc["B"].append(B), acc["f0"].append(f0)
        np.savez_compressed(out, **{k: np.concatenate(v) for k, v in acc.items()})
        print(name, sum(len(x) for x in acc["f0"]), flush=True)


def _grid_keys():
    d142 = _mod("d142_flags")
    out = []
    for k, steps in STEPS.items():
        for b in steps:
            b3, b4 = (b, 150.0) if k == 3 else (120.0, b)
            out += [(k, b, s, f0, b1, sd, Fs, b3, b4) for s, Fs in d142.SETS.items() for f0 in d142.F0S for b1 in GRID_B1
                    for sd in SEEDS]
    return out


def _audio(Fs, f0, b1, b3, b4, seed):
    """D-155's clip (D-142's seed formula) with B2 = 90 and B3, B4 as given."""
    from vocal_analysis.synth import sampler, tract, voice

    C = int(3.0 * FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / FS
    pf = np.full((C, tract.N_POLES), np.nan)
    pb = pf.copy()
    pf[:, :4], pb[:, :4] = Fs, (b1, 90.0, b3, b4)
    rng = np.random.default_rng(int(1420 + 1000 * seed + f0 + b1 + Fs[0]))
    tr = voice.Trajectories(f0=f0 * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12), rd=np.full(C, 1.0),
                            voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1), whisper_amp=np.zeros(C),
                            fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
    return 0.5 * y / np.abs(y).max()


def cmd_grid() -> None:
    import torch

    from vocal_analysis.models.infer import analyse, load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    models = {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in FIVEB}
    res = []
    for (k, b, s, f0, b1, sd, Fs, b3, b4) in _grid_keys():
        y = _audio(Fs, f0, b1, b3, b4, sd).astype(np.float32)
        row = {"k": k, "b": b, "set": s, "f0": f0, "b1": b1, "seed": sd}
        for m, md in models.items():
            r = analyse(md, y, dev)
            n = len(r["voiced_p"])
            i = np.arange(int(0.2 * n), int(0.8 * n))
            i = i[(r["voiced_p"][i] > 0.5) & np.isfinite(r["f0"][i]) & (i * 160 >= 512) & (i * 160 + 512 <= len(y))]
            i = i[np.all(np.isfinite(r["formants"][i, :4]), 1)]
            Fm = r["formants"][i, :4]
            if not len(i):
                continue
            row[m] = {"fail": float(np.mean(np.abs(np.log(Fm[:, 1] / Fs[2])) < np.abs(np.log(Fm[:, 1] / Fs[1])))),
                      "e3": float(np.median(np.abs(np.log2(Fm[:, 2] / Fs[2])) * 1200)),
                      "e4": float(np.median(np.abs(np.log2(Fm[:, 3] / Fs[3])) * 1200))}
        res.append(row)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "grid.json").write_text(json.dumps(res), encoding="utf-8")
    print("grid", len(res), flush=True)


def _labels():
    d = ROOT / "data/synth/train_v2/flat"
    n = len(np.load(d / "label_sonorant.npy", mmap_mode="r"))
    idx = np.arange(0, n, 10)
    son = np.asarray(np.load(d / "label_sonorant.npy", mmap_mode="r")[idx]) > 0.5
    voi = np.asarray(np.load(d / "label_voiced.npy", mmap_mode="r")[idx]) > 0.5
    F = np.asarray(np.load(d / "label_formants.npy", mmap_mode="r")[idx, :4])
    B = np.asarray(np.load(d / "label_bandwidths.npy", mmap_mode="r")[idx, :4])
    f0 = np.asarray(np.load(d / "label_f0.npy", mmap_mode="r")[idx])
    s = son & voi
    return F[s], B[s], f0[s]


def _pct(B, F, k, band, f0, low):
    lo, hi = band
    with np.errstate(invalid="ignore"):
        s = (F[:, k - 1] >= lo) & (F[:, k - 1] < hi) & np.isfinite(B[:, k - 1])
        if low:
            s &= f0 <= 150
    b = B[s, k - 1]
    return (int(s.sum()), *(np.percentile(b, [50, 90, 99]).tolist() if len(b) else [np.nan] * 3))


def cmd_report() -> None:
    P = {n: np.load(CACHE / f"{n}.npz") for n in REAL + ("train",)}
    LF, LB, Lf0 = _labels()
    lines = ["D-156: bandwidth coverage by one instrument (Praat Burg on Praat-voiced frames), real voices against train_v2",
             "  per formant and band of Praat's Fk: n frames, Praat Bk 50th / 90th / 99th percentile (Hz); train_v2 labels for context",
             "  GAP: real 90th above Praat-on-train 99th; COVERED: real 90th at or below Praat-on-train 90th; THIN between", ""]
    res = {"coverage": {}, "per_formant": {}, "grid": {}}
    for low in (False, True):
        lines.append(f"  {'f0 <= 150 Hz' if low else 'all voiced frames'}")
        for k, bands in BANDS.items():
            for band in bands:
                tr = _pct(P["train"]["B"], P["train"]["F"], k, band, P["train"]["f0"], low)
                lab = _pct(LB, LF, k, band, Lf0, low)
                cells = [f"train {tr[0]:6d} {tr[1]:4.0f}/{tr[2]:4.0f}/{tr[3]:5.0f} (labels {lab[2]:4.0f}/{lab[3]:4.0f})"]
                for n in REAL:
                    r = _pct(P[n]["B"], P[n]["F"], k, band, P[n]["f0"], low)
                    tot = int(np.isfinite(P[n]["F"][:, k - 1]).sum() if not low else
                              (np.isfinite(P[n]["F"][:, k - 1]) & (P[n]["f0"] <= 150)).sum())
                    v = ("few" if r[0] < 500 else "GAP" if r[2] > tr[3] else "COVERED" if r[2] <= tr[2] else "THIN")
                    res["coverage"][f"{'low' if low else 'all'}|B{k}|{band[0]}-{band[1]}|{n}"] = {
                        "n": r[0], "share": r[0] / max(tot, 1), "p": r[1:], "train": tr, "labels": lab, "verdict": v}
                    cells.append(f"{n} {r[0]:6d} {r[1]:4.0f}/{r[2]:4.0f}/{r[3]:5.0f} {v}")
                lines.append(f"    B{k} {band[0]:4d}-{band[1]:4d}: " + "; ".join(cells))
        lines.append("")
    # per formant verdicts (all voiced frames; f0 <= 150 reported in the json)
    lines.append("  per formant (bands with >= 10 % of a set's frames and >= 500 frames):")
    for low in (False, True):
        for k in BANDS:
            per = {}
            for n in REAL:
                vs = [c["verdict"] for key, c in res["coverage"].items()
                      if key.startswith(f"{'low' if low else 'all'}|B{k}|") and key.endswith(f"|{n}") and c["share"] >= 0.10 and c["n"] >= 500]
                per[n] = "GAP" if "GAP" in vs else "COVERED" if vs and all(v == "COVERED" for v in vs) else "THIN" if vs else "few"
            res["per_formant"][f"{'low' if low else 'all'}|B{k}"] = per
            lines.append(f"    {'f0<=150' if low else 'all':7s} B{k}: " + ", ".join(f"{n} {v}" for n, v in per.items()))
    # part B
    G = json.loads((CACHE / "grid.json").read_text(encoding="utf-8"))
    lines += ["", "  part B: D-155's grid (B2 90) with B3 or B4 stepped; 5b seed mean, all f0, B1 150 and 360"]
    for k, steps in STEPS.items():
        st = {}
        for b in steps:
            rows = [g for g in G if g["k"] == k and g["b"] == b]
            st[b] = {q: float(np.mean([np.mean([g[m][q] for g in rows if m in g]) for m in FIVEB])) for q in ("fail", "e3", "e4")}
            lo = [g for g in rows if g["f0"] <= 140]
            st[b]["fail_low"] = float(np.mean([np.mean([g[m]["fail"] for g in lo if m in g]) for m in FIVEB]))
        e = f"e{k}"
        df, de = st[steps[-1]]["fail"] - st[steps[0]]["fail"], st[steps[-1]][e] - st[steps[0]][e]
        ratio = st[steps[-1]][e] / max(st[steps[0]][e], 1e-9)
        v = ("MATTERS" if df >= 0.2 or (de >= 100 and ratio >= 2) else "NO EFFECT" if df < 0.05 and de < 25 else "PARTLY")
        res["grid"][f"B{k}"] = {"steps": st, "d_fail": df, "d_err": de, "verdict": v}
        lines.append(f"    B{k}: " + "  ".join(f"{b:.0f}: fail {s['fail']:.2f} (f0<=140 {s['fail_low']:.2f}), F3 err {s['e3']:.0f} c, "
                                             f"F4 err {s['e4']:.0f} c" for b, s in st.items()))
        lines.append(f"        widest minus narrowest: failure {df:+.2f}, F{k} error {de:+.0f} c (x{ratio:.1f}) -> {v}")
    lines += ["", "Reading (registered): a formant needs a training change only if it is a gap or thin in part A and matters in part B"]
    for k in (3, 4):
        a = res["per_formant"][f"all|B{k}"]
        lines.append(f"  B{k}: part A " + ", ".join(f"{n} {v}" for n, v in a.items()) + f"; part B {res['grid'][f'B{k}']['verdict']}")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d156.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d156.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "praat":
        cmd_praat(sys.argv[2].split(",") if len(sys.argv) > 2 else list(REAL) + ["train"])
    else:
        {"grid": cmd_grid, "report": cmd_report}[cmd]()
