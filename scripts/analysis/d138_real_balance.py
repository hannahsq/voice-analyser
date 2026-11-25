"""D-138: is a real voice's formant level balance outside training, and does the dropped formant (F3 read as F2) come
with the voice source or the spectral envelope? Inference only on a recording session outside the repo; its per-take
numbers go to OUT (outside the repo, e.g. runs/d138_private/). The training reference is committed.

    python scripts/analysis/d138_real_balance.py trainref          -> reports/analysis/d138/d138_train.{txt,npz}
    python scripts/analysis/d138_real_balance.py levels REC OUT FILES  -> OUT/levels.json   (part 1, all usable sustained takes)
    python scripts/analysis/d138_real_balance.py swap REC OUT FILES    -> OUT/swap.json     (part 2)
    python scripts/analysis/d138_real_balance.py report OUT            -> OUT/d138.txt
REC holds the session's manifest.csv and recordings; FILES is a comma-separated list of the recordings to use.
"""

from __future__ import annotations

import csv
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reports/analysis/d138"
FS = 16000
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")
NFFT, WIN = 2048, 1024  # 64 ms frames
BANDS = {"low": (90.0, 140.0), "mid": (180.0, 270.0)}
SYN_BW = (60.0, 90.0, 120.0, 150.0)
FAIL, OK = 0.5, 0.2
B = 2000


def levels(frame: np.ndarray, F) -> np.ndarray:
    """Largest spectral magnitude (dB) within 1/6 octave of each formant."""
    s = 20 * np.log10(np.abs(np.fft.rfft(frame * np.hanning(len(frame)), NFFT)) + 1e-9)
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    return np.array([s[(f >= x * 2 ** (-1 / 6)) & (f <= x * 2 ** (1 / 6))].max() for x in F])


def cmd_trainref() -> None:
    d = ROOT / "data/synth/train_v2/flat"
    audio = np.load(d / "audio.npy", mmap_mode="r")
    ao, fo = np.load(d / "audio_offsets.npy"), np.load(d / "frame_offsets.npy")
    lab = {k: np.load(d / f"label_{k}.npy", mmap_mode="r") for k in ("formants", "f0", "voiced", "sonorant")}
    rng = np.random.default_rng(138)
    rows = []  # f0, F1, F2, F3, L1-L2, L2-L3
    for c in rng.choice(len(ao) - 1, 3000, replace=False):
        x = np.asarray(audio[ao[c]:ao[c + 1]], np.float64)
        fr = np.arange(fo[c], fo[c + 1])
        f0 = np.asarray(lab["f0"][fr])
        ok = (np.asarray(lab["voiced"][fr]) > 0.5) & (np.asarray(lab["sonorant"][fr]) > 0.5) & np.isfinite(f0)
        ok &= np.any([(f0 >= lo) & (f0 <= hi) for lo, hi in BANDS.values()], 0)
        idx = np.flatnonzero(ok)
        idx = idx[(idx * 160 - WIN // 2 >= 0) & (idx * 160 + WIN // 2 <= len(x))]
        for i in rng.permutation(idx)[:20]:
            F = np.asarray(lab["formants"][fo[c] + i, :3])
            if not np.all(np.isfinite(F)) or F[2] > 7000:
                continue
            L = levels(x[i * 160 - WIN // 2:i * 160 + WIN // 2], F)
            rows.append([f0[i], *F, L[0] - L[1], L[1] - L[2]])
    R = np.array(rows)
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "d138_train.npz", rows=R, cols=np.array(["f0", "F1", "F2", "F3", "L1-L2", "L2-L3"]))
    lines = ["D-138 training reference: train_v2, 3000 random clips (seed 138), up to 20 voiced sonorant frames each; level "
             "differences (dB) at the label formants, largest magnitude within 1/6 octave of each (64 ms Hann)", ""]
    for b, (lo, hi) in BANDS.items():
        inb = (R[:, 0] >= lo) & (R[:, 0] <= hi)
        cell = inb & (R[:, 1] >= 350) & (R[:, 1] <= 600) & (R[:, 2] / R[:, 1] < 2.2)
        for name, m in (("all frames", inb), ("cell F1 350-600, F2/F1 < 2.2", cell)):
            q = lambda k: " / ".join(f"{v:+.1f}" for v in np.percentile(R[m, k], [2.5, 25, 50, 75, 97.5]))  # noqa: E731
            lines.append(f"f0 {lo:g}-{hi:g} Hz, {name} (n = {m.sum()}): L1-L2 {q(4)}; L2-L3 {q(5)}  (2.5/25/50/75/97.5th)")
    (REF / "d138_train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _load(rec: Path, files: list[str]):
    import soundfile as sf
    from scipy.signal import resample_poly

    man = [r for r in csv.DictReader(open(rec / "manifest.csv", encoding="utf-8-sig"))
           if r["file"] in files and r["kind"] == "sustained vowel" and r["usable"] == "yes"]
    audio = {}
    for f in files:
        y, sr = sf.read(rec / f, dtype="float64", always_2d=True)
        g = gcd(sr, FS)
        audio[f] = resample_poly(y.mean(1), FS // g, sr // g)
    takes = []
    for r in man:
        y = audio[r["file"]][int(float(r["start_s"]) * FS):int(float(r["end_s"]) * FS)]
        takes.append((f"{files.index(r['file']) + 1}_{r['take_list_no']}", r, y / np.abs(y).max() * 0.5))
    return takes


def _models():
    import torch

    from vocal_analysis.models.infer import load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    return dev, {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in TEACHERS + STUDENTS}


def _mid(n):
    return slice(int(0.2 * n), int(0.8 * n))


def _praat(y):
    import parselmouth

    snd = parselmouth.Sound(y, FS)
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)
    n = int(len(y) / 160)
    ts = np.arange(n)[_mid(n)] * 0.01
    F = np.array([[fm.get_value_at_time(k, t) for k in (1, 2, 3, 4)] for t in ts])
    Bw = np.array([[fm.get_bandwidth_at_time(k, t) for k in (1, 2, 3, 4)] for t in ts])
    return np.nanmedian(F, 0), np.nanmedian(Bw, 0)


def _read(models, dev, y, F):
    """Per model: failure share, median F1 and F2 over voiced frames of the middle 60 %; and the teacher s1 f0 track."""
    from vocal_analysis.models.infer import analyse

    out, f0 = {}, None
    for m, mod in models.items():
        r = analyse(mod, y.astype(np.float32), dev)
        s = _mid(len(r["voiced_p"]))
        v = r["voiced_p"][s] > 0.5
        f2 = r["formants"][s][v]
        drop = np.abs(np.log(f2[:, 1] / F[2])) < np.abs(np.log(f2[:, 1] / F[1]))
        out[m] = {"fail": float(drop.mean()) if len(drop) else float("nan"), "F1": float(np.median(f2[:, 0])) if len(f2) else None,
                  "F2": float(np.median(f2[:, 1])) if len(f2) else None, "n": int(v.sum())}
        if m == TEACHERS[0]:
            f0 = (r["f0"], r["voiced_p"])
    return out, f0


def cmd_levels(rec: Path, out: Path, files: list[str]) -> None:
    dev, models = _models()
    res = []
    for name, r, y in _load(rec, files):
        F, Bw = _praat(y)
        if not np.all(np.isfinite(F[:3])):
            continue
        rd, _ = _read(models, dev, y, F)
        n = int(len(y) / 160)
        L = np.array([levels(y[i * 160 - WIN // 2:i * 160 + WIN // 2], F[:3]) for i in range(n)[_mid(n)]
                      if i * 160 - WIN // 2 >= 0 and i * 160 + WIN // 2 <= len(y)])
        res.append({"take": name, "vowel": r["item"], "f0": float(r["f0_median_hz"]), "F": F.tolist(), "Bw": Bw.tolist(),
                    "L12": float(np.median(L[:, 0] - L[:, 1])), "L23": float(np.median(L[:, 1] - L[:, 2])), "read": rd})
        print(name, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "levels.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def _tfail(t):
    return float(np.mean([t["read"][m]["fail"] for m in TEACHERS]))


def cmd_swap(rec: Path, out: Path, files: list[str]) -> None:
    from scipy.linalg import solve_toeplitz
    from scipy.signal import lfilter

    from vocal_analysis.synth import tract, voice

    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    fail = {k for k, t in lv.items() if _tfail(t) >= FAIL}
    vowels = {lv[k]["vowel"] for k in fail}
    ctrl = {k for k, t in lv.items() if t["vowel"] in vowels and _tfail(t) < OK}
    dev, models = _models()
    res = []
    for name, r, y in _load(rec, files):
        if name not in fail | ctrl:
            continue
        t = lv[name]
        F = np.array(t["F"])
        # LPC, one per take, from the summed autocorrelation of 32 ms frames of the pre-emphasised middle
        n = len(y)
        z = lfilter([1, -0.97], [1], y)[int(0.2 * n):int(0.8 * n)]
        ac = np.zeros(19)
        for i in range(0, len(z) - 512, 160):
            w = z[i:i + 512] * np.hanning(512)
            ac += np.array([w[:512 - k] @ w[k:] for k in range(19)])
        a = np.r_[1.0, -solve_toeplitz(ac[:18], ac[1:19])]
        resid = lfilter(a, [1], y)
        # the cascade: control-rate trajectories over the take
        C = n // voice.CTRL + 1
        tc = np.arange(C) * voice.CTRL / FS
        _, (f0t, vp) = _read({TEACHERS[0]: models[TEACHERS[0]]}, dev, y, F)
        tf = np.arange(len(f0t)) * 0.01
        ok = vp > 0.5
        f0c = np.interp(tc, tf[ok], f0t[ok])
        amp = np.clip(np.minimum(tc, tc[-1] - tc) / 0.02, 0, 1)
        pf = np.full((C, tract.N_POLES), np.nan)
        pb = np.full((C, tract.N_POLES), np.nan)

        def traj(with_tract):
            if with_tract:
                pf[:, :4], pb[:, :4] = F[:4], SYN_BW
            return voice.Trajectories(f0=f0c, rd=np.full(C, 1.0), voice_amp=amp, whisper_amp=np.zeros(C),
                                      fric_amp=np.zeros(C), poles_f=pf.copy(), poles_b=pb.copy(),
                                      zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))

        src = voice.synthesize(traj(False), np.random.default_rng(138), jitter=0.01, shimmer_db=0.5)["total"]
        m = len(src)
        arms = {"O": y[:m],
                "A": voice.tract_filter(np.r_[resid, np.zeros(max(0, m - n))][:m], traj(True)),
                "B": lfilter([1], a, src),
                "C": voice.synthesize(traj(True), np.random.default_rng(138), jitter=0.01, shimmer_db=0.5)["total"]}
        row = {"take": name, "vowel": t["vowel"], "group": "fail" if name in fail else "control"}
        for k, x in arms.items():
            x = 0.5 * x / (np.abs(x).max() + 1e-12)
            row[k], _ = _read(models, dev, x, F)
        res.append(row)
        print(name, flush=True)
    (out / "swap.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_report(out: Path) -> None:
    from scipy.stats import spearmanr

    R = np.load(REF / "d138_train.npz")["rows"]
    lv = json.loads((out / "levels.json").read_text(encoding="utf-8"))
    lines = ["D-138 (private: per-take numbers; not for the repo)", "", "Part 1: takes, teacher / student failure share, "
             "F2/F1, L1-L2, L2-L3 (dB), percentile of each in train_v2's cell at the take's f0 band"]
    for t in sorted(lv, key=lambda t: -_tfail(t)):
        b = "low" if t["f0"] < 160 else "mid"
        lo, hi = BANDS[b]
        cell = (R[:, 0] >= lo) & (R[:, 0] <= hi) & (R[:, 1] >= 350) & (R[:, 1] <= 600) & (R[:, 2] / R[:, 1] < 2.2)
        p = [100 * np.mean(R[cell, 4] < t["L12"]), 100 * np.mean(R[cell, 5] < t["L23"])]
        st = np.nanmean([t["read"][m]["fail"] for m in STUDENTS])
        F = t["F"]
        lines.append(f"  {t['take']:6s} {t['vowel']:4s} f0 {t['f0']:5.0f}  fail {_tfail(t):4.0%} / {st:4.0%}  F {F[0]:4.0f} {F[1]:4.0f} "
                     f"{F[2]:4.0f} (B {t['Bw'][0]:3.0f} {t['Bw'][1]:3.0f} {t['Bw'][2]:3.0f})  F2/F1 {F[1] / F[0]:4.2f}  "
                     f"L1-L2 {t['L12']:+5.1f} (p{p[0]:3.0f})  L2-L3 {t['L23']:+5.1f} (p{p[1]:3.0f})")
    fs = np.array([_tfail(t) for t in lv])
    for k in ("L12", "L23"):
        rho, pv = spearmanr(fs, [t[k] for t in lv])
        lines.append(f"  Spearman failure share vs {k}: {rho:+.2f} (p {pv:.3f}, n {len(lv)})")
    rho, pv = spearmanr(fs, [t["F"][1] / t["F"][0] for t in lv])
    lines.append(f"  Spearman failure share vs F2/F1: {rho:+.2f} (p {pv:.3f})")
    sw = out / "swap.json"
    if sw.exists():
        rows = json.loads(sw.read_text(encoding="utf-8"))
        rng = np.random.default_rng(1380)
        lines += ["", "Part 2: mean failure share per arm (teacher seed mean; students mean) [95 % bootstrap over takes]"]
        for g in ("fail", "control"):
            rs = [r for r in rows if r["group"] == g]
            lines.append(f"  {g} takes (n = {len(rs)}):")
            for arm in "OABC":
                for who, names in (("teacher", TEACHERS), ("students", STUDENTS)):
                    v = np.array([np.nanmean([r[arm][m]["fail"] for m in names]) for r in rs])
                    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
                    lines.append(f"    {arm} {who:8s} {v.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"
                                 + ("  per take " + " ".join(f"{x:.2f}" for x in v) if who == "teacher" else ""))
    (out / "d138.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "trainref":
        cmd_trainref()
    elif cmd == "levels":
        cmd_levels(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    elif cmd == "swap":
        cmd_swap(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    elif cmd == "report":
        cmd_report(Path(sys.argv[2]))
