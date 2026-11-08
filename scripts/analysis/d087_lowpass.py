"""D-087: does removing F4 (and above) from the input change real-speech F1? Existing models, no training.
PTDB-TUG closed-phase frames (cache/m3w/ptdb_egg_sub.npz, dev speakers only), each file low-passed at 16 kHz before
the model (8th-order Butterworth, sosfilt: the channel model's own filter, synth/channel.py) at each cutoff, F1-F3
read at the closed-phase frames. Same audio path, resampling and 400-frame windows as add_model.py, GPU fp32.
Writes reports/analysis/d087/lowpass_<model>.npz (frame index, F1-F3 per cutoff) and decide_d087.txt.
Args: scratch dir (cache/m3w), then model=checkpoint pairs. --decide-only re-reads the saved npz."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np

CUTS = [None, 7500.0, 4500.0, 3500.0]
O = Path("reports/analysis/d087")


def tag(c):
    return "none" if c is None else f"lp{int(c)}"


def frames(S):
    import json
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    return d, np.flatnonzero(dev)


def infer(S, name, ck):
    import soundfile as sf
    import torch
    from scipy.signal import butter, resample_poly, sosfilt
    from vocal_analysis.models.infer import DEFAULT_WINDOW, analyse, load_teacher

    out_path = O / f"lowpass_{name}.npz"
    if out_path.exists():
        print(f"{name}: already scored")
        return
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    model = load_teacher(ck, dev)[0]
    d, idx = frames(S)
    groups = {}
    for i in idx:
        groups.setdefault((d["speaker"][i], d["utt"][i]), []).append(i)
    res = {tag(c): np.full((len(idx), 3), np.nan) for c in CUTS}
    pos = {i: k for k, i in enumerate(idx)}
    root = Path("data/ptdb_tug")
    for n, ((spk, utt), ii) in enumerate(groups.items()):
        x, sr = sf.read(root / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{utt}.wav")
        x16 = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr)
        for c in CUTS:
            y = x16 if c is None else sosfilt(butter(8, c, "lowpass", fs=16000, output="sos"), x16)
            r = analyse(model, y.astype(np.float32), dev, window_frames=DEFAULT_WINDOW)
            j = np.clip(np.round(d["t"][ii] / 0.010).astype(int), 0, len(r["formants"]) - 1)
            res[tag(c)][[pos[i] for i in ii]] = r["formants"][j, :3]
        if (n + 1) % 100 == 0:
            print(f"  {name}: {n + 1}/{len(groups)} files", flush=True)
    O.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, idx=idx, **res)
    print(f"{name}: done", flush=True)


def decide(S, names):
    d, _ = frames(S)
    rng = np.random.default_rng(0)
    lines, verdict = [], {}
    for name in names:
        z = dict(np.load(O / f"lowpass_{name}.npz"))
        idx = z["idx"]
        stored = d[name][idx, 0]
        ok = np.isfinite(stored) & np.isfinite(z["none"][:, 0])
        rep = float(np.percentile(np.abs(stored[ok] - z["none"][ok, 0]), 99))
        lines.append(f"{name}: reproduction (none vs the cached column), 99th percentile |dF1| {rep:.2f} Hz")
        for sex in ("f", "m"):
            m = (d["sex"][idx] == sex) & np.isfinite(d["cp"][idx, 0])
            spk = d["speaker"][idx][m]
            us = np.unique(spk)
            cp = d["cp"][idx][m]
            for c in CUTS[1:]:
                for f in range(3):
                    a, b = z["none"][m, f], z[tag(c)][m, f]
                    k = np.isfinite(a) & np.isfinite(b) & np.isfinite(cp[:, f])

                    def stat(sel):
                        return (np.median(np.abs(b[sel] - cp[sel, f])) - np.median(np.abs(a[sel] - cp[sel, f])),
                                np.median(b[sel] - cp[sel, f]) - np.median(a[sel] - cp[sel, f]))
                    dabs, dbias = stat(k)
                    by = {s: np.flatnonzero(k & (spk == s)) for s in us}
                    boot = [stat(np.concatenate([by[s] for s in rng.choice(us, len(us))]))[0] for _ in range(1000)]
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    lines.append(f"  {sex} F{f + 1} {tag(c)}: |err| {dabs:+.1f} Hz [{lo:+.1f}, {hi:+.1f}], bias {dbias:+.1f} Hz "
                                 f"({len(us)} speakers, {k.sum()} frames)")
                    if f == 0:
                        verdict[(name, sex, tag(c))] = (dabs, lo, hi)
    print("\n".join(lines))

    def moves(sex, t):
        v = [verdict[(n, sex, t)] for n in names]
        return all(x[0] >= 2 and x[1] > 0 for x in v)
    ctrl = any(abs(verdict[(n, s, "lp7500")][0]) >= 2 for n in names for s in "fm")
    women = moves("f", "lp4500") or moves("f", "lp3500")
    print(f"control (7.5 kHz moves F1 |err| by >= 2 Hz anywhere): {ctrl}")
    print(f"women's F1 |err| rises >= 2 Hz (interval above 0) at 4.5 or 3.5 kHz on every seed: {women}")
    print(f"men's at 3.5 kHz: {moves('m', 'lp3500')}")
    if ctrl:
        print("DECISION: inconclusive (the 7.5 kHz control moves F1: the filter itself matters, not F4)")
    elif women:
        print("DECISION: F4 visibility matters for women's real-speech F1")
    else:
        print("DECISION: F4 visibility does not drive women's real-speech F1")


def main():
    args = [a for a in sys.argv[1:] if a != "--decide-only"]
    S, models = Path(args[0]), dict(a.split("=", 1) for a in args[1:])
    if "--decide-only" not in sys.argv:
        for name, ck in models.items():
            infer(S, name, ck)
    decide(S, list(models))


if __name__ == "__main__":
    main()
