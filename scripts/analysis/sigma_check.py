"""Does the full-distribution sigma (decode, 2026-10-09) widen where evidence is missing, and is it calibrated?
1. Synthetic suite (exact labels): median sigma per case and formant; coverage |err| < 1 / 2 sigma; error by sigma
   quintile. 2. PTDB-TUG closed-phase subset (fusion_test frames): coverage against closed-phase F1-F3.
Args: scratch dir, then name=ckpt pairs. Run from the voice-analyser-m3 worktree (uses its decode)."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.device import setup_device
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth.shards import ShardReader

S = Path(sys.argv[1])
CK = dict(a.split("=", 1) for a in sys.argv[2:])
SUITE = Path("../voice-analyser/data/synth/suite_v1")
dev = setup_device()


def cents(a, b):
    return np.abs(1200 * np.log2(a / b))


def summary(name, err_c, sig):
    ok = np.isfinite(err_c) & np.isfinite(sig)
    e, s = err_c[ok], sig[ok]
    q = np.percentile(s, [20, 40, 60, 80])
    bins = np.digitize(s, q)
    by_q = " ".join(f"{np.median(e[bins == k]):4.0f}" for k in range(5))
    return (f"{name:14s} sigma med {np.median(s):5.0f}  p90 {np.percentile(s, 90):5.0f}   within 1/2 sigma "
            f"{100 * np.mean(e < s):4.1f} / {100 * np.mean(e < 2 * s):4.1f} %   median |err| by sigma quintile: {by_q}")


for mname, ck in CK.items():
    model = load_teacher(ck, dev)[0]
    print(f"\n===== {mname}: synthetic suite, F1-F3 on sonorant frames (cents; Gaussian: 68 / 95 %)")
    for case in sorted(p for p in SUITE.iterdir() if (p / "index.json").exists()):
        r = ShardReader(case)
        E, G = [], []
        for i in range(len(r)):
            audio, lab, _ = r[i]
            res = analyse(model, audio, dev)
            n = min(len(res["t"]), len(lab["t"]))
            son = lab["sonorant"][:n]
            E.append(cents(res["formants"][:n, :3][son], lab["formants"][:n, :3][son]))
            G.append(res["formant_sigma_cents"][:n, :3][son])
        E, G = np.concatenate(E), np.concatenate(G)
        print(f"  {case.name}")
        for k in range(3):
            print("    " + summary(f"F{k + 1}", E[:, k], G[:, k]))

    print(f"\n===== {mname}: PTDB-TUG closed-phase frames (fusion_test subset), F1-F3")
    d = dict(np.load(S / "ptdb_egg.npz"))
    ok = np.isfinite(d["cp"][:, :3]).all(1)
    rng = np.random.default_rng(1)
    E, G, SX = [], [], []
    for spk in np.unique(d["speaker"]):
        for utt in rng.choice(np.unique(d["utt"][ok & (d["speaker"] == spk)]), 10, replace=False):
            m = ok & (d["utt"] == utt)
            x, sr = sf.read(Path("../voice-analyser/data/ptdb_tug") / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{utt}.wav")
            res = analyse(model, resample_poly(x, 16000, sr).astype(np.float32), dev)
            j = np.clip(np.round(d["t"][m] / 0.01).astype(int), 0, len(res["t"]) - 1)
            E.append(cents(res["formants"][j, :3], d["cp"][m, :3]))
            G.append(res["formant_sigma_cents"][j, :3])
            SX += [spk[0]] * m.sum()
    E, G, SX = np.concatenate(E), np.concatenate(G), np.array(SX)
    for sx in "MF":
        for k in range(3):
            print("    " + summary(f"{sx} F{k + 1}", E[SX == sx, k], G[SX == sx, k]))
