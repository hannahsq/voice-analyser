"""D-019 EGG test on CMU ARCTIC bdl / jmk / slt: closed-phase covariance LPC (reference), whole-cycle covariance LPC
(same estimator, open phases included), Praat Burg (sex-based ceiling, the LPC-reference analogue) and teachers
v1 / d06k / d16k, at the monophthong frames of every file. Saves per-frame arrays to <scratch>/arctic_egg.npz.
Run from the voice-analyser-m3 worktree. Args: scratch dir."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.eval import closed_phase as cp
from vocal_analysis.eval.datasets import cmu_arctic as ca
from vocal_analysis.eval.teacher_eval import teacher_baselines

S = Path(sys.argv[1])
ROOT = Path("../voice-analyser/data/cmu_arctic")
R = Path("../voice-analyser/runs")
CKPTS = {"v1": R / "teacher_v1/ckpt_020000.pt", "d06k": R / "teacher_v2_d06k/ckpt_010000.pt",
         "d16k": R / "teacher_v2_d16k/ckpt_020000.pt"}
FS_CP, ORDER = 10000, 12


def praat(x16, times, sex):
    import parselmouth

    f = parselmouth.Sound(x16.astype(np.float64), 16000).to_formant_burg(maximum_formant=5000.0 if sex == "m" else 5500.0)
    out = np.full((len(times), 3), np.nan)
    for i, t in enumerate(times):
        for k in range(3):
            v = f.get_value_at_time(k + 1, float(t))
            out[i, k] = v if v == v else np.nan
    return out


if __name__ == "__main__":
    teachers = {k: teacher_baselines(str(v), k)[0] for k, v in CKPTS.items()}
    cols = {k: [] for k in ("voice", "phone", "utt", "t", "f0", "cq", "ncyc", "cp", "cycle", "praat", *CKPTS)}
    rows = ca.scan(ROOT)
    lags = {}
    for i, r in enumerate(rows):
        it = ca.load(ROOT, r)
        if not len(it.times):
            continue
        x, sr = sf.read(it.audio_path)
        sp10, eg10 = resample_poly(x[:, 0], FS_CP, sr), resample_poly(x[:, 1], FS_CP, sr)
        res = cp.analyse(sp10, eg10, FS_CP, it.times, order=ORDER)
        lags.setdefault(it.meta["voice"], []).append(res["lag"])
        x16 = resample_poly(x[:, 0], 16000, sr).astype(np.float32)
        n = len(it.times)
        cols["voice"] += [it.meta["voice"]] * n
        cols["phone"] += it.meta["phones"]
        cols["utt"] += [Path(r.path).stem] * n
        cols["t"].append(it.times)
        for k in ("f0", "cq"):
            cols[k].append(res[k])
        cols["ncyc"].append(res["n_cycles"])
        cols["cp"].append(res["cp"][:, :3])
        cols["cycle"].append(res["cycle"][:, :3])
        cols["praat"].append(praat(x16, it.times, it.meta["sex"]))
        for k, b in teachers.items():
            cols[k].append(b.predict(x16, 16000, it.times, {})[:, :3])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    out = {k: (np.array(v) if isinstance(v[0], str) else np.concatenate(v)) for k, v in cols.items()}
    np.savez(S / "arctic_egg.npz", **out)
    for v, l in lags.items():
        print(f"{v}: speech lag median {np.median(l) / FS_CP * 1000:.2f} ms (IQR {np.percentile(l, 25) / FS_CP * 1000:.2f}-"
              f"{np.percentile(l, 75) / FS_CP * 1000:.2f})")
    print(f"saved {len(out['t'])} frames")
