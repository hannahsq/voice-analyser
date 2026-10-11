"""D-019 closed-phase test on PTDB-TUG's laryngograph channel (20 speakers, 10 women): closed-phase covariance LPC,
whole-cycle LPC, Praat Burg (sex-based ceiling) and teachers v1 / d06k / d16k / sf20k at vowel-like frames.

Frames (no phone labels in PTDB-TUG): voiced runs of the reference f0 lasting >= 80 ms; in each, the frame of
highest speech energy and the frames 20 ms either side, kept if within 15 dB of the utterance's loudest frame.
Every second utterance. Saves <scratch>/ptdb_egg.npz; progress every 50 files. Args: scratch dir.
Run from the voice-analyser-m3 worktree."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.eval import closed_phase as cp
from vocal_analysis.eval.datasets import ptdb_tug as pt
from vocal_analysis.eval.teacher_eval import teacher_baselines

S = Path(sys.argv[1])
ROOT = Path("../voice-analyser/data/ptdb_tug")
R = Path("../voice-analyser/runs")
CKPTS = {"v1": R / "teacher_v1/ckpt_020000.pt", "d06k": R / "teacher_v2_d06k/ckpt_010000.pt",
         "d16k": R / "teacher_v2_d16k/ckpt_020000.pt", "sf20k": R / "teacher_v2_sf/ckpt_020000.pt"}
FS_CP, ORDER = 10000, 12


def frames(f0_ref, x16):
    """Vowel-like frame times (s) from the reference f0 (frame i at 20 ms + 10 ms * i) and speech energy."""
    v = f0_ref > 0
    t = pt.FRAME_OFFSET_S + pt.HOP_S * np.arange(len(f0_ref))
    hop = 160
    e = np.array([10 * np.log10(np.mean(x16[max(0, int(tt * 16000) - hop): int(tt * 16000) + hop] ** 2) + 1e-12) for tt in t])
    out, i = [], 0
    while i < len(v):
        if not v[i]:
            i += 1
            continue
        j = i
        while j < len(v) and v[j]:
            j += 1
        if (j - i) * pt.HOP_S >= 0.08:
            k = i + int(np.argmax(e[i:j]))
            for kk in (k - 2, k, k + 2):
                if i <= kk < j and e[kk] >= e.max() - 15:
                    out.append(t[kk])
        i = j
    return np.array(sorted(set(out)))


def praat(x16, times, sex):
    import parselmouth

    f = parselmouth.Sound(x16.astype(np.float64), 16000).to_formant_burg(maximum_formant=5000.0 if sex == "m" else 5500.0)
    out = np.full((len(times), 3), np.nan)
    for i, t in enumerate(times):
        for k in range(3):
            val = f.get_value_at_time(k + 1, float(t))
            out[i, k] = val if val == val else np.nan
    return out


if __name__ == "__main__":
    teachers = {k: teacher_baselines(str(v), k)[0] for k, v in CKPTS.items()}
    cols = {k: [] for k in ("speaker", "sex", "utt", "t", "f0", "cq", "ncyc", "cp", "cycle", "praat", *CKPTS)}
    rows = pt.scan(ROOT)[::2]
    lags = {}
    for i, r in enumerate(rows):
        mic = ROOT / r.path
        lar = Path(str(mic).replace("MIC", "LAR").replace("mic_", "lar_"))
        f0_ref = pt.load(ROOT, r).f0
        x, sr = sf.read(mic)
        egg, _ = sf.read(lar)
        x16 = resample_poly(x, 16000, sr).astype(np.float32)
        times = frames(np.nan_to_num(f0_ref), x16)
        if not len(times):
            continue
        sp10, eg10 = resample_poly(x, FS_CP, sr), resample_poly(egg, FS_CP, sr)
        res = cp.analyse(sp10, eg10, FS_CP, times, order=ORDER)
        spk = r.speaker_id.split(":")[-1]
        sex = "f" if "FEMALE" in r.path else "m"
        lags.setdefault(spk, []).append(res["lag"])
        n = len(times)
        cols["speaker"] += [spk] * n
        cols["sex"] += [sex] * n
        cols["utt"] += [mic.stem] * n
        cols["t"].append(times)
        for k in ("f0", "cq"):
            cols[k].append(res[k])
        cols["ncyc"].append(res["n_cycles"])
        cols["cp"].append(res["cp"][:, :3])
        cols["cycle"].append(res["cycle"][:, :3])
        cols["praat"].append(praat(x16, times, sex))
        for k, b in teachers.items():
            cols[k].append(b.predict(x16, 16000, times, {})[:, :3])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    out = {k: (np.array(v) if isinstance(v[0], str) else np.concatenate(v)) for k, v in cols.items()}
    np.savez(S / "ptdb_egg.npz", **out)
    for s, l in sorted(lags.items()):
        print(f"{s}: speech lag median {np.median(l) / FS_CP * 1000:.2f} ms")
    print(f"saved {len(out['t'])} frames")
