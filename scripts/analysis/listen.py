"""Listening samples for the open-phase effects in the sensitivity probe. Args: out dir. Run from the m3 worktree."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv

sv.SEC = 1.5
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
VOWELS = {"i": 0, "ae": 3, "A": 4, "u": 7}
FACTORS = [
    ("01_baseline", {}),
    ("02_open-F1+10pc", {"op_f1": 0.10}),
    ("03_open-F1+20pc_exaggerated", {"op_f1": 0.20}),
    ("04_open-B1x4", {"op_b1": 4.0}),
    ("05_open-F1+10pc_B1x3", {"op_f1": 0.10, "op_b1": 3.0}),
    ("06_static-B1x3", {"b1_mult": 3.0}),
    ("07_Rd0.5_pressed", {"rd": 0.5}),
    ("08_Rd2.5_breathy", {"rd": 2.5}),
    ("09_subglottal_open-phase", {"subglottal": 1.0}),
]
FADE, GAP = int(0.03 * sv.FS), np.zeros(int(0.4 * sv.FS), np.float32)


def finish(y):
    y = y * (0.1 / np.sqrt(np.mean(y**2)))  # equal RMS (-20 dBFS), so loudness doesn't give the answer away
    r = np.linspace(0, 1, FADE)
    y[:FADE] *= r
    y[-FADE:] *= r[::-1]
    return y.astype(np.float32)


for sex in ("men", "women"):
    for v, k in VOWELS.items():
        d = OUT / f"{sex}_{v}"
        d.mkdir(exist_ok=True)
        base = None
        for name, change in FACTORS:
            y = finish(sv.render(sv.PB[sex][k], sex, sv.BASE | change, np.random.default_rng([k, 7])))
            sf.write(d / f"{name}.wav", y, sv.FS, subtype="PCM_16")
            if base is None:
                base = y
            else:  # A/B/A/B: baseline, variant, baseline, variant
                sf.write(d / f"AB_{name}.wav", np.concatenate([base, GAP, y, GAP, base, GAP, y]), sv.FS, subtype="PCM_16")
print("done", OUT)
