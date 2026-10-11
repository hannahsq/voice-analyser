import sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.eval import invariance as inv
from vocal_analysis.eval.teacher_eval import teacher_baselines
from vocal_analysis.baselines.formants import praat_default
R = Path("../voice-analyser/runs")
ck = dict(a.split("=", 1) for a in sys.argv[1:])
clips = inv.probe_clips()
print(f"{len(clips)} probe clips; true F1 by (vowel, length): " + ", ".join(
    f"{c.vowel}/{c.length_cm:g} {c.formants[0]:.0f}" for c in clips[:: len(inv.F0S_HZ)]))
models = {"praat_default": praat_default} | {k: teacher_baselines(str(R / v), k)[0].predict for k, v in ck.items()}
for name, p in models.items():
    r = inv.run(p, clips)
    print(f"\n{name}: F1 slope vs log2 f0, median {r['median_slope']:+.1f} Hz/oct, median |slope| {r['median_abs_slope']:.1f}; f0 <= 250 Hz: median {r['median_slope_low_f0']:+.1f}, |slope| {r['median_abs_slope_low_f0']:.1f}")
    print("  median F1 error by f0: " + "  ".join(f"{f:.0f}:{e:+.0f}" for f, e in zip(r["f0s"], r["median_error_by_f0"])))
    print("  slopes: " + "  ".join(f"{v}/{L:g} {s:+.0f}" for (v, L), s in r["slopes"].items()))
