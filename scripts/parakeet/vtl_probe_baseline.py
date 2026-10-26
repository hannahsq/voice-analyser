"""D-021 item 4, feasibility: the formant-derived baseline on the f0-invariance probe's clips (eval.invariance: Story's
vowels through the lossy tube at 13, 15.5 and 18 cm, f0 swept 80-400 Hz; true length known, nothing else moves).
Per clip, the calibrated uniform-tube length (backend.vtl from teacher v3 seed 1 and 2's mean F1-F4, median over
0.3-0.7 s; calibration in log2 fitted on the head cache's tube crops' true F1-F4) and from the true F1-F4 for
reference. Reported: slope of log2 length against log2 f0 per (vowel, length), cents per octave (median and median
|slope|, all f0 and f0 <= 250 Hz); vowel spread (SD across vowels of the per-(vowel, length) mean, cents); error
against the true length (median, cents). Args: head cache dir, out txt."""
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.backend.vtl import apparent_vtl
from vocal_analysis.eval import invariance as inv
from vocal_analysis.eval.teacher_eval import teacher_baselines

CACHE, OUTF = Path(sys.argv[1]), sys.argv[2]
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
v = np.load(CACHE / "vtl.npz")
son = np.load(CACHE / "sonorant.npy", mmap_mode="r")
fm = np.load(CACHE / "formants.npy", mmap_mode="r")
a4, yc = [], []
for i in np.flatnonzero(v["tube"] & ~v["from_f0"]):
    s = np.asarray(son[i]) > 0.5
    if s.sum() >= 3:
        a4.append(np.nanmedian(apparent_vtl(np.asarray(fm[i])[s])))
        yc.append(np.log2(v["length_cm"][i]))
cal = np.polyfit(np.log2(a4), yc, 1)

clips = inv.probe_clips()
times = np.arange(0.3, 0.7001, 0.01)
preds = [teacher_baselines(f"runs/conformer_v2r2_s{k}_d16k/ckpt_last.pt", f"s{k}")[0].predict for k in (1, 2)]
est = {"labeller v3 formants": [], "true formants": []}
for c in clips:
    f = np.exp(np.mean([np.log(pr(c.audio, inv.FS, times, {})[:, :4]) for pr in preds], 0))
    est["labeller v3 formants"].append(np.nanmedian(np.polyval(cal, np.log2(apparent_vtl(f)))))
    est["true formants"].append(np.polyval(cal, np.log2(apparent_vtl(c.formants[None]))[0]))
f0 = np.log2([c.f0 for c in clips])
truth = np.log2([c.length_cm for c in clips])
keys = sorted({(c.vowel, c.length_cm) for c in clips})
kid = np.array([keys.index((c.vowel, c.length_cm)) for c in clips])
p(f"Probe: {len(clips)} clips; calibration slope {cal[0]:.3f} on {len(yc)} tube crops")
for name, e in est.items():
    e = np.array(e)
    sl, sl_lo = [], []
    for k in range(len(keys)):
        m = kid == k
        sl.append(1200 * np.polyfit(f0[m], e[m], 1)[0])
        lo = m & (f0 <= np.log2(inv.LOW_F0_MAX_HZ))
        sl_lo.append(1200 * np.polyfit(f0[lo], e[lo], 1)[0])
    sl, sl_lo = np.array(sl), np.array(sl_lo)
    lo = f0 <= np.log2(inv.LOW_F0_MAX_HZ)
    vs = [np.std([1200 * np.mean(e[(kid == keys.index((vw, L))) & lo]) for vw in inv.VOWELS]) for L in inv.LENGTHS_CM]
    p(f"\n{name}: slope vs log2 f0, median {np.median(sl):+.0f} c/oct, median |slope| {np.median(np.abs(sl)):.0f}; "
      f"f0 <= 250 Hz: median {np.median(sl_lo):+.0f}, |slope| {np.median(np.abs(sl_lo)):.0f}")
    p(f"  vowel spread (f0 <= 250 Hz, SD across vowels) by length: " + "  ".join(f"{L:g}: {s:.0f}" for L, s in zip(inv.LENGTHS_CM, vs)) + " cents")
    p(f"  error vs true length, median by length (f0 <= 250 Hz): " + "  ".join(
        f"{L:g}: {1200 * np.median((e - truth)[lo & (truth == np.log2(L))]):+.0f}" for L in inv.LENGTHS_CM) + " cents")
