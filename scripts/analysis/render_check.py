"""Render check (D-026 fail branch, pre-registered 2026-10-15): do the two tube renderers put the same resonances
where the labels say, and is the closed-phase reference itself unbiased on audio whose poles are known?

Story's 10 vowels, men (17.5 cm, f0 110) and women (15 cm, f0 210), static, 1 s; label = the closed-glottis poles
(`tube.poles`). Renders:
  A  the standard path (train_v2's tube clips): LF at Rd 1.0 through the pole cascade (`voice.synthesize`);
  B  the interactive renderer without feedback, LF-derived area pulse at Rd 1.0 (its flow is the LF flow,
     r 0.998): the same source as A through the full tube transfer instead of the cascade;
  C  the interactive renderer without feedback, Rosenberg pulse (OQ 0.5, sq 1.3);
  D  C with the load's feedback.
Measures:
  analytic: the transfer's magnitude peaks minus the poles (B-D's filter; by construction near 0);
  models: median over frames 30-70 of F1-F3 minus the poles, median over vowels, mean over each arm's seeds
    (models named <arm>_s<k>); mix never saw the interactive renderer, so B - A for mix measures the renderers'
    difference as a train_v2 model hears it;
  closed phase: `eval.closed_phase.analyse` (10 kHz, order 12, as ptdb_egg.py) on C and D with the glottal area as
    the EGG (contact falls as the area rises), F1-F3 minus the poles: the reference method's own bias.
Summary lines: RENDER <sex> F1 F2 F3 (mix B - A, cents); CPREF <sex> <render> F1 F2 F3 (Hz);
  DECISION: mismatch if any |RENDER| >= 10 cents, else ok.
Args: out file, name=ckpt ... ($VA_DEVICE)."""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.signal import resample_poly

from vocal_analysis.eval import closed_phase as cp
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import interactive as it
from vocal_analysis.synth import tract, tube, voice
from vocal_analysis.synth.coupling import Glottis

sys.stdout.reconfigure(encoding="utf-8")
FS, FS_CP, ORDER, DUR = 16000, 10000, 12, 1.0
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
arm = {k: re.sub(r"_s\d+$", "", k) for k in models}
arms = list(dict.fromkeys(arm.values()))
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731
SPK = {"men": (17.5, 1.0, 1.0, 110.0), "women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3, 210.0)}
RENDERS = ("A cascade LF", "B tube LF", "C tube Rosenberg", "D C + feedback")
cents = lambda a, b: 1200 * np.log2(a / b)  # noqa: E731

model_err = defaultdict(lambda: defaultdict(list))  # (sex, render) -> model -> [(3,) Hz]
model_cents = defaultdict(lambda: defaultdict(list))
peak_err, cp_err = defaultdict(list), defaultdict(list)
n = int(DUR * FS)
for sex, (L, asc, phar, f0) in SPK.items():
    s = L / 17.5
    g = Glottis(0.12 * s * s, -0.1, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
    cyc = np.arange(n) * f0 / FS
    nb = n // 16 + 1
    pulses = {"B tube LF": it.lf_area_pulse(cyc, 1.0, g), "C tube Rosenberg": it.area_pulse(cyc, 0.5, 1.3, 0.0)}
    for v, (q1, q2) in tube.STORY_VOWELS.items():
        sh = tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc)
        F, B = tube.poles(sh, tract.F_MAX_HZ)
        F, B = F[:tract.N_POLES], B[:tract.N_POLES]
        # analytic peaks of the transfer near each of F1-F3
        fg = np.arange(50.0, 5000.0, 0.5)
        mag = np.abs(tube.transfer(fg, sh))
        for k in range(3):
            w = (fg > F[k] * 0.8) & (fg < F[k] * 1.2)
            peak_err[sex, k].append(fg[w][np.argmax(mag[w])] - F[k])
        C = n // voice.CTRL + 1
        pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
        pf[:, :len(F)], pb[:, :len(F)] = F, B
        traj = voice.Trajectories(f0=np.full(C, f0), rd=np.full(C, 1.0), voice_amp=np.ones(C), whisper_amp=np.zeros(C),
                                  fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)),
                                  zeros_b=np.zeros((C, 0)))
        ys = {"A cascade LF": voice.synthesize(traj, np.random.default_rng(0))["total"][:n]}
        resp = it.tract_responses(sh, FS, sub_scale=1 / s)
        area = {}
        for name, pulse in pulses.items():
            ys[name] = it.render(g.peak_area_cm2 * pulse, [resp], np.zeros(nb, int), np.zeros(nb), g, coupled=False)["pressure"]
            area[name] = g.peak_area_cm2 * pulse
        ys["D C + feedback"] = it.render(area["C tube Rosenberg"], [resp], np.zeros(nb, int), np.zeros(nb), g, coupled=True)["pressure"]
        area["D C + feedback"] = area["C tube Rosenberg"]
        for name in RENDERS:
            y = (0.5 * ys[name] / np.abs(ys[name]).max()).astype(np.float32)
            for k, m in models.items():
                est = np.nanmedian(analyse(m, y, dev)["formants"][30:71, :3], 0)
                model_err[sex, name][k].append(est - F[:3])
                model_cents[sex, name][k].append(cents(est, F[:3]))
        times = np.arange(0.3, 0.71, 0.01)
        for name in ("C tube Rosenberg", "D C + feedback"):
            sp, eg = resample_poly(ys[name], FS_CP, FS), resample_poly(-area[name], FS_CP, FS)
            r = cp.analyse(sp, eg, FS_CP, times, order=ORDER)
            cp_err[sex, name].append(np.nanmedian(r["cp"][:, :3], 0) - F[:3])

p("Render check (D-026): Story's 10 vowels, static; label = closed-glottis poles")
p("\nAnalytic: transfer magnitude peak minus pole (Hz), median over vowels, F1 / F2 / F3")
for sex in SPK:
    p(f"  {sex:5s} " + " / ".join(f"{np.median(peak_err[sex, k]):+6.1f}" for k in range(3)))
for unit, res in (("Hz", model_err), ("cents", model_cents)):
    p(f"\nModels: F1 / F2 / F3 minus poles ({unit}), median over vowels, mean over each arm's seeds")
    for (sex, name), r in res.items():
        p(f"  {sex:5s} {name:17s}: " + "   ".join(
            f"{a} " + " ".join(f"{x:+5.0f}" for x in np.mean([np.median(r[k], 0) for k in models if arm[k] == a], 0)) for a in arms))
p("\nClosed-phase reference on the renders (glottal area as EGG): F1 / F2 / F3 minus poles (Hz), median over vowels")
for (sex, name), r in cp_err.items():
    p(f"  {sex:5s} {name:17s}: " + " / ".join(f"{x:+6.1f}" for x in np.nanmedian(r, 0)))
p("")
worst = 0.0
ref = "mix" if "mix" in arms else arms[0]
for sex in SPK:
    d = np.mean([np.median(model_cents[sex, "B tube LF"][k], 0) - np.median(model_cents[sex, "A cascade LF"][k], 0)
                 for k in models if arm[k] == ref], 0)
    worst = max(worst, float(np.max(np.abs(d))))
    p(f"RENDER {sex} " + " ".join(f"{x:+.1f}" for x in d))
for (sex, name), r in cp_err.items():
    p(f"CPREF {sex} {name.split()[0]} " + " ".join(f"{x:+.1f}" for x in np.nanmedian(r, 0)))
p(f"DECISION: {'mismatch' if worst >= 10 else 'ok'}")
