"""Shortcut probe (D-026): does a model correct for source-tract interaction only when the feedback is there? Story's
10 vowels through `synth.interactive`, men (17.5 cm, f0 110) and women (15 cm, f0 210), peak area 0.12 cm^2 scaled
by length, no gap; pulses: Rosenberg (OQ 0.5, speed quotient 1.3) and LF-derived (Rd 1.0, 1.7); each with and
without the load's feedback. Per model, median over vowels of F1-F3 minus the closed-glottis poles.
Summary per arm (models named <arm>_s<k>; the reference arm, default mix, applies no correction):
  SHORTCUT <arm> = mean over sexes, pulses and F1-F3 of |error without feedback - the reference arm's|: the correction
    applied where there is nothing to correct (0 for a model that detects the feedback);
  COUPLED <arm> = mean |error with feedback|, Rosenberg pulse only.
The LF-derived pulse with feedback counts the interaction twice (`interactive.lf_area_pulse`), so its feedback rows
are not a target; without feedback it reproduces the LF flow exactly, which is what SHORTCUT uses.
Args: out file, reference arm, name=ckpt ... ($VA_DEVICE)."""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import interactive as it
from vocal_analysis.synth import tube
from vocal_analysis.synth.coupling import Glottis

sys.stdout.reconfigure(encoding="utf-8")
FS = 16000
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
ref_arm = sys.argv[2]
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[3:]}
arm = {k: re.sub(r"_s\d+$", "", k) for k in models}
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731
SPK = {"men": (17.5, 1.0, 1.0, 110.0), "women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3, 210.0)}
PULSES = {"Rosenberg": None, "LF Rd 1.0": 1.0, "LF Rd 1.7": 1.7}
res = defaultdict(dict)  # (sex, pulse, coupled) -> model -> (3,) error
for sex, (L, asc, phar, f0) in SPK.items():
    s = L / 17.5
    g = Glottis(0.12 * s * s, -0.1, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
    shapes = {v: tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc) for v, (q1, q2) in tube.STORY_VOWELS.items()}
    poles = {v: tube.poles(sh)[0][:3] for v, sh in shapes.items()}
    resp = {v: it.tract_responses(sh, FS, sub_scale=1 / s) for v, sh in shapes.items()}
    cyc = np.arange(FS) * f0 / FS
    nb = FS // 16 + 1
    for pname, rd in PULSES.items():
        pulse = it.area_pulse(cyc, 0.5, 1.3, 0.0) if rd is None else it.lf_area_pulse(cyc, rd, g)
        for coupled in (False, True):
            err = defaultdict(list)
            for v in shapes:
                y = it.render(g.peak_area_cm2 * pulse, [resp[v]], np.zeros(nb, int), np.zeros(nb), g, coupled=coupled)["pressure"]
                y = (0.5 * y / np.abs(y).max()).astype(np.float32)
                for k, m in models.items():
                    err[k].append(np.nanmedian(analyse(m, y, dev)["formants"][30:71, :3], 0) - poles[v])
            for k in models:
                res[sex, pname, coupled][k] = np.median(err[k], 0)
arms = list(dict.fromkeys(arm.values()))
p("Model F1 / F2 / F3 minus closed-glottis poles (Hz), median over Story's 10 vowels, mean over each arm's seeds")
for (sex, pname, coupled), r in res.items():
    p(f"  {sex:5s} {pname:10s} {'feedback   ' if coupled else 'no feedback'}: " + "   ".join(
        f"{a} " + " ".join(f"{x:+5.0f}" for x in np.mean([r[k] for k in models if arm[k] == a], 0)) for a in arms))
p("")
for a in arms:
    sc, cp = [], []
    for (sex, pname, coupled), r in res.items():
        mean = np.mean([r[k] for k in models if arm[k] == a], 0)
        if coupled:
            if pname == "Rosenberg":
                cp.append(np.abs(mean))
        else:
            sc.append(np.abs(mean - np.mean([r[k] for k in models if arm[k] == ref_arm], 0)))
    p(f"SHORTCUT {a} {np.mean(sc):.1f}")
    p(f"COUPLED {a} {np.mean(cp):.1f}")
