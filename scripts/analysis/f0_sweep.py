"""f0 sweep (D-026, pre-registered 2026-10-15 before running): with the tract fixed, does a model's F3 fall as f0
rises? On real voices v2m d16k's women's F3 underread is concentrated in the two highest-f0 speakers (260 / 294 Hz:
-54 / -49 cents against full-size train_v2, the other four -16 to -27). If the interactive sets shifted the model's
prior and the prior takes over where harmonics sample F3 sparsely, v2m's F3 should fall with f0 on a fixed tract.

Story's 10 vowels, women's tract (15 cm, as shortcut_probe.py), f0 150 / 210 / 260 / 300 / 350 / 400 Hz, 1 s static.
Renders: A the standard path (LF Rd 1.0 through the pole cascade), D the interactive tube with feedback (Rosenberg,
OQ 0.5, sq 1.3). Per model, median over frames 30-70 of F1-F3 minus the closed-glottis poles, in cents; median over
vowels, mean over each arm's seeds (models named <arm>_s<k>).
Reading fixed beforehand: "prior shift" if, on both renders, v2md16k's F3 change from 150 to 300 Hz is at least
15 cents more negative than mix's; otherwise "no prior shift".
Summary: SWEEP <render> <arm> <F3 change 150 -> 300, cents>; DECISION.
Args: out file, name=ckpt ... ($VA_DEVICE)."""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import interactive as it
from vocal_analysis.synth import tract, tube, voice
from vocal_analysis.synth.coupling import Glottis

sys.stdout.reconfigure(encoding="utf-8")
FS, N = 16000, 16000
F0S = (150.0, 210.0, 260.0, 300.0, 350.0, 400.0)
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
arm = {k: re.sub(r"_s\d+$", "", k) for k in models}
arms = list(dict.fromkeys(arm.values()))
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731

L, asc, phar = 15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3
s = L / 17.5
g = Glottis(0.12 * s * s, -0.1, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
res = defaultdict(lambda: defaultdict(list))  # (render, f0) -> model -> [(3,) cents]
shapes = {v: tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc) for v, (q1, q2) in tube.STORY_VOWELS.items()}
for v, sh in shapes.items():
    F, B = tube.poles(sh, tract.F_MAX_HZ)
    F, B = F[:tract.N_POLES], B[:tract.N_POLES]
    resp = it.tract_responses(sh, FS, sub_scale=1 / s)
    C = N // voice.CTRL + 1
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    pf[:, :len(F)], pb[:, :len(F)] = F, B
    for f0 in F0S:
        traj = voice.Trajectories(f0=np.full(C, f0), rd=np.full(C, 1.0), voice_amp=np.ones(C), whisper_amp=np.zeros(C),
                                  fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
        cyc = np.arange(N) * f0 / FS
        ys = {"A": voice.synthesize(traj, np.random.default_rng(0))["total"][:N],
              "D": it.render(g.peak_area_cm2 * it.area_pulse(cyc, 0.5, 1.3, 0.0), [resp], np.zeros(N // 16 + 1, int),
                             np.zeros(N // 16 + 1), g, coupled=True)["pressure"]}
        for r, y in ys.items():
            y = (0.5 * y / np.abs(y).max()).astype(np.float32)
            for k, m in models.items():
                est = np.nanmedian(analyse(m, y, dev)["formants"][30:71, :3], 0)
                res[r, f0][k].append(1200 * np.log2(est / F[:3]))
    print(f"  {v} done", flush=True)


def mean_arm(r, f0, a):
    return np.mean([np.nanmedian(res[r, f0][k], 0) for k in models if arm[k] == a], 0)


p("f0 sweep (D-026): Story's 10 vowels, women's tract (15 cm); model minus closed-glottis poles, cents, F1 / F2 / F3")
p("A standard path (LF Rd 1.0, pole cascade); D interactive tube with feedback (Rosenberg)")
for r in ("A", "D"):
    p(f"\nRender {r}")
    for a in arms:
        p(f"  {a:9s} " + "   ".join(f"{f0:3.0f}: " + " ".join(f"{x:+4.0f}" for x in mean_arm(r, f0, a)) for f0 in F0S))
p("")
verdict = []
for r in ("A", "D"):
    ch = {a: float(mean_arm(r, 300.0, a)[2] - mean_arm(r, 150.0, a)[2]) for a in arms}
    for a, x in ch.items():
        p(f"SWEEP {r} {a} {x:+.1f}")
    if "v2md16k" in ch and "mix" in ch:
        verdict.append(ch["v2md16k"] - ch["mix"] <= -15)
p(f"DECISION: {'prior shift' if verdict and all(verdict) else 'no prior shift'}")
