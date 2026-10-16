"""Rd sweep (D-026, pre-registered 2026-10-16 before running): is the high-f0 F3 underread a tilt effect? On real
voices v2m d16k reads the two highest-f0 dev women 36-62 cents below closed phase, and the f0 sweep showed it lost
train_v2's upward f0 -> F3 lean. Candidate: the steep spectral tilt of breathier voices pulls F3's evidence down when
harmonics are sparse, and the interactive sets (Rosenberg, a fixed closure corner) never show the model that tilt.

Story's 10 vowels, women's tract (15 cm), the standard path (LF through the pole cascade; Rd sets the tilt, its return
phase above all), Rd 0.5 / 1.0 / 1.5 / 2.0 / 2.5 at f0 210 and 300 Hz, 1 s static. Per model, median over frames 30-70
of F1-F3 minus the closed-glottis poles, cents; median over vowels, mean over each arm's seeds (<arm>_s<k>).
Reading fixed beforehand: "tilt" if at f0 300 v2md16k's F3 change from Rd 0.5 to 2.5 is at least 15 cents more
negative than full's (full-size train_v2, which reads real high-f0 women near closed phase); otherwise "no tilt".
Summary: RDSWEEP <f0> <arm> <F3 change Rd 0.5 -> 2.5>; DECISION.
Args: out file, name=ckpt ... ($VA_DEVICE)."""
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import tract, tube, voice

sys.stdout.reconfigure(encoding="utf-8")
N = 16000
F0S, RDS = (210.0, 300.0), (0.5, 1.0, 1.5, 2.0, 2.5)
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
arm = {k: re.sub(r"_s\d+$", "", k) for k in models}
arms = list(dict.fromkeys(arm.values()))
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731

L, asc, phar = 15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3
res = defaultdict(lambda: defaultdict(list))  # (f0, rd) -> model -> [(3,) cents]
C = N // voice.CTRL + 1
for v, (q1, q2) in tube.STORY_VOWELS.items():
    sh = tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc)
    F, B = tube.poles(sh, tract.F_MAX_HZ)
    F, B = F[:tract.N_POLES], B[:tract.N_POLES]
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    pf[:, :len(F)], pb[:, :len(F)] = F, B
    for f0 in F0S:
        for rd in RDS:
            traj = voice.Trajectories(f0=np.full(C, f0), rd=np.full(C, rd), voice_amp=np.ones(C), whisper_amp=np.zeros(C),
                                      fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
            y = voice.synthesize(traj, np.random.default_rng(0))["total"][:N]
            y = (0.5 * y / np.abs(y).max()).astype(np.float32)
            for k, m in models.items():
                est = np.nanmedian(analyse(m, y, dev)["formants"][30:71, :3], 0)
                res[f0, rd][k].append(1200 * np.log2(est / F[:3]))
    print(f"  {v} done", flush=True)


def mean_arm(f0, rd, a):
    return np.mean([np.nanmedian(res[f0, rd][k], 0) for k in models if arm[k] == a], 0)


p("Rd sweep (D-026): Story's 10 vowels, women's tract (15 cm), standard path; model minus closed-glottis poles, cents, F1 / F2 / F3")
for f0 in F0S:
    p(f"\nf0 {f0:.0f} Hz")
    for a in arms:
        p(f"  {a:9s} " + "   ".join(f"Rd {rd:.1f}: " + " ".join(f"{x:+4.0f}" for x in mean_arm(f0, rd, a)) for rd in RDS))
p("")
ch = {}
for f0 in F0S:
    for a in arms:
        ch[f0, a] = float(mean_arm(f0, 2.5, a)[2] - mean_arm(f0, 0.5, a)[2])
        p(f"RDSWEEP {f0:.0f} {a} {ch[f0, a]:+.1f}")
tilt = ("v2md16k" in arms and "full" in arms and ch[300.0, "v2md16k"] - ch[300.0, "full"] <= -15)
p(f"DECISION: {'tilt' if tilt else 'no tilt'}")
