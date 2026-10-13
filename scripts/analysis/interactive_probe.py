"""Do the models read F1 high, and do seeds split as on real voices, when the source interacts with the tract (D-026)?
Story's 10 vowels through `synth.interactive`, men (17.5 cm, f0 120) and women (15 cm, f0 210); Rd 1.0 / 2.0 (open
quotient from the LF te; Rosenberg area pulse, speed quotient 1.3); closed-phase gap 0 or Holmberg's (`coupling.gap_from_open_quotient`, offset -0.1); with the
load (coupled) or without (the same area waveform and tract, no feedback: the control). Per model and condition: median
over vowels of F1 minus the closed-glottis fR1 (the label). Then the seed contrasts (real-voice women: +14.7, +12.2,
+11.8, +8.2, +12.7 Hz) and, across models, Spearman rho of each condition's error with the real closed-phase bias
(from probe_vs_real.txt). Args: out file, models file (name=ckpt ...), probe_vs_real.txt."""
import os
import re
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import interactive as it
from vocal_analysis.synth import tube
from vocal_analysis.synth.coupling import Glottis, gap_from_open_quotient
from vocal_analysis.synth.glottal import open_phase

sys.stdout.reconfigure(encoding="utf-8")
FS = 16000
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
specs = open(sys.argv[2]).read().split()
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in specs}
names = list(models)
real = {}
for line in open(sys.argv[3], encoding="utf-8"):
    m = re.match(r"^(\S+)\s+([+-]\d+\.\d)\s+([+-]\d+\.\d)\s*$", line)
    if m:
        real[m.group(1)] = (float(m.group(2)), float(m.group(3)))
SPK = {"men": (17.5, 1.0, 1.0, 120.0), "women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3, 210.0)}
CONTRASTS = [("mix_conformer_s1", ["mix_conformer_s0", "mix_conformer_s2"]), ("m3_conformer_s0_d16k", ["m3_conformer_s2_d16k"]),
             ("v2_s1", ["v2_s0", "v2_s2"]), ("v2c_conformer_s1", ["v2c_conformer_s0", "v2c_conformer_s2"]),
             ("m3_conformer_s0", ["m3_conformer_s1"])]
conf = np.array(["conformer" in n for n in names])
n = FS
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731
p("Interactive source (synth.interactive): F1 minus the closed-glottis fR1 (Hz), median over Story's 10 vowels")
for sex, (L, asc, phar, f0) in SPK.items():
    s = L / 17.5
    g = Glottis(0.1 * s * s, -0.1, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
    shapes = {v: tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc) for v, (q1, q2) in tube.STORY_VOWELS.items()}
    fr1 = {v: tube.poles(sh)[0][0] for v, sh in shapes.items()}
    resp = {v: it.tract_responses(sh, FS, sub_scale=1 / s) for v, sh in shapes.items()}
    p(f"\n{sex}: L {L} cm, f0 {f0:.0f} Hz, peak area {g.peak_area_cm2:.3f} cm2")
    p(f"{'condition':30s} {'all':>6s} {'conf':>6s} {'GRU':>6s} | seed contrasts (real women +14.7 +12.2 +11.8 +8.2 +12.7) | rho with real bias: all / Conformers")
    for rd in (1.0, 2.0):
        te = open_phase(rd)
        for gap_on in (False, True):
            gap = float(gap_from_open_quotient(te, g.gap_offset)) if gap_on else 0.0
            area = g.peak_area_cm2 * it.area_pulse(np.arange(n) * f0 / FS, te, 1.3, gap)
            for coupled in (False, True):
                errs = {k: [] for k in names}
                for v in shapes:
                    nb = n // 16 + 1
                    y = it.render(area, [resp[v]], np.zeros(nb, int), np.zeros(nb), g, coupled=coupled)["pressure"]
                    y = (0.5 * y / np.abs(y).max()).astype(np.float32)
                    for k, m in models.items():
                        errs[k].append(np.nanmedian(analyse(m, y, dev)["formants"][30:71, 0]) - fr1[v])
                e = np.array([np.median(errs[k]) for k in names])
                con = " ".join(f"{e[names.index(b)] - np.mean([e[names.index(a)] for a in o]):+6.1f}" for b, o in CONTRASTS)
                r = np.array([real[k][0 if sex == "women" else 1] for k in names])
                cond = f"Rd {rd} gap {gap:.2f} {'coupled' if coupled else 'uncoupled'}"
                p(f"{cond:30s} {np.median(e):+6.1f} {np.median(e[conf]):+6.1f} {np.median(e[~conf]):+6.1f} | {con} | "
                  f"{spearmanr(e, r).statistic:+.2f} / {spearmanr(e[conf], r[conf]).statistic:+.2f}")
    p("  per model, last condition: " + "  ".join(f"{k} {np.median(errs[k]):+.0f}" for k in names))
