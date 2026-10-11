"""D-023 render check: tube vowels rendered with glottal coupling at fixed peak areas; Praat (and the teacher) read
F1-F3 against the closed-glottis resonances. Expect (Birkholz et al. 2019): Praat F1 rises with peak area, more for
close vowels (fR1 < 350 Hz); F2, F3 not systematically. Args: out file [name=ckpt]. Run from the m3 worktree."""
import sys

import numpy as np
import torch

from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import coupling, tube
from vocal_analysis.synth.voice import CTRL, FS, Trajectories, synthesize

SPK = {"men": (17.5, 1.0, 1.0, 120.0), "women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3, 210.0)}
AREAS = (0.0, 0.05, 0.1, 0.2)
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731
model = None
if len(sys.argv) > 2:
    torch.set_num_threads(6)
    model = load_teacher(sys.argv[2].split("=", 1)[1], torch.device("cpu"))[0]
times = np.arange(0.3, 0.7001, 0.01)
c = int(1.0 * FS / CTRL) + 1
for sex, (L, asc, phar, f0) in SPK.items():
    s = L / 17.5
    g = coupling.Glottis(0.1, 0.0, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
    p(f"\n{sex}, f0 {f0:.0f} Hz, Rd 1.0, no gap. Error vs closed-glottis fR (Hz) at peak area (cm2, at 17.5 cm; x{s * s:.2f} here)")
    p(f"{'vowel':6s}{'fR1':>6s} | " + " | ".join(f"{a:4.2f}: Praat F1 F2 F3" + ("  teacher F1 F2 F3" if model else "") for a in AREAS))
    for v, (q1, q2) in tube.STORY_VOWELS.items():
        shape = tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc)
        F, B = tube.poles(shape)
        F, B = F[:8], B[:8]
        pf, pb = np.full((c, 8), np.nan), np.full((c, 8), np.nan)
        pf[:, :len(F)], pb[:, :len(B)] = F, B
        cells = []
        for a in AREAS:
            area = a * s * s
            if area > 0:
                g.peak_area_cm2 = area
                df, bm = coupling.open_shifts(shape, F, B, g)
                cpl = {"df": np.tile(df[:5], (c, 1)), "bm": np.tile(bm[:5], (c, 1)), "gap": 0.0}
            else:
                cpl = None
            tr = Trajectories(f0=np.full(c, f0), rd=np.full(c, 1.0), voice_amp=np.ones(c), whisper_amp=np.zeros(c),
                              fric_amp=np.zeros(c), poles_f=pf, poles_b=pb, zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
            y = synthesize(tr, np.random.default_rng(0), jitter=0.005, shimmer_db=0.2, hnr_db=30.0, coupling=cpl)["total"]
            y = (0.5 * y / np.abs(y).max()).astype(np.float32)
            pr = np.nanmedian(praat_default(y, FS, times, {})[:, :3], axis=0) - F[:3]
            cell = f"      {pr[0]:+4.0f} {pr[1]:+4.0f} {pr[2]:+4.0f}"
            if model:
                te = np.nanmedian(analyse(model, y, torch.device("cpu"))["formants"][30:71, :3], axis=0) - F[:3]
                cell += f"         {te[0]:+4.0f} {te[1]:+4.0f} {te[2]:+4.0f}"
            cells.append(cell)
        p(f"{v:6s}{F[0]:6.0f} | " + " | ".join(cells))
