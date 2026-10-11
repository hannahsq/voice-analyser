"""D-023 probe, extended: does glottal coupling make the teacher read F1 high, once the voice is breathier and the
glottis doesn't close fully? Story's vowels, women (f0 210) and men (f0 120); Rd 1.0 / 1.6 / 2.2; peak opening 0.1 and
0.2 cm2 (at 17.5 cm); closed-phase gap 0 / 0.3 / 0.6 of the peak. Per condition, the median over vowels of each reader's
F1 minus the closed-glottis fR1, and the change against the same Rd without coupling (the coupling effect alone).
Args: out file, name=ckpt ... Run from the m3 worktree."""
import sys

import numpy as np
import torch

from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import coupling, tube
from vocal_analysis.synth.voice import CTRL, FS, Trajectories, synthesize

SPK = {"women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3, 210.0), "men": (17.5, 1.0, 1.0, 120.0)}
RDS, AREAS, GAPS = (1.0, 1.6, 2.2), (0.1, 0.2), (0.0, 0.3, 0.6)
torch.set_num_threads(6)
dev = torch.device("cpu")
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
out = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731
times = np.arange(0.3, 0.7001, 0.01)
c = int(1.0 * FS / CTRL) + 1
readers = ["praat", *models]


def f1_errors(y, F1):
    errs = {"praat": np.nanmedian(praat_default(y, FS, times, {})[:, 0]) - F1}
    for n, m in models.items():
        errs[n] = np.nanmedian(analyse(m, y, dev)["formants"][30:71, 0]) - F1
    return errs


for sex, (L, asc, phar, f0) in SPK.items():
    s = L / 17.5
    g = coupling.Glottis(0.1, 0.0, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
    res = {}  # (rd, area, gap) -> reader -> list over vowels; area 0 = no coupling
    for v, (q1, q2) in tube.STORY_VOWELS.items():
        shape = tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc)
        F, B = tube.poles(shape)
        F, B = F[:8], B[:8]
        pf, pb = np.full((c, 8), np.nan), np.full((c, 8), np.nan)
        pf[:, :len(F)], pb[:, :len(B)] = F, B
        for rd in RDS:
            conds = [(rd, 0.0, 0.0)] + [(rd, a, gp) for a in AREAS for gp in GAPS]
            for key in conds:
                _, a, gp = key
                cpl = None
                if a > 0:
                    g.peak_area_cm2 = a * s * s
                    df, bm = coupling.open_shifts(shape, F, B, g)
                    cpl = {"df": np.tile(df[:5], (c, 1)), "bm": np.tile(bm[:5], (c, 1)), "gap": gp}
                tr = Trajectories(f0=np.full(c, f0), rd=np.full(c, rd), voice_amp=np.ones(c), whisper_amp=np.zeros(c),
                                  fric_amp=np.zeros(c), poles_f=pf, poles_b=pb, zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
                y = synthesize(tr, np.random.default_rng(0), jitter=0.005, shimmer_db=0.2, hnr_db=30.0, coupling=cpl)["total"]
                y = (0.5 * y / np.abs(y).max()).astype(np.float32)
                for r, e in f1_errors(y, F[0]).items():
                    res.setdefault(key, {}).setdefault(r, []).append(e)
    p(f"\n{sex}, f0 {f0:.0f} Hz: F1 minus closed-glottis fR1, median over {len(tube.STORY_VOWELS)} vowels (Hz);"
      f" in brackets, the change against the same Rd without coupling")
    p(f"{'Rd':>4s} {'peak':>5s} {'gap':>4s} | " + " | ".join(f"{r:>18s}" for r in readers))
    for key in sorted(res):
        rd, a, gp = key
        base = res[(rd, 0.0, 0.0)]
        cells = []
        for r in readers:
            v = np.median(res[key][r])
            d = np.median(np.array(res[key][r]) - np.array(base[r]))
            cells.append(f"{v:+7.0f}" + (f" ({d:+5.0f})" if a > 0 else "        ") + "   ")
        p(f"{rd:4.1f} {a:5.2f} {gp:4.1f} | " + " | ".join(f"{c_:>18s}" for c_ in cells))
