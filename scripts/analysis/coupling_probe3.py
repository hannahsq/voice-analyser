"""D-023 probe, noise: does aspiration noise alone (no coupling) make the teacher read F1 high, as breathy voices are
noisier as well as more open? Story's vowels, women (f0 210) and men (f0 120); Rd 1.0 / 2.2; HNR 30 / 15 / 8 dB; coupling
off, or on (peak 0.1 cm2 at 17.5 cm, closed-phase gap 0.3 of the peak). Per condition, the median over vowels of each
reader's F1 minus the closed-glottis fR1 (each render: median over 3 noise seeds), and in brackets the change against
the same Rd at HNR 30 dB without coupling. Args: out file, name=ckpt ... Run from the m3 worktree."""
import sys

import numpy as np
import torch

from vocal_analysis.baselines.formants import praat_default
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import coupling, tube
from vocal_analysis.synth.voice import CTRL, FS, Trajectories, synthesize

SPK = {"women": (15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3, 210.0), "men": (17.5, 1.0, 1.0, 120.0)}
RDS, HNRS, PEAK, GAP = (1.0, 2.2), (30.0, 15.0, 8.0), 0.1, 0.3
torch.set_num_threads(16)
dev = torch.device(__import__("os").environ.get("VA_DEVICE", "cpu"))  # cuda: same readings, much faster
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
    g = coupling.Glottis(PEAK * s * s, 0.0, 1.4 * s, 0.3 * s, 8000.0, 1 / s)
    res = {}  # (rd, hnr, coupled) -> reader -> list over vowels
    for v, (q1, q2) in tube.STORY_VOWELS.items():
        shape = tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc)
        F, B = tube.poles(shape)
        F, B = F[:8], B[:8]
        pf, pb = np.full((c, 8), np.nan), np.full((c, 8), np.nan)
        pf[:, :len(F)], pb[:, :len(B)] = F, B
        df, bm = coupling.open_shifts(shape, F, B, g)
        for rd in RDS:
            for hnr in HNRS:
                for on in (False, True):
                    cpl = {"df": np.tile(df[:5], (c, 1)), "bm": np.tile(bm[:5], (c, 1)), "gap": GAP} if on else None
                    tr = Trajectories(f0=np.full(c, f0), rd=np.full(c, rd), voice_amp=np.ones(c), whisper_amp=np.zeros(c),
                                      fric_amp=np.zeros(c), poles_f=pf, poles_b=pb, zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
                    per_seed = []
                    for seed in range(3):
                        y = synthesize(tr, np.random.default_rng(seed), jitter=0.005, shimmer_db=0.2, hnr_db=hnr, coupling=cpl)["total"]
                        per_seed.append(f1_errors((0.5 * y / np.abs(y).max()).astype(np.float32), F[0]))
                    for r in readers:
                        res.setdefault((rd, hnr, on), {}).setdefault(r, []).append(float(np.median([e[r] for e in per_seed])))
    p(f"\n{sex}, f0 {f0:.0f} Hz: F1 minus closed-glottis fR1, median over {len(tube.STORY_VOWELS)} vowels (Hz); in brackets, "
      f"the change against the same Rd at HNR 30 dB without coupling")
    p(f"{'Rd':>4s} {'HNR':>4s} {'coupling':>9s} | " + " | ".join(f"{r:>18s}" for r in readers))
    for key in sorted(res, key=lambda k: (k[0], -k[1], k[2])):
        rd, hnr, on = key
        base = res[(rd, 30.0, False)]
        cells = [f"{np.median(res[key][r]):+7.0f} ({np.median(np.array(res[key][r]) - np.array(base[r])):+5.0f})" for r in readers]
        p(f"{rd:4.1f} {hnr:4.0f} {'on' if on else 'off':>9s} | " + " | ".join(f"{c_:>18s}" for c_ in cells))
