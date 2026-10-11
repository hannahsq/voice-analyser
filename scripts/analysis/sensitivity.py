"""Sensitivity probe (D-022 follow-up): which real-voice property does the teacher read as higher F1?

Vowels with known closed-phase formants (Peterson & Barney Table II, men at f0 120 Hz, women at 210 Hz; F4-F5 from a
uniform tube of 17.5 / 15 cm; Hawks & Miller bandwidths at 132 Hz) are rendered in the time domain: an LF flow-derivative
pulse train through a cascade of resonators whose frequencies and bandwidths can change every sample, so effects within
the glottal cycle are real. One factor is changed at a time from a baseline; the label is always the closed-phase F1.
Reported: the median over vowels of the teacher's F1 error (Hz), CPU fp32. Args: name=ckpt pairs. Run from the m3 worktree.
"""
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import filters, glottal

FS = 16000
SEC = 1.0
PB = {"men": [(270, 2290, 3010), (390, 1990, 2550), (530, 1840, 2480), (660, 1720, 2410), (730, 1090, 2440),
              (640, 1190, 2390), (440, 1020, 2240), (300, 870, 2240)],
      "women": [(310, 2790, 3310), (430, 2480, 3070), (610, 2330, 2990), (860, 2050, 2850), (850, 1220, 2810),
                (760, 1400, 2780), (470, 1160, 2680), (370, 950, 2670)]}
F0 = {"men": 120.0, "women": 210.0}
LEN = {"men": 17.5, "women": 15.0}

BASE = dict(rd=1.0, b1_mult=1.0, op_f1=0.0, op_b1=1.0, hnr=None, jitter=0.0, subglottal=0.0, sg_open_only=True)
FACTORS = [
    ("baseline", {}),
    ("B1 x0.5", {"b1_mult": 0.5}), ("B1 x2", {"b1_mult": 2.0}), ("B1 x3", {"b1_mult": 3.0}),
    ("Rd 0.5 (pressed)", {"rd": 0.5}), ("Rd 1.7 (breathy-ish)", {"rd": 1.7}), ("Rd 2.5 (breathy)", {"rd": 2.5}),
    ("open phase: B1 x2", {"op_b1": 2.0}), ("open phase: B1 x4", {"op_b1": 4.0}),
    ("open phase: F1 +5 %", {"op_f1": 0.05}), ("open phase: F1 +10 %", {"op_f1": 0.10}),
    ("open phase: F1 +10 %, B1 x3", {"op_f1": 0.10, "op_b1": 3.0}),
    ("aspiration HNR 20 dB", {"hnr": 20.0}), ("aspiration HNR 10 dB", {"hnr": 10.0}),
    ("jitter 2 %", {"jitter": 0.02}),
    ("subglottal pair (open phase)", {"subglottal": 1.0}), ("subglottal pair (always)", {"subglottal": 1.0, "sg_open_only": False}),
]


def render(f123, sex, p, rng):
    n = int(SEC * FS)
    f0 = F0[sex]
    # glottal pulses: per-period LF derivative, period jittered
    src, phase, open_mask = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    te = glottal.open_phase(p["rd"])
    i = 0
    while i < n:
        per = int(round(FS / f0 * np.exp(rng.normal(0, p["jitter"])))) if p["jitter"] else int(round(FS / f0))
        w = glottal.lf_waveform(p["rd"], per)
        m = min(per, n - i)
        src[i:i + m] = w[:m]
        phase[i:i + m] = np.arange(m) / per
        open_mask[i:i + m] = phase[i:i + m] < te
        i += per
    src /= np.abs(src).max()
    if p["hnr"] is not None:  # aspiration, pulsed with the open phase as in voice.synthesize
        noise = np.diff(rng.standard_normal(n + 1)) * np.where(open_mask, 1.0, 0.25)
        noise *= np.sqrt(np.mean(src**2) / np.mean(noise**2)) * 10 ** (-p["hnr"] / 20)
        src = src + noise
    L = LEN[sex]
    upper = (2 * np.arange(4, 6) - 1) * 35000.0 / (4 * L)
    freqs = np.array([*f123, *upper], float)
    bws = filters.hawks_miller_bandwidth(freqs, 132.0)
    F = np.repeat(freqs[:, None], n, 1)
    B = np.repeat(bws[:, None], n, 1)
    B[0] *= p["b1_mult"]
    F[0] = np.where(open_mask, F[0] * (1 + p["op_f1"]), F[0])
    B[0] = np.where(open_mask, B[0] * p["op_b1"], B[0])
    is_zero = np.zeros(len(freqs), bool)
    if p["subglottal"]:  # weak pole-zero pairs near 600 / 1400 Hz (scaled); open-only: zero == pole when closed (cancels)
        s = 17.5 / L
        extra_f, extra_z = [], []
        for base in (600.0, 1400.0):
            pf = base * s
            extra_f.append(np.full(n, pf))
            zf = pf + 50.0
            extra_z.append(np.where(open_mask, zf, pf) if p["sg_open_only"] else np.full(n, zf))
        F = np.vstack([F, *extra_f, *extra_z])
        B = np.vstack([B, np.full((4, n), 150.0)])
        is_zero = np.r_[is_zero, False, False, True, True]
    y = filters.filter_tv(src, F, B, FS, is_zero)
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


if __name__ == "__main__":
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    models = {k: load_teacher(v, dev)[0] for k, v in (a.split("=", 1) for a in sys.argv[1:])}
    t0, t1 = int(0.3 / 0.01), int(0.7 / 0.01)
    print("median over vowels of teacher F1 - closed-phase F1 (Hz); men f0 120 Hz, women f0 210 Hz")
    print(f"{'factor':32s} " + "  ".join(f"{m:>8s} men {m:>8s} women" for m in models))
    base_err = {}
    for name, change in FACTORS:
        p = BASE | change
        cells = []
        for m, model in models.items():
            for sex in ("men", "women"):
                errs = []
                for v, f123 in enumerate(PB[sex]):
                    rng = np.random.default_rng([v, 7])
                    y = render(f123, sex, p, rng)
                    est = np.nanmedian(analyse(model, y, dev)["formants"][t0:t1, 0])
                    errs.append(est - f123[0])
                e = float(np.median(errs))
                if name == "baseline":
                    base_err[(m, sex)] = e
                cells.append(f"{e:+6.0f} ({e - base_err[(m, sex)]:+4.0f})")
        print(f"{name:32s} " + "  ".join(cells), flush=True)
    print("\n(in brackets: change from the baseline row)")
