"""Sensitivity probe, all formants: open-phase shifts and widenings of F1, F2 or F3, read back as the teacher's F1-F3 errors
against the closed-phase values. Same renders as sensitivity.py. Args: name=ckpt pairs. Run from the m3 worktree."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sensitivity as sv
from vocal_analysis.models.infer import analyse, load_teacher
from vocal_analysis.synth import filters, glottal

FACTORS = [("baseline", {})] + [
    (f"open phase: F{k + 1} +{int(pc * 100)} %", {"op_f": {k: pc}}) for k in range(3) for pc in (0.05, 0.10)] + [
    (f"open phase: B{k + 1} x4", {"op_b": {k: 4.0}}) for k in range(3)] + [
    ("open phase: F1-F3 +10 %", {"op_f": {0: 0.1, 1: 0.1, 2: 0.1}}),
    ("open phase: F1 +10 %, F2/F3 +3 %", {"op_f": {0: 0.1, 1: 0.03, 2: 0.03}})]


def render(f123, sex, p, rng):
    n = int(sv.SEC * sv.FS)
    f0 = sv.F0[sex]
    src, open_mask = np.zeros(n), np.zeros(n, bool)
    te = glottal.open_phase(1.0)
    per = int(round(sv.FS / f0))
    w = glottal.lf_waveform(1.0, per)
    for i in range(0, n, per):
        m = min(per, n - i)
        src[i:i + m] = w[:m]
        open_mask[i:i + m] = np.arange(m) / per < te
    src /= np.abs(src).max()
    L = sv.LEN[sex]
    upper = (2 * np.arange(4, 6) - 1) * 35000.0 / (4 * L)
    freqs = np.array([*f123, *upper], float)
    F = np.repeat(freqs[:, None], n, 1)
    B = np.repeat(filters.hawks_miller_bandwidth(freqs, 132.0)[:, None], n, 1)
    for k, pc in p.get("op_f", {}).items():
        F[k] = np.where(open_mask, F[k] * (1 + pc), F[k])
    for k, mult in p.get("op_b", {}).items():
        B[k] = np.where(open_mask, B[k] * mult, B[k])
    y = filters.filter_tv(src, F, B, sv.FS, np.zeros(len(freqs), bool))
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    models = {k: load_teacher(v, dev)[0] for k, v in (a.split("=", 1) for a in sys.argv[1:])}
    t0, t1 = 30, 70
    print("median over vowels of teacher Fk - closed-phase Fk (Hz), change from baseline in brackets; men f0 120, women 210")
    for m, model in models.items():
        print(f"\n{m}\n{'factor':34s} " + "  ".join(f"{s:>5s} F{k}     " for s in ("men", "women") for k in (1, 2, 3)))
        base = {}
        for name, change in FACTORS:
            cells = []
            for sex in ("men", "women"):
                errs = []
                for v, f123 in enumerate(sv.PB[sex]):
                    y = render(f123, sex, change, np.random.default_rng([v, 7]))
                    est = np.nanmedian(analyse(model, y, dev)["formants"][t0:t1, :3], axis=0)
                    errs.append(est - np.array(f123))
                e = np.median(errs, axis=0)
                for k in range(3):
                    base.setdefault((sex, k), e[k]) if name == "baseline" else None
                    cells.append(f"{e[k]:+5.0f} ({e[k] - base[(sex, k)]:+4.0f})")
            print(f"{name:34s} " + "  ".join(cells), flush=True)
