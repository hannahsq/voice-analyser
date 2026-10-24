"""D-034, reported only (pre-registered 2026-10-24 in decisions.md): a controlled f0 x F1 grid of sustained vowels, to tell
whether the students' loss of fidelity to Parakeet on high voices follows F1 / f0 or f0 alone.
Stimuli: the standard path (LF Rd 1.0, pole cascade, no channel, peak 0.5); f0 in F0S crossed with F1 in F1S, F2-F5 and
bandwidths fixed; a slow +-3 % f0 drift (0.5 Hz sinusoid, random phase); four variants per cell (drift phase, synthesiser
seed). Measure: Parakeet layer 5 at the eight 10 ms phases (audio shifted by 10q ms), each student at frame 8e + q - 4,
e = 1..49; NE = mean over dimensions of squared error / held_synth_v2's per-dimension target variance. Also Parakeet's
own variation over the clip (same normalisation) and its grid-pattern share.
Rule: per student, NE = a + b_f log2 f0 + b_1 log2 F1 on the 25 cell means; NE range < 0.05 -> "no effect"; else
rho = -b_1 / b_f: >= 0.67 "F1 / f0 drives it", <= 0.33 "f0 drives it", otherwise "both"; bootstrap over variants; both
students must agree.
Args: v4 cache dir, out prefix, then LABEL=CKPT=STATS_DIR per student. Writes <prefix>.txt and <prefix>_cells.csv.
Run from the main checkout (VA_DEVICE=cuda)."""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_cache as D  # noqa: E402
import load  # noqa: E402
from student import load_student  # noqa: E402

from vocal_analysis.features.vqt import HOP, TorchVQT  # noqa: E402
from vocal_analysis.synth import tract, voice  # noqa: E402

F0S = (150.0, 250.0, 400.0, 600.0, 800.0)
F1S = (300.0, 450.0, 600.0, 800.0, 1000.0)
UPPER = (1700.0, 2800.0, 3800.0, 4800.0)
BWS = (80.0, 100.0, 140.0, 180.0, 220.0)
VARIANTS, E4, FS = 4, 49, 16000
NTOT = D.NS + 8 * HOP


def render(f0, f1, v):
    rng = np.random.default_rng([int(f0), int(f1), v, 4242])
    C = NTOT // voice.CTRL + 1
    t = np.arange(C) * voice.CTRL / FS
    f0t = f0 * (1 + 0.03 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi)))
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    pf[:, :5], pb[:, :5] = (f1, *UPPER), BWS
    traj = voice.Trajectories(f0=f0t, rd=np.full(C, 1.0), voice_amp=np.ones(C), whisper_amp=np.zeros(C), fric_amp=np.zeros(C),
                              poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(traj, rng)["total"][:NTOT]
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


def main():
    C4, PRE, SPECS = Path(sys.argv[1]), sys.argv[2], [a.split("=", 2) for a in sys.argv[3:]]
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    vq = TorchVQT(dev)
    st = np.load(C4 / "stats.npz")
    MU4, SD4 = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    # the fixed reference: held_synth_v2's per-dimension target variance (standardised)
    tg = np.load(C4 / "held_synth_v2" / "target.npy", mmap_mode="r")
    s1, s2, n = np.zeros(1024), np.zeros(1024), 0
    for i in range(0, len(tg), 32):
        t = ((np.asarray(tg[i:i + 32], np.float32) - st["mean"]) / st["std"]).reshape(-1, 1024).astype(np.float64)
        s1, s2, n = s1 + t.sum(0), s2 + (t ** 2).sum(0), n + len(t)
    VREF = torch.tensor(s2 / n - (s1 / n) ** 2, dtype=torch.float32, device=dev)
    fe, enc = load.feature_extractor(), load.encoder(dev)
    enc.layers = enc.layers[:D.LAYER]
    students = []
    for label, ck, sdir in SPECS:
        s = np.load(Path(sdir) / "stats.npz")
        students.append((label, load_student(ck, dev).requires_grad_(False), *(torch.tensor(s[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))))
    j0 = torch.arange(1, E4 + 1, device=dev) * 8 - 4

    def parakeet(ctxs):
        f = fe(list(ctxs), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[D.LAYER].float()
        return (h[:, 1:1 + E4] - MU4) / SD4

    cells = [(f0, f1) for f0 in F0S for f1 in F1S]
    ne = {lab: np.zeros((len(cells), VARIANTS)) for lab, *_ in students}
    pvar, pgrid = np.zeros((len(cells), VARIANTS)), np.zeros((len(cells), VARIANTS))
    for c, (f0, f1) in enumerate(cells):
        xs = [render(f0, f1, v) for v in range(VARIANTS)]
        t = torch.stack([parakeet([x[q * HOP:] for x in xs]) for q in range(8)], 1)  # (V, 8, 49, 1024)
        mean = t.mean((1, 2), keepdim=True)
        pvar[c] = (((t - mean) ** 2).mean((1, 2)) / VREF).mean(-1).cpu().numpy()
        comb = t.mean(2, keepdim=True) - mean
        pgrid[c] = ((comb ** 2).sum((1, 2, 3)) * E4 / ((t - mean) ** 2).sum((1, 2, 3))).cpu().numpy()
        audio = torch.from_numpy(np.stack([x[:D.NS] for x in xs])).to(dev)
        for lab, m, MU, SD in students:
            with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16):
                y = m(vq.log_power_with_context(audio, D.CROP)).float()
            y = (y * SD + MU - MU4) / SD4
            err = torch.stack([y[:, j0 + q] - t[:, q] for q in range(8)], 1)  # (V, 8, 49, 1024)
            ne[lab][c] = ((err ** 2).mean((1, 2)) / VREF).mean(-1).cpu().numpy()
        D.log(f"f0 {f0:.0f} F1 {f1:.0f}: " + ", ".join(f"{lab} NE {ne[lab][c].mean():.3f}" for lab in ne) + f"; Parakeet variation {pvar[c].mean():.3f}, grid share {pgrid[c].mean():.3f}")

    out = open(f"{PRE}.txt", "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")

    say("D-034 controlled f0 x F1 grid (reported only; pre-registered 2026-10-24). NE = squared error / held_synth_v2's target variance")
    for lab in list(ne) + ["parakeet_variation", "parakeet_grid_share"]:
        M = (ne[lab] if lab in ne else pvar if lab == "parakeet_variation" else pgrid).mean(1).reshape(len(F0S), len(F1S))
        say(f"\n{lab}: rows f0, columns F1 " + " ".join(f"{f1:6.0f}" for f1 in F1S))
        for i, f0 in enumerate(F0S):
            say(f"  f0 {f0:4.0f} Hz                    " + " ".join(f"{x:6.3f}" for x in M[i]))
    lf = np.log2([f0 for f0, _ in cells])
    l1 = np.log2([f1 for _, f1 in cells])
    X = np.column_stack([np.ones(len(cells)), lf, l1])

    def fit(y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        return b, 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))

    rng = np.random.default_rng(0)
    readings = {}
    say("\nRule: NE = a + b_f log2 f0 + b_1 log2 F1 on the 25 cell means; rho = -b_1 / b_f")
    for lab in ne:
        y = ne[lab].mean(1)
        (a, bf, b1), r2 = fit(y)
        rho = -b1 / bf if bf != 0 else np.nan
        boots = []
        for _ in range(1000):
            yb = ne[lab][np.arange(len(cells))[:, None], rng.integers(0, VARIANTS, (len(cells), VARIANTS))].mean(1)
            (_, bfb, b1b), _ = fit(yb)
            boots.append(-b1b / bfb if bfb != 0 else np.nan)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rng_ne = y.max() - y.min()
        reading = ("no effect on controlled vowels" if rng_ne < 0.05 else "F1 / f0 drives it" if rho >= 0.67 else "f0 drives it" if rho <= 0.33 else "both")
        readings[lab] = reading
        say(f"  {lab}: NE range {rng_ne:.3f}; b_f {bf:+.3f} per octave of f0, b_1 {b1:+.3f} per octave of F1 (fit R2 {r2:.2f}); rho {rho:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {reading}")
    final = next(iter(readings.values())) if len(set(readings.values())) == 1 else "students disagree"
    say(f"\nREADING: {final}")
    out.close()
    with open(f"{PRE}_cells.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["model", "f0_hz", "f1_hz", "f1_over_f0", "ne", "ne_lo", "ne_hi"])
        for c, (f0, f1) in enumerate(cells):
            for lab, arr in list(ne.items()) + [("parakeet_variation", pvar), ("parakeet_grid_share", pgrid)]:
                w.writerow([lab, f0, f1, round(f1 / f0, 3), round(float(arr[c].mean()), 4), round(float(arr[c].min()), 4), round(float(arr[c].max()), 4)])


if __name__ == "__main__":
    main()
