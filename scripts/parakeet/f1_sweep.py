"""D-034, reported only (pre-registered 2026-10-24 in decisions.md): a fine F1 sweep at fixed f0. Sustained vowels as
fxf1_grid.py but with a +-1 % drift; f0 in F0S, F1 / f0 from 0.6 to 3.4 in steps of 0.1 (150 <= F1 <= 1500 Hz); four
variants per cell.
  A. Teacher v3's F1 (whole-file analyse, median over frames 30-370) against the cascade's F1, signed, in cents. Hannah's
     prediction (harmonic attraction) per coincidence F1 = k f0: signed error crosses zero from + to - (error at k - 0.2
     > 0 > error at k + 0.2, interpolated crossing within +-0.1 of k) and |error| at k below both cells at k +- 0.3.
  B. Students' normalised error against Parakeet (as fxf1_grid.py) peaks at k: the cell nearest k exceeds the mean of
     the cells at k +- 0.3 by more than twice the across-variant SD.
Readings over the (f0, k) pairs: >= 75 % -> confirmed, <= 25 % -> not, otherwise mixed.
Source via $VA_RD (default 1.0) and $VA_HNR (aspiration, dB HNR; default none).
Args: v4 cache dir, teacher checkpoint, out prefix, then LABEL=CKPT=STATS_DIR per student. Writes <prefix>.txt and
<prefix>_cells.csv. Run from the main checkout (VA_DEVICE=cuda)."""
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
from vocal_analysis.models.infer import analyse, load_teacher  # noqa: E402
from vocal_analysis.synth import tract, voice  # noqa: E402

F0S = (220.0, 300.0, 450.0, 600.0)
RATIOS = np.round(np.arange(6, 35) / 10, 1)
UPPER = (1700.0, 2800.0, 3800.0, 4800.0)
BWS = (80.0, 100.0, 140.0, 180.0, 220.0)
VARIANTS, E4, FS = 4, 49, 16000
NTOT = D.NS + 8 * HOP
RD = float(os.environ.get("VA_RD", 1.0))  # source: LF Rd (1.0 modal)
HNR = None if os.environ.get("VA_HNR") is None else float(os.environ["VA_HNR"])  # aspiration, dB; None = none


def render(f0, f1, v):
    rng = np.random.default_rng([int(f0), int(round(f1)), v, 4343])
    C = NTOT // voice.CTRL + 1
    t = np.arange(C) * voice.CTRL / FS
    f0t = f0 * (1 + 0.01 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi)))
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    pf[:, :5], pb[:, :5] = (f1, *UPPER), BWS
    traj = voice.Trajectories(f0=f0t, rd=np.full(C, RD), voice_amp=np.ones(C), whisper_amp=np.zeros(C), fric_amp=np.zeros(C),
                              poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(traj, rng, hnr_db=HNR)["total"][:NTOT]
    return (0.5 * y / np.abs(y).max()).astype(np.float32)


def main():
    C4, TCK, PRE, SPECS = Path(sys.argv[1]), sys.argv[2], sys.argv[3], [a.split("=", 2) for a in sys.argv[4:]]
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    vq = TorchVQT(dev)
    st = np.load(C4 / "stats.npz")
    MU4, SD4 = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    tg = np.load(C4 / "held_synth_v2" / "target.npy", mmap_mode="r")
    s1, s2, n = np.zeros(1024), np.zeros(1024), 0
    for i in range(0, len(tg), 32):
        t = ((np.asarray(tg[i:i + 32], np.float32) - st["mean"]) / st["std"]).reshape(-1, 1024).astype(np.float64)
        s1, s2, n = s1 + t.sum(0), s2 + (t ** 2).sum(0), n + len(t)
    VREF = torch.tensor(s2 / n - (s1 / n) ** 2, dtype=torch.float32, device=dev)
    teacher = load_teacher(TCK, dev)[0]
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

    cells = [(f0, r) for f0 in F0S for r in RATIOS if 150 <= f0 * r <= 1500]
    res = {}  # (f0, ratio) -> {name: (VARIANTS,) array}
    for f0, r in cells:
        f1 = f0 * r
        xs = [render(f0, f1, v) for v in range(VARIANTS)]
        out = {"teacher_cents": np.array([1200 * np.log2(np.nanmedian(analyse(teacher, x, dev)["formants"][30:371, 0]) / f1) for x in xs])}
        t = torch.stack([parakeet([x[q * HOP:] for x in xs]) for q in range(8)], 1)  # (V, 8, 49, 1024)
        out["parakeet_variation"] = (((t - t.mean((1, 2), keepdim=True)) ** 2).mean((1, 2)) / VREF).mean(-1).cpu().numpy()
        audio = torch.from_numpy(np.stack([x[:D.NS] for x in xs])).to(dev)
        for lab, m, MU, SD in students:
            with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16):
                y = m(vq.log_power_with_context(audio, D.CROP)).float()
            y = (y * SD + MU - MU4) / SD4
            err = torch.stack([y[:, j0 + q] - t[:, q] for q in range(8)], 1)
            out[lab] = ((err ** 2).mean((1, 2)) / VREF).mean(-1).cpu().numpy()
        res[f0, r] = out
        D.log(f"f0 {f0:.0f} F1/f0 {r:.1f} (F1 {f1:.0f}): teacher {out['teacher_cents'].mean():+.0f} cents; " + ", ".join(f"{lab} {out[lab].mean():.3f}" for lab, *_ in students))

    fo = open(f"{PRE}.txt", "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        fo.write(s + "\n")

    say(f"D-034 fine F1 sweep at fixed f0 (reported only; pre-registered 2026-10-24); source Rd {RD}, aspiration {HNR if HNR is not None else 'none'} dB HNR")
    names = ["teacher_cents"] + [lab for lab, *_ in students] + ["parakeet_variation"]
    for f0 in F0S:
        say(f"\nf0 {f0:.0f} Hz   F1/f0 " + " ".join(f"{name:>14s}" for name in names))
        for r in RATIOS:
            if (f0, r) in res:
                say(f"            {r:4.1f}  " + " ".join(f"{res[f0, r][name].mean():14.3f}" for name in names))

    def at(f0, r, name):
        r = round(r, 1)
        return res[f0, r][name] if (f0, r) in res else None

    pairs = [(f0, k) for f0 in F0S for k in (1, 2, 3) if all(at(f0, k + d, "teacher_cents") is not None for d in (-0.3, -0.2, 0.0, 0.2, 0.3))]
    say("\nA. teacher v3, harmonic attraction at F1 = k f0 (signed error crosses + to - within +-0.1 of k, |error| at k below k +- 0.3)")
    hits_a = []
    for f0, k in pairs:
        e = {round(d, 1): at(f0, k + d, "teacher_cents").mean() for d in np.round(np.arange(-3, 4) / 10, 1)}
        cross = e[-0.2] > 0 > e[0.2]
        zc = None
        if cross:
            ds = sorted(e)
            for a, b in zip(ds, ds[1:]):
                if e[a] > 0 >= e[b]:
                    zc = a + (b - a) * e[a] / (e[a] - e[b])
                    break
        ok_cross = cross and zc is not None and abs(zc) <= 0.1
        ok_min = abs(e[0.0]) < abs(e[-0.3]) and abs(e[0.0]) < abs(e[0.3])
        hits_a.append(ok_cross and ok_min)
        say(f"  f0 {f0:.0f} k {k}: error at k-0.3..k+0.3 " + " ".join(f"{e[d]:+.0f}" for d in sorted(e)) + f" cents; crossing {'%+.2f' % zc if zc is not None else 'none'}; "
            f"zero crossing {'yes' if ok_cross else 'no'}, |error| minimum {'yes' if ok_min else 'no'}")
    frac = np.mean(hits_a)
    read_a = "harmonic attraction" if frac >= 0.75 else "no attraction" if frac <= 0.25 else "mixed"
    say(f"  {sum(hits_a)} of {len(pairs)} pairs -> READING A: {read_a}")
    for k in (1, 2, 3):
        h = [x for (f, kk), x in zip(pairs, hits_a) if kk == k]
        if h:
            say(f"    k = {k}: {sum(h)} of {len(h)}")

    say("\nB. students: error peaks at F1 = k f0 (cell at k above the mean of k +- 0.3 by > 2 across-variant SD)")
    reads_b = {}
    for lab, *_ in students:
        hits = []
        for f0, k in pairs:
            c, lo, hi = at(f0, k, lab), at(f0, k - 0.3, lab), at(f0, k + 0.3, lab)
            ex = c.mean() - (lo.mean() + hi.mean()) / 2
            hits.append(ex > 2 * c.std(ddof=1))
            say(f"  {lab} f0 {f0:.0f} k {k}: at k {c.mean():.3f}, flanks {lo.mean():.3f} / {hi.mean():.3f}, excess {ex:+.3f} (2 SD {2 * c.std(ddof=1):.3f}) -> {'peak' if hits[-1] else 'no'}")
        f = np.mean(hits)
        reads_b[lab] = "coincidence peaks confirmed" if f >= 0.75 else "not confirmed" if f <= 0.25 else "mixed"
        say(f"  {lab}: {sum(hits)} of {len(pairs)} -> {reads_b[lab]}")
    say(f"  READING B: {next(iter(reads_b.values())) if len(set(reads_b.values())) == 1 else 'students differ: ' + ', '.join(f'{k} {v}' for k, v in reads_b.items())}")
    fo.close()
    with open(f"{PRE}_cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "f0_hz", "f1_over_f0", "f1_hz", "value", "sd"])
        for (f0, r), out in res.items():
            for name, arr in out.items():
                w.writerow([name, f0, r, round(f0 * r, 1), round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4)])


if __name__ == "__main__":
    main()
