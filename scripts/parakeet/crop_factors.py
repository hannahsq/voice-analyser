"""D-034, reported only: which held-out synthetic crops do the students reproduce Parakeet badly on, and why. Synthetic
crops carry their generator labels, so each crop of held_synth_v2 / held_synth_v2s (distil_cache_v4.py's draws, checked
against the cached audio) gets voice features, and they are set against the per-crop R^2 from phase_spread.py (512 crops,
eight 10 ms phases) and subhop.py (first 128 crops, forty 2 ms offsets), and against Parakeet's own grid instability (the
crop's fixed 8-periodic pattern, as teacher_ceiling.py, per phase).
  1. Students agree? Spearman of crop mean R^2, v2 vs v4; Pearson of each crop's offset profile (deviation from its mean),
     v2 vs v4, median over crops, against the same with the crops shuffled.
  2. Parakeet's grid: per crop, correlation over phases between a student's R^2 deviation and Parakeet's grid-pattern
     energy at that phase (share of the crop's variance).
  3. Features: Spearman of crop mean R^2 with each feature; mean R^2 by f0 band and by F1 / f0; the worst tenth for both
     students against the rest.
Args: v4 cache dir, phase_spread csv, subhop crops csv, out prefix. Writes <prefix>.txt and <prefix>_crops.csv (one row per
crop and set: features, R^2 per student). Run from the main checkout."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_cache as D  # noqa: E402

from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import SynthCrops, split_ids  # noqa: E402

C4, PS, SH, PRE = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
SETS = ("held_synth_v2", "held_synth_v2s")
N = 512


class Labels(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return N

    def __getitem__(self, i):
        x, lab = self.ds[i]
        return x, lab


def features(lab):
    v = lab["voiced"].astype(bool)
    f0, F = lab["f0"], lab["formants"]
    out = {"voiced_frac": float(v.mean()), "sonorant_frac": float(lab["sonorant"].astype(bool).mean()),
           "whisper_frac": float(lab["whisper"].astype(bool).mean())}
    if v.sum() >= 10:
        out |= {"f0_hz": float(np.median(f0[v])), "f1_hz": float(np.median(F[v, 0])), "f2_hz": float(np.median(F[v, 1])),
                "b1_hz": float(np.median(lab["bandwidths"][v, 0])), "f1_over_f0": float(np.median(F[v, 0] / f0[v])),
                "rd": float(np.median(lab["rd"][v]))}
    return out


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1]), int(ok.sum())


def prof_corr(A, B):
    """row-wise Pearson of two (crops, offsets) arrays after removing each row's mean"""
    a, b = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    flat = Path(cfg3["train"]["data"]) / "flat"
    _, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    v2s = Path("data/synth/train_v2s")
    base = {"held_synth_v2": SynthCrops(flat, val_ids[:512], D.CROP, channel, seed=12345, fixed=True),
            "held_synth_v2s": SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                         D.CROP, channel, seed=12345, fixed=True)}
    st = np.load(C4 / "stats.npz")
    MU, SD = st["mean"], st["std"]

    r10 = defaultdict(lambda: np.full((N, 8), np.nan))
    for r in csv.DictReader(open(PS)):
        r10[(r["student"], r["set"])][int(r["crop"]), int(r["phase"])] = float(r["r2"])
    r2ms = defaultdict(dict)
    for r in csv.DictReader(open(SH)):
        r2ms[(r["model"], r["set"])].setdefault(int(r["crop"]), np.full(40, np.nan))[int(r["offset_ms"]) // 2] = float(r["r2"])

    out = open(f"{PRE}.txt", "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")

    rows = []
    say("D-034 crop factors: v2 step 12,000 and v4 step 5,000 on the held-out synthetic crops (reported only)")
    for name in SETS:
        say(f"\n=== {name}")
        audio = np.load(C4 / name / "audio.npy", mmap_mode="r")
        feats, worst_diff = [], 0.0
        dl = torch.utils.data.DataLoader(Labels(base[name]), batch_size=16, num_workers=6, collate_fn=D.as_list)
        i = 0
        for b in dl:
            for x, lab in b:
                worst_diff = max(worst_diff, float(np.abs(x - np.asarray(audio[i], np.float32)).max()))
                feats.append(features(lab))
                i += 1
        say(f"labels for {i} crops; audio against the cache: max |difference| {worst_diff:.4f} (fp16 storage)")

        # Parakeet's grid pattern per crop and phase, as a share of the crop's variance
        tgt = np.load(C4 / name / "target.npy", mmap_mode="r")
        grid = np.zeros((N, 8))
        for s in range(0, N, 32):
            t = (np.asarray(tgt[s:s + 32], np.float32) - MU) / SD  # (b, 8, 49, 1024)
            comb = t.mean(2) - t.mean((1, 2))[:, None]  # (b, 8, 1024)
            tot = ((t - t.mean((1, 2), keepdims=True)) ** 2).sum((1, 2, 3))  # (b,)
            grid[s:s + len(t)] = 49 * (comb ** 2).sum(-1) / tot[:, None] * 8
        say(f"Parakeet's grid pattern: median share of a crop's variance {np.median(grid.mean(1)):.4f}, 90th percentile {np.percentile(grid.mean(1), 90):.4f}")

        A2, A4 = r10[("v2", name)], r10[("v4", name)]
        m2, m4 = A2.mean(1), A4.mean(1)
        say("\n1. Do the students agree?")
        say(f"  crop mean R2, v2 vs v4: Spearman {spearman(m2, m4)[0]:+.3f} (512 crops)")
        rng = np.random.default_rng(0)
        pc = prof_corr(A2, A4)
        sh = prof_corr(A2, A4[rng.permutation(N)])
        say(f"  10 ms phase profiles (8 phases), v2 vs v4 same crop: median r {np.median(pc):+.3f}; crops shuffled {np.median(sh):+.3f}")
        if ("v2", name) in r2ms:
            P2, P4 = (np.array([r2ms[(s, name)][c] for c in sorted(r2ms[(s, name)])]) for s in ("v2", "v4"))
            pc2 = prof_corr(P2, P4)
            sh2 = prof_corr(P2, P4[rng.permutation(len(P4))])
            say(f"  2 ms offset profiles (40 offsets, {len(P2)} crops): median r {np.median(pc2):+.3f}; crops shuffled {np.median(sh2):+.3f}")

        say("\n2. Do the students' bad phases line up with Parakeet's grid pattern?")
        for lab_, A in (("v2", A2), ("v4", A4)):
            cg = prof_corr(A, grid)
            say(f"  {lab_}: median over crops of r(R2 deviation, grid energy) across phases {np.median(cg):+.3f}; "
                f"crop mean R2 vs crop grid share Spearman {spearman(A.mean(1), grid.mean(1))[0]:+.3f}")

        say("\n3. What do the hard crops have in common?")
        keys = ["voiced_frac", "sonorant_frac", "whisper_frac", "f0_hz", "f1_hz", "f2_hz", "b1_hz", "f1_over_f0", "rd"]
        fv = {k: np.array([f.get(k, np.nan) for f in feats]) for k in keys}
        say("  Spearman with crop mean R2      v2       v4    v4-v2 gain")
        for k in keys:
            say(f"    {k:14s}             {spearman(fv[k], m2)[0]:+.3f}   {spearman(fv[k], m4)[0]:+.3f}   {spearman(fv[k], m4 - m2)[0]:+.3f}   (n {spearman(fv[k], m2)[1]})")
        say("  mean R2 by median voiced f0      n     v2     v4")
        for lo, hi in ((0, 150), (150, 250), (250, 400), (400, 2000)):
            k = (fv["f0_hz"] >= lo) & (fv["f0_hz"] < hi)
            if k.sum():
                say(f"    {lo:4d}-{hi:<4d} Hz              {k.sum():4d}  {m2[k].mean():.3f}  {m4[k].mean():.3f}")
        say("  mean R2 by median F1 / f0 (harmonics per F1)")
        for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 4), (4, 100)):
            k = (fv["f1_over_f0"] >= lo) & (fv["f1_over_f0"] < hi)
            if k.sum():
                say(f"    {lo:4.1f}-{hi:<5.1f}                 {k.sum():4d}  {m2[k].mean():.3f}  {m4[k].mean():.3f}")
        worst = (m2 <= np.percentile(m2, 10)) & (m4 <= np.percentile(m4, 10))
        say(f"  worst tenth for both students: {worst.sum()} crops; medians, those crops / the rest")
        for k in keys:
            say(f"    {k:14s} {np.nanmedian(fv[k][worst]):9.3f} / {np.nanmedian(fv[k][~worst]):9.3f}")
        say("  Parakeet's grid share by median voiced f0 (median over crops)")
        for lo, hi in ((0, 150), (150, 250), (250, 400), (400, 2000)):
            k = (fv["f0_hz"] >= lo) & (fv["f0_hz"] < hi)
            if k.sum():
                say(f"    {lo:4d}-{hi:<4d} Hz              {k.sum():4d}  {np.median(grid[k].mean(1)):.4f}")
        for c in range(N):
            rows.append({"set": name, "crop": c, **{k: round(float(fv[k][c]), 4) for k in keys}, "grid_share": round(float(grid[c].mean()), 5),
                         "r2_v2": round(float(m2[c]), 4), "r2_v4": round(float(m4[c]), 4), "worst_both": int(worst[c])})
    # the two sets render the same validation clips (train_v2s re-renders train_v2's voices with the interactive source),
    # so crop i is the same voice in both: compare them crop by crop
    say("\n=== held_synth_v2 against held_synth_v2s, crop by crop")
    by = {s: [r for r in rows if r["set"] == s] for s in SETS}
    for k in ("f0_hz", "voiced_frac", "rd", "f1_hz"):
        x, y = (np.array([r[k] for r in by[s]], float) for s in SETS)
        ok = np.isfinite(x) & np.isfinite(y)
        say(f"  {k:12s} correlation {np.corrcoef(x[ok], y[ok])[0, 1]:.3f}, identical {np.mean(np.isclose(x[ok], y[ok], rtol=1e-3)):.0%}")
    rng = np.random.default_rng(1)
    for s1 in ("v2", "v4"):
        say(f"  crop mean R2, {s1}: Spearman {spearman(r10[(s1, SETS[0])].mean(1), r10[(s1, SETS[1])].mean(1))[0]:+.3f}")
    for s1 in ("v2", "v4"):
        for s2 in ("v2", "v4"):
            A, B = r10[(s1, SETS[0])], r10[(s2, SETS[1])]
            say(f"  10 ms phase profiles, {s1} on v2 vs {s2} on v2s: median r {np.median(prof_corr(A, B)):+.3f}; shuffled {np.median(prof_corr(A, B[rng.permutation(N)])):+.3f}")
            if (s1, SETS[0]) in r2ms and (s2, SETS[1]) in r2ms:
                P, Q = (np.array([r2ms[key][c] for c in sorted(r2ms[key])]) for key in ((s1, SETS[0]), (s2, SETS[1])))
                say(f"  2 ms offset profiles, {s1} on v2 vs {s2} on v2s: median r {np.median(prof_corr(P, Q)):+.3f}; shuffled {np.median(prof_corr(P, Q[rng.permutation(len(Q))])):+.3f}")
    with open(f"{PRE}_crops.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.close()


if __name__ == "__main__":
    main()
