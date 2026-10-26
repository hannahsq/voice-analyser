"""D-034 (pre-registered 2026-10-26; reported only): head-subspace fidelity. For each head (H5, S80, S4x80; six seeds
each) and each set of features (students and their noise controls, as scoring caches mirroring emb_v1), at the reference
frames: R^2 of the head's input projection (LayerNorm then the 1024 -> 128 map, interpolated to the frame as the head
does) against the same on Parakeet's layer 5 (per dimension, Parakeet's variance, mean over the 128); the whole-vector R^2
at the same frames; and the median |F1 - F1 on Parakeet| in cents, head alone. Groups: PTDB-TUG closed phase women / men,
Hillenbrand, synthetic v2 / v2s. Reading 1 (per student and head family, PTDB-TUG women): subspace R^2(student) minus
R^2(its noise), six seeds, 95 % t-interval.
Args: emb_v1 dir, out csv, then FEAT=dir ... and NAME#k=ckpt ... (FEAT names: v2 v2n v4 v4n). Run from the main checkout."""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head import interp_time, load_head  # noqa: E402

from vocal_analysis.models.infer import decode_outputs  # noqa: E402

E1, OUT = Path(sys.argv[1]), sys.argv[2]
FEATS = dict(a.split("=", 1) for a in sys.argv[3:] if "#" not in a.split("=", 1)[0])
HEADS = [a.split("=", 1) for a in sys.argv[3:] if "#" in a.split("=", 1)[0]]
SETS = ("gold_ptdb_cp", "gold_hillenbrand", "synth_train_v2", "synth_train_v2s")
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
torch.backends.cudnn.enabled = False


def load(d, name):
    return np.load(Path(d) / name / "emb.npy", mmap_mode="r"), np.load(Path(d) / name / "emb_off.npy")


def run(m, emb, eo, mel, ro):
    """per reference frame: projection (N, 128), raw layer 5 (N, 1024) and log2 F1"""
    Z, X, F = [], [], []
    for i in range(len(eo) - 1):
        e = torch.from_numpy(np.asarray(emb[eo[i]:eo[i + 1], 5], np.float32)).to(dev)[None]
        mp = torch.from_numpy(np.asarray(mel[ro[i]:ro[i + 1]], np.float32)).to(dev)
        with torch.inference_mode():
            Z.append(interp_time(m.proj(m.norm(e)), mp, m.ratio, m.offset)[0].float().cpu().numpy())
            X.append(interp_time(e, mp, m.ratio, m.offset)[0].float().cpu().numpy())
            d = decode_outputs({k: v[0].float().cpu().numpy() for k, v in m(e, mp).items()})
        F.append(np.log2(d["formants"][:, 0]))
    return np.concatenate(Z), np.concatenate(X), np.concatenate(F)


def r2(a, ref):
    v = ref.var(0)
    k = v > 1e-12
    return float((1 - ((a - ref) ** 2).mean(0)[k] / v[k]).mean())


rows = []
for name in SETS:
    ro, mel = np.load(E1 / name / "ref_off.npy"), np.load(E1 / name / "mel.npy")
    groups = {"all": slice(None)}
    if name == "gold_ptdb_cp":
        sex = np.load(E1 / name / "sex.npy")
        groups = {"women": sex == "f", "men": sex == "m"}
    pe, po = load(E1, name)
    fe = {f: load(d, name) for f, d in FEATS.items()}
    for f, (_, o) in fe.items():
        assert np.array_equal(o, po), (f, name)
    for hn, ck in HEADS:
        m = load_head(ck, dev)
        Zp, Xp, Fp = run(m, pe, po, mel, ro)
        for f, (emb, eo) in fe.items():
            Zf, Xf, Ff = run(m, emb, eo, mel, ro)
            for g, k in groups.items():
                d = np.abs(Ff[k] - Fp[k]) * 1200
                d = d[np.isfinite(d)]
                rows.append([f, hn.split("#")[0], hn.split("#")[1], name, g, round(r2(Zf[k], Zp[k]), 4), round(r2(Xf[k], Xp[k]), 4), round(float(np.median(d)), 1)])
        print(f"[{name}] {hn} done", flush=True)
with open(OUT, "w", newline="") as fo:
    csv.writer(fo).writerows([["features", "heads", "seed", "set", "group", "subspace_r2", "whole_r2", "f1_med_cents"], *rows])

# summary: six-seed means; reading 1
T = defaultdict(list)
for f, h, s, name, g, zr, xr, c in rows:
    T[(f, h, name, g)].append((zr, xr, c))
print("\nsix-seed means: subspace R2 / whole-vector R2 / median |F1 - F1 on Parakeet| (cents)")
for (f, h, name, g), v in sorted(T.items()):
    a = np.array(v)
    print(f"  {f:4s} {h:6s} {name:17s} {g:6s} {a[:, 0].mean():.3f} / {a[:, 1].mean():.3f} / {a[:, 2].mean():6.1f}")
print("\nReading 1 (PTDB-TUG women): subspace R2(student) - R2(noise), six seeds paired")
for st in ("v2", "v4"):
    for h in sorted({r[1] for r in rows}):
        a = np.array([x[0] for x in T.get((st, h, "gold_ptdb_cp", "women"), [])])
        b = np.array([x[0] for x in T.get((st + "n", h, "gold_ptdb_cp", "women"), [])])
        if len(a) == 0 or len(a) != len(b):
            continue
        dd = a - b
        half = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
        lo, hi = dd.mean() - half, dd.mean() + half
        v = "error kept out of the head's subspace" if lo > 0 else "error concentrated in it" if hi < 0 else "like noise"
        print(f"  {st} x {h}: {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v}")
