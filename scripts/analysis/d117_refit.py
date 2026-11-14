"""D-117: does refitting the VTL head on a branch arm's frozen trunk recover the Hillenbrand length reliability its
joint head lost (D-114 to D-116)? Heads: D-037's refit (d037_head.py, three head seeds, seed-mean estimator as
d037_score.py); trunks cached by d037_cache.py (parts train, val, probe, hb).

    check:  python scripts/analysis/d117_refit.py check <old cache dir> <new cache dir>    (a few crops rebuilt today)
    export: python scripts/analysis/d117_refit.py export name=<trunk cache>[:<head prefix>] ...
            -> cache/d117/refit/<name>.npz (with a head prefix: the refits' seed mean) or cache/d117/joint/<name>.npz
               (without: the trunk's own joint head, from the cache), in d115_reliability.py's per-frame format
    report: python scripts/analysis/d117_refit.py report > reports/analysis/d117/d117.txt
Report: the precondition (the joint heads from the caches against vtl_score_joint.py's reliabilities); per arm (d114,
d109, f5) against 5b, seeds paired, seed means, 2000-draw speaker bootstrap: reliability change (mean of men and women)
for the joint heads and for the refits; then D-115's decomposition on the refits (d115_reliability.py report).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("cache/d117")
JOINT_REPORTED = {"d040_5b_s1_b20k": 0.228, "d040_5b_s2_b20k": 0.261, "d114_s1": 0.215, "d114_s2": 0.236,
                  "d109_s1": 0.223, "d109_s2": 0.241, "d090_f5_s1": 0.234, "d090_f5_s2": 0.261}


def check(old, new):
    """d037_cache.py's train part rebuilt today (VA_LIMIT crops) against the old cache's first crops."""
    old, new = Path(old) / "train", Path(new) / "train"
    for f in sorted(p.name for p in new.iterdir() if p.suffix == ".npy"):
        a = np.load(new / f, mmap_mode="r")
        b = np.load(old / f, mmap_mode="r")[:len(a)]
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        same = np.array_equal(a, b, equal_nan=True)
        d = np.nanmax(np.abs(a - b)) if a.size else 0.0
        print(f"  {f}: shape {a.shape}, identical {same}, max |diff| {d:.3g}")


def export(pairs):
    import d037_score as S

    for name, spec in pairs.items():
        t, _, prefix = spec.partition(":")
        t = Path(t)
        z = np.load(t / "hb" / "files.npz")
        if prefix:
            f, F = np.load(t / "hb" / "feats.npy"), np.load(t / "hb" / "log2F.npy")
            mu, var = S.refit_frames(S.load_heads(prefix), f.reshape(-1, 128), F.reshape(-1, 4))
            d = OUT / "refit"
        else:
            mu, var = np.load(t / "hb" / "vtl_mu.npy").reshape(-1), np.exp(np.load(t / "hb" / "vtl_logvar.npy")).reshape(-1)
            d = OUT / "joint"
        d.mkdir(parents=True, exist_ok=True)
        f115 = np.load("cache/d115/files.npz")
        assert np.array_equal(f115["spk"], z["speaker"]), "file order differs from D-115's"
        if not (d / "files.npz").exists():
            np.savez(d / "files.npz", **{k: f115[k] for k in f115.files})
        np.savez(d / f"{name}.npz", mu=mu.reshape(-1), sigma=np.sqrt(var.reshape(-1)), off=z["off"])
        print(f"{name}: {len(z['off']) - 1} files -> {d}")


def report():
    import d115_reliability as D

    rng = np.random.default_rng(20261117)
    print("D-117: the VTL head refitted on each branch arm's frozen trunk (D-037's recipe, three head seeds) against 5b's refits")
    out = {}
    for kind in ("joint", "refit"):
        D.CACHE = OUT / kind
        f = np.load(D.CACHE / "files.npz")
        fs, fv = f["spk"], f["vowel"]
        spks = {G: sorted(set(fs[f["group"] == G])) for G in "mw"}
        names = [n for n in list(D.BASE.values()) + [p.format(s) for p in D.ARMS.values() for s in (1, 2)] if (D.CACHE / f"{n}.npz").exists()]
        E = {n: D.per_file(n) for n in names}

        def rel(e, draw):
            return np.mean([D.parts(e, fs, fv, draw[G])["rel"] for G in "mw"])
        pt = {n: rel(E[n], spks) for n in names}
        if kind == "joint":
            print("\nPrecondition, joint heads from the trunk caches against vtl_score_joint.py: " + "; ".join(
                f"{n} {pt[n]:.3f} ({JOINT_REPORTED[n]:.3f})" for n in names if n in JOINT_REPORTED))
        print(f"\n{kind} heads, reliability (mean of men and women): " + "; ".join(f"{n} {pt[n]:.3f}" for n in names))
        boots = [{G: [spks[G][j] for j in rng.integers(0, len(spks[G]), len(spks[G]))] for G in "mw"} for _ in range(2000)]
        for arm, pat in D.ARMS.items():
            if not all(pat.format(s) in E for s in (1, 2)):
                continue
            d = lambda draw: np.mean([rel(E[pat.format(s)], draw) - rel(E[D.BASE[s]], draw) for s in (1, 2)])  # noqa: E731
            bs = [d(b) for b in boots]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            per = [rel(E[pat.format(s)], spks) - rel(E[D.BASE[s]], spks) for s in (1, 2)]
            out[(kind, arm)] = (d(spks), lo, hi)
            print(f"  {arm} - 5b ({kind}): {d(spks):+.3f} [{lo:+.3f}, {hi:+.3f}]; seed 1 {per[0]:+.3f}, seed 2 {per[1]:+.3f}")
    print("\nSummary (registered readings use d114; f5 is the null arm's refit noise; d109 reported):")
    for arm in ("d114", "d109", "f5"):
        if ("refit", arm) in out and ("joint", arm) in out:
            j, r = out[("joint", arm)], out[("refit", arm)]
            print(f"  {arm}: joint {j[0]:+.3f} [{j[1]:+.3f}, {j[2]:+.3f}] -> refit {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]; "
                  f"refit passes the guard's bar (lower bound > -0.02): {r[1] > -0.02}; at least half the joint cost recovered: {r[0] >= j[0] / 2}")
    sys.stdout.flush()
    print("\n---- D-115's decomposition on the refits ----", flush=True)
    env = {**os.environ, "D115_CACHE": str(OUT / "refit"), "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, str(Path(__file__).parent / "d115_reliability.py"), "report"], env=env, check=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "export":
        export(dict(a.split("=", 1) for a in sys.argv[2:]))
    else:
        report()
