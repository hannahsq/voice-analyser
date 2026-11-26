"""D-145: does a trunk that keeps the source place a wide F1 better? D-080's A1 (A0 plus the opening head, which learned
an Rd proxy) against A0, paired by seed, with 5b for reference. D-143 stage A's clips and probes. CPU, inference only.

    python scripts/analysis/d145_source_probe.py run      -> reports/analysis/d145/d145.{json,txt}
    python scripts/analysis/d145_source_probe.py report   -> reports/analysis/d145/d145.txt
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d145"
FIVE_B = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
A0 = ("d080_a0_s1", "d080_a0_s2")
A1 = ("d080_a1_s1", "d080_a1_s2")
MODELS = FIVE_B + A0 + A1
B = 2000


def _d143():
    spec = importlib.util.spec_from_file_location("d143", ROOT / "scripts/analysis/d143_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def clip_sets(d143):
    """D-143's clips exactly (same draws and seeds), with each clip's Rd kept; the grid's Rd is 1.0."""
    d142 = d143._d142()
    rng = np.random.default_rng(1430)
    sets = {"train": [], "test": [], "grid": []}
    rd = {"train": [], "test": [], "grid": []}
    for name, n, test, base in (("train", d143.N_TRAIN, False, 143000), ("test", d143.N_TEST, True, 243000)):
        for i in range(n):
            p = d143.draw(rng, test)
            sets[name].append((f"{name}{i}", d143.clip(p, base + i), np.array(p["F"])))
            rd[name].append(p["rd"])
    for sname, Fs in d142.SETS.items():
        for f0 in d142.F0S:
            for b1 in d142.B1S:
                for s in d143.GRID_SEEDS:
                    sets["grid"].append((f"{sname}|f{f0:g}|b{b1:g}|s{s}", d142._grid_audio(Fs, f0, b1, s), np.array(Fs)))
                    rd["grid"].append(1.0)
    return sets, {k: np.array(v) for k, v in rd.items()}


def _models():
    import torch

    from vocal_analysis.models.infer import load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    return dev, {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}


def _r2(p, y):
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def cmd_run() -> None:
    from vocal_analysis.synth.glottal import source_h_db

    d143 = _d143()
    sets, rd = clip_sets(d143)
    dev, models = _models()
    rng = np.random.default_rng(1450)
    res = {}
    for m, mod in models.items():
        tr, te, gr = (d143.features(mod, dev, sets[k]) for k in ("train", "test", "grid"))
        r = {}
        # (1) the source: log Rd and the source's H1-H2 per frame, from h
        for tname, f in (("log_rd", np.log), ("h1h2", lambda x: (lambda s: s[:, 0] - s[:, 1])(source_h_db(x, 2)))):
            ytr, yte = f(rd["train"][tr["cid"]]), f(rd["test"][te["cid"]])
            p, a = d143.fit(tr["H"], ytr, tr["cid"])
            perm = rng.permutation(len(rd["train"]))
            pn, _ = d143.fit(tr["H"], f(rd["train"][perm][tr["cid"]]), tr["cid"])
            r[tname] = {"r2": _r2(p(te["H"]), yte), "null_r2": _r2(pn(te["H"]), yte), "alpha": a}
        # (2) the wide F1, as D-143 stage A: per test clip, failure share (own heads) and the probe's median |error|
        ph, _ = d143.fit(tr["H"], tr["y"], tr["cid"])
        st = d143._clip_stats(te, ph(te["H"]))
        r["test_clips"] = {"fail": st[:, 0].tolist(), "h_err": st[:, 1].tolist(),
                           "heads_err": d143._clip_stats(te, te["C"][:, 0])[:, 1].tolist(),
                           "cid": np.unique(te["cid"]).tolist()}
        # (3) the grid with the model's own heads
        sg = d143._clip_stats(gr, gr["C"][:, 0])
        r["grid"] = {"keys": [sets["grid"][j][0] for j in np.unique(gr["cid"])], "fail": sg[:, 0].tolist()}
        res[m] = r
        print(m, flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d145.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    cmd_report(res)


def cmd_report(res=None) -> None:
    res = res or json.loads((REP / "d145.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1451)
    lines = ["D-145: the source in the frame vector, and a wide F1 from it; D-080's A1 (opening head) against A0, paired by "
             "seed; 5b for reference", "", "(1) The source from h (ridge probe, test clips; R2, and a clip-shuffled null's):"]
    for m in MODELS:
        lines.append(f"  {m:18s} log Rd {res[m]['log_rd']['r2']:.2f} (null {res[m]['log_rd']['null_r2']:.2f}); "
                     f"source H1-H2 {res[m]['h1h2']['r2']:.2f} (null {res[m]['h1h2']['null_r2']:.2f})")
    # the fixed clip set: test clips 5b fails (seed mean >= 0.5) among those every model scored
    tc = {m: {c: i for i, c in enumerate(res[m]["test_clips"]["cid"])} for m in MODELS}
    common = sorted(set.intersection(*(set(v) for v in tc.values())))
    col = lambda m, k: np.array([res[m]["test_clips"][k][tc[m][c]] for c in common])
    f5 = np.mean([col(m, "fail") for m in FIVE_B], 0)
    fix = f5 >= 0.5
    err = {m: col(m, "h_err")[fix] for m in MODELS}
    lines += ["", f"(2) A wide F1 from h on the {int(fix.sum())} test clips 5b fails (the same clips for every model): median "
              "over clips of the probe's median |F1 error| (cents); the model's own heads' error and failure share there"]
    for m in MODELS:
        he = col(m, "heads_err")[fix]
        fl = col(m, "fail")[fix]
        lines.append(f"  {m:18s} probe {np.median(err[m]):4.0f}; heads {np.median(he):4.0f}, failing (own heads) {np.mean(fl >= 0.5):.0%}")
    verdict = []
    lines.append("  A1 against A0, paired by seed (per-clip difference of the probe's error, median [95 % bootstrap over clips]):")
    for a0, a1 in zip(A0, A1):
        d = err[a1] - err[a0]
        bs = [np.median(d[i]) for i in [rng.integers(0, len(d), len(d)) for _ in range(B)]]
        ratio = np.median(err[a1]) / np.median(err[a0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict.append((ratio, hi))
        lines.append(f"    {a1} - {a0}: {np.median(d):+.0f} [{lo:+.0f}, {hi:+.0f}] c; ratio of medians {ratio:.2f}")
    ds = err[A0[1]] - err[A0[0]]
    bs = [np.median(ds[i]) for i in [rng.integers(0, len(ds), len(ds)) for _ in range(B)]]
    lines.append(f"    noise floor, A0 seed 2 - seed 1: {np.median(ds):+.0f} [{np.percentile(bs, 2.5):+.0f}, "
                 f"{np.percentile(bs, 97.5):+.0f}] c; ratio {np.median(err[A0[1]]) / np.median(err[A0[0]]):.2f}")
    strong = [r <= 0.75 and hi < 0 for r, hi in verdict]
    weak = [r <= 0.9 and hi < 0 for r, hi in verdict]
    if all(strong):
        out = "HELPS: a trunk that keeps the source places a wide F1 better; a source head is worth a training arm"
    elif not any(weak):
        out = "NO HELP: keeping the source (as D-080's head did) doesn't make a wide F1 readable"
    else:
        out = "PARTLY or SPLIT between seeds: reported, Hannah decides"
    lines += [f"  Reading (registered): {out}", "",
              "(3) The tipping grid (seeds 20-22) with each model's own heads: mean failure share in the primary cells "
              "(f0 110-140, B1 250-500) and overall"]
    for m in MODELS:
        keys = res[m]["grid"]["keys"]
        fl = np.array(res[m]["grid"]["fail"])
        f0 = np.array([float(k.split("|")[1][1:]) for k in keys])
        b1 = np.array([float(k.split("|")[2][1:]) for k in keys])
        cell = (f0 <= 140) & (b1 >= 250) & (b1 <= 500)
        lines.append(f"  {m:18s} primary {fl[cell].mean():.2f}; all {fl.mean():.2f}")
    (REP / "d145.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run()
    elif cmd == "report":
        cmd_report()
