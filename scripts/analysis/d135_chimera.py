"""D-135: the tract-length head (and F1-F3) on CHIMERA renders: Story's ten shapes through CHIMERA's airway (its area
override, a local fork), and CHIMERA's own articulatory postures. Labels are CHIMERA's closed-glottis resonances.

Needs the local CHIMERA label driver (not in this repo; MIT, `docs/considering.md`), path in $CHIMERA_DRIVER.

    python scripts/analysis/d135_chimera.py render             -> data/probes/chimera_v1/{story,posture}/
    python scripts/analysis/d135_chimera.py score NAME CKPT    -> reports/probes/chimera/NAME.npz ($VA_DEVICE)
    python scripts/analysis/d135_chimera.py analyse            -> reports/analysis/d135/d135.txt
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/probes/chimera_v1"
RAW = ROOT / "reports/probes/chimera"
OUT = ROOT / "reports/analysis/d135"
DRIVER = Path(os.environ.get("CHIMERA_DRIVER", ROOT / "scratchpad/chimera_driver/target/release/chimera-driver.exe"))
FS_RENDER = 48000
LENGTHS = (13.0, 14.5, 15.6, 17.0, 18.5)
F0S = (110.0, 180.0, 280.0)
N_POSTURES = 300
SECS = 1.5
EDGE = 20
LO = (1.2, 5.0, 0.05, 0.12, 0.0, 0.0, -0.9, -1.3, 0.0)  # tbX tbY jaw lipAp lipProt lipWidth root larynx curl
HI = (4.6, 8.8, 2.3, 2.0, 1.4, 1.0, 0.9, 1.3, 1.0)
PNAMES = ("tbX", "tbY", "jaw", "lipAp", "lipProt", "lipWidth", "root", "larynx", "curl")
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")
BACK = ("o", "u", "U", "O")
B = 1000


def _tag(v):
    return "".join(c + "^" if c.isupper() else c for c in v)  # Windows names ignore case


def _female(L):
    return float(np.clip((17.5 - L) / 3.0, 0.0, 1.0))


def cmd_render() -> None:
    from scipy.signal import resample_poly

    from vocal_analysis.synth import tube

    rng = np.random.default_rng(135)
    jobs = {"story": [], "posture": []}
    meta = {"story": [], "posture": []}
    for L in LENGTHS:
        for v, q in tube.STORY_VOWELS.items():
            A = tube.vowel_areas(*q)
            for f0 in F0S:
                name = f"L{L:g}_{_tag(v)}_f{f0:g}"
                jobs["story"].append(f"{name},story,{_female(L):.4f},{f0:g},{SECS},{L:g}," + ",".join(f"{a:.6f}" for a in A))
                meta["story"].append({"name": name, "length": L, "vowel": v, "f0": f0})
    for i in range(N_POSTURES):
        p = [rng.uniform(lo, hi) for lo, hi in zip(LO, HI)]
        female = rng.uniform(0, 1)
        f0 = float(rng.choice(F0S))
        name = f"p{i:03d}"
        jobs["posture"].append(f"{name},posture,{female:.4f},{f0:g},{SECS}," + ",".join(f"{x:.5f}" for x in p))
        meta["posture"].append({"name": name, "female": female, "f0": f0, **dict(zip(PNAMES, p))})
    for arm in ("story", "posture"):
        d = CACHE / arm
        if (d / "meta.json").exists():
            continue
        tmp = d / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "jobs.csv").write_text("name,kind,female,f0,secs,vals\n" + "\n".join(jobs[arm]) + "\n", encoding="utf-8")
        subprocess.run([str(DRIVER), "jobs", str(tmp / "jobs.csv"), str(tmp), str(FS_RENDER)], check=True,
                       stderr=subprocess.DEVNULL)
        labels = {r["name"]: r for r in csv.DictReader(open(tmp / "labels.csv", encoding="utf-8"))}
        for it in meta[arm]:
            x = np.fromfile(tmp / f"{it['name']}.f32", dtype=np.float32).astype(np.float64)
            y = resample_poly(x, 1, FS_RENDER // 16000)
            n = len(y)
            fade = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / 320.0, 0, 1)
            y = 0.5 * y * fade / (np.abs(y * fade).max() + 1e-12)
            r = labels[it["name"]]
            it.update({k: float(r[k]) for k in ("F1", "F2", "F3", "F4", "length_cm", "min_area_cm2")})
            np.savez_compressed(d / f"{it['name']}.npz", audio=np.round(y * 32000).astype(np.int16))
            (tmp / f"{it['name']}.f32").unlink()
        (d / "meta.json").write_text(json.dumps({"items": meta[arm]}, indent=1))
        print(f"{arm}: {len(meta[arm])} clips", flush=True)


def cmd_score(name: str, ckpt: str) -> None:
    import torch

    from vocal_analysis.models.infer import DEFAULT_WINDOW, analyse, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ckpt, dev)
    causal = bool(model.cfg.causal)
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for arm in ("story", "posture"):
        items = json.loads((CACHE / arm / "meta.json").read_text())["items"]
        est = np.full((len(items), 5), np.nan)
        for i, it in enumerate(items):
            d = np.load(CACHE / arm / f"{it['name']}.npz")
            r = analyse(model, d["audio"].astype(np.float32) / 32000, dev, window_frames=None if causal else DEFAULT_WINDOW)
            s = slice(EDGE, len(r["sonorant_p"]) - EDGE)
            est[i] = ([np.nanmedian(r["vtl_log2cm"][s])] + [np.nanmedian(np.log2(r["formants"][s, k])) for k in range(3)]
                      + [np.mean(r["sonorant_p"][s] > 0.5)])
        out[arm] = est
    np.savez_compressed(RAW / f"{name}.npz", **out)
    print(f"{name}: scored")


def cmd_analyse() -> None:
    from vocal_analysis.synth import tube

    rng = np.random.default_rng(1350)
    est = {n: np.load(RAW / f"{n}.npz") for n in TEACHERS + STUDENTS}
    groups = [(n, [n]) for n in TEACHERS] + [("teacher (seed mean)", list(TEACHERS)), ("students (mean of four)", list(STUDENTS))]
    lines = ["D-135: CHIMERA renders; length bias = median estimate - CHIMERA's length (cm); F1-F3 error in cents against "
             "CHIMERA's closed-glottis resonances", ""]
    # arm 1: Story's shapes
    items = json.loads((CACHE / "story" / "meta.json").read_text())["items"]
    vowels = list(tube.STORY_VOWELS)
    V = np.array([it["vowel"] for it in items])
    L = np.array([it["length_cm"] for it in items])
    T = np.array([[it["F1"], it["F2"], it["F3"]] for it in items])
    cellkey = np.array([f"{it['length']}_{it['f0']}" for it in items])
    cells = np.unique(cellkey)
    back = np.isin(V, BACK)
    lines.append(f"Arm 1, Story's ten shapes through CHIMERA: {len(items)} clips")
    for gname, names in groups:
        bias = np.mean([2 ** est[n]["story"][:, 0] - L for n in names], 0)
        per_v = {v: np.mean(bias[V == v]) for v in vowels}
        c0 = np.mean(bias[back]) - np.mean(bias[~back])
        boots = []
        for _ in range(B):
            w = np.zeros(len(items))
            for c in rng.choice(cells, len(cells)):
                w[cellkey == c] += 1
            boots.append(np.sum(w * bias * back) / np.sum(w * back) - np.sum(w * bias * ~back) / np.sum(w * ~back))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        lines.append(f"  {gname:24s} mean {bias.mean():+.2f}; range {max(per_v.values()) - min(per_v.values()):.2f} cm; back - others "
                     f"{c0:+.2f} [{lo:+.2f}, {hi:+.2f}]; by vowel " + ", ".join(f"{v} {per_v[v]:+.2f}" for v in vowels))
    lines.append("  F1-F3 median error (cents), teacher seed mean / students, by vowel:")
    for v in vowels:
        m = V == v
        parts = []
        for k in range(3):
            te = np.median(np.mean([1200 * (est[n]["story"][m, k + 1] - np.log2(T[m, k])) for n in TEACHERS], 0))
            st = np.median(np.mean([1200 * (est[n]["story"][m, k + 1] - np.log2(T[m, k])) for n in STUDENTS], 0))
            parts.append(f"F{k + 1} {te:+5.0f} / {st:+5.0f}")
        lines.append(f"    {v:3s} " + " ".join(parts))
    lines.append("")
    # arm 2: CHIMERA's postures
    items = json.loads((CACHE / "posture" / "meta.json").read_text())["items"]
    ok = np.array([it["min_area_cm2"] >= 0.1 for it in items])
    items = [it for it, k in zip(items, ok) if k]
    L = np.array([it["length_cm"] for it in items])
    T = np.array([[it["F1"], it["F2"], it["F3"]] for it in items])
    X = np.array([[it[p] for p in PNAMES] for it in items])
    lines.append(f"Arm 2, CHIMERA's own postures: {len(items)} of {len(ok)} open (narrowest area >= 0.1 cm^2); lengths "
                 f"{L.min():.1f}-{L.max():.1f} cm")
    Z = np.column_stack([np.ones(len(L)), L - L.mean(), (X - X.mean(0)) / X.std(0)])
    for gname, names in groups:
        bias = np.mean([2 ** est[n]["posture"][ok][:, 0] - L for n in names], 0)
        coef = np.linalg.lstsq(Z, bias, rcond=None)[0]
        boots = []
        for _ in range(B):
            i = rng.integers(0, len(L), len(L))
            boots.append(np.linalg.lstsq(Z[i], bias[i], rcond=None)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        span = {p: coef[2 + j] * 2.56 for j, p in enumerate(PNAMES)}  # 10th to 90th percentile of a uniform, in SDs
        lines.append(f"  {gname:24s} mean bias {bias.mean():+.2f} cm; per cm of true length {coef[1]:+.2f}; change in bias from "
                     "the 10th to 90th percentile of each articulator [95 % bootstrap]:")
        lines.append("    " + "; ".join(f"{p} {span[p]:+.2f} [{lo[2 + j] * 2.56:+.2f}, {hi[2 + j] * 2.56:+.2f}]" for j, p in enumerate(PNAMES)))
    te = [np.median(np.mean([1200 * (est[n]["posture"][ok][:, k + 1] - np.log2(T[:, k])) for n in TEACHERS], 0)) for k in range(3)]
    st = [np.median(np.mean([1200 * (est[n]["posture"][ok][:, k + 1] - np.log2(T[:, k])) for n in STUDENTS], 0)) for k in range(3)]
    lines.append("  F1-F3 median error (cents), teacher / students: " + ", ".join(f"F{k + 1} {te[k]:+.0f} / {st[k]:+.0f}" for k in range(3)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d135.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        cmd_render()
    elif cmd == "score":
        cmd_score(sys.argv[2], sys.argv[3])
    elif cmd == "analyse":
        cmd_analyse()
