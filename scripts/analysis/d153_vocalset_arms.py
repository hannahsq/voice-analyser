"""D-153: D-080's coupled arms and D-098's arms on D-152's measure (VocalSet's held-out men, f0 <= 150 Hz, D-141's cell
by Praat), on D-152's fixed frames. CPU, inference only.

    python scripts/analysis/d153_vocalset_arms.py run [MODEL,MODEL,...]   -> cache/d153/<model>.npz (one process per list)
    python scripts/analysis/d153_vocalset_arms.py report                  -> reports/analysis/d153/d153.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d153"
CACHE = ROOT / "cache/d153"
D152 = ROOT / "cache/d152"
REF = "d040_5b_s2_b20k"  # D-152's frame selection (voicing, f0)
NULL = ("d040_5b", "d090_f5", "d090_f57")
COUPLED = ("d080_a0", "d080_a1")
D098 = ("d098_base_b20k", "d098_heads_b20k")
FS = 16000
MIN_FRAMES = 10


def run_name(arm, s):
    return f"d040_5b_s{s}_b20k" if arm == "d040_5b" else f"{arm}_s{s}"


NEW = [run_name(a, s) for a in NULL[1:] + COUPLED for s in (1, 2)] + list(D098)


def _units():
    """D-152's primary files and frames: [(singer, file meta, frame indices into the file's frames, Praat F1-F3)]."""
    out = []
    for p in sorted(D152.glob("male*.npz")):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        P, f0, fid = z["P"], z["f0"], z["fid"]
        v = z[f"{REF}|v"] > 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = v & np.all(np.isfinite(P), 1) & (P[:, 0] >= 350) & (P[:, 0] <= 650) & (P[:, 1] / P[:, 0] < 2.2) & (f0 <= 150)
        for u, mt in enumerate(meta):
            k = fid == u
            s = cell[k]
            if s.sum() >= MIN_FRAMES:
                out.append({"singer": p.stem, "meta": mt, "idx": np.where(s)[0], "P": P[k][s],
                            "ref": {m: z[f"{m}|F"][k][s] for m in ("d040_5b_s1_b20k", "d040_5b_s2_b20k")}})
    return out


def cmd_run(models: list[str]) -> None:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import analyse, load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    units = _units()
    CACHE.mkdir(parents=True, exist_ok=True)
    for m in models:
        out = CACHE / f"{m}.npz"
        if out.exists():
            continue
        md = load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]
        F = []
        for u in units:
            y, sr = sf.read(ROOT / "data/vocalset" / u["meta"]["path"], dtype="float64", always_2d=True)
            g = gcd(sr, FS)
            y = resample_poly(y.mean(1), FS // g, sr // g).astype(np.float32)
            F.append(analyse(md, y, dev)["formants"][u["idx"], :3])
        np.savez_compressed(out, **{str(i): f for i, f in enumerate(F)})
        print(m, flush=True)


def _share(F, P):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.mean(np.abs(np.log(F[:, 1] / P[:, 2])) < np.abs(np.log(F[:, 1] / P[:, 1]))))


def cmd_report() -> None:
    units = _units()
    per = {}
    for m in [run_name(a, s) for a in NULL + COUPLED for s in (1, 2)] + list(D098):
        if m.startswith("d040_5b"):
            Fs = [u["ref"][m] for u in units]
        else:
            z = np.load(CACHE / f"{m}.npz")
            Fs = [z[str(i)] for i in range(len(units))]
        per[m] = np.array([_share(F, u["P"]) for F, u in zip(Fs, units)])
    vowels = np.array([u["meta"]["vowel"] for u in units])
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(units), (1000, len(units)))

    def diff(a, b):
        d = per[a] - per[b]
        bs = d[boot].mean(1)
        return float(d.mean()), *np.percentile(bs, [2.5, 97.5]).tolist()

    lines = ["D-153: coupled and D-098 arms on D-152's measure (VocalSet held-out men, f0 <= 150 Hz, D-141's cell by Praat)",
             f"  {len(units)} files from {len({u['singer'] for u in units})} singers; frames fixed (D-152's: 5b seed 2's voicing and f0); "
             "a frame fails if the run's F2 is nearer Praat's F3 than its F2", "",
             f"  {'run':22s} mean share  failing files  /o/   /u/   /a/    minus 5b same seed [95 %]"]
    res = {"runs": {}}
    for m in per:
        s = 1 if "_s1" in m or m.startswith("d098") else 2
        five = run_name("d040_5b", s)
        nf = per[m] >= 0.5
        byv = {v: f"{nf[vowels == v].sum()}/{(vowels == v).sum()}" for v in ("o", "u", "a")}
        d = diff(m, five) if m != five else None
        res["runs"][m] = {"mean": float(per[m].mean()), "failing": int(nf.sum()), "by_vowel": byv, "minus_5b": d}
        lines.append(f"  {m:22s} {per[m].mean():9.3f}  {nf.sum():6d} of {len(nf)}  {byv['o']:5s} {byv['u']:5s} {byv['a']:5s}  "
                     + ("" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"))
    mean = {m: per[m].mean() for m in per}
    verdicts = []
    lines.append("")
    for s in (1, 2):
        nulls = [mean[run_name(a, s)] for a in NULL]
        cp = [mean[run_name(a, s)] for a in COUPLED]
        v = "below" if max(cp) < min(nulls) else "within" if min(nulls) <= min(cp) else "mixed"
        verdicts.append(v)
        lines.append(f"  seed {s}: null {min(nulls):.3f}-{max(nulls):.3f}; A0 {cp[0]:.3f}, A1 {cp[1]:.3f} -> {v}")
    coupled = ("COUPLED LOWER ON REAL VOICES" if verdicts == ["below", "below"] else
               "NOT LOWER" if "within" in verdicts else "MIXED: reported, Hannah decides")
    n1 = min(mean[run_name(a, 1)] for a in NULL)
    pos = {m: ("below the null" if mean[m] < n1 else "within") for m in D098}
    hb = diff("d098_heads_b20k", "d098_base_b20k")
    lines.append(f"  D-098 (seed-1 null from {n1:.3f}): base {mean['d098_base_b20k']:.3f} {pos['d098_base_b20k']}; "
                 f"heads {mean['d098_heads_b20k']:.3f} {pos['d098_heads_b20k']}; heads minus base {hb[0]:+.3f} [{hb[1]:+.3f}, {hb[2]:+.3f}]")
    if mean["d098_heads_b20k"] >= mean["d098_base_b20k"]:
        heads = "NOT LOWER"
    elif pos["d098_heads_b20k"] == "below the null" and hb[2] < 0:
        heads = "HEADS LOWER"
    else:
        heads = "UNCLEAR: reported, Hannah decides"
    res["reading"] = {"coupled": coupled, "seeds": verdicts, "d098_position": pos, "heads_minus_base": hb, "heads": heads}
    lines += ["", f"Reading (registered): coupled {coupled}; D-098 heads {heads}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d153.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d153.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2].split(",") if len(sys.argv) > 2 else NEW)
    else:
        cmd_report()
