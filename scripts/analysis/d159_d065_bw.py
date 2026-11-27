"""D-159: D-065's partial bandwidth widening (bw) on D-153's VocalSet measure, D-155's B2 grid and the bandwidth
read-out, against 5b and D-065's gain arm (the from-scratch null). CPU, inference only.

    python scripts/analysis/d159_d065_bw.py run [MODEL,...]   -> cache/d153/<m>.npz, cache/d155/<m>.json, cache/d159/<m>.json
    python scripts/analysis/d159_d065_bw.py report            -> reports/analysis/d159/d159.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d159"
CACHE = ROOT / "cache/d159"
FIVEB = {1: "d040_5b_s1_b20k", 2: "d040_5b_s2_b20k"}
BW = {1: "d065_bw_s1_b20k", 2: "d065_bw_s2_b20k"}
GAIN = {1: "d065_gain_s1_b20k", 2: "d065_gain_s2_b20k"}
NEW = list(BW.values()) + list(GAIN.values())


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _readout(models):
    """Median log2 (model B / true B) for B1 and B2 per D-155 grid clip at low f0, D-155's frame selection."""
    import torch

    from vocal_analysis.models.infer import analyse, load_teacher

    d155 = _mod("d155_b2_grid")
    z = np.load(ROOT / "cache/d155/clips.npz")
    ks = [k for k in d155._keys() if k[1] <= 140]
    dev = torch.device("cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    for m in models:
        out = CACHE / f"{m}.json"
        if out.exists():
            continue
        md = load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]
        res = {}
        for k in ks:
            y = z[d155._key(k)]
            r = analyse(md, y.astype(np.float32), dev)
            n = len(r["voiced_p"])
            i = np.arange(int(0.2 * n), int(0.8 * n))
            i = i[(r["voiced_p"][i] > 0.5) & np.isfinite(r["f0"][i]) & (i * 160 >= 512) & (i * 160 + 512 <= len(y))]
            i = i[np.all(np.isfinite(r["formants"][i, :4]), 1) & np.all(np.isfinite(r["bandwidths"][i, :2]), 1)]
            if len(i):
                B = r["bandwidths"][i]
                res[d155._key(k)] = [float(np.median(np.log2(B[:, 0] / k[2]))), float(np.median(np.log2(B[:, 1] / k[3])))]
        out.write_text(json.dumps(res), encoding="utf-8")
        print("readout", m, flush=True)


def cmd_run(models: list[str]) -> None:
    import torch

    torch.set_num_threads(6)
    _mod("d153_vocalset_arms").cmd_run(models)
    _mod("d155_b2_grid").cmd_run(models)
    _readout(models)


def cmd_report() -> None:
    d153 = _mod("d153_vocalset_arms")
    d155 = _mod("d155_b2_grid")
    units = d153._units()
    per = {}
    for m in list(FIVEB.values()) + NEW:
        if m.startswith("d040_5b"):
            Fs = [u["ref"][m] for u in units]
        else:
            zz = np.load(ROOT / f"cache/d153/{m}.npz")
            Fs = [zz[str(i)] for i in range(len(units))]
        per[m] = np.array([d153._share(F, u["P"]) for F, u in zip(Fs, units)])
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(units), (1000, len(units)))
    lines = ["D-159: D-065's partial bandwidth widening (bw, 0.4-2.5 on cascade and tube) against 5b; gain as the from-scratch null",
             f"  (1) VocalSet (D-153's measure: {len(units)} files, fixed frames): mean failure share; minus 5b of the seed [95 %]"]
    res = {"vocalset": {}, "grid": {}, "readout": {}}
    verdict = {}
    for s in (1, 2):
        five = FIVEB[s]
        lines.append(f"    seed {s}: 5b {per[five].mean():.3f}")
        for m in (BW[s], GAIN[s]):
            d = per[m] - per[five]
            lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
            res["vocalset"][m] = {"mean": float(per[m].mean()), "minus_5b": [float(d.mean()), float(lo), float(hi)]}
            lines.append(f"      {m:20s} {per[m].mean():.3f}  {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        res["vocalset"][five] = {"mean": float(per[five].mean())}
        b = res["vocalset"][BW[s]]
        verdict[s] = b["minus_5b"][2] < 0 and b["mean"] < res["vocalset"][GAIN[s]]["mean"]
        verdict[f"below{s}"] = b["minus_5b"][2] < 0
    # (2) the B2 grid
    ks = d155._keys()
    lines += ["", "  (2) D-155's B2 grid, low f0 (110, 140), all B1, both sets: failure by B2 (90 / 150 / 250 / 400; 250-400)"]
    for s in (1, 2):
        for m in (FIVEB[s], BW[s], GAIN[s]):
            sc = json.loads((ROOT / f"cache/d155/{m}.json").read_text(encoding="utf-8"))
            by = {b2: float(np.mean([sc[d155._key(k)] for k in ks if k[1] <= 140 and k[3] == b2 and d155._key(k) in sc]))
                  for b2 in d155.B2S}
            wide = float(np.mean([sc[d155._key(k)] for k in ks if k[1] <= 140 and k[3] >= 250 and d155._key(k) in sc]))
            res["grid"][m] = {"by_b2": by, "wide": wide}
            lines.append(f"    {m:20s} " + " / ".join(f"{v:.2f}" for v in by.values()) + f";  250-400 {wide:.2f}")
    # (3) the read-out
    lines += ["", "  (3) bandwidth read-out on the grid's low-f0 clips: median log2(model / true) for B1 and B2, by (B1, B2)"]
    cells = [(150.0, 90.0), (150.0, 250.0), (360.0, 90.0), (360.0, 250.0)]
    for s in (1, 2):
        for m in (FIVEB[s], BW[s], GAIN[s]):
            ro = json.loads((CACHE / f"{m}.json").read_text(encoding="utf-8"))
            row = {}
            for b1, b2 in cells:
                v = np.array([ro[d155._key(k)] for k in ks if k[1] <= 140 and k[2] == b1 and k[3] == b2 and d155._key(k) in ro])
                row[f"{b1:.0f}/{b2:.0f}"] = np.median(v, 0).tolist() if len(v) else [np.nan, np.nan]
            res["readout"][m] = row
            lines.append(f"    {m:20s} " + "  ".join(f"B1 {c.split('/')[0]} B2 {c.split('/')[1]}: {v[0]:+.2f} / {v[1]:+.2f}"
                                                  for c, v in row.items()))
    if verdict[1] and verdict[2]:
        out = "MOVES: a partial widening already lowers the real-voice failure"
    elif not verdict["below1"] and not verdict["below2"]:
        out = "NO MOVE: the partial widening doesn't lower the real-voice failure on either seed"
    else:
        out = "MIXED: reported, Hannah decides"
    res["reading"] = out
    lines += ["", f"Reading (registered, on (1)): {out}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d159.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d159.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2].split(",") if len(sys.argv) > 2 else NEW + list(FIVEB.values()))
    else:
        cmd_report()
