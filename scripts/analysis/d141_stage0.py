"""D-141 stage 0: does D-065's bw arm (every bandwidth x log-uniform 0.4-2.5) already resist the dropped formant?
D-142's tipping grid at seeds 10-12 and D-142's flag B, for the bw arm's seeds and 5b's.

    python scripts/analysis/d141_stage0.py synth                 -> reports/analysis/d141/stage0_synth.json
    python scripts/analysis/d141_stage0.py private REC OUT FILES -> OUT/d141_stage0_private.json   (outside the repo)
    python scripts/analysis/d141_stage0.py report [OUT]          -> reports/analysis/d141/stage0.txt (aggregates only)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d141"
FIVE_B = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
BW = ("d065_bw_s1_b20k", "d065_bw_s2_b20k")
SEEDS = (10, 11, 12)
B = 2000


def _d142():
    spec = importlib.util.spec_from_file_location("d142", ROOT / "scripts/analysis/d142_flags.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _models():
    import torch

    from vocal_analysis.models.infer import load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    return dev, {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in FIVE_B + BW}


def cmd_synth() -> None:
    d142 = _d142()
    prior = d142.Prior()
    dev, models = _models()
    res = {}
    for sname, Fs in d142.SETS.items():
        items = [(f"{sname}|f{f0:g}|b{b1:g}|s{s}", d142._grid_audio(Fs, f0, b1, s), np.array(Fs))
                 for f0 in d142.F0S for b1 in d142.B1S for s in SEEDS]
        res.update(d142._score_group(items, models, dev, prior))
        print(sname, flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "stage0_synth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_private(rec: Path, out: Path, files: list[str]) -> None:
    d142 = _d142()
    d138 = d142._mod("d138", "scripts/analysis/d138_real_balance.py")
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    prior = d142.Prior()
    dev, models = _models()
    groups = {}
    for name, r, y in d138._load(rec, files):
        if name in lv:
            groups.setdefault(name.split("_")[0], []).append((name, y, np.array(lv[name]["F"])))
    res = {}
    for items in groups.values():
        res.update(d142._score_group(items, models, dev, prior))
    (out / "d141_stage0_private.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def _b1_distribution():
    """train_v2's B1 in the cell (F1 350-650, F2/F1 < 2.2), and train_bw_v1's approximated (stage 0's note)."""
    from vocal_analysis.synth import tract

    d = ROOT / "data/synth/train_v2/flat"
    fo = np.load(d / "frame_offsets.npy")
    tube = np.repeat(np.load(d / "clip_tube.npy"), np.diff(fo))
    idx = np.arange(0, len(tube), 37)
    F = np.asarray(np.load(d / "label_formants.npy", mmap_mode="r")[idx, :2])
    Bw = np.asarray(np.load(d / "label_bandwidths.npy", mmap_mode="r")[idx, 0])
    f0 = np.asarray(np.load(d / "label_f0.npy", mmap_mode="r")[idx])
    son = np.asarray(np.load(d / "label_sonorant.npy", mmap_mode="r")[idx]) > 0.5
    m = son & np.isfinite(Bw) & np.isfinite(f0) & (F[:, 0] >= 350) & (F[:, 0] <= 650) & (F[:, 1] / F[:, 0] < 2.2)
    rng = np.random.default_rng(1410)
    fac = np.exp(rng.uniform(np.log(0.4), np.log(2.5), m.sum()))
    t = tube[idx][m]
    bw = np.where(t, Bw[m] * fac, tract.bandwidth_prior(F[m, 0], f0[m]) * fac)
    out = []
    for name, x in (("train_v2", Bw[m]), ("train_bw_v1 (approx.)", bw)):
        out.append(f"  {name:22s} n {len(x)}: median {np.median(x):.0f}, 90th {np.percentile(x, 90):.0f}, 99th "
                   f"{np.percentile(x, 99):.0f} Hz; above 250 Hz {np.mean(x > 250):.2%}, above 360 Hz {np.mean(x > 360):.2%}")
    return out


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def cmd_report(out: Path | None) -> None:
    syn = json.loads((REP / "stage0_synth.json").read_text(encoding="utf-8"))
    prv = json.loads((out / "d141_stage0_private.json").read_text(encoding="utf-8")) if out else None
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))} if out else {}
    rng = np.random.default_rng(1411)
    keys = list(syn)
    f0 = np.array([float(k.split("|")[1][1:]) for k in keys])
    b1 = np.array([float(k.split("|")[2][1:]) for k in keys])
    cell = (f0 <= 140) & (b1 >= 250) & (b1 <= 500)
    lines = ["D-141 stage 0: D-065's bw arm against 5b on D-142's tipping grid (seeds 10-12) and flag B", "",
             "F1 bandwidth labels in the cell (F1 350-650 Hz, F2/F1 < 2.2), every 37th sonorant frame:"] + _b1_distribution() + [""]
    lines.append("Failure share (F2 nearer the true F3 than the true F2), mean over clips, by f0 (rows) and B1 (columns):")
    fail = {m: np.array([syn[k][m]["fail"] for k in keys]) for m in FIVE_B + BW}
    for m in FIVE_B + BW:
        lines.append(f"  {m}:")
        for sname in ("O-like", "6-like"):
            s = np.array([k.startswith(sname) for k in keys])
            for f in sorted(set(f0)):
                lines.append(f"    {sname} f0 {f:3.0f}: " + "  ".join(
                    f"B1 {b:3.0f} {fail[m][s & (f0 == f) & (b1 == b)].mean():4.0%}" for b in sorted(set(b1))))
    m5 = np.mean([fail[m][cell] for m in FIVE_B], 0)
    mb = np.mean([fail[m][cell] for m in BW], 0)
    bs = [np.mean(mb[i]) / max(np.mean(m5[i]), 1e-9) for i in [rng.integers(0, cell.sum(), cell.sum()) for _ in range(B)]]
    ratio = mb.mean() / max(m5.mean(), 1e-9)
    lines += ["", f"Primary cells (f0 110-140, B1 250-500; {cell.sum()} clips), seed means: 5b {m5.mean():.2f}, bw {mb.mean():.2f}; "
              f"ratio {ratio:.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]",
              "  by seed: " + ", ".join(f"{m} {fail[m][cell].mean():.2f}" for m in FIVE_B + BW),
              f"  -> the bw arm {'HELPS' if ratio <= 0.5 else 'does not help'} by the registered reading (ratio <= 0.5)", ""]
    lines.append("Flag B (D-142): share of grid clips flagged (above the 95th percentile of the model's not-failing clips), "
                 "and AUC against failing (>= 0.5) vs not (< 0.2):")
    for m in FIVE_B + BW:
        sc = np.array([syn[k][m]["B"] for k in keys])
        fl = fail[m]
        thr = np.percentile(sc[fl < 0.2], 95)
        auc = _auc(sc[fl >= 0.5], sc[fl < 0.2]) if (fl >= 0.5).any() else float("nan")
        lines.append(f"  {m}: {np.mean(sc > thr):.0%} flagged; failing clips {int((fl >= 0.5).sum())}; AUC {auc:.2f}")
    if prv:
        lines += ["", "Private recording (aggregates only): D-138's failing takes (teacher seed mean >= 0.5 on 5b) - mean failure "
                  "share per model; and how many of all takes fail"]
        fails = [k for k, t in lv.items() if np.mean([t["read"][m]["fail"] for m in FIVE_B]) >= 0.5 and k in prv]
        for m in FIVE_B + BW:
            lines.append(f"  {m}: failing takes {np.mean([prv[k][m]['fail'] for k in fails]):.2f} (n {len(fails)}); "
                         f"all takes failing (>= 0.5): {sum(prv[k][m]['fail'] >= 0.5 for k in prv)} of {len(prv)}")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "stage0.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "synth":
        cmd_synth()
    elif cmd == "private":
        cmd_private(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    elif cmd == "report":
        cmd_report(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
