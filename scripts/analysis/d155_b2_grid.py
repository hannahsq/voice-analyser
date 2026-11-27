"""D-155: the tipping grid with B2 steps (90 / 150 / 250 / 400 Hz), scored by 5b and D-153's arms; and train_v2's
B2 coverage near the failing vowels. CPU, inference only.

    python scripts/analysis/d155_b2_grid.py render                -> cache/d155/clips.npz
    python scripts/analysis/d155_b2_grid.py run [MODEL,MODEL,...] -> cache/d155/<model>.json (one process per list)
    python scripts/analysis/d155_b2_grid.py coverage              -> cache/d155/coverage.json
    python scripts/analysis/d155_b2_grid.py report                -> reports/analysis/d155/d155.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d155"
CACHE = ROOT / "cache/d155"
FS = 16000
B2S = (90.0, 150.0, 250.0, 400.0)
SEEDS = (20, 21, 22)  # D-143's grid seeds, as D-147
FIVEB = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
ARMS = ("d090_f5_s1", "d090_f5_s2", "d090_f57_s1", "d090_f57_s2", "d080_a0_s1", "d080_a0_s2", "d080_a1_s1", "d080_a1_s2",
        "d098_base_b20k", "d098_heads_b20k")


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _keys():
    d142 = _mod("d142_flags")
    return [(s, f0, b1, b2, sd, Fs) for s, Fs in d142.SETS.items() for f0 in d142.F0S for b1 in d142.B1S for b2 in B2S
            for sd in SEEDS]


def _audio(Fs, f0, b1, b2, seed):
    """D-142's `_grid_audio` with B2 as a parameter; the random seed is D-142's, so B2 = 90 gives its clips exactly."""
    from vocal_analysis.synth import sampler, tract, voice

    C = int(3.0 * FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / FS
    pf = np.full((C, tract.N_POLES), np.nan)
    pb = pf.copy()
    pf[:, :4], pb[:, :4] = Fs, (b1, b2, 120, 150)
    rng = np.random.default_rng(int(1420 + 1000 * seed + f0 + b1 + Fs[0]))
    tr = voice.Trajectories(f0=f0 * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12), rd=np.full(C, 1.0),
                            voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1), whisper_amp=np.zeros(C),
                            fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
    return 0.5 * y / np.abs(y).max()


def _key(k):
    return f"{k[0]}|f{k[1]:.0f}|b{k[2]:.0f}|B{k[3]:.0f}|s{k[4]}"


def cmd_render() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    ks = _keys()
    np.savez(CACHE / "clips.npz", **{_key(k): _audio(k[5], k[1], k[2], k[3], k[4]).astype(np.float32) for k in ks})
    print(len(ks), "clips")


def cmd_run(models: list[str]) -> None:
    import torch

    from vocal_analysis.models.infer import load_teacher

    d147 = _mod("d147_branch_spread")
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    z = np.load(CACHE / "clips.npz")
    ks = _keys()
    items = [(None, z[_key(k)], k[5]) for k in ks]
    for m in models:
        out = CACHE / f"{m}.json"
        if out.exists():
            continue
        md = load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]
        f = d147.fail_by_clip(md, dev, items)
        out.write_text(json.dumps({_key(ks[j]): v for j, v in f.items()}), encoding="utf-8")
        print(m, flush=True)


def cmd_coverage() -> None:
    d = ROOT / "data/synth/train_v2/flat"
    fo = np.load(d / "frame_offsets.npy")
    tube = np.repeat(np.load(d / "clip_tube.npy"), np.diff(fo))
    idx = np.arange(0, len(tube), 10)
    son = np.asarray(np.load(d / "label_sonorant.npy", mmap_mode="r")[idx]) > 0.5
    voi = np.asarray(np.load(d / "label_voiced.npy", mmap_mode="r")[idx]) > 0.5
    f0 = np.asarray(np.load(d / "label_f0.npy", mmap_mode="r")[idx])
    F = np.asarray(np.load(d / "label_formants.npy", mmap_mode="r")[idx, :2])
    B = np.asarray(np.load(d / "label_bandwidths.npy", mmap_mode="r")[idx, :2])
    t = tube[idx]
    base = son & voi & np.all(np.isfinite(F), 1) & np.all(np.isfinite(B), 1)
    with np.errstate(invalid="ignore"):
        near = base & (f0 <= 150) & (F[:, 0] >= 400) & (F[:, 0] <= 600) & (F[:, 1] >= 700) & (F[:, 1] <= 1100)
    res = {}
    for name, sel in (("near", near), ("all_sonorant", base)):
        for part, p in (("all", np.ones(len(t), bool)), ("tube", t), ("cascade", ~t)):
            s = sel & p
            b1, b2 = B[s, 0], B[s, 1]
            res[f"{name}|{part}"] = {"frames": int(s.sum()), "b2_median": float(np.median(b2)) if s.any() else float("nan"),
                                     **{f"b2_gt{x}": float(np.mean(b2 > x)) if s.any() else float("nan") for x in (150, 250, 400)},
                                     "b1_b2_gt250": float(np.mean((b1 > 250) & (b2 > 250))) if s.any() else float("nan")}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def cmd_report() -> None:
    from scipy.stats import spearmanr

    sc = {m: json.loads((CACHE / f"{m}.json").read_text(encoding="utf-8")) for m in FIVEB + ARMS}
    ks = _keys()

    def mean(m, sel):
        v = [sc[m][_key(k)] for k in ks if sel(k) and _key(k) in sc[m]]
        return float(np.mean(v)) if v else float("nan")

    low = lambda k: k[1] <= 140  # noqa: E731
    lines = ["D-155: the tipping grid with B2 steps (D-142's clips, seeds 20-22; failure = F2 nearer the true F3 than the true F2)", ""]
    res = {"fiveb": {}, "arms": {}}
    # render check: B2 = 90 is D-147's grid; its primary cells (f0 110-140, B1 250-500)
    chk = {m: mean(m, lambda k: low(k) and 250 <= k[2] <= 500 and k[3] == 90) for m in FIVEB}
    lines.append("  render check (B2 = 90, D-147's primary cells; D-147: s1 0.34, s2 0.17): "
                 + ", ".join(f"{m} {v:.2f}" for m, v in chk.items()))
    lines += ["", "  5b, low f0 (110, 140), all B1, both sets; failure by B2:"]
    by = {}
    for m in FIVEB + ("mean",):
        ms = FIVEB if m == "mean" else (m,)
        by[m] = {b2: float(np.mean([mean(x, lambda k, b2=b2: low(k) and k[3] == b2) for x in ms])) for b2 in B2S}
        lines.append(f"    {m:18s} " + "  ".join(f"B2 {b2:.0f}: {v:.2f}" for b2, v in by[m].items()))
    res["fiveb"]["low_by_b2"] = by
    lines += ["", "  5b seed mean by set, B1 and B2 (low f0 | f0 180-220):"]
    for s in ("O-like", "6-like"):
        for b1 in (150.0, 250.0, 360.0, 500.0, 700.0):
            cells = []
            for b2 in B2S:
                lo = np.mean([mean(m, lambda k: k[0] == s and low(k) and k[2] == b1 and k[3] == b2) for m in FIVEB])
                hi = np.mean([mean(m, lambda k: k[0] == s and not low(k) and k[2] == b1 and k[3] == b2) for m in FIVEB])
                cells.append(f"{lo:.2f}|{hi:.2f}")
            lines.append(f"    {s:6s} B1 {b1:3.0f}: " + "  ".join(f"B2 {b2:.0f} {c}" for b2, c in zip(B2S, cells)))
    # arms
    d153 = json.loads((ROOT / "reports/analysis/d153/d153.json").read_text(encoding="utf-8"))["runs"]
    lines += ["", "  arms, low f0 (reported): failure at B2 = 90 and at B2 250-400; D-153's VocalSet measure"]
    g90, gw, vs = [], [], []
    for m in FIVEB + ARMS:
        a, b = mean(m, lambda k: low(k) and k[3] == 90), mean(m, lambda k: low(k) and k[3] >= 250)
        res["arms"][m] = {"b2_90": a, "b2_wide": b, "vocalset": d153[m]["mean"]}
        g90.append(a), gw.append(b), vs.append(d153[m]["mean"])
        lines.append(f"    {m:18s} B2 90 {a:.3f}   B2 250-400 {b:.3f}   VocalSet {d153[m]['mean']:.3f}")
    r90, rw = spearmanr(g90, vs)[0], spearmanr(gw, vs)[0]
    res["arms"]["spearman"] = {"b2_90": float(r90), "b2_wide": float(rw)}
    lines.append(f"    Spearman with VocalSet across the 12 runs: B2 90 {r90:+.2f}, B2 250-400 {rw:+.2f} (reported only)")
    # coverage
    cov = json.loads((CACHE / "coverage.json").read_text(encoding="utf-8"))
    res["coverage"] = cov
    lines += ["", "  train_v2 coverage (every 10th sonorant voiced frame); near = f0 <= 150, F1 400-600, F2 700-1100:"]
    for k, v in cov.items():
        lines.append(f"    {k:22s} {v['frames']:8d} frames; B2 median {v['b2_median']:.0f}; above 150 {v['b2_gt150']:.1%}, "
                     f"250 {v['b2_gt250']:.1%}, 400 {v['b2_gt400']:.1%}; B1 and B2 above 250 {v['b1_b2_gt250']:.1%}")
    # readings
    d = {m: by[m][400.0] - by[m][90.0] for m in by}
    if d["mean"] >= 0.2 and min(d[m] for m in FIVEB) >= 0.1 and by["mean"][250.0] >= by["mean"][90.0]:
        b2 = "B2 MATTERS: a wide B2 tips the dropped formant on the grid"
    elif d["mean"] < 0.05:
        b2 = "NO B2 EFFECT"
    else:
        b2 = "PARTLY: reported, Hannah decides"
    share = cov["near|all"]["b2_gt250"]
    covr = "B2 GAP" if share < 0.02 else "COVERED" if share >= 0.10 else "THIN"
    res["reading"] = {"b2": b2, "delta": d, "coverage": covr, "near_b2_gt250": share}
    lines += ["", f"Reading (registered): grid {b2} (delta B2 400 - 90: seed mean {d['mean']:+.2f}, "
              + ", ".join(f"{m} {d[m]:+.2f}" for m in FIVEB) + f"); coverage {covr} ({share:.1%} of near frames above 250 Hz)"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d155.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d155.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2].split(",") if len(sys.argv) > 2 else list(FIVEB + ARMS))
    else:
        {"render": cmd_render, "coverage": cmd_coverage, "report": cmd_report}[cmd]()
