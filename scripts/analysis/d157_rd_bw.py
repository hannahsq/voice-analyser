"""D-157: does the source's slope (Rd) change the wide-bandwidth failure (D-155's grid with Rd steps), and does train_v2
cover Rd and wide bandwidths together at low f0? CPU, inference only.

    python scripts/analysis/d157_rd_bw.py run [MODEL,MODEL]   -> cache/d157/<model>.json
    python scripts/analysis/d157_rd_bw.py coverage            -> cache/d157/coverage.json
    python scripts/analysis/d157_rd_bw.py report              -> reports/analysis/d157/d157.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d157"
CACHE = ROOT / "cache/d157"
FS = 16000
FIVEB = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
RDS = (0.6, 1.0, 1.8, 2.7)
B2S = (90.0, 250.0)
B1S = (150.0, 360.0)
SEEDS = (20, 21, 22)


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _keys():
    d142 = _mod("d142_flags")
    return [(s, f0, b1, b2, rd, sd, Fs) for s, Fs in d142.SETS.items() for f0 in d142.F0S for b1 in B1S for b2 in B2S
            for rd in RDS for sd in SEEDS]


def _key(k):
    return f"{k[0]}|f{k[1]:.0f}|b{k[2]:.0f}|B{k[3]:.0f}|r{k[4]:.1f}|s{k[5]}"


def _d155_key(k):
    return f"{k[0]}|f{k[1]:.0f}|b{k[2]:.0f}|B{k[3]:.0f}|s{k[5]}"


def _audio(Fs, f0, b1, b2, rd, seed):
    """D-155's clip (D-142's seed formula; Rd doesn't enter it) at the given Rd."""
    from vocal_analysis.synth import sampler, tract, voice

    C = int(3.0 * FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / FS
    pf = np.full((C, tract.N_POLES), np.nan)
    pb = pf.copy()
    pf[:, :4], pb[:, :4] = Fs, (b1, b2, 120, 150)
    rng = np.random.default_rng(int(1420 + 1000 * seed + f0 + b1 + Fs[0]))
    tr = voice.Trajectories(f0=f0 * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12), rd=np.full(C, rd),
                            voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1), whisper_amp=np.zeros(C),
                            fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
    return 0.5 * y / np.abs(y).max()


def cmd_run(models: list[str]) -> None:
    import torch

    from vocal_analysis.models.infer import load_teacher

    d147 = _mod("d147_branch_spread")
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    ks = _keys()
    items = [(None, _audio(k[6], k[1], k[2], k[3], k[4], k[5]), k[6]) for k in ks]
    CACHE.mkdir(parents=True, exist_ok=True)
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
    rd = np.asarray(np.load(d / "label_rd.npy", mmap_mode="r")[idx], float).reshape(len(idx), -1)[:, 0]
    B = np.asarray(np.load(d / "label_bandwidths.npy", mmap_mode="r")[idx, :2])
    t = tube[idx]
    base = son & voi & np.isfinite(rd) & np.all(np.isfinite(B), 1)
    res = {}
    with np.errstate(invalid="ignore"):
        for name, sel in (("low_f0", base & (f0 <= 150)), ("all_f0", base)):
            for part, p in (("all", np.ones(len(t), bool)), ("tube", t), ("cascade", ~t)):
                s = sel & p
                r, b = rd[s], B[s]
                hi = r >= 1.8
                res[f"{name}|{part}"] = {"frames": int(s.sum()), "rd_p5_50_95": np.percentile(r, [5, 50, 95]).tolist() if s.any() else [],
                                         "rd_ge1.8": float(hi.mean()) if s.any() else float("nan"),
                                         "rd_ge2.4": float((r >= 2.4).mean()) if s.any() else float("nan"),
                                         "hi_rd_b1_gt250": float((b[hi, 0] > 250).mean()) if hi.any() else float("nan"),
                                         "hi_rd_b2_gt150": float((b[hi, 1] > 150).mean()) if hi.any() else float("nan")}
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_report() -> None:
    sc = {m: json.loads((CACHE / f"{m}.json").read_text(encoding="utf-8")) for m in FIVEB}
    ref = {m: json.loads((ROOT / f"cache/d155/{m}.json").read_text(encoding="utf-8")) for m in FIVEB}
    ks = _keys()
    chk = [(sc[m][_key(k)], ref[m][_d155_key(k)]) for m in FIVEB for k in ks if k[4] == 1.0 and _key(k) in sc[m]]
    mism = sum(abs(a - b) > 1e-9 for a, b in chk)
    lines = ["D-157: Rd (source slope) against the wide-bandwidth failure (D-155's grid; failure = F2 nearer the true F3 than "
             "the true F2)", f"  render check (Rd 1.0 clips against D-155's scores, per clip): {mism} of {len(chk)} differ", ""]

    def mean(m, sel):
        v = [sc[m][_key(k)] for k in ks if sel(k) and _key(k) in sc[m]]
        return float(np.mean(v)) if v else float("nan")

    low = lambda k: k[1] <= 140  # noqa: E731
    tab = {}
    for m in FIVEB:
        tab[m] = {rd: {b2: mean(m, lambda k, rd=rd, b2=b2: low(k) and k[4] == rd and k[3] == b2) for b2 in B2S} for rd in RDS}
    tab["mean"] = {rd: {b2: float(np.mean([tab[m][rd][b2] for m in FIVEB])) for b2 in B2S} for rd in RDS}
    lines.append("  low f0 (110, 140), B1 150 and 360, both sets: failure at B2 90 / 250, and the B2 effect")
    stats = {}
    for m in FIVEB + ("mean",):
        eff = {rd: tab[m][rd][250.0] - tab[m][rd][90.0] for rd in RDS}
        slope = max(tab[m][rd][90.0] for rd in RDS) - min(tab[m][rd][90.0] for rd in RDS)
        stats[m] = {"b2_effect": eff, "range": max(eff.values()) - min(eff.values()), "slope_effect": slope}
        lines.append(f"    {m:18s} " + "  ".join(f"Rd {rd}: {tab[m][rd][90.0]:.2f} / {tab[m][rd][250.0]:.2f} ({eff[rd]:+.2f})" for rd in RDS)
                     + f"   B2-effect range {stats[m]['range']:.2f}, slope effect {slope:.2f}")
    lines += ["", "  5b seed mean by set and B1 (low f0 | f0 180-220), B2 90 / 250:"]
    for s in ("O-like", "6-like"):
        for b1 in B1S:
            cells = []
            for rd in RDS:
                c = [np.mean([mean(m, lambda k: k[0] == s and sel(k) and k[2] == b1 and k[4] == rd and k[3] == b2) for m in FIVEB])
                     for sel in (low, lambda k: not low(k)) for b2 in B2S]
                cells.append(f"Rd {rd}: {c[0]:.2f}/{c[1]:.2f} | {c[2]:.2f}/{c[3]:.2f}")
            lines.append(f"    {s:6s} B1 {b1:3.0f}: " + "   ".join(cells))
    cov = json.loads((CACHE / "coverage.json").read_text(encoding="utf-8"))
    lines += ["", "  train_v2 coverage (every 10th sonorant voiced frame; labels):"]
    for k, v in cov.items():
        p = v["rd_p5_50_95"]
        lines.append(f"    {k:16s} {v['frames']:8d} frames; Rd 5/50/95th {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}; Rd >= 1.8 {v['rd_ge1.8']:.1%}, "
                     f">= 2.4 {v['rd_ge2.4']:.1%}; of Rd >= 1.8: B1 > 250 {v['hi_rd_b1_gt250']:.1%}, B2 > 150 {v['hi_rd_b2_gt150']:.1%}")
    st = stats["mean"]
    if st["range"] >= 0.15 and all(stats[m]["range"] >= 0.10 for m in FIVEB):
        out = "JOINT: the source slope changes how a wide B2 reads; a bandwidth arm should vary them together"
    elif st["range"] < 0.05 and st["slope_effect"] < 0.10:
        out = "SEPARATE: the slope neither tips the failure nor changes the wide-B2 effect"
    else:
        out = "PARTLY: reported, Hannah decides"
    res = {"render_mismatches": mism, "table": {m: {str(rd): v for rd, v in t.items()} for m, t in tab.items()},
           "stats": {m: {**s, "b2_effect": {str(k): v for k, v in s["b2_effect"].items()}} for m, s in stats.items()},
           "coverage": cov, "reading": out}
    lines += ["", f"Reading (registered; seed mean B2-effect range {st['range']:.2f}, slope effect {st['slope_effect']:.2f}): {out}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d157.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d157.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2].split(",") if len(sys.argv) > 2 else list(FIVEB))
    else:
        {"coverage": cmd_coverage, "report": cmd_report}[cmd]()
