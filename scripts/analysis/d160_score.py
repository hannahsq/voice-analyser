"""D-160: scoring and decision for the wide-bandwidth and off-line-Ra arm (decay branches `d160_s{1,2}` from 5b).
CPU, inference only. The guards come from the GPU chain (`chains/d160.sh`, D-065's decide script); this script reads
them if present.

    python scripts/analysis/d160_score.py coverage                -> cache/d160/coverage.json (before training: part A)
    python scripts/analysis/d160_score.py run MODEL[,MODEL...]    -> cache/d160/... and the shared grid caches
    python scripts/analysis/d160_score.py private REC OUT FILES   -> OUT/d160_private.json (counts only reach the repo)
    python scripts/analysis/d160_score.py report [OUT]            -> reports/analysis/d160/d160.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d160"
CACHE = ROOT / "cache/d160"
FS = 16000
ARMS = {1: "d160_s1", 2: "d160_s2"}
FIVEB = {1: "d040_5b_s1_b20k", 2: "d040_5b_s2_b20k"}
NULLS = {s: (FIVEB[s], f"d090_f5_s{s}", f"d090_f57_s{s}") for s in (1, 2)}
SETS = {"v2": ROOT / "data/synth/train_v2/flat", "v2w": ROOT / "data/synth/train_v2w/flat"}
N_VAL = 512
FRONT = ("iy", "ih", "ei", "eh", "ae")
BANDS = {1: ((300, 450), (450, 650), (650, 900)), 2: ((700, 1100), (1100, 1700), (1700, 2500)),
         3: ((2000, 2500), (2500, 3000), (3000, 3500)), 4: ((3000, 3500), (3500, 4000), (4000, 4500))}


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dev():
    import torch

    torch.set_num_threads(6)
    return torch.device("cpu")


def _load(m, dev):
    from vocal_analysis.models.infer import load_teacher

    return load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]


# ----------------------------------------------------------------------------- part A, before training

def _label_rows(flat, every=10):
    fo = np.load(flat / "frame_offsets.npy")
    n = int(fo[-1])
    idx = np.arange(0, n, every)
    son = np.asarray(np.load(flat / "label_sonorant.npy", mmap_mode="r")[idx]) > 0.5
    voi = np.asarray(np.load(flat / "label_voiced.npy", mmap_mode="r")[idx]) > 0.5
    F = np.asarray(np.load(flat / "label_formants.npy", mmap_mode="r")[idx, :4])
    B = np.asarray(np.load(flat / "label_bandwidths.npy", mmap_mode="r")[idx, :4])
    f0 = np.asarray(np.load(flat / "label_f0.npy", mmap_mode="r")[idx])
    s = son & voi
    return F[s], B[s], f0[s]


def cmd_coverage():
    res = {}
    for name, flat in SETS.items():
        F, B, f0 = _label_rows(flat)
        with np.errstate(invalid="ignore"):
            near = (f0 <= 150) & (F[:, 0] >= 400) & (F[:, 0] <= 600) & (F[:, 1] >= 700) & (F[:, 1] <= 1100)
        r = {"near": {"frames": int(near.sum()), "b1_gt250": float(np.mean(B[near, 0] > 250)), "b1_gt360": float(np.mean(B[near, 0] > 360)),
                      "b2_gt150": float(np.mean(B[near, 1] > 150)), "b2_gt250": float(np.mean(B[near, 1] > 250)),
                      "b3_gt400": float(np.mean(B[near, 2] > 400)), "b4_gt300": float(np.mean(B[near, 3] > 300))}}
        for low in (False, True):
            for k, bands in BANDS.items():
                for lo, hi in bands:
                    with np.errstate(invalid="ignore"):
                        s = (F[:, k - 1] >= lo) & (F[:, k - 1] < hi) & np.isfinite(B[:, k - 1])
                        if low:
                            s &= f0 <= 150
                    b = B[s, k - 1]
                    r[f"{'low' if low else 'all'}|B{k}|{lo}-{hi}"] = [int(s.sum()), *(np.percentile(b, [50, 90, 99]).tolist() if len(b) else [np.nan] * 3)]
        res[name] = r
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v["near"] for k, v in res.items()}, indent=1))


# ----------------------------------------------------------------------------- per-model inference

def _vocalset(m, md, dev):
    """D-152's primary files (D-153's selection): the model's F1-F4, length and voicing on every frame."""
    import soundfile as sf
    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import analyse

    out = CACHE / f"vs_{m}.npz"
    if out.exists():
        return
    units = _mod("d153_vocalset_arms")._units()
    acc = {}
    for i, u in enumerate(units):
        y, sr = sf.read(ROOT / "data/vocalset" / u["meta"]["path"], dtype="float64", always_2d=True)
        g = gcd(sr, FS)
        r = analyse(md, resample_poly(y.mean(1), FS // g, sr // g).astype(np.float32), dev)
        acc[f"F{i}"], acc[f"L{i}"], acc[f"v{i}"] = r["formants"][:, :4], 2 ** r["vtl_log2cm"], r["voiced_p"]
    np.savez_compressed(out, **acc)


def _d156_grid(m, md, dev):
    """D-156's part B (B3 or B4 stepped, B2 90) for one model, scored as D-156."""
    from vocal_analysis.models.infer import analyse

    out = CACHE / f"d156_{m}.json"
    if out.exists():
        return
    d156 = _mod("d156_upper_bw")
    res = []
    for (k, b, s, f0, b1, sd, Fs, b3, b4) in d156._grid_keys():
        y = d156._audio(Fs, f0, b1, b3, b4, sd).astype(np.float32)
        r = analyse(md, y, dev)
        n = len(r["voiced_p"])
        i = np.arange(int(0.2 * n), int(0.8 * n))
        i = i[(r["voiced_p"][i] > 0.5) & np.isfinite(r["f0"][i]) & (i * 160 >= 512) & (i * 160 + 512 <= len(y))]
        i = i[np.all(np.isfinite(r["formants"][i, :4]), 1)]
        if not len(i):
            continue
        Fm = r["formants"][i, :4]
        res.append({"k": k, "b": b, "f0": f0, "fail": float(np.mean(np.abs(np.log(Fm[:, 1] / Fs[2])) < np.abs(np.log(Fm[:, 1] / Fs[1])))),
                    "e3": float(np.median(np.abs(np.log2(Fm[:, 2] / Fs[2])) * 1200)),
                    "e4": float(np.median(np.abs(np.log2(Fm[:, 3] / Fs[3])) * 1200))})
    out.write_text(json.dumps(res), encoding="utf-8")


def _val(m, md, dev):
    """Synthetic validation (train_v2's first 512 validation clips, and the same clips in train_v2w), clean audio:
    per sonorant voiced frame the labels, the model's F1-F4 and its formant spread."""
    from vocal_analysis.models.infer import analyse
    from vocal_analysis.train.data import split_ids

    out = CACHE / f"val_{m}.npz"
    if out.exists():
        return
    acc = {}
    for name, flat in SETS.items():
        a = np.load(flat / "audio.npy", mmap_mode="r")
        ao, fo = np.load(flat / "audio_offsets.npy"), np.load(flat / "frame_offsets.npy")
        lab = {k: np.load(flat / f"label_{k}.npy", mmap_mode="r") for k in ("formants", "bandwidths", "f0", "voiced", "sonorant")}
        ids = split_ids(len(ao) - 1)[1][:N_VAL]
        rows = []
        for i in ids:
            y = np.asarray(a[ao[i]:ao[i + 1]], np.float32) / 32767.0
            r = analyse(md, y, dev)
            L = {k: np.asarray(v[fo[i]:fo[i + 1]]) for k, v in lab.items()}
            n = min(len(r["formants"]), len(L["f0"]))
            s = (L["sonorant"][:n] > 0.5) & (L["voiced"][:n] > 0.5)
            rows.append(np.c_[L["formants"][:n, :4][s], L["bandwidths"][:n, :4][s], L["f0"][:n][s], r["formants"][:n, :4][s],
                              r["formant_sigma_cents"][:n, :4][s]])
        acc[name] = np.concatenate(rows)
    np.savez_compressed(out, **acc)


def _hillenbrand(m, md, dev):
    """Hillenbrand dev tokens, as D-150: the model's F1-F3 at each labelled time (front vowels used in the report)."""
    from vocal_analysis.models.infer import analyse

    out = CACHE / f"hill_{m}.json"
    if out.exists():
        return
    d150 = _mod("d150_dev_dropped")
    res = []
    for row, it, y in d150._tokens():
        if it.meta["vowel"] not in FRONT:
            continue
        r = analyse(md, y, dev)
        fr = []
        for j, t in enumerate(it.times):
            ref = it.formants[j, :3]
            if np.isfinite(t) and np.all(np.isfinite(ref)):
                v = d150._reading(r, t)
                fr.append({"ref": ref.tolist(), "read": None if v is None else v.tolist()})
        res.append({"speaker": row.speaker_id, "vowel": it.meta["vowel"], "frames": fr})
    out.write_text(json.dumps(res), encoding="utf-8")


def cmd_run(models):
    dev = _dev()
    for m in models:
        md = _load(m, dev)
        _vocalset(m, md, dev)
        print(m, "vocalset", flush=True)
        _mod("d155_b2_grid").cmd_run([m])
        _mod("d157_rd_bw").cmd_run([m])
        _mod("d158_lf_offline").cmd_run([m])
        _mod("d159_d065_bw")._readout([m])
        _d156_grid(m, md, dev)
        print(m, "grids", flush=True)
        _val(m, md, dev)
        _hillenbrand(m, md, dev)
        print(m, "done", flush=True)


def cmd_private(rec: Path, out: Path, files: list[str]):
    """The private recording's takes (D-149's form): per-take failure for the arms; counts only reach the repo."""
    d138 = _mod("d138_real_balance")
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    dev = _dev()
    models = {m: _load(m, dev) for m in ARMS.values()}
    res = {}
    for name, _, y in d138._load(rec, files):
        if name not in lv:
            continue
        t = lv[name]
        rd, _ = d138._read(models, dev, y, np.array(t["F"], float))
        f5 = float(np.mean([t["read"][m]["fail"] for m in FIVEB.values()]))
        res[name] = {"group": "fail" if f5 >= d138.FAIL else "ok" if f5 < d138.OK else "mid", "5b": f5,
                     **{m: rd[m]["fail"] for m in ARMS.values()}}
    (out / "d160_private.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- report

def _vs_shares(m, units):
    """Per-file failure share on D-153's fixed frames: the arm's own inference, or 5b's from D-152's cache."""
    d153 = _mod("d153_vocalset_arms")
    if m in FIVEB.values():
        return np.array([d153._share(u["ref"][m], u["P"]) for u in units])
    if (ROOT / f"cache/d153/{m}.npz").exists() and not (CACHE / f"vs_{m}.npz").exists():
        z = np.load(ROOT / f"cache/d153/{m}.npz")
        return np.array([d153._share(z[str(i)], u["P"]) for i, u in enumerate(units)])
    z = np.load(CACHE / f"vs_{m}.npz")
    return np.array([d153._share(z[f"F{i}"][u["idx"], :3], u["P"]) for i, u in enumerate(units)])


def _flag_auc(m, units, fail):
    """Flag B (D-142's, seed 2's threshold) per file: the median score over voiced frames, length from the singer's other
    files (leave-one-out); AUC of failing (>= 0.5) against OK (< 0.2) files by the run's own failure."""
    prior = _mod("d142_flags").Prior()
    z = np.load(CACHE / f"vs_{m}.npz")
    singers = np.array([u["singer"] for u in units])
    med = np.array([np.nanmedian(z[f"L{i}"][z[f"v{i}"] > 0.5]) if (z[f"v{i}"] > 0.5).any() else np.nan for i in range(len(units))])
    sc = []
    for i in range(len(units)):
        others = (singers == singers[i]) & (np.arange(len(units)) != i)
        L = np.nanmedian(med[others])
        v = (z[f"v{i}"] > 0.5) & np.all(np.isfinite(z[f"F{i}"]), 1)
        sc.append(float(np.nanmedian(prior.score(z[f"F{i}"][v], np.full(v.sum(), L)))) if v.any() else np.nan)
    sc = np.array(sc)
    pos, neg = sc[(fail >= 0.5) & np.isfinite(sc)], sc[(fail < 0.2) & np.isfinite(sc)]
    if len(pos) < 5 or len(neg) < 5:
        return None, len(pos), len(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())), len(pos), len(neg)


def cmd_report(out_private: Path | None = None):
    d153 = _mod("d153_vocalset_arms")
    d155 = _mod("d155_b2_grid")
    units = d153._units()
    null = json.loads((ROOT / "reports/analysis/d153/d153.json").read_text(encoding="utf-8"))["runs"]
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(units), (1000, len(units)))
    lines = ["D-160: wide low-f0 bandwidths and an off-line return phase (train_v2w in train_v2's place), decay branches from 5b", ""]
    res = {"vocalset": {}, "grid": {}}
    # manipulation check
    ks = d155._keys()
    lines.append("  Manipulation check: D-155's B2 grid, low f0 (110, 140), failure at B2 250-400 (and at B2 90)")
    man = {}
    for s in (1, 2):
        for m in (FIVEB[s], ARMS[s]):
            sc = json.loads((ROOT / f"cache/d155/{m}.json").read_text(encoding="utf-8"))
            wide = float(np.mean([sc[d155._key(k)] for k in ks if k[1] <= 140 and k[3] >= 250 and d155._key(k) in sc]))
            narrow = float(np.mean([sc[d155._key(k)] for k in ks if k[1] <= 140 and k[3] == 90 and d155._key(k) in sc]))
            res["grid"][m] = {"wide": wide, "b2_90": narrow}
            lines.append(f"    {m:18s} {wide:.3f} ({narrow:.3f})")
        man[s] = res["grid"][FIVEB[s]]["wide"] - res["grid"][ARMS[s]]["wide"]
    learned = (man[1] + man[2]) / 2 >= 0.2 and min(man.values()) >= 0.1
    lines.append(f"    fall: seed 1 {man[1]:+.3f}, seed 2 {man[2]:+.3f}, mean {(man[1] + man[2]) / 2:+.3f} -> "
                 + ("LEARNED" if learned else "NOT LEARNED"))
    # VocalSet
    lines += ["", f"  VocalSet (D-153's measure, {len(units)} files): mean failure share; null runs; arm minus 5b [95 %]"]
    verdict = []
    for s in (1, 2):
        nulls = {m: null[m]["mean"] for m in NULLS[s]}
        a = _vs_shares(ARMS[s], units)
        f5 = _vs_shares(FIVEB[s], units)
        d = a - f5
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        res["vocalset"][ARMS[s]] = {"mean": float(a.mean()), "nulls": nulls, "minus_5b": [float(d.mean()), float(lo), float(hi)],
                                    "failing": int((a >= 0.5).sum())}
        below = a.mean() < min(nulls.values())
        verdict.append("lower" if below and hi < 0 else "within" if min(nulls.values()) <= a.mean() else "mixed")
        lines.append(f"    seed {s}: nulls " + ", ".join(f"{k} {v:.3f}" for k, v in nulls.items())
                     + f"; arm {a.mean():.3f} ({int((a >= 0.5).sum())} failing files), minus 5b {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict[-1]}")
        vow = np.array([u["meta"]["vowel"] for u in units])
        lines.append("      failing files by vowel, 5b / arm: " + ", ".join(
            f"{v} {int((f5[vow == v] >= 0.5).sum())}/{int((a[vow == v] >= 0.5).sum())}" for v in ("o", "u", "a")))
        if (CACHE / f"vs_{ARMS[s]}.npz").exists():
            auc = _flag_auc(ARMS[s], units, a)
            res["vocalset"][ARMS[s]]["flag_auc"] = auc
            lines.append(f"      flag B AUC (arm's own failing / OK files): " + (f"{auc[0]:.2f} ({auc[1]} / {auc[2]})" if auc[0] is not None
                                                                              else f"too few ({auc[1]} / {auc[2]})"))
    if not learned:
        vs = "NOT LEARNED: the VocalSet result is reported but uninformative"
    elif verdict == ["lower", "lower"]:
        vs = "LOWER ON REAL VOICES"
    elif "within" in verdict:
        vs = "NOT LOWER"
    else:
        vs = "MIXED: reported, Hannah decides"
    # reported: D-156 part B
    lines += ["", "  D-156 part B (B2 90; B1 150 and 360; all f0): failure / F3 or F4 error (c) by step, 5b -> arm"]
    d156 = _mod("d156_upper_bw")
    for s in (1, 2):
        for m in (FIVEB[s], ARMS[s]):
            p = CACHE / f"d156_{m}.json"
            if not p.exists():
                continue
            G = json.loads(p.read_text(encoding="utf-8"))
            cells = []
            for k, steps in d156.STEPS.items():
                for b in steps:
                    rows = [g for g in G if g["k"] == k and g["b"] == b]
                    cells.append(f"B{k} {b:.0f}: {np.mean([g['fail'] for g in rows]):.2f}/{np.median([g[f'e{k}'] for g in rows]):.0f}")
            lines.append(f"    {m:18s} " + "  ".join(cells))
    # reported: read-out ratios (D-159's cells)
    lines += ["", "  bandwidth read-out, low-f0 grid clips: median log2(model / true) B1 / B2 at (B1, B2)"]
    for s in (1, 2):
        for m in (FIVEB[s], ARMS[s]):
            p = ROOT / f"cache/d159/{m}.json"
            if not p.exists():
                continue
            ro = json.loads(p.read_text(encoding="utf-8"))
            cells = []
            for b1, b2 in ((150.0, 90.0), (150.0, 250.0), (360.0, 90.0), (360.0, 250.0)):
                v = np.array([ro[d155._key(k)] for k in ks if k[1] <= 140 and k[2] == b1 and k[3] == b2 and d155._key(k) in ro])
                med = np.median(v, 0)
                cells.append(f"({b1:.0f},{b2:.0f}) {med[0]:+.2f}/{med[1]:+.2f}")
            lines.append(f"    {m:18s} " + "  ".join(cells))
    # reported: resolvability and front vowels on synthetic validation
    lines += ["", "  synthetic validation (512 clips, clean): pair (Fk, Fk+1) unresolvable if spacing < mean bandwidth or < f0;",
              "  median |error| of the pair's two formants (c) and median spread (c); front vowels (label F2 > 1800): F2 / F3 |error| (c)"]
    for name in SETS:
        for s in (1, 2):
            for m in (FIVEB[s], ARMS[s]):
                p = CACHE / f"val_{m}.npz"
                if not p.exists():
                    continue
                X = np.load(p)[name]
                Fl, Bl, f0, Fp, sg = X[:, :4], X[:, 4:8], X[:, 8], X[:, 9:13], X[:, 13:17]
                with np.errstate(invalid="ignore", divide="ignore"):
                    err = np.abs(np.log2(Fp / Fl)) * 1200
                    cells = []
                    for k in range(3):
                        un = ((Fl[:, k + 1] - Fl[:, k]) < (Bl[:, k] + Bl[:, k + 1]) / 2) | ((Fl[:, k + 1] - Fl[:, k]) < f0)
                        ok = np.all(np.isfinite(err[:, k:k + 2]), 1)
                        e_r, e_u = np.median(err[ok & ~un, k:k + 2]), np.median(err[ok & un, k:k + 2]) if (ok & un).any() else np.nan
                        s_r, s_u = np.median(sg[ok & ~un, k:k + 2]), np.median(sg[ok & un, k:k + 2]) if (ok & un).any() else np.nan
                        cells.append(f"F{k + 1}F{k + 2} res {e_r:.0f}c/{s_r:.0f}c, unres ({(ok & un).mean():.1%}) {e_u:.0f}c/{s_u:.0f}c")
                    fv = Fl[:, 1] > 1800
                    cells.append(f"front F2 {np.nanmedian(err[fv, 1]):.0f}c F3 {np.nanmedian(err[fv, 2]):.0f}c")
                lines.append(f"    {name:3s} {m:18s} " + "; ".join(cells))
    # reported: Hillenbrand front vowels
    lines += ["", "  Hillenbrand dev front vowels (iy ih ei eh ae): median |F2| / |F3| error against the hand-checked values (c)"]
    d150 = json.loads((ROOT / "reports/analysis/d150/d150.json").read_text(encoding="utf-8"))
    for s in (1, 2):
        for m in (FIVEB[s], ARMS[s]):
            if m in FIVEB.values():
                fr = [(f["ref"], f.get(m)) for t in d150 if t["vowel"] in FRONT for f in t["frames"]]
            elif (CACHE / f"hill_{m}.json").exists():
                fr = [(f["ref"], f["read"]) for t in json.loads((CACHE / f"hill_{m}.json").read_text(encoding="utf-8")) for f in t["frames"]]
            else:
                continue
            fr = [(np.array(r), np.array(v)) for r, v in fr if v is not None]
            e2 = np.median([abs(np.log2(v[1] / r[1])) * 1200 for r, v in fr])
            e3 = np.median([abs(np.log2(v[2] / r[2])) * 1200 for r, v in fr])
            lines.append(f"    {m:18s} F2 {e2:.0f} c, F3 {e3:.0f} c (n {len(fr)})")
    # reported: coverage (part A) and private counts
    cov = CACHE / "coverage.json"
    if cov.exists():
        c = json.loads(cov.read_text(encoding="utf-8"))
        lines += ["", "  coverage near the failing vowels (f0 <= 150, F1 400-600, F2 700-1100; labels, every 10th frame):"]
        for name, v in c.items():
            n = v["near"]
            lines.append(f"    {name:3s} {n['frames']} frames: B1 > 250 {n['b1_gt250']:.1%}, > 360 {n['b1_gt360']:.1%}; B2 > 150 {n['b2_gt150']:.1%}, "
                         f"> 250 {n['b2_gt250']:.1%}; B3 > 400 {n['b3_gt400']:.1%}; B4 > 300 {n['b4_gt300']:.1%}")
    if out_private is not None and (out_private / "d160_private.json").exists():
        P = json.loads((out_private / "d160_private.json").read_text(encoding="utf-8"))
        fail = [v for v in P.values() if v["group"] == "fail"]
        ok = [v for v in P.values() if v["group"] == "ok"]
        lines += ["", f"  a private recording: {len(fail)} takes 5b fails, {len(ok)} it reads; counts by arm (fails < 0.5 / new failures >= 0.5):"]
        for m in ARMS.values():
            lines.append(f"    {m}: reads {sum(v[m] < 0.5 for v in fail)} of {len(fail)} failing takes; {sum(v[m] >= 0.5 for v in ok)} new failures")
    # guards (from the GPU chain)
    g = REP / "decide_guards.txt"
    if g.exists():
        lines += ["", "  guards (D-051's, D-065's script; its primary line is D-065's bw criterion, reported only):"]
        lines += ["    " + ln for ln in g.read_text(encoding="utf-8").splitlines() if ln.strip()][:60]
    res["reading"] = {"learned": learned, "manipulation": man, "vocalset": vs, "per_seed": verdict}
    lines += ["", f"Reading (registered): {vs}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d160.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d160.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "coverage":
        cmd_coverage()
    elif cmd == "run":
        cmd_run(sys.argv[2].split(","))
    elif cmd == "private":
        cmd_private(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    else:
        cmd_report(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
