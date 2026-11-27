"""D-152: the dropped formant and flag B (D-142) on VocalSet's held-out singers' sung vowels, against Praat.
CPU, inference only.

    python scripts/analysis/d152_vocalset_flag.py run      -> cache/d152/*.npz (per frame, not committed), reports/analysis/d152/d152.{txt,json}
    python scripts/analysis/d152_vocalset_flag.py report   -> reports/analysis/d152/d152.{txt,json}
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d152"
CACHE = ROOT / "cache/d152"
MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")  # flag on seed 2 (D-142's); the failure rule on the seed mean
FS = 16000
MIN_FRAMES = 10
B1_BANDS = ((0, 150), (150, 250), (250, 400), (400, np.inf))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _files():
    rows = [r for r in csv.DictReader(open(ROOT / "data/manifests/vocalset.csv", encoding="utf-8")) if r["split"] == "val"]
    out = []
    for r in rows:
        parts = r["path"].split("/")
        singer, task, tech = parts[1], parts[2], parts[3]
        if task == "excerpts" or tech == "lip_trill":
            continue
        out.append({"path": r["path"], "singer": singer, "task": task, "tech": tech, "vowel": parts[-1][:-4].split("_")[-1]})
    return out


def cmd_run() -> None:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import analyse, load_teacher

    d151 = _mod("d151_ptdb_flag")
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    models = {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}
    files = _files()
    CACHE.mkdir(parents=True, exist_ok=True)
    for singer in sorted({f["singer"] for f in files}):
        out = CACHE / f"{singer}.npz"
        if out.exists():
            continue
        acc = {k: [] for k in ("fid", "f0", "P", "B1")}
        acc.update({f"{m}|{k}": [] for m in MODELS for k in ("F", "L", "v")})
        meta = []
        for j, f in enumerate([f for f in files if f["singer"] == singer]):
            y, sr = sf.read(ROOT / "data/vocalset" / f["path"], dtype="float64", always_2d=True)
            g = gcd(sr, FS)
            y = resample_poly(y.mean(1), FS // g, sr // g).astype(np.float32)
            res = {m: analyse(md, y, dev) for m, md in models.items()}
            n = len(res[MODELS[0]]["voiced_p"])
            P, B1 = d151._praat(y, 5000.0 if singer.startswith("male") else 5500.0, n)
            acc["fid"].append(np.full(n, j))
            acc["f0"].append(res[MODELS[0]]["f0"])
            acc["P"].append(P)
            acc["B1"].append(B1)
            for m in MODELS:
                acc[f"{m}|F"].append(res[m]["formants"][:, :4])
                acc[f"{m}|L"].append(2 ** res[m]["vtl_log2cm"])
                acc[f"{m}|v"].append(res[m]["voiced_p"])
            meta.append(f)
        np.savez_compressed(out, meta=json.dumps(meta), **{k: np.concatenate(v) for k, v in acc.items()})
        print(singer, flush=True)
    cmd_report()


def _auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def cmd_report() -> None:
    prior = _mod("d142_flags").Prior()
    thr = _mod("d151_ptdb_flag").thresholds()[MODELS[0]]
    units, frames = [], []  # per file per block; per primary cell frame
    for p in sorted(CACHE.glob("*.npz")):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        fid, P, B1, f0 = z["fid"], z["P"], z["B1"], z["f0"]
        v = z[f"{MODELS[0]}|v"] > 0.5
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = v & np.all(np.isfinite(P), 1) & (P[:, 0] >= 350) & (P[:, 0] <= 650) & (P[:, 1] / P[:, 0] < 2.2)
            fail = {m: np.abs(np.log(z[f"{m}|F"][:, 1] / P[:, 2])) < np.abs(np.log(z[f"{m}|F"][:, 1] / P[:, 1])) for m in MODELS}
        # flag B, seed 2: the file's median score, length from the singer's other files (leave-one-out)
        m0 = MODELS[0]
        F, L = z[f"{m0}|F"], z[f"{m0}|L"]
        ok = v & np.all(np.isfinite(F), 1)
        um = np.array([np.nanmedian(L[(fid == u) & ok]) if ((fid == u) & ok).any() else np.nan for u in range(len(meta))])
        Lrec = np.array([np.nanmedian(np.delete(um, u)) for u in range(len(um))])
        B = np.full(len(v), np.nan)
        B[ok] = prior.score(F[ok], Lrec[fid[ok]])
        male = meta[0]["singer"].startswith("male")
        blocks = {"primary": (male, lambda x: x <= 150), "men_150_250": (male, lambda x: (x > 150) & (x <= 250)),
                  "women_le300": (not male, lambda x: x <= 300), "all_gt300": (True, lambda x: x > 300)}
        for u, mt in enumerate(meta):
            k = fid == u
            score = float(np.nanmedian(B[k & ok])) if (k & ok).any() else float("nan")
            units.append({"block": "file", "sex": "M" if male else "F", "score": score})
            for b, (use, sel) in blocks.items():
                if not use:
                    continue
                s = k & cell & sel(f0)
                if s.sum() < MIN_FRAMES:
                    continue
                share = float(np.mean([fail[m][s].mean() for m in MODELS]))
                units.append({"block": b, "singer": mt["singer"], "task": mt["task"], "tech": mt["tech"], "vowel": mt["vowel"],
                              "sex": "M" if male else "F", "fail": share, "s2fail": float(fail[m0][s].mean()), "score": score,
                              "b1": float(np.nanmedian(B1[s]))})
        if male:
            s = cell & (f0 <= 150)
            frames.append(np.c_[B1[s], np.mean([fail[m][s] for m in MODELS], 0)])
    lines = ["D-152: the dropped formant and flag B on VocalSet's held-out singers (scales, arpeggios, long tones; no lip trills)",
             f"  reference Praat Burg; frames voiced, in D-141's cell by Praat, f0 by 5b seed 2; a file counts with >= {MIN_FRAMES} "
             "such frames, fails at >= 0.5 (5b seed mean), OK below 0.2", ""]
    res = {"threshold": thr, "blocks": {}}
    titles = {"primary": "primary: men, f0 <= 150 Hz", "men_150_250": "men, f0 150-250 Hz", "women_le300": "women, f0 <= 300 Hz",
              "all_gt300": "everyone, f0 above 300 Hz (Praat unreliable)"}
    for b, title in titles.items():
        U = [x for x in units if x["block"] == b]
        nf = [x for x in U if x["fail"] >= 0.5]
        row = {"files": len(U), "singers": len({x["singer"] for x in U}), "failing": len(nf),
               "failing_singers": len({x["singer"] for x in nf}), "ok": sum(x["fail"] < 0.2 for x in U),
               "mean_fail": float(np.mean([x["fail"] for x in U])) if U else float("nan"),
               "b1_median": float(np.nanmedian([x["b1"] for x in U])) if U else float("nan")}
        res["blocks"][b] = row
        lines.append(f"  {title}: {row['files']} files from {row['singers']} singers; failing {row['failing']} "
                     f"({row['failing'] / max(row['files'], 1):.0%}) from {row['failing_singers']} singers; OK {row['ok']}; "
                     f"mean failure share {row['mean_fail']:.1%}; files' median Praat B1 {row['b1_median']:.0f} Hz")
    U = [x for x in units if x["block"] == "primary"]
    lines += ["", "  primary block, by vowel / task / technique (failing of counted files):"]
    for key in ("vowel", "task", "tech"):
        cnt = {}
        for x in U:
            c = cnt.setdefault(x[key], [0, 0])
            c[0] += x["fail"] >= 0.5
            c[1] += 1
        res["blocks"][f"primary_by_{key}"] = cnt
        lines.append(f"    {key}: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(cnt.items())))
    fr = np.concatenate(frames) if frames else np.zeros((0, 2))
    fr = fr[np.isfinite(fr[:, 0])]
    lines.append(f"  primary frames: {len(fr)}; Praat B1 median {np.median(fr[:, 0]):.0f} Hz, above 250 Hz {np.mean(fr[:, 0] > 250):.0%}")
    band = []
    for lo, hi in B1_BANDS:
        s = (fr[:, 0] >= lo) & (fr[:, 0] < hi)
        band.append(f"{lo:.0f}-{hi:.0f} Hz {fr[s, 1].mean():.0%} (n {s.sum()})" if s.any() else f"{lo:.0f}-{hi:.0f} Hz none")
    res["blocks"]["primary_frames_by_b1"] = band
    lines.append("    frame failure share by Praat B1: " + "; ".join(band))
    # flag B
    lines += ["", f"  flag B (5b seed 2, threshold {thr:.2f}), file median score:"]
    for sx in "FM":
        A = [x["score"] for x in units if x["block"] == "file" and x["sex"] == sx and np.isfinite(x["score"])]
        okf = [x["score"] for x in units if x["block"] in ("primary", "men_150_250", "women_le300") and x["sex"] == sx
               and x["fail"] < 0.2]
        lines.append(f"    {'women' if sx == 'F' else 'men':5s}: files above {np.mean(np.array(A) > thr):.0%} of {len(A)}; "
                     f"OK files (any block below 300 Hz) above {np.mean(np.array(okf) > thr) if okf else float('nan'):.0%} of {len(okf)}")
    pos = [x["score"] for x in U if x["fail"] >= 0.5 and np.isfinite(x["score"])]
    neg = [x["score"] for x in U if x["fail"] < 0.2 and np.isfinite(x["score"])]
    if len(pos) >= 5 and len(neg) >= 5:
        auc = _auc(pos, neg)
        rng = np.random.default_rng(0)
        bs = [_auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(1000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        flag = "HOLDS" if auc >= 0.8 and lo > 0.5 else "DOESN'T HOLD"
        lines.append(f"    primary: AUC failing ({len(pos)}) against OK ({len(neg)}) {auc:.2f} [{lo:.2f}, {hi:.2f}]")
        res["flag_auc"] = [auc, lo, hi]
    else:
        flag = "NOT TESTABLE"
        lines.append(f"    primary: {len(pos)} failing, {len(neg)} OK files; too few for the AUC")
    p = res["blocks"]["primary"]
    if p["files"] and p["failing"] / p["files"] >= 0.10 and p["failing_singers"] >= 2:
        fail = "PRESENT: the dropped formant appears on other singers' held vowels"
    elif p["failing"] <= 1:
        fail = "RARE: at most one file fails"
    else:
        fail = "SOME: reported, Hannah decides"
    res["reading"] = {"failure": fail, "flag": flag}
    lines += ["", f"Reading (registered): failure {fail}; flag B {flag}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d152.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d152.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
