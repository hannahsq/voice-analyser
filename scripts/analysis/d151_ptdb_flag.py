"""D-151: flag B (D-142's recording-level plausibility) swept over PTDB-TUG's dev tier. Where does it fire on read speech,
and do the flagged frames look like the dropped formant against Praat? CPU, inference only; dev tier only.

    python scripts/analysis/d151_ptdb_flag.py run      -> cache/d151/*.npz (per frame, not committed), reports/analysis/d151/d151.{txt,json}
    python scripts/analysis/d151_ptdb_flag.py report   -> reports/analysis/d151/d151.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d151"
CACHE = ROOT / "cache/d151"
MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")  # deciding, reported (D-142's order)
RUN = 5  # a flagged run: at least 5 consecutive voiced frames (50 ms) above the threshold


def _d142():
    spec = importlib.util.spec_from_file_location("d142", ROOT / "scripts/analysis/d142_flags.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def thresholds():
    """D-142's: the 95th percentile of the not-failing (< 0.2) grid clips' B scores, per model."""
    d = json.loads((ROOT / "reports/analysis/d142/d142_synth.json").read_text(encoding="utf-8"))
    grid = {k: v for k, v in d.items() if not k.startswith("story")}
    return {m: float(np.percentile([v[m]["B"] for v in grid.values() if m in v and v[m]["fail"] < 0.2], 95)) for m in MODELS}


def _praat(y, ceiling, n):
    import parselmouth

    fm = parselmouth.Sound(y.astype(np.float64), 16000).to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                                                      maximum_formant=ceiling)
    ts = np.arange(n) * 0.01
    F = np.array([[fm.get_value_at_time(k, t) for k in (1, 2, 3)] for t in ts], float)
    B1 = np.array([fm.get_bandwidth_at_time(1, t) for t in ts], float)
    return F, B1


def cmd_run() -> None:
    import torch

    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry
    from vocal_analysis.models.infer import analyse, load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    models = {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}
    mod = registry()["ptdb_tug"]
    rows = TI.filter_rows(read_manifest(str(ROOT / "data/manifests/ptdb_tug.csv")), ("dev",))
    CACHE.mkdir(parents=True, exist_ok=True)
    for spk in sorted({r.speaker_id for r in rows}):
        out = CACHE / f"{spk.replace(':', '_')}.npz"
        if out.exists():
            continue
        sex = spk.split(":")[-1][0]
        acc = {k: [] for k in ("utt", "f0", "P", "B1")}
        acc.update({f"{m}|{k}": [] for m in MODELS for k in ("F", "L", "v")})
        for j, r in enumerate([r for r in rows if r.speaker_id == spk]):
            y = load_audio(mod.load(ROOT / "data/ptdb_tug", r))
            res = {m: analyse(md, y, dev) for m, md in models.items()}
            n = len(res[MODELS[0]]["voiced_p"])
            P, B1 = _praat(y, 5000.0 if sex == "M" else 5500.0, n)
            acc["utt"].append(np.full(n, j))
            acc["f0"].append(res[MODELS[0]]["f0"])
            acc["P"].append(P)
            acc["B1"].append(B1)
            for m in MODELS:
                acc[f"{m}|F"].append(res[m]["formants"][:, :4])
                acc[f"{m}|L"].append(2 ** res[m]["vtl_log2cm"])
                acc[f"{m}|v"].append(res[m]["voiced_p"])
        np.savez_compressed(out, sex=sex, **{k: np.concatenate(v) for k, v in acc.items()})
        print(spk, flush=True)
    cmd_report()


def _runs(mask):
    """Frames belonging to runs of >= RUN consecutive True."""
    out = np.zeros_like(mask)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= RUN:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def cmd_report() -> None:
    prior = _d142().Prior()
    thr = thresholds()
    res = {"threshold": thr, "speakers": {}}
    agg = {m: {s: {"utt_flag": [], "frames": 0, "flagged": 0, "drop_f": [], "drop_u": [], "f0_f": [], "f0_u": [],
                   "b1_f": [], "b1_u": [], "f1err_f": [], "f1err_u": []} for s in "FM"} for m in MODELS}
    for p in sorted(CACHE.glob("*.npz")):
        z = np.load(p)
        sex = str(z["sex"])
        for m in MODELS:
            v = z[f"{m}|v"] > 0.5
            F, L, utt = z[f"{m}|F"], z[f"{m}|L"], z["utt"]
            # the recording's length: median over the speaker's other utterances' median readings (leave-one-out)
            um = np.array([np.nanmedian(L[(utt == u) & v]) if ((utt == u) & v).any() else np.nan for u in np.unique(utt)])
            Lrec = np.array([np.nanmedian(np.delete(um, u)) for u in range(len(um))])
            B = np.full(len(v), np.nan)
            ok = v & np.all(np.isfinite(F), 1)
            B[ok] = prior.score(F[ok], Lrec[utt[ok]])
            above = ok & (B > thr[m])
            fl = np.zeros(len(v), bool)
            for u in np.unique(utt):
                k = utt == u
                fl[k] = _runs(above[k])
                if ok[k].any():
                    agg[m][sex]["utt_flag"].append(float(np.nanmedian(B[k & ok]) > thr[m]))
            P = z["P"]
            pok = ok & np.all(np.isfinite(P), 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                drop = np.abs(np.log(F[:, 1] / P[:, 2])) < np.abs(np.log(F[:, 1] / P[:, 1]))
                f1e = np.abs(np.log2(F[:, 0] / P[:, 0])) * 1200
            a = agg[m][sex]
            a["frames"] += int(ok.sum())
            a["flagged"] += int(fl.sum())
            for tag, sel in (("f", pok & fl), ("u", pok & ~fl)):
                a[f"drop_{tag}"].append(drop[sel])
                a[f"f0_{tag}"].append(z["f0"][sel])
                a[f"b1_{tag}"].append(z["B1"][sel])
                a[f"f1err_{tag}"].append(f1e[sel])
    lines = ["D-151: flag B (D-142) over PTDB-TUG's dev tier (12 speakers, read sentences)",
             f"  threshold per model: D-142's (95th percentile of not-failing grid clips): " +
             ", ".join(f"{m} {t:.2f}" for m, t in thr.items()),
             f"  a flagged run: >= {RUN} consecutive voiced frames above it; 'dropped against Praat': the model's F2 nearer "
             "Praat's F3 than its F2 (Praat Burg, ceiling 5000 Hz men, 5500 women)", ""]
    verdict = None
    for m in MODELS:
        lines.append(f"  {m}")
        tot = {"frames": 0, "flagged": 0, "df": [], "du": []}
        for s in "FM":
            a = agg[m][s]
            cat = {k: np.concatenate(a[k]) if a[k] else np.zeros(0) for k in a if k.endswith(("_f", "_u"))}
            df, du = cat["drop_f"], cat["drop_u"]
            tot["frames"] += a["frames"]
            tot["flagged"] += a["flagged"]
            tot["df"].append(df)
            tot["du"].append(du)
            row = {"utterances_flagged": float(np.mean(a["utt_flag"])) if a["utt_flag"] else float("nan"),
                   "frames": a["frames"], "flagged_share": a["flagged"] / max(a["frames"], 1),
                   "drop_flagged": float(df.mean()) if len(df) else float("nan"), "drop_unflagged": float(du.mean()) if len(du) else float("nan"),
                   "f0_flagged": float(np.nanmedian(cat["f0_f"])) if len(df) else float("nan"),
                   "f0_unflagged": float(np.nanmedian(cat["f0_u"])),
                   "praat_b1_flagged": float(np.nanmedian(cat["b1_f"])) if len(df) else float("nan"),
                   "praat_b1_unflagged": float(np.nanmedian(cat["b1_u"])),
                   "f1err_flagged": float(np.nanmedian(cat["f1err_f"])) if len(df) else float("nan"),
                   "f1err_unflagged": float(np.nanmedian(cat["f1err_u"]))}
            res["speakers"].setdefault(m, {})[s] = row
            lines.append(f"    {'women' if s == 'F' else 'men':5s}: utterances flagged (median score) {row['utterances_flagged']:.1%}; "
                         f"frames in flagged runs {row['flagged_share']:.2%} of {row['frames']}")
            lines.append(f"           dropped against Praat: flagged {row['drop_flagged']:.1%}, unflagged {row['drop_unflagged']:.1%}; "
                         f"median f0 {row['f0_flagged']:.0f} / {row['f0_unflagged']:.0f} Hz; Praat B1 {row['praat_b1_flagged']:.0f} / "
                         f"{row['praat_b1_unflagged']:.0f} Hz; |F1 - Praat F1| {row['f1err_flagged']:.0f} / {row['f1err_unflagged']:.0f} c")
        df, du = np.concatenate(tot["df"]), np.concatenate(tot["du"])
        share = tot["flagged"] / max(tot["frames"], 1)
        pf, pu = (df.mean() if len(df) else 0.0), (du.mean() if len(du) else 0.0)
        lines.append(f"    all: frames in flagged runs {share:.2%}; dropped against Praat {pf:.1%} flagged, {pu:.1%} unflagged")
        if m == MODELS[0]:
            if share < 0.001:
                verdict = "NOTHING FLAGGED: flag B hardly fires on these voices"
            elif pf >= 0.2 and pf >= 5 * pu:
                verdict = "PICKS UP DROPPED FORMANTS: the flagged frames are mostly the failure's pattern against Praat"
            elif pf < 2 * pu:
                verdict = "FLAGS SOMETHING ELSE: the flagged frames are not the failure's pattern"
            else:
                verdict = "PARTLY: reported, Hannah decides"
    lines += ["", f"Reading (registered; {MODELS[0]}): {verdict}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d151.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d151.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
