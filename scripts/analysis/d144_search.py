"""D-144: a joint search at inference for the dropped formant. Per clip or take, two hypotheses are fitted to the
reconstruction loss (D-109's, as D-143 stage C):
  A: the model's F1-F4 is right (its lowest formant refit by a factor of 0.85-1.15, B1 free);
  B: the model skipped F1 (a new F1 inserted at 300-700 Hz with B1 free; the model's F1-F3 move up one slot).
B is chosen when its loss is lower than A's by more than a margin set on the synthetic clips. Inference only.

    python scripts/analysis/d144_search.py synth                 -> reports/analysis/d144/synth.json
    python scripts/analysis/d144_search.py private REC OUT FILES -> OUT/d144_private.json   (outside the repo)
    python scripts/analysis/d144_search.py report [OUT]          -> reports/analysis/d144/d144.txt (aggregates only)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d144"
MODELS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
SEEDS = (30, 31, 32)
PROFILES = ("offset", "tilt")
A_FAC = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
B1G = tuple(float(x) for x in np.round(np.exp(np.linspace(np.log(50), np.log(700), 12))))
F1G = tuple(float(x) for x in range(300, 701, 25))
FALSE_FIX = 0.05
B = 2000


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _models():
    import torch

    from vocal_analysis.models.infer import load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    return dev, {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in MODELS}


def _frames(r, y):
    n = len(r["voiced_p"])
    sel = np.zeros(n, bool)
    for i in range(int(0.2 * n), int(0.8 * n)):
        sel[i] = (r["voiced_p"][i] > 0.5 and np.isfinite(r["f0"][i]) and i * 160 >= 512 and i * 160 + 512 <= len(y)
                  and np.all(np.isfinite(r["formants"][i, :4])) and np.all(np.isfinite(r["bandwidths"][i, :4])))
    return sel


def _losses(cands, lab, profile):
    """cands: list of (F (T, 4), B (T, 4)) -> mean loss per candidate over the frames every candidate scores."""
    import torch

    from vocal_analysis.train.reconstruction import recon_loss_fb

    out = []
    for s in range(0, len(cands), 64):
        ch = cands[s:s + 64]
        F = torch.from_numpy(np.stack([c[0] for c in ch])).float()
        Bw = torch.from_numpy(np.stack([c[1] for c in ch])).float()
        lb = {k: v.expand(len(ch), *v.shape[1:]) for k, v in lab.items()}
        with torch.no_grad():
            out.append(recon_loss_fb(F, Bw, lb, fmax=4000.0, hpc=True, per_frame=True, profile=profile).numpy())
    L = np.concatenate(out)
    ok = np.all(np.isfinite(L), 0)
    return L[:, ok].mean(1) if ok.any() else np.full(len(L), np.nan)


def search(model, dev, y):
    """The model's reading of y, then both hypotheses under each source profile."""
    import torch

    from vocal_analysis.models.infer import analyse
    from vocal_analysis.train.reconstruction import harmonic_targets

    r = analyse(model, y.astype(np.float32), dev)
    sel = _frames(r, y)
    if sel.sum() < 10:
        return None
    f0 = r["f0"].astype(np.float64)
    amp, ok = harmonic_targets(y, np.where(sel, f0, np.nan), sel, snr_min_db=10.0)
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)[None]
    lab = {"db": t(amp[sel]), "ok": t(ok[sel], torch.bool), "f0": t(f0[sel]), "mask": t(np.ones(sel.sum()), torch.bool)}
    Fm, Bm = r["formants"][sel, :4].astype(np.float64), r["bandwidths"][sel, :4].astype(np.float64)
    m1 = float(np.median(Fm[:, 0]))

    def cand_a(a, b1):
        F, Bw = Fm.copy(), Bm.copy()
        F[:, 0] = np.minimum(F[:, 0] * a, 0.95 * F[:, 1])
        Bw[:, 0] = b1
        return F, Bw

    def cand_b(f1, b1):
        return (np.c_[np.full(len(Fm), f1), Fm[:, :3]], np.c_[np.full(len(Fm), b1), Bm[:, :3]])

    out = {"fail_frames": Fm, "f0": float(np.median(f0[sel])), "n": int(sel.sum())}
    for prof in PROFILES:
        A = [(a, b) for a in A_FAC for b in B1G]
        Bc = [(f, b) for f in F1G for b in B1G if f < 0.9 * m1]
        la = _losses([cand_a(*p) for p in A], lab, prof)
        ia = int(np.nanargmin(la))
        res = {"A": float(la[ia]), "A_fac": A[ia][0], "A_b1": A[ia][1]}
        if Bc:
            lb = _losses([cand_b(*p) for p in Bc], lab, prof)
            ib = int(np.nanargmin(lb))
            f1, b1 = Bc[ib]
            k = B1G.index(b1)
            fine = [(f, b) for f in np.arange(f1 - 20, f1 + 21, 5.0) for b in B1G[max(0, k - 1):k + 2]
                    if 250 <= f < 0.9 * m1]
            lf = _losses([cand_b(*p) for p in fine], lab, prof)
            j = int(np.nanargmin(lf))
            res.update(B=float(min(lf[j], lb[ib])), B_f1=float(fine[j][0] if lf[j] <= lb[ib] else f1),
                       B_b1=float(fine[j][1] if lf[j] <= lb[ib] else b1))
        else:
            res.update(B=float("nan"), B_f1=float("nan"), B_b1=float("nan"))
        out[prof] = res
    return out


def _fail(Fm, true):
    return float(np.mean(np.abs(np.log(Fm[:, 1] / true[2])) < np.abs(np.log(Fm[:, 1] / true[1]))))


def _run(groups, models, dev):
    """groups: lists of (key, audio, true F1-F4 or reference); flag B from D-142's code, one 'recording' per group."""
    d142 = _mod("d142", "scripts/analysis/d142_flags.py")
    prior = d142.Prior()
    res = {}
    for items in groups:
        fb = d142._score_group(items, models, dev, prior)
        for key, y, true in items:
            for m, mod in models.items():
                s = search(mod, dev, y)
                if s is None or key not in fb or m not in fb[key]:
                    continue
                Fm = s.pop("fail_frames")
                s.update(fail=_fail(Fm, true), flagB=fb[key][m]["B"], true=[float(x) for x in true[:4]],
                         read_F1=float(np.median(Fm[:, 0])))
                res.setdefault(key, {})[m] = s
            print(".", end="", flush=True)
    return res


def cmd_synth() -> None:
    d142 = _mod("d142", "scripts/analysis/d142_flags.py")
    d131 = _mod("d131", "scripts/analysis/d131_context.py")
    from vocal_analysis.synth import tube

    dev, models = _models()
    groups = [[(f"{sname}|f{f0:g}|b{b1:g}|s{s}", d142._grid_audio(Fs, f0, b1, s), np.array(Fs))
               for f0 in d142.F0S for b1 in d142.B1S for s in SEEDS] for sname, Fs in d142.SETS.items()]
    story = []
    for v, q in tube.STORY_VOWELS.items():
        f, bw = tube.poles(tube.build_shape(*q, 15.6), f_max_hz=7600.0)
        k = min(d131.d130.N_P, len(f))
        F = np.full((2, d131.d130.N_P), np.nan)
        Bw = F.copy()
        F[:, :k], Bw[:, :k] = f[:k], bw[:k]
        for f0 in (110.0, 220.0):
            tag = "".join(c + "^" if c.isupper() else c for c in v)
            y, _, _ = d131.render_jit(F, Bw, np.array([0.0, 2.0]), f0, 1440 + int(f0) + len(story))
            story.append((f"story|{tag}|f{f0:g}", y.astype(np.float64), np.array(f[:4])))
    groups.append(story)
    res = _run(groups, models, dev)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "synth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_private(rec: Path, out: Path, files: list[str]) -> None:
    d138 = _mod("d138", "scripts/analysis/d138_real_balance.py")
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    dev, models = _models()
    groups = {}
    for name, _, y in d138._load(rec, files):
        if name in lv:
            groups.setdefault(name.split("_")[0], []).append((name, y, np.array(lv[name]["F"][:4], float)))
    res = _run(list(groups.values()), models, dev)
    for name, v in res.items():
        v["group"] = ("fail" if np.mean([lv[name]["read"][m]["fail"] for m in d138.TEACHERS]) >= 0.5 else
                      "ok" if np.mean([lv[name]["read"][m]["fail"] for m in d138.TEACHERS]) < 0.2 else "mid")
        v["vowel"], v["take_f0"] = lv[name]["vowel"], float(lv[name]["f0"])
    (out / "d144_private.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def _cents(a, b):
    return abs(1200 * np.log2(a / b))


def cmd_report(out: Path | None) -> None:
    syn = json.loads((REP / "synth.json").read_text(encoding="utf-8"))
    prv = json.loads((out / "d144_private.json").read_text(encoding="utf-8")) if out else {}
    lines = ["D-144: joint search for the dropped formant (B: an F1 inserted below the model's; A: the model's reading), "
             "per clip or take; B chosen when A's loss exceeds B's by more than the margin", ""]
    verdict = {}
    for m in MODELS:
        S = {k: v[m] for k, v in syn.items() if m in v}
        grid = {k: v for k, v in S.items() if not k.startswith("story")}
        okg = [v for v in grid.values() if v["fail"] < 0.2]
        thr_b = np.percentile([v["flagB"] for v in okg], 95)
        normal = okg + [v for k, v in S.items() if k.startswith("story") and v["fail"] < 0.2]
        P = {k: v[m] for k, v in prv.items() if m in v}
        dec = {}
        lines.append(f"{m}: flag B threshold {thr_b:.2f} (95th percentile of not-failing grid clips)")
        for prof in PROFILES:
            d_norm = np.array([v[prof]["A"] - v[prof]["B"] for v in normal if np.isfinite(v[prof]["B"])])
            margin = max(0.0, float(np.percentile(d_norm, 100 * (1 - FALSE_FIX)))) if len(d_norm) else 0.0
            pick = lambda v, gate: bool(np.isfinite(v[prof]["B"]) and v[prof]["A"] - v[prof]["B"] > margin
                                        and (not gate or v["flagB"] > thr_b))
            hit = lambda v, gate, tol: pick(v, gate) and _cents(v[prof]["B_f1"], v["true"][0]) <= tol
            lines.append(f"  profile {prof}: margin {margin:.2f} (synthetic false-fix share {FALSE_FIX:.0%} on "
                         f"{len(d_norm)} normal clips)")
            for gate in (True, False):
                g = "gated by flag B" if gate else "ungated"
                fs = [v for v in grid.values() if v["fail"] >= 0.5]
                sh = np.mean([hit(v, gate, 100) for v in fs]) if fs else float("nan")
                sf = np.mean([pick(v, gate) for v in normal])
                row = f"    {g}: synthetic hits {sh:.0%} of {len(fs)} failing (within 100 c), fixes on normal {sf:.0%}"
                if P:
                    pf = [v for v in P.values() if prv_group(prv, P, v) == "fail"]
                    po = [v for v in P.values() if prv_group(prv, P, v) == "ok"]
                    ph = sum(hit(v, gate, 150) for v in pf)
                    pff = sum(pick(v, gate) for v in po)
                    row += f"; private hits {ph} of {len(pf)} (within 150 c of Praat), false fixes {pff} of {len(po)}"
                    if gate and prof == "offset":
                        verdict[m] = dict(S=sh >= 0.8, P=ph >= 5 and pff <= 2)
                lines.append(row)
            dec[prof] = {k: pick(v, True) for k, v in list(grid.items()) + [(k, v) for k, v in S.items() if k.startswith("story")]}
            dec[prof + "_p"] = {k: pick(v, True) for k, v in P.items()}
            if prof == "offset":
                chosen = [v for v in list(S.values()) + list(P.values()) if pick(v, False)]
                if chosen:
                    h = np.array([abs(v[prof]["B_f1"] / v["f0"] - round(v[prof]["B_f1"] / v["f0"])) for v in chosen])
                    lines.append(f"    harmonic check (ungated B choices, n {len(h)}): mean distance of the inserted F1 / f0 "
                                 f"from an integer {h.mean():.2f} (0.25 if unrelated; under 0.15 would suggest locking)")
        agree_s = np.mean([dec["offset"][k] == dec["tilt"][k] for k in dec["offset"]])
        agree_p = np.mean([dec["offset_p"][k] == dec["tilt_p"][k] for k in dec["offset_p"]]) if P else float("nan")
        lines.append(f"  source check: gated decisions agree between profiles on {agree_s:.0%} of synthetic clips"
                     + (f", {agree_p:.0%} of private takes" if P else ""))
        if m in verdict:
            verdict[m]["R"] = agree_s >= 0.9 and agree_p >= 0.9
        if P:  # the same vowel's higher-pitched takes that the models read correctly
            pf = [(k, v) for k, v in P.items() if prv[k]["group"] == "fail" and np.isfinite(v["offset"]["B"])]
            ds = []
            for k, v in pf:
                hi = [w["read_F1"] for kk, w in P.items() if prv[kk]["vowel"] == prv[k]["vowel"] and prv[kk]["group"] == "ok"
                      and prv[kk]["take_f0"] >= 1.5 * prv[k]["take_f0"]]
                if hi:
                    ds.append(_cents(v["offset"]["B_f1"], np.median(hi)))
            if ds:
                lines.append(f"  higher-pitched same-vowel takes: median |inserted F1 - their F1| {np.median(ds):.0f} c (n {len(ds)})")
        lines.append("")
    if prv:
        both = [k for k in prv if all(m in prv[k] for m in MODELS)]
        diff = [k for k in both if abs(prv[k][MODELS[0]]["fail"] - prv[k][MODELS[1]]["fail"]) >= 0.3]
        good = sum((prv[k][MODELS[0]]["offset"]["A"] < prv[k][MODELS[1]]["offset"]["A"]) ==
                   (prv[k][MODELS[0]]["fail"] < prv[k][MODELS[1]]["fail"]) for k in diff)
        lines.append(f"Models as candidates (reported): takes where the seeds' failure shares differ by >= 0.3: {len(diff)}; "
                     f"the loss (A, refit) picks the less-failing seed on {good}")
        ok = all(all(v.values()) for v in verdict.values()) and len(verdict) == len(MODELS)
        lines.append(f"Reading (registered; gated, offset profile, both models): {'WORKS' if ok else 'NOT SHOWN'} "
                     f"({ {m: v for m, v in verdict.items()} })")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d144.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def prv_group(prv, P, v):
    for k, w in P.items():
        if w is v:
            return prv[k]["group"]
    return None


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "synth":
        cmd_synth()
    elif cmd == "private":
        cmd_private(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    elif cmd == "report":
        cmd_report(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
