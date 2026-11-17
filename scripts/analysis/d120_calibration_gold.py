"""D-120: D-071's stage B. Teacher v3.1's formant sigma and voicing probability on the dev gold sets, and what a sigma
gate would keep. Descriptive; dev tier only; nothing is fitted on a gold set.

    analyse: VA_DEVICE=cuda python scripts/analysis/d120_calibration_gold.py analyse     (cache/d120/<model>.npz)
    report:  python scripts/analysis/d120_calibration_gold.py report > reports/analysis/d120/d120_stage_b.txt   (CPU)

Hillenbrand dev: every file's reference frames through analyse() (windowed, the default). PTDB-TUG formants: the windowed
closed-phase cache (cache/m3w/ptdb_egg_sub.npz), which already holds both seeds' F1-F3 and formant sigma; D-091's frames.
PTDB-TUG voicing: every third dev utterance per speaker, voiced_p at every reference frame against the laryngograph.

Measures: (1) D-071's cell statistics (its `_cell` and `_ece`, imported; speakers in place of clips); (2) the reference SD
that would make 1-sigma coverage nominal; (3) Spearman's correlation of sigma with |error| and the gate curve; (4) what a
gate keeps by f0 band and sex; (5) F1 within 100 cents of f0 or 2 f0 against the rest.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")  # v3.1 first
CACHE = Path("cache/d120")
EGG = Path("cache/m3w/ptdb_egg_sub.npz")
MIN_SPK, MIN_FRAMES, SEED = 5, 200, 5120
Q1 = 0.6826894921370859
GATES = (0.25, 0.50, 0.75, 1.00)
BANDS = ("<150", "150-300", "300-500", ">500")
GROUPS = {"m": "men", "w": "women", "c": "children"}


# --- GPU -------------------------------------------------------------------------------------------------------------
def analyse(names):
    import torch

    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.device import setup_device
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry
    from vocal_analysis.models.infer import analyse as run, load_teacher

    dev = setup_device()
    mods = registry()
    CACHE.mkdir(parents=True, exist_ok=True)
    hb = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        ok = np.isfinite(item.times)
        if ok.any():
            f0 = np.asarray(item.f0, float)
            f0 = float(np.nanmax(f0)) if np.isfinite(f0).any() else np.nan  # labelled at steady state only: the file's value
            hb.append((item, item.times[ok], np.asarray(item.formants, float)[ok], f0, row.speaker_id, item.meta.get("group", "?")))
    by = {}
    for row in TI.filter_rows(read_manifest("data/manifests/ptdb_tug.csv"), ("dev",)):
        by.setdefault(row.speaker_id, []).append(row)
    pt = [(row, spk) for spk, rows in sorted(by.items()) for row in rows[::3]]
    for name in names:
        out = CACHE / f"{name}.npz"
        if out.exists():
            continue
        model = load_teacher(f"runs/{name}/ckpt_last.pt", dev)[0]
        H = {k: [] for k in ("pred", "sigma", "ref", "f0", "spk", "grp")}
        for item, t, ref, f0, spk, g in hb:
            r = run(model, load_audio(item, 16000).astype(np.float32), dev)
            idx = np.clip(np.round(t / 0.01).astype(int), 0, len(r["f0"]) - 1)
            H["pred"].append(r["formants"][idx, :4]), H["sigma"].append(r["formant_sigma_cents"][idx, :4]), H["ref"].append(ref[:, :4])
            H["f0"].append(np.full(len(idx), f0))
            H["spk"] += [spk] * len(idx)
            H["grp"] += [g] * len(idx)
        print(f"  {name} hillenbrand: {len(hb)} files", file=sys.stderr, flush=True)
        V = {k: [] for k in ("p", "y", "spk")}
        for row, spk in pt:
            item = mods["ptdb_tug"].load(Path("data/ptdb_tug"), row)
            k = np.isfinite(item.times)
            r = run(model, load_audio(item, 16000).astype(np.float32), dev)
            idx = np.clip(np.round(item.times[k] / 0.01).astype(int), 0, len(r["f0"]) - 1)
            f0 = np.asarray(item.f0, float)[k]
            V["p"].append(r["voiced_p"][idx]), V["y"].append(np.isfinite(f0) & (f0 > 0))
            V["spk"] += [spk] * len(idx)
        print(f"  {name} ptdb voicing: {len(pt)} utterances", file=sys.stderr, flush=True)
        np.savez(out, hb_pred=np.concatenate(H["pred"]), hb_sigma=np.concatenate(H["sigma"]), hb_ref=np.concatenate(H["ref"]),
                 hb_f0=np.concatenate(H["f0"]), hb_spk=np.array(H["spk"]), hb_grp=np.array(H["grp"]),
                 v_p=np.concatenate(V["p"]), v_y=np.concatenate(V["y"]), v_spk=np.array(V["spk"]))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# --- CPU -------------------------------------------------------------------------------------------------------------
def load_sets(name):
    """-> {set: dict(err (N, F) cents, sigma (N, F) cents, spk, grp ('m' / 'w' / 'c'), f0, ref_f1, gaps {label: (N, F) cents})}"""
    z = np.load(CACHE / f"{name}.npz")
    ref = np.where(z["hb_ref"] > 0, z["hb_ref"], np.nan)
    with np.errstate(all="ignore"):
        sets = {"Hillenbrand dev": {"err": 1200 * np.log2(z["hb_pred"] / ref), "sigma": z["hb_sigma"].astype(float), "spk": z["hb_spk"],
                                    "grp": np.array([g if g in "mw" else "c" for g in z["hb_grp"]]), "f0": z["hb_f0"], "ref_f1": ref[:, 0], "gaps": {}}}
        d = dict(np.load(EGG, allow_pickle=False))
        tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
        ok = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
        ok &= np.isfinite(d["cp"][:, 0]) & np.isfinite(d["cycle"][:, 0]) & np.isfinite(d["praat"][:, 0]) & (d["f0"] > 0)  # D-091's frames
        ok &= np.all(np.isfinite(d[name]), 1)
        sets["PTDB-TUG dev (closed-phase reference)"] = {
            "err": 1200 * np.log2(d[name][ok] / d["cp"][ok]), "sigma": d[f"{name}__sigma"][ok], "spk": d["speaker"][ok],
            "grp": np.where(d["sex"][ok] == "f", "w", "m"), "f0": d["f0"][ok], "ref_f1": d["cp"][ok, 0],
            "gaps": {"whole-cycle LPC - closed phase": 1200 * np.log2(d["cycle"][ok] / d["cp"][ok]),
                     "Praat - closed phase": 1200 * np.log2(d["praat"][ok] / d["cp"][ok])}}
    return sets, (z["v_p"].astype(float), z["v_y"].astype(bool), z["v_spk"])


def s_ref(e, sg):
    """the reference SD (cents) that makes 1-sigma coverage nominal: mean(|e| <= sqrt(sigma^2 + s^2)) = Q1; 0 if already there"""
    cov = lambda s: np.mean(np.abs(e) <= np.sqrt(sg ** 2 + s ** 2))  # noqa: E731
    if cov(0.0) >= Q1:
        return 0.0
    lo, hi = 0.0, 4000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if cov(mid) >= Q1 else (mid, hi)
    return hi


def spk_draws(spk, rng, n):
    us = np.unique(spk)
    idx = {s: np.where(spk == s)[0] for s in us}
    for _ in range(n):
        yield np.concatenate([idx[s] for s in rng.choice(us, len(us))])


def pct(v):
    return "[{:.2f}, {:.2f}]".format(*np.percentile(v, [2.5, 97.5]))


def report(names):
    from scipy.stats import spearmanr

    import d071_calibration as D071
    from vocal_analysis.eval.metrics import f0_band_labels

    D071.MIN_CLIPS = MIN_SPK  # D-071's `_cell` with speakers as the resampling unit (D-120's change for gold)
    L = ["D-120 (D-071 stage B): teacher v3.1's formant sigma and voicing probability on the dev gold sets; descriptive",
         "z = error in cents / formant_sigma_cents; s = the 0.683 quantile of |z| (1 = nominal; > 1 sigma too narrow, or the reference's "
         "own error); intervals: speaker bootstrap", "on a gold set s and the flag include the reference's error: they are not read as miscalibration", ""]
    for name in names:
        if not (CACHE / f"{name}.npz").exists():
            continue
        rng = np.random.default_rng(SEED)
        sets, (vp, vy, vspk) = load_sets(name)
        L.append(f"=================== {name}{' (v3.1; decides)' if name == MODELS[0] else ' (beside)'} ===================")
        for sname, S in sets.items():
            nF = S["err"].shape[1]
            band = f0_band_labels(S["f0"])
            L.append(f"== {sname}: {len(S['spk'])} frames, {len(np.unique(S['spk']))} speakers")
            L.append("  (1, 2) cells: n frames / speakers; s [95 %]; 1-sigma and 2-sigma coverage (2-sigma implied by s); RMS z; "
                     "s_ref = the reference SD making coverage nominal, cents [95 %]; flags as D-071 prints them")
            for g in [*GROUPS, "all"]:
                gm = np.ones(len(S["spk"]), bool) if g == "all" else S["grp"] == g
                if not gm.any():
                    continue
                for b in ["all", *BANDS]:
                    m = gm & ((band == b) if b != "all" else True)
                    for j in range(nF):
                        k = m & np.isfinite(S["err"][:, j]) & np.isfinite(S["sigma"][:, j]) & (S["sigma"][:, j] > 0)
                        if k.sum() < MIN_FRAMES:
                            continue
                        e, sg, sp = S["err"][k, j], S["sigma"][k, j], S["spk"][k]
                        c = D071._cell(e / sg, sp, rng)
                        if c is None:
                            continue
                        sr = s_ref(e, sg)
                        srb = [s_ref(e[i], sg[i]) for i in spk_draws(sp, rng, 200)]
                        gaps = "".join(f"; {lab} {np.quantile(np.abs(v[k, j][np.isfinite(v[k, j])]), Q1):.0f} c"
                                       for lab, v in S["gaps"].items()) if b == "all" else ""
                        L.append(f"    {GROUPS.get(g, 'all'):8s} f0 {b:8s} F{j + 1}: n {c['frames']:5d} / {c['clips']:2d}; s {c['s']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]; "
                                 f"coverage {c['c1']:.3f} / {c['c2']:.3f} ({c['c2_implied']:.3f}); RMS z {c['rms']:.2f}; s_ref {sr:.0f} c "
                                 f"[{np.percentile(srb, 2.5):.0f}, {np.percentile(srb, 97.5):.0f}]; median sigma {np.median(sg):.0f} c, median |error| {np.median(np.abs(e)):.0f} c"
                                 + (" FLAG" if c["flag"] else "") + (" tails" if c["tails"] else "")
                                 + (f"   0.683 quantile of |reference gap|{gaps}" if gaps else ""))
            L.append("  (3) does sigma sort frames by error? Spearman's rho(sigma, |error|) [95 %]; gate curve: median |error| (c) of the frames "
                     "kept in order of sigma at 25 / 50 / 75 / 100 %; of the 25 % dropped at the 75 % gate")
            for g in [*GROUPS, "all"]:
                gm = np.ones(len(S["spk"]), bool) if g == "all" else S["grp"] == g
                for j in range(nF):
                    k = gm & np.isfinite(S["err"][:, j]) & np.isfinite(S["sigma"][:, j])
                    if k.sum() < MIN_FRAMES or len(np.unique(S["spk"][k])) < MIN_SPK:
                        continue
                    ae, sg, sp = np.abs(S["err"][k, j]), S["sigma"][k, j], S["spk"][k]
                    rho = spearmanr(sg, ae).statistic
                    rb = [spearmanr(sg[i], ae[i]).statistic for i in spk_draws(sp, rng, 300)]
                    o = np.argsort(sg)
                    kept = [np.median(ae[o[:max(1, int(q * len(o)))]]) for q in GATES]
                    dropped = np.median(ae[o[int(0.75 * len(o)):]])
                    lo = np.percentile(rb, 2.5)
                    stands = lo > 0 and kept[1] <= 0.8 * kept[3]
                    L.append(f"    {GROUPS.get(g, 'all'):8s} F{j + 1}: rho {rho:+.2f} {pct(rb)}; kept " + " / ".join(f"{x:.0f}" for x in kept)
                             + f"; dropped at 75 % {dropped:.0f}; 50 % gate's kept against all {kept[1] / kept[3]:.2f} -> "
                             + ("something to stand on" if stands else "not by the reading's bar"))
            L.append("  (4) what a gate keeps: share of each group's frames kept at the 50 % / 75 % gate (threshold over the set's frames, per formant)")
            for j in range(min(nF, 3)):
                k = np.isfinite(S["err"][:, j]) & np.isfinite(S["sigma"][:, j])
                th = np.quantile(S["sigma"][k, j], [0.5, 0.75])
                cells = []
                for lab, m in [*[(GROUPS[g], S["grp"] == g) for g in GROUPS], *[(f"f0 {b}", band == b) for b in BANDS]]:
                    mm = m & k
                    if mm.sum() >= MIN_FRAMES:
                        cells.append(f"{lab} {np.mean(S['sigma'][mm, j] <= th[0]):.2f} / {np.mean(S['sigma'][mm, j] <= th[1]):.2f} (n {int(mm.sum())})")
                L.append(f"    F{j + 1}: " + "; ".join(cells))
            L.append("  (5) F1 within 100 cents of f0 or 2 f0 (reference F1) against the rest: n; s; 1-sigma coverage; median |error| c; median sigma c")
            with np.errstate(all="ignore"):
                near = np.minimum(np.abs(1200 * np.log2(S["ref_f1"] / S["f0"])), np.abs(1200 * np.log2(S["ref_f1"] / (2 * S["f0"])))) <= 100
            for g in [*GROUPS, "all"]:
                gm = np.ones(len(S["spk"]), bool) if g == "all" else S["grp"] == g
                parts = []
                for lab, m in (("near a harmonic", near), ("the rest", ~near)):
                    k = gm & m & np.isfinite(S["err"][:, 0]) & np.isfinite(S["sigma"][:, 0]) & np.isfinite(S["f0"]) & (S["f0"] > 0)
                    if k.sum() < MIN_FRAMES:
                        parts.append(f"{lab}: n {int(k.sum())} (too few)")
                        continue
                    zz = S["err"][k, 0] / S["sigma"][k, 0]
                    parts.append(f"{lab}: n {int(k.sum())}; s {np.quantile(np.abs(zz), Q1):.2f}; coverage {np.mean(np.abs(zz) <= 1):.3f}; "
                                 f"median |error| {np.median(np.abs(S['err'][k, 0])):.0f}; median sigma {np.median(S['sigma'][k, 0]):.0f}")
                if gm.any():
                    L.append(f"    {GROUPS.get(g, 'all'):8s} " + " | ".join(parts))
            L.append("")
        L.append("== PTDB-TUG dev, voicing: voiced_p at every reference frame against the laryngograph's voicing (every third utterance)")
        sex = np.array(["w" if s.split(":")[-1].startswith("F") else "m" for s in vspk])
        for g in ("m", "w", "all"):
            k = np.ones(len(vp), bool) if g == "all" else sex == g
            p, y, sp = vp[k], vy[k], vspk[k]
            eb = [D071._ece(p[i], y[i]) for i in spk_draws(sp, rng, 300)]
            L.append(f"    {GROUPS.get(g, 'all'):8s} n {int(k.sum())}; ECE {D071._ece(p, y):.3f} {pct(eb)} (D-071's bar 0.05); reference voiced share {y.mean():.3f}, "
                     f"mean voiced_p {p.mean():.3f}; accuracy at 0.5 {np.mean((p > 0.5) == y):.3f}")
        L.append("")
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "analyse":
        analyse(sys.argv[2:] or MODELS)
    elif cmd == "report":
        report(sys.argv[2:] or MODELS)
    else:
        print(__doc__)
        sys.exit(2)
