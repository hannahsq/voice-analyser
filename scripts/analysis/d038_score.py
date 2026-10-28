"""D-038 (draft) scoring: the fine-tuned formant heads (d038_head.py) on one trunk's D-037 cache, every formant reported.
Per arm (seed mean of the decoded log2 Hz over its seeds; also each seed alone) and for the trunk's own heads (T0, the
cached F1-F4):
- Hillenbrand dev F1-F4 MAE (Hz) at the reference frames, and the F3 / F4 mean (the decision's measure);
- PTDB-TUG dev closed phase (cp part): F1-F3 median signed error and median |error| by sex;
- synthetic: train_v2's held-out crops (val part), sonorant frames, F1-F4 MAE on tube clips and on all clips;
- the invariance probe: per formant, the median |slope| (Hz per octave of f0, f0 <= 250 Hz) of the clip's median estimate
  over 0.3-0.7 s, and the median error against the probe's true formants by length;
- the single-formant-shift probe (shift part, eval.invariance.shift_clips): for each tract, f0 and moved formant n, the
  slope of each estimate (log2 of the clip's median over 0.3-0.7 s) against log2 of the shift factor, with the unshifted
  clip at factor 1: *follow* = formant n's slope (ideal 1), *drag* = the mean |slope| of the other three (ideal 0); and
  the trunk's VTL head's slope (arm-independent; ideal 0). Paired bootstraps over the 18 tracts (vowel x length).
Paired speaker bootstraps (2,000) of each arm minus F (and Fi minus Fv): Hillenbrand MAE per formant and the F3 / F4 mean
(resampling speakers, frame-weighted), women's closed-phase F1 median |error|.
Usage: python d038_score.py <cache dir> <runs dir (holding F_s*, Fv_s*, Fi_s*)> <report .txt>"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d038_head as D  # noqa: E402

from vocal_analysis.eval import invariance as inv  # noqa: E402

B = 2000
ARMS = D.ARMS
GUARD_HZ = (4.0, 9.0, 20.0, 50.0)  # D-036's Hillenbrand guard-rails F1-F3, and F4's added at D-038's registration
CP_GUARD_HZ = 5.0
FOLLOW_DROP = (0.05, 0.02, 0.02, 0.02)  # shift probe: largest allowed drop of the follow slope, F1-F4
DRAG_RISE = 0.01  # shift probe: largest allowed rise of the drag
F0_RISE = 5.0  # f0 probe: largest allowed rise of a formant's |slope|, cents per octave


def part(cache, name):
    d = cache / name
    mu = np.load(d / "vtl_mu.npy") if (d / "vtl_mu.npy").exists() else None
    return np.load(d / "feats.npy"), mu, np.load(d / "vtl_logvar.npy") if mu is not None else None, np.load(d / "log2F.npy")


def load_head(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = D.build_model(ck["trunk"], ck["arm"], dev)
    model.load_state_dict(ck["state"])
    stats = None if ck["stats"] is None else tuple(np.asarray(s, np.float32) for s in ck["stats"])
    return model.eval(), ck["arm"], stats


def run_head(model, arm, stats, P, dev):
    """P = (feats, mu, logvar, _) -> log2 Hz (..., 4)"""
    h, mu, lv, _ = P
    shape = h.shape[:-1]
    h2 = h.reshape(-1, 1, h.shape[-1]) if h.ndim == 2 else h
    x = None if arm == "F" else D.standardise(arm, stats, mu.reshape(h2.shape[:-1]), lv.reshape(h2.shape[:-1]))
    return D.log2_hz(D.predict(model, h2, x, dev, chunk=64 if h.ndim == 3 else 8192)).reshape(*shape, 4)


def boot_speakers(err_a, err_b, spk, rng, stat=np.mean):
    """paired: stat over the resampled speakers' frames of (a - b); err_* (n,), NaN-free"""
    us = np.unique(spk)
    idx = [np.flatnonzero(spk == s) for s in us]
    obs = stat(err_a) - stat(err_b)
    bs = []
    for _ in range(B):
        j = np.concatenate([idx[k] for k in rng.integers(len(us), size=len(us))])
        bs.append(stat(err_a[j]) - stat(err_b[j]))
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def shift_slopes(p, sm):
    """p (clips, T, 4) log2 values; -> {n: (tract keys, follow (k,), drag (k,), slopes (k, 4))}"""
    est = np.nanmedian(p[:, 30:71], 1)
    out = {}
    for n in range(4):
        keys, fol, drag, allsl = [], [], [], []
        for v, L, f0 in sorted({(a, b, c) for a, b, c in zip(sm["vowel"], sm["length_cm"], sm["f0"])}):
            g = (sm["vowel"] == v) & (sm["length_cm"] == L) & (sm["f0"] == f0)
            i = np.flatnonzero(g & ((sm["formant"] == n) | (sm["formant"] == -1)))
            if len(i) < 3:
                continue
            x = np.log2(sm["factor"][i])
            s = np.array([np.polyfit(x, est[i, m], 1)[0] for m in range(4)])
            keys.append(f"{v}|{L}"), fol.append(s[n]), drag.append(np.mean(np.abs(np.delete(s, n)))), allsl.append(s)
        out[n] = (np.array(keys), np.array(fol), np.array(drag), np.array(allsl))
    return out


def boot_tracts(a, b, keys, rng):
    """paired: mean over the resampled tracts' rows of (a - b)"""
    us = np.unique(keys)
    idx = [np.flatnonzero(keys == u) for u in us]
    bs = []
    for _ in range(B):
        j = np.concatenate([idx[k] for k in rng.integers(len(us), size=len(us))])
        bs.append(np.mean(a[j] - b[j]))
    return float(np.mean(a - b)), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def main():
    cache, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    parts = {p: part(cache, p) for p in ("hb", "cp", "val", "probe", "shift") if (cache / p / "done").exists()}
    preds = {"T0": {p: parts[p][3] for p in parts}}  # the trunk's own heads (cached)
    seeds = {}
    for arm in ARMS:
        runs_a = sorted(p for p in runs.glob(f"{arm}_s*") if (p / "head.pt").exists())
        if not runs_a:
            continue
        per = []
        for r in runs_a:
            model, a, stats = load_head(r / "head.pt", dev)
            per.append({p: run_head(model, a, stats, parts[p], dev) for p in parts})
            preds[f"{arm}_{r.name.split('_')[-1]}"] = per[-1]
        seeds[arm] = [r.name for r in runs_a]
        preds[arm] = {p: np.mean([q[p] for q in per], 0) for p in parts}

    # references
    hb_ref = np.load("cache/emb_v1/gold_hillenbrand/ref_formants.npy")
    files = np.load(cache / "hb" / "files.npz")
    hb_spk = np.repeat(files["speaker"], np.diff(files["off"]))
    cpf = np.load(cache / "cp" / "frames.npz")
    V = D.val_data(cache)
    val_fm, val_son, val_tube = V[1], V[2], V[6]
    clips = inv.probe_clips()
    meta = np.load(cache / "probe" / "meta.npz")
    assert np.allclose([c.f0 for c in clips], meta["f0"]) and np.allclose([c.length_cm for c in clips], meta["length_cm"])
    truth = np.array([c.formants[:4] for c in clips])
    sm = dict(np.load(cache / "shift" / "meta.npz")) if "shift" in parts else None
    shifts, probe_c = {}, {}

    res, lines = {}, []
    say = lambda s="": lines.append(s)  # noqa: E731
    say(f"D-038 scoring, cache {cache}, runs {runs}; seeds {json.dumps(seeds)}")
    flags = {}
    for arm in ARMS:
        for r in seeds.get(arm, []):
            c = json.loads((runs / r / "curve.json").read_text())
            flags[r] = c["overfit_flag"]
    say(f"overfit flags: {flags}")
    names = [k for k in preds]
    for k in names:
        p, m = preds[k], {}
        hz = 2 ** p["hb"]
        e = np.abs(hz - hb_ref)
        m["hb_mae"] = [float(np.nanmean(e[:, n])) for n in range(4)]
        m["hb_f34"] = float(np.mean(m["hb_mae"][2:]))
        cp = 2 ** p["cp"][:, :3] - cpf["cp"]
        for g in ("f", "m"):
            s = (cpf["sex"] == g) & np.isfinite(cpf["cp"]).all(1)
            m[f"cp_bias_{g}"] = np.median(cp[s], 0).tolist()
            m[f"cp_abs_{g}"] = np.median(np.abs(cp[s]), 0).tolist()
        ve = np.abs(2 ** p["val"] - val_fm)
        for name, sel in (("tube", val_tube), ("all", np.ones_like(val_tube))):
            s = val_son & sel[:, None]
            m[f"val_mae_{name}"] = [float(np.nanmean(ve[..., n][s])) for n in range(4)]
        est = np.nanmedian(2 ** p["probe"][:, 30:71], 1)  # (144, 4)
        keys = sorted({(c.vowel, c.length_cm) for c in clips})
        sl = []
        for key in keys:
            i = [j for j, c in enumerate(clips) if (c.vowel, c.length_cm) == key and c.f0 <= inv.LOW_F0_MAX_HZ]
            x = np.log2([clips[j].f0 for j in i])
            sl.append([np.polyfit(x, est[i, n], 1)[0] for n in range(4)])
        m["probe_abs_slope"] = np.median(np.abs(sl), 0).tolist()
        slc = []  # the same slopes in cents per octave, per tract, for the paired f0 guard
        for key in keys:
            i = [j for j, c in enumerate(clips) if (c.vowel, c.length_cm) == key and c.f0 <= inv.LOW_F0_MAX_HZ]
            x = np.log2([clips[j].f0 for j in i])
            slc.append([np.polyfit(x, 1200 * np.log2(est[i, n]), 1)[0] for n in range(4)])
        probe_c[k] = (np.array([f"{v}|{L}" for v, L in keys]), np.abs(np.array(slc)))
        m["probe_abs_slope_cents"] = np.median(np.abs(slc), 0).tolist()
        m["probe_err_by_len"] = {str(L): np.median((est - truth)[meta["length_cm"] == L], 0).tolist() for L in inv.LENGTHS_CM}
        if "shift" in p:
            sh = shift_slopes(p["shift"], sm)
            m["shift_follow"] = [float(np.median(sh[n][1])) for n in range(4)]
            m["shift_drag"] = [float(np.median(sh[n][2])) for n in range(4)]
            shifts[k] = sh
        res[k] = m
        say(f"\n{k}")
        say("  Hillenbrand MAE F1-F4 (Hz): " + " / ".join(f"{v:.1f}" for v in m["hb_mae"]) + f"; F3/F4 mean {m['hb_f34']:.2f}")
        for g in ("f", "m"):
            say(f"  closed phase {g}: bias F1-F3 " + " / ".join(f"{v:+.1f}" for v in m[f"cp_bias_{g}"])
                + "; |error| " + " / ".join(f"{v:.1f}" for v in m[f"cp_abs_{g}"]))
        say("  synthetic MAE F1-F4 (Hz), tube: " + " / ".join(f"{v:.1f}" for v in m["val_mae_tube"])
            + "; all: " + " / ".join(f"{v:.1f}" for v in m["val_mae_all"]))
        say("  probe median |slope| F1-F4 (Hz/oct, f0 <= 250): " + " / ".join(f"{v:.1f}" for v in m["probe_abs_slope"])
            + "; error by length 13/15.5/18 cm: " + "; ".join(" / ".join(f"{v:+.0f}" for v in m["probe_err_by_len"][str(L)])
                                                          for L in inv.LENGTHS_CM))
        if "shift_follow" in m:
            say("  shift probe, moved F1-F4: follow slope " + " / ".join(f"{v:.3f}" for v in m["shift_follow"])
                + "; drag (mean |slope| of the others) " + " / ".join(f"{v:.3f}" for v in m["shift_drag"]))

    rng = np.random.default_rng(0)
    say("\nPaired against F (speaker bootstrap, 2,000): Hillenbrand MAE F1 / F2 / F3 / F4, F3/F4 mean; women's cp F1 |error|")
    pairs = [(a, "F") for a in ("Fv", "Fi", "T0") if a in preds and "F" in preds]
    pairs += [("Fi", "Fv")] if {"Fi", "Fv"} <= set(preds) else []
    pairs += [(f"F_{s.split('_')[-1]}", "F_s0") for s in seeds.get("F", [])[1:]]  # the fine-tuning seed noise
    wf = (cpf["sex"] == "f") & np.isfinite(cpf["cp"][:, 0])
    paired = {}
    for a, b in pairs:
        ea, eb = np.abs(2 ** preds[a]["hb"] - hb_ref), np.abs(2 ** preds[b]["hb"] - hb_ref)
        d = {}
        for n in range(4):
            ok = np.isfinite(hb_ref[:, n])
            d[f"F{n + 1}"] = boot_speakers(ea[ok, n], eb[ok, n], hb_spk[ok], rng)
        ok = np.isfinite(hb_ref[:, 2:4]).all(1)
        d["F3F4"] = boot_speakers(ea[ok, 2:4].mean(1), eb[ok, 2:4].mean(1), hb_spk[ok], rng)
        ca = np.abs(2 ** preds[a]["cp"][wf, 0] - cpf["cp"][wf, 0])
        cb = np.abs(2 ** preds[b]["cp"][wf, 0] - cpf["cp"][wf, 0])
        d["cp_f_F1"] = boot_speakers(ca, cb, cpf["speaker"][wf], rng, np.median)
        paired[f"{a}-{b}"] = d
        say(f"  {a} - {b}: " + "  ".join(f"{k} {fmt(v)}" for k, v in d.items()))
    say("\nf0 probe, paired over tracts (2,000): |slope| of each formant against log2 f0 (f0 <= 250 Hz), cents per octave")
    for a, b in pairs:
        ka, sa = probe_c[a]
        _, sb = probe_c[b]
        d = {f"F{n + 1}_f0slope": boot_tracts(sa[:, n], sb[:, n], ka, rng) for n in range(4)}
        paired.setdefault(f"{a}-{b}", {}).update(d)
        say(f"  {a} - {b}: " + "  ".join(f"{kk} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for kk, v in d.items()))
    if shifts:
        if parts["shift"][1] is not None:  # the trunk's VTL head on the shift probe (arm-independent)
            vs = shift_slopes(np.repeat(parts["shift"][1][..., None], 4, -1), sm)
            say("\nshift probe, the trunk's VTL head (log2 cm per log2 factor), moved F1-F4: "
                + " / ".join(f"{np.median(vs[n][1]):+.3f}" for n in range(4)))
        say("\nshift probe, paired over tracts (2,000): follow and drag per moved formant")
        for a, b in pairs:
            if a in shifts and b in shifts:
                d = {}
                for n in range(4):
                    ka, fa, da, _ = shifts[a][n]
                    _, fb, db, _ = shifts[b][n]
                    d[f"F{n + 1}_follow"] = boot_tracts(fa, fb, ka, rng)
                    d[f"F{n + 1}_drag"] = boot_tracts(da, db, ka, rng)
                paired.setdefault(f"{a}-{b}", {}).update(d)
                say(f"  {a} - {b}: " + "  ".join(f"{kk} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for kk, v in d.items()))
    def rule(d):
        """D-038's registered rule on one arm's paired differences against F -> (pass, {criterion: pass})"""
        c = {"primary F3/F4": d["F3F4"][2] < 0,
             "guard-rails": all(d[f"F{n + 1}"][0] <= GUARD_HZ[n] for n in range(4)) and d["cp_f_F1"][0] <= CP_GUARD_HZ,
             "shift": all(d[f"F{n + 1}_follow"][0] >= -FOLLOW_DROP[n] and d[f"F{n + 1}_drag"][0] <= DRAG_RISE for n in range(4)),
             "f0": all(d[f"F{n + 1}_f0slope"][0] <= F0_RISE for n in range(4))}
        return all(c.values()), c

    if {"Fv", "F"} <= set(preds):
        ok, crit = rule(paired["Fv-F"])
        unreliable = any(v for k, v in flags.items() if k.split("_")[0] in ("F", "Fv"))
        say(f"\n  criteria Fv: {crit}")
        say(f"  DECISION Fv: {'joins item 5' if ok else 'does not join item 5'}"
            + (" (UNRELIABLE: a flagged F or Fv seed)" if unreliable else ""))
        if "Fi" in preds:
            oki, criti = rule(paired["Fi-F"])
            dv = paired["Fi-Fv"]
            say(f"  criteria Fi: {criti}")
            say(f"  DECISION form: {'1/L' if oki and dv['F3F4'][2] < 0 else 'log2 L'} (Fi passes: {oki}; Fi - Fv F3/F4 {fmt(dv['F3F4'])})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"seeds": seeds, "flags": flags, "arms": res,
                                                    "paired": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in paired.items()}}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
