"""D-038 (draft) scoring: the fine-tuned formant heads (d038_head.py) on one trunk's D-037 cache, every formant reported.
Per arm (seed mean of the decoded log2 Hz over its seeds; also each seed alone) and for the trunk's own heads (T0, the
cached F1-F4):
- Hillenbrand dev F1-F4 MAE (Hz) at the reference frames, and the F3 / F4 mean (the decision's measure);
- PTDB-TUG dev closed phase (cp part): F1-F3 median signed error and median |error| by sex;
- synthetic: train_v2's held-out crops (val part), sonorant frames, F1-F4 MAE on tube clips and on all clips;
- the invariance probe: per formant, the median |slope| (Hz per octave of f0, f0 <= 250 Hz) of the clip's median estimate
  over 0.3-0.7 s, and the median error against the probe's true formants by length.
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
GUARD_HZ = (4.0, 9.0, 20.0)  # D-036's Hillenbrand guard-rails, F1-F3
CP_GUARD_HZ = 5.0


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


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def main():
    cache, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    parts = {p: part(cache, p) for p in ("hb", "cp", "val", "probe")}
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
        m["probe_err_by_len"] = {str(L): np.median((est - truth)[meta["length_cm"] == L], 0).tolist() for L in inv.LENGTHS_CM}
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
    if {"Fv", "F"} <= set(preds):
        d = paired["Fv-F"]
        ok = (d["F3F4"][2] < 0 and all(d[f"F{n + 1}"][0] <= GUARD_HZ[n] for n in range(3)) and d["cp_f_F1"][0] <= CP_GUARD_HZ)
        unreliable = any(v for k, v in flags.items() if k.split("_")[0] in ("F", "Fv"))
        say(f"\n  DECISION Fv: {'joins item 5' if ok else 'does not join item 5'}"
            + (" (UNRELIABLE: a flagged F or Fv seed)" if unreliable else ""))
        if "Fi" in preds:
            di, dv = paired["Fi-F"], paired["Fi-Fv"]
            oki = (di["F3F4"][2] < 0 and all(di[f"F{n + 1}"][0] <= GUARD_HZ[n] for n in range(3)) and di["cp_f_F1"][0] <= CP_GUARD_HZ)
            say(f"  DECISION form: {'1/L' if oki and dv['F3F4'][2] < 0 else 'log2 L'} (Fi passes: {oki}; Fi - Fv F3/F4 {fmt(dv['F3F4'])})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"seeds": seeds, "flags": flags, "arms": res,
                                                    "paired": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in paired.items()}}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
