"""Render, cache and score the probes (D-050).

    python -m vocal_analysis.eval.probes.run render <probe> [--per-group 6]
    python -m vocal_analysis.eval.probes.run score <probe> <ckpt> [<ckpt> ...]   ($VA_DEVICE=cuda)

Probes: counterfactual (1: length, f0), shift (2), timing (4), voice (6), bandwidth (7), polezero (8), chain (9), articulation (10). Renders are cached under
data/probes/<probe>_v1/ so every model hears identical audio; scores go to reports/probes/<probe>/<run>.{txt,json}.

Each condition is a dict with its arm and settings. "Slope" arms vary one factor (alpha, r, bw): every output's
per-item slope against it, with its ideal (1 equivariant, 0 invariant; formants and bandwidths are read against the
labels' own change). "Change" arms (pole-zero pairs, signal chain) are compared with the reference condition: every
output should be unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from . import counterfactual as cf
from . import measures as M

CACHE = Path("data/probes")
REPORTS = Path("reports/probes")
STORES = {"zeroth_val": ("data/resynth/zeroth_korean_val_d16k", "data/zeroth_korean", 2)}
OUTPUTS = ("F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "VTL", "VTLi", "f0")
REF = {"arm": "ref"}
PROBES = {
    "counterfactual": [REF] + [{"arm": "length", "alpha": a} for a in cf.ALPHAS if a != 1.0]
                      + [{"arm": "f0", "r": r} for r in cf.RATIOS if r != 1.0],
    "bandwidth": [REF] + [{"arm": "bw", "bw": s} for s in (0.7, 0.85, 1.2, 1.45, 2.0)],
    # probe 4's time-stretch arm (D-050 amendment 2), on resynthesis: its reference is the resynthesised clip
    "stretch": [REF] + [{"arm": "stretch", "stretch": s} for s in (0.8, 0.9, 1.1, 1.25)],
    # probe 12, arm A (D-050 amendment 4): the length arm with every bandwidth held at its alpha = 1 value
    "bwhold": [REF] + [{"arm": "length_bwhold", "alpha": a, "hold_bw": True} for a in cf.ALPHAS if a != 1.0],
    "polezero": [REF, {"arm": "nasal", "extras": "nasal"}, {"arm": "subglottal", "extras": "subglottal"}],
    "chain": [REF] + [{"arm": "gain", "chain": ("gain", g)} for g in (0.1, 0.5, 2.0)]
             + [{"arm": "dc", "chain": ("dc", 0.05)}, {"arm": "clip", "chain": ("clip", 0.5)},
                {"arm": "resample", "chain": ("resample", 44100)}],
    # probe 13 (D-050 amendment 5): one arm per gain (probe 9's single gain arm reports only its first condition)
    "gain": [REF] + [{"arm": f"gain_{g}", "chain": ("gain", g)} for g in (0.01, 0.03, 0.1, 0.3, 2.0)],
    # probe 4: timing, on the original clips (time-stretch left out: it needs a pitch-preserving method chosen)
    "timing": [REF] + [{"arm": f"offset_{k}", "chain": ("offset", k)} for k in (20, 40, 80)]
              + [{"arm": "polarity", "chain": ("polarity", 0)}, {"arm": "pad_2s", "chain": ("pad", 2.0)}],
    # probe 6: voice quality and effort, synthetic (only the source settings differ; reference Rd 1.2, HNR 25 dB,
    # jitter 0.005, shimmer 0.3 dB)
    "voice": [REF] + [{"arm": f"rd_{v}", "rd": v} for v in (0.5, 0.8, 1.8, 2.5)]
             + [{"arm": f"hnr_{v}", "hnr": v} for v in (10.0, 15.0, 40.0)]
             + [{"arm": "jitter_0.02", "jitter": 0.02}, {"arm": "shimmer_1.0", "shimmer": 1.0}],
    # probe 2: one formant moved alone (D-038's factors), on the real source
    "shift": [REF] + [{"arm": f"shiftF{k + 1}", "shift": (k, s)} for k in range(4) for s in (0.9, 0.95, 1.05, 1.1)],
    # probe 10: synthetic tube clips; larynx (pg) and lip (pm) changes held fixed, cm at Story's size
    "articulation": [REF] + [{"arm": "larynx", "pg": v} for v in (-0.4, -0.2, 0.2, 0.4)]
                    + [{"arm": "lips", "pm": v} for v in (-0.25, 0.4, 0.8)],
}
ARTIC_ITEMS, ARTIC_SEED = 60, 5052  # seed used by no other set
VOICE_SEED = 5053
VOICE_REF = {"rd": 1.2, "hnr": 25.0, "jitter": 0.005, "shimmer": 0.3}
SLOPE_ARMS = {"length": "alpha", "f0": "r", "bw": "bw", "stretch": "stretch", "length_bwhold": "alpha", "larynx": "label", "lips": "label"}
# outputs read against their labels' own change, per arm: only where the labels are meant to move (bandwidth labels
# follow the prior, which depends on the formants and f0); elsewhere against the arm's factor. In the f0 arm the
# formant labels move only by the time-warp's rounding (<= 0.004 octave), which must not become the x of a slope
LABEL_X = {"length": "FB", "f0": "B", "bw": "B", "larynx": "FB", "lips": "FB", "length_bwhold": "F"}
SHIFT_ARMS = {f"shiftF{k}": f"F{k}" for k in range(1, 5)}  # probe 2: the moved formant
# ideal slope per arm: formants / bandwidths against the labels' change, VTL and f0 against log2 of the factor
IDEAL = {"length": {"F": 1.0, "B": 1.0, "VTL": 1.0, "VTLi": 1.0, "f0": 0.0},
         "f0": {"F": 0.0, "B": 1.0, "VTL": 0.0, "VTLi": 0.0, "f0": 1.0},
         "bw": {"F": 0.0, "B": 1.0, "VTL": 0.0, "VTLi": 0.0, "f0": 0.0},
         "stretch": {"F": 0.0, "B": 0.0, "VTL": 0.0, "VTLi": 0.0, "f0": 0.0},
         "length_bwhold": {"F": 1.0, "B": 0.0, "VTL": 1.0, "VTLi": 1.0, "f0": 0.0},
         # probe 10: x is the instantaneous length's own change; the anatomical head should not follow it
         "larynx": {"F": 1.0, "B": 1.0, "VTL": 0.0, "VTLi": 1.0, "f0": 0.0},
         "lips": {"F": 1.0, "B": 1.0, "VTL": 0.0, "VTLi": 1.0, "f0": 0.0}}


def _cond(c) -> dict:
    """Conditions as dicts (counterfactual_v1's meta stored (arm, alpha, r) tuples)."""
    if isinstance(c, dict):
        return c
    arm, a, r = c
    return {"arm": arm} | ({"alpha": a} if arm == "length" else {}) | ({"r": r} if arm == "f0" else {})


def render_articulation(log=print) -> Path:
    """Probe 10: ARTIC_ITEMS synthetic tube clips (train_v2's settings, 3-4 s, length not drawn from f0), each
    rendered under every condition; conditions differ only in the held larynx / lip change, so every other draw is
    shared."""
    import dataclasses

    from ...synth.sampler import SynthConfig, generate

    out = CACHE / "articulation_v1"
    out.mkdir(parents=True, exist_ok=True)
    conds = PROBES["articulation"]
    base = dataclasses.replace(SynthConfig(), duration_s=(3.0, 4.0), p_tube=1.0, p_length_from_f0=0.0, articulation="fixed")
    meta = {"conditions": conds, "seed": ARTIC_SEED, "items": []}
    for i in range(ARTIC_ITEMS):
        arrays = {}
        for c, cd in enumerate(conds):
            clip = generate(i, dataclasses.replace(base, artic_pg_cm=cd.get("pg", 0.0), artic_pm_cm=cd.get("pm", 0.0)), ARTIC_SEED)
            lab = clip.labels
            arrays[f"audio_{c}"], arrays[f"scored_{c}"], arrays[f"voiced_{c}"] = clip.audio, lab["sonorant"], lab["voiced"]
            arrays[f"label_formants_{c}"] = lab["formants"][:, :4].astype(np.float32)
            arrays[f"label_bandwidths_{c}"] = lab["bandwidths"][:, :4].astype(np.float32)
            arrays[f"label_inst_{c}"] = lab["length_inst_cm"].astype(np.float32)
        np.savez_compressed(out / f"synth_{i:03d}.npz", **arrays)
        meta["items"].append({"file": f"synth_{i:03d}.npz", "key": f"synth:{ARTIC_SEED}:{i}", "group": f"synth:{i}",
                              "beta": float("nan")})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"articulation: {ARTIC_ITEMS} items")
    return out


def render_voice(log=print) -> Path:
    """Probe 6: ARTIC_ITEMS synthetic clips (train_v2's settings, 3-4 s, no Rd wander); conditions set Rd, HNR,
    jitter and shimmer to fixed values (each still one uniform draw, so every other draw is shared)."""
    import dataclasses

    from ...synth.sampler import SynthConfig, generate

    out = CACHE / "voice_v1"
    out.mkdir(parents=True, exist_ok=True)
    conds = PROBES["voice"]
    meta = {"conditions": conds, "seed": VOICE_SEED, "items": []}
    for i in range(ARTIC_ITEMS):
        arrays = {}
        for c, cd in enumerate(conds):
            v = VOICE_REF | {k: cd[k] for k in VOICE_REF if k in cd}
            cfg = dataclasses.replace(SynthConfig(), duration_s=(3.0, 4.0), rd=(v["rd"], v["rd"]), rd_wander_sd=(0.0, 0.0),
                                      hnr_db=(v["hnr"], v["hnr"]), jitter=(v["jitter"], v["jitter"]),
                                      shimmer_db=(v["shimmer"], v["shimmer"]))
            clip = generate(i, cfg, VOICE_SEED)
            lab = clip.labels
            arrays[f"audio_{c}"], arrays[f"scored_{c}"], arrays[f"voiced_{c}"] = clip.audio, lab["sonorant"], lab["voiced"]
            arrays[f"label_formants_{c}"] = lab["formants"][:, :4].astype(np.float32)
            arrays[f"label_bandwidths_{c}"] = lab["bandwidths"][:, :4].astype(np.float32)
        np.savez_compressed(out / f"synth_{i:03d}.npz", **arrays)
        meta["items"].append({"file": f"synth_{i:03d}.npz", "key": f"synth:{VOICE_SEED}:{i}", "group": f"synth:{i}",
                              "beta": float("nan")})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"voice: {ARTIC_ITEMS} items")
    return out


def render_probe(probe: str, per_group: int = 6, log=print) -> Path:
    if probe == "articulation":
        return render_articulation(log)
    if probe == "voice":
        return render_voice(log)
    out = CACHE / f"{probe}_v1"
    out.mkdir(parents=True, exist_ok=True)
    conds = PROBES[probe]
    meta = {"conditions": conds, "per_group": per_group, "stores": {}, "items": []}
    for name, (store, root, part) in STORES.items():
        items = cf.load_items(store, root, per_group, part)
        meta["stores"][name] = store
        for k, it in enumerate(items):
            arrays = {"original": it.audio.astype(np.float32)} if probe == "counterfactual" else {}
            _, ref_lab = cf.render(it)
            for c, cd in enumerate(conds):
                if probe in ("chain", "timing", "gain"):  # on the original clip; frames chosen as in the reference resynthesis
                    y = cf.chain(it.audio, *cd["chain"]) if "chain" in cd else it.audio.astype(np.float32)
                    lab = ref_lab | {"formants": np.full_like(ref_lab["formants"], np.nan),
                                     "bandwidths": np.full_like(ref_lab["bandwidths"], np.nan)}
                    if "chain" in cd and cd["chain"][0] == "pad":  # the clip's frames move by the padding
                        npad = int(round(cd["chain"][1] * 100))
                        pad = lambda m: np.r_[np.zeros(npad, bool), m, np.zeros(npad, bool)]  # noqa: E731
                        lab = lab | {"scored": pad(lab["scored"]), "voiced": pad(lab["voiced"]),
                                     "formants": np.full((len(lab["scored"]) + 2 * npad, lab["formants"].shape[1]), np.nan),
                                     "bandwidths": np.full((len(lab["scored"]) + 2 * npad, lab["bandwidths"].shape[1]), np.nan)}
                else:
                    y, lab = cf.render(it, cd.get("alpha", 1.0), cd.get("r", 1.0), cd.get("bw", 1.0), cd.get("extras"),
                                       tuple(cd["shift"]) if "shift" in cd else None, cd.get("stretch", 1.0),
                                       cd.get("hold_bw", False))
                arrays[f"audio_{c}"], arrays[f"scored_{c}"], arrays[f"voiced_{c}"] = y, lab["scored"], lab["voiced"]
                arrays[f"label_formants_{c}"] = lab["formants"][:, :4].astype(np.float32)
                arrays[f"label_bandwidths_{c}"] = lab["bandwidths"][:, :4].astype(np.float32)
            np.savez_compressed(out / f"{name}_{k:03d}.npz", **arrays)
            meta["items"].append({"file": f"{name}_{k:03d}.npz", "key": it.key, "group": f"{name}:{it.group}", "beta": it.beta})
        log(f"{probe} / {name}: {len(items)} items")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def _summary(r: dict, scored: np.ndarray, voiced: np.ndarray) -> dict:
    """One clip's model outputs -> one value per output: medians over the scored (pure re-filtered) frames."""
    n = min(len(scored), len(r["f0"]))
    m, v = scored[:n], (scored & voiced)[:n]
    med = lambda a, mask: float(np.nanmedian(a[:n][mask])) if mask.any() else np.nan  # noqa: E731
    out = {f"F{k + 1}": med(np.log2(r["formants"][:, k]), m) for k in range(4)}
    out |= {f"B{k + 1}": med(np.log2(r["bandwidths"][:, k]), m) for k in range(4)}
    out |= {f"sF{k + 1}": med(r["formant_sigma_cents"][:, k], m) for k in range(4)}
    out["f0"] = med(np.log2(r["f0"]), v)
    out["VTL"] = med(r["vtl_log2cm"], m) if "vtl_log2cm" in r else np.nan
    out["VTLi"] = med(r["vtl_inst_log2cm"], m) if "vtl_inst_log2cm" in r else np.nan
    out["sVTL"] = med(r["vtl_sigma"], m) if "vtl_sigma" in r else np.nan
    return out


KEYS = [f"F{k}" for k in range(1, 5)] + [f"B{k}" for k in range(1, 5)] + [f"sF{k}" for k in range(1, 5)] + ["f0", "VTL", "VTLi", "sVTL"]


def score_probe(probe: str, ckpt: str, log=print) -> dict:
    import torch

    from ...models.infer import analyse, load_teacher

    src = CACHE / f"{probe}_v1"
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    conds = [_cond(c) for c in meta["conditions"]]
    extra = probe == "counterfactual"  # last column: the original clip
    n_items = len(meta["items"])
    Y = {k: np.full((n_items, len(conds) + extra), np.nan) for k in KEYS}
    L = np.full((n_items, len(conds), 4), np.nan)
    LB = np.full_like(L, np.nan)
    LI = np.full((n_items, len(conds)), np.nan)  # probe 10: log2 instantaneous length label
    for i, it in enumerate(meta["items"]):
        d = np.load(src / it["file"])
        for c in range(len(conds)):
            s = _summary(analyse(model, d[f"audio_{c}"], dev), d[f"scored_{c}"], d[f"voiced_{c}"])
            for k in KEYS:
                Y[k][i, c] = s[k]
            m = d[f"scored_{c}"]
            if m.any():
                with np.errstate(all="ignore"):
                    L[i, c] = np.nanmedian(np.log2(d[f"label_formants_{c}"][m]), 0)
                    if f"label_bandwidths_{c}" in d:
                        LB[i, c] = np.nanmedian(np.log2(d[f"label_bandwidths_{c}"][m]), 0)
                    if f"label_inst_{c}" in d:
                        LI[i, c] = np.nanmedian(np.log2(d[f"label_inst_{c}"][m]))
        if extra:
            s = _summary(analyse(model, d["original"], dev), d["scored_0"], d["voiced_0"])
            for k in KEYS:
                Y[k][i, -1] = s[k]
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{n_items}")
    return {"conds": conds, "Y": Y, "L": L, "LB": LB, "LI": LI, "groups": np.array([it["group"] for it in meta["items"]]),
            "beta": np.array([it["beta"] for it in meta["items"]], float)}


def analyse_probe(res: dict) -> dict:
    conds, Y, L, LB, groups, beta = res["conds"], res["Y"], res["L"], res["LB"], res["groups"], res["beta"]
    out = {}
    arms = list(dict.fromkeys(c["arm"] for c in conds if c["arm"] != "ref"))
    for arm in arms:
        cols = [0] + [c for c, cd in enumerate(conds) if cd["arm"] == arm]
        if arm in SLOPE_ARMS or arm in SHIFT_ARMS:
            if arm in SHIFT_ARMS:  # probe 2: x is the moved formant's own label change, for every output
                moved = int(SHIFT_ARMS[arm][1]) - 1
                factor = L[:, cols, moved] - L[:, [0], moved]
            elif SLOPE_ARMS[arm] == "label":  # probe 10: per item, the instantaneous length label's own change
                factor = res["LI"][:, cols] - res["LI"][:, [0]]
            else:
                factor = np.log2([conds[c].get(SLOPE_ARMS[arm], 1.0) for c in cols])
            subsets = [("all", np.ones(len(beta), bool))]
            if (np.abs(beta) >= cf.SMALL_BETA).any():  # nan beta (synthetic) never qualifies
                subsets.append(("small_beta", np.abs(beta) < cf.SMALL_BETA))
            for subset, keep in subsets:
                for o in OUTPUTS:
                    y = Y[o][keep][:, cols]
                    if not np.isfinite(y).any():
                        continue
                    kind = o if o in ("VTL", "VTLi", "f0") else o[0]
                    if kind in ("F", "B") and kind in LABEL_X.get(arm, ""):
                        lab = (L if kind == "F" else LB)[keep][:, cols, int(o[1]) - 1]
                        x = lab - lab[:, [0]]
                    else:
                        x = _x(factor, keep, y.shape)
                    s = M.summarise_slopes(y, x, groups[keep])
                    ideal = (1.0 if o == SHIFT_ARMS[arm] else 0.0) if arm in SHIFT_ARMS else IDEAL[arm][kind]
                    out[f"{arm}/{subset}/{o}"] = s.__dict__ | {"ideal": ideal}
        else:  # change against the reference: ideal 0 for every output
            for o in OUTPUTS:
                d = Y[o][:, cols[1]] - Y[o][:, 0]
                if np.isfinite(d).any():
                    out[f"{arm}/change/{o}"] = dict(zip(("point", "low", "high"), M.group_bootstrap(d, groups))) | {"ideal": 0.0}
        for o in ("F1", "F2", "F3", "F4", "VTL"):
            sg = "sF" + o[1] if o != "VTL" else "sVTL"
            if np.isfinite(Y[sg]).any():
                g = M.sigma_growth(Y[sg][:, cols], 0)
                out[f"{arm}/sigma_growth/{o}"] = {"per_condition_median": np.nanmedian(g, 0).tolist(),
                                                  "conditions": [conds[c] for c in cols]}
    if Y["F1"].shape[1] > len(conds):  # counterfactual: the original recording minus its alpha = r = 1 resynthesis.
        # Not an artefact check: the resynthesis replaces the real tract with the tracker's smoothed tracks and
        # prior bandwidths, so this is "real tract against those"
        for o in OUTPUTS:
            d = Y[o][:, -1] - Y[o][:, 0]
            if np.isfinite(d).any():
                out[f"original_vs_resynth/{o}"] = dict(zip(("point", "low", "high"), M.group_bootstrap(d, groups)))
    return out


def _x(factor: np.ndarray, keep: np.ndarray, shape) -> np.ndarray:
    """The imposed change as (items, conditions): a per-condition factor broadcast, or a per-item one subset."""
    return np.broadcast_to(factor, shape) if factor.ndim == 1 else factor[keep]


def _label(c: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in c.items() if k != "arm") or "ref"


def report(probe: str, name: str, a: dict, beta: np.ndarray) -> str:
    lines = [f"Probe '{probe}' (D-050): {name}",
             f"items {len(beta)}; beta median {np.nanmedian(beta):+.2f}, |beta| < {cf.SMALL_BETA}: {int((np.abs(beta) < cf.SMALL_BETA).sum())}",
             "slopes: point [95 % group bootstrap]; shuffle null median [2.5, 97.5 %]; ideal. Changes: log2 (x 1200 = cents)"]
    for k, v in a.items():
        if "/sigma_growth/" in k:
            lines.append(f"  {k:26s} sigma log2 change: " + " ".join(f"{_label(c)}:{g:+.2f}" for c, g in zip(v["conditions"], v["per_condition_median"])))
        elif "/change/" in k or k.startswith("original_vs_resynth/"):
            ideal = f"  ideal {v['ideal']:g}" if "ideal" in v else ""
            lines.append(f"  {k:26s} {v['point']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}]{ideal}")
        else:
            lines.append(f"  {k:26s} {v['point']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]  null {v['null_median']:+.3f} "
                         f"[{v['null_low']:+.3f}, {v['null_high']:+.3f}]  n {v['n_items']}/{v['n_groups']}  ideal {v['ideal']:g}")
    return "\n".join(lines)


def reanalyse(probe: str, name: str) -> str:
    """Rebuild a report from its saved raw scores (after a change to the analysis only)."""
    src = CACHE / f"{probe}_v1"
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    out = REPORTS / probe
    d = np.load(out / f"{name}_raw.npz")
    res = {"conds": [_cond(c) for c in meta["conditions"]], "Y": {k[2:]: d[k] for k in d.files if k.startswith("Y_")},
           "L": d["L"], "LB": d["LB"], "LI": d["LI"] if "LI" in d.files else None, "groups": d["groups"], "beta": d["beta"]}
    for k in KEYS:  # raw files from before an output existed
        res["Y"].setdefault(k, np.full_like(res["Y"]["F1"], np.nan))
    a = analyse_probe(res)
    (out / f"{name}.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
    text = report(probe, name, a, res["beta"])
    (out / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(argv) >= 2 and argv[0] == "render" and argv[1] in PROBES:
        pg = int(argv[argv.index("--per-group") + 1]) if "--per-group" in argv else 6
        render_probe(argv[1], pg)
        return 0
    if len(argv) >= 3 and argv[0] == "score" and argv[1] in PROBES:
        probe = argv[1]
        for ckpt in argv[2:]:
            name = Path(ckpt).parent.name
            res = score_probe(probe, ckpt)
            a = analyse_probe(res)
            out = REPORTS / probe
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"{name}_raw.npz", L=res["L"], LB=res["LB"], LI=res["LI"], groups=res["groups"], beta=res["beta"],
                     **{f"Y_{k}": v for k, v in res["Y"].items()})
            (out / f"{name}.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
            text = report(probe, name, a, res["beta"])
            (out / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            print(text)
        return 0
    if len(argv) >= 3 and argv[0] == "reanalyse" and argv[1] in PROBES:
        for name in argv[2:]:
            print(reanalyse(argv[1], name))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
