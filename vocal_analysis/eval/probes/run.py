"""Render, cache and score the probes (D-050).

    python -m vocal_analysis.eval.probes.run render counterfactual [--per-group 6]
    python -m vocal_analysis.eval.probes.run score counterfactual <ckpt> [<ckpt> ...]   ($VA_DEVICE=cuda)

Renders are cached under data/probes/<probe>/ so every model hears identical audio; scores go to
reports/probes/<probe>/<run>.json and a text summary.
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
OUTPUTS = ("F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "VTL", "f0")


def render_counterfactual(per_group: int = 6, log=print) -> Path:
    out = CACHE / "counterfactual_v1"
    out.mkdir(parents=True, exist_ok=True)
    conds = cf.conditions()
    meta = {"conditions": conds, "per_group": per_group, "stores": {}, "items": []}
    for name, (store, root, part) in STORES.items():
        items = cf.load_items(store, root, per_group, part)
        meta["stores"][name] = store
        for k, it in enumerate(items):
            path = out / f"{name}_{k:03d}.npz"
            arrays = {"original": it.audio.astype(np.float32)}
            for c, (arm, a, r) in enumerate(conds):
                y, lab = cf.render(it, a, r)
                arrays[f"audio_{c}"], arrays[f"scored_{c}"] = y, lab["scored"]
                arrays[f"voiced_{c}"], arrays[f"label_formants_{c}"] = lab["voiced"], lab["formants"][:, :4].astype(np.float32)
                arrays[f"label_bandwidths_{c}"] = lab["bandwidths"][:, :4].astype(np.float32)
            np.savez_compressed(path, **arrays)
            meta["items"].append({"file": path.name, "key": it.key, "group": f"{name}:{it.group}", "beta": it.beta})
        log(f"{name}: {len(items)} items")
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
    out["sVTL"] = med(r["vtl_sigma"], m) if "vtl_sigma" in r else np.nan
    return out


def score_counterfactual(ckpt: str, log=print) -> dict:
    import torch

    from ...models.infer import analyse, load_teacher

    src = CACHE / "counterfactual_v1"
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    conds = [tuple(c) for c in meta["conditions"]]
    keys = [f"F{k}" for k in range(1, 5)] + [f"B{k}" for k in range(1, 5)] + [f"sF{k}" for k in range(1, 5)] + ["f0", "VTL", "sVTL"]
    Y = {k: np.full((len(meta["items"]), len(conds) + 1), np.nan) for k in keys}  # last column: the original clip
    L = np.full((len(meta["items"]), len(conds), 4), np.nan)
    LB = np.full_like(L, np.nan)
    for i, it in enumerate(meta["items"]):
        d = np.load(src / it["file"])
        for c in range(len(conds)):
            s = _summary(analyse(model, d[f"audio_{c}"], dev), d[f"scored_{c}"], d[f"voiced_{c}"])
            for k in keys:
                Y[k][i, c] = s[k]
            m = d[f"scored_{c}"]
            L[i, c] = np.nanmedian(np.log2(d[f"label_formants_{c}"][m]), 0) if m.any() else np.nan
            LB[i, c] = np.nanmedian(np.log2(d[f"label_bandwidths_{c}"][m]), 0) if m.any() else np.nan
        s = _summary(analyse(model, d["original"], dev), d["scored_0"], d["voiced_0"])
        for k in keys:
            Y[k][i, -1] = s[k]
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(meta['items'])}")
    return {"conds": conds, "Y": Y, "L": L, "LB": LB, "groups": np.array([it["group"] for it in meta["items"]]),
            "beta": np.array([it["beta"] for it in meta["items"]], float)}


def analyse_counterfactual(res: dict) -> dict:
    """Slopes per arm and output, on all items and on |beta| < SMALL_BETA; and the resynthesis artefact check."""
    conds, Y, L, LB, groups, beta = res["conds"], res["Y"], res["L"], res["LB"], res["groups"], res["beta"]
    out = {}
    for arm, j in (("length", 1), ("f0", 2)):
        cols = [0] + [c for c, cd in enumerate(conds) if cd[0] == arm]
        factor = np.log2([conds[c][j] for c in cols])  # log2 alpha or log2 r
        for subset, keep in (("all", np.ones(len(beta), bool)), ("small_beta", np.abs(beta) < cf.SMALL_BETA)):
            for o in OUTPUTS:
                y = Y[o][keep][:, cols]
                if not np.isfinite(y).any():
                    continue
                if o.startswith("F") and o != "f0":  # x: the labels' own change (exact), ideal 1 / ideal 0 for f0 arm
                    n = int(o[1]) - 1
                    x = L[keep][:, cols, n] - L[keep][:, [0], n] if arm == "length" else np.broadcast_to(factor, y.shape)
                    ideal = 1.0 if arm == "length" else 0.0
                elif o == "VTL":
                    x, ideal = np.broadcast_to(factor, y.shape), (1.0 if arm == "length" else 0.0)
                elif o == "f0":
                    x, ideal = np.broadcast_to(factor, y.shape), (0.0 if arm == "length" else 1.0)
                else:  # bandwidths: x is the labels' own change (the prior moves them with the formants and f0)
                    n = int(o[1]) - 1
                    x, ideal = LB[keep][:, cols, n] - LB[keep][:, [0], n], 1.0
                s = M.summarise_slopes(y, x, groups[keep])
                out[f"{arm}/{subset}/{o}"] = s.__dict__ | {"ideal": ideal}
        for o in ("F1", "F2", "F3", "F4", "VTL"):
            sg = "sF" + o[1] if o != "VTL" else "sVTL"
            if np.isfinite(Y[sg]).any():
                g = M.sigma_growth(Y[sg][:, cols], 0)
                out[f"{arm}/sigma_growth/{o}"] = {"per_condition_median": np.nanmedian(g, 0).tolist(), "factor": factor.tolist()}
    # the original recording minus its alpha = r = 1 resynthesis. Not an artefact check: the resynthesis replaces the
    # real tract with the tracker's smoothed tracks and prior bandwidths, so this is "real tract against those"
    for o in OUTPUTS:
        d = Y[o][:, -1] - Y[o][:, 0]
        if np.isfinite(d).any():
            out[f"original_vs_resynth/{o}"] = dict(zip(("point", "low", "high"), M.group_bootstrap(d, groups)))
    return out


def report(name: str, a: dict, beta: np.ndarray) -> str:
    lines = [f"Counterfactual resynthesis probe (D-050): {name}",
             f"items {len(beta)}; |beta| < {cf.SMALL_BETA}: {int((np.abs(beta) < cf.SMALL_BETA).sum())}; beta median {np.nanmedian(beta):+.2f}",
             "slope [95 % group bootstrap]; shuffle null median [2.5, 97.5 %]; ideal"]
    for k, v in a.items():
        if "/sigma_growth/" in k:
            lines.append(f"  {k:24s} sigma log2 change by factor: " + " ".join(f"{f:+.2f}:{g:+.2f}" for f, g in zip(v["factor"], v["per_condition_median"])))
        elif k.startswith("original_vs_resynth/"):
            lines.append(f"  {k:24s} original - resynthesis {v['point']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}] (log2)")
        else:
            ideal = "" if v["ideal"] is None else f"  ideal {v['ideal']:g}"
            lines.append(f"  {k:24s} {v['point']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]  null {v['null_median']:+.3f} "
                         f"[{v['null_low']:+.3f}, {v['null_high']:+.3f}]  n {v['n_items']}/{v['n_groups']}{ideal}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if argv[:2] == ["render", "counterfactual"]:
        pg = int(argv[argv.index("--per-group") + 1]) if "--per-group" in argv else 6
        render_counterfactual(pg)
        return 0
    if argv[:2] == ["score", "counterfactual"]:
        for ckpt in argv[2:]:
            name = Path(ckpt).parent.name
            res = score_counterfactual(ckpt)
            a = analyse_counterfactual(res)
            out = REPORTS / "counterfactual"
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / f"{name}_raw.npz", L=res["L"], LB=res["LB"], groups=res["groups"], beta=res["beta"],
                     **{f"Y_{k}": v for k, v in res["Y"].items()})
            (out / f"{name}.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
            text = report(name, a, res["beta"])
            (out / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            print(text)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
