"""D-147: is D-080's lower failure (A0, A1 against 5b, D-145 part 3) outside the spread of 5b's other decay branches?
Every run below is a 4000-step branch from `d040_5b_s{1,2}/ckpt_016000.pt` (5b's own b20k included), so the spread
among the branches that left the data alone is the null for "the coupled set made the difference". CPU, inference only.

    python scripts/analysis/d147_branch_spread.py run      -> reports/analysis/d147/d147.{json,txt}
    python scripts/analysis/d147_branch_spread.py report   -> reports/analysis/d147/d147.txt
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d147"
# deciding: 5b, D-090's arms (heads above F4 only; F1-F4 heads and data as 5b's), D-080's coupled arms
NULL = ("d040_5b", "d090_f5", "d090_f57")
COUPLED = ("d080_a0", "d080_a1")
# reported, deciding nothing: other branches without the coupled set, which change the F1-F4 read-out or add a loss
OTHER = ("d056", "d072_factorised", "d072_retrieve", "d086_std", "d086_ramp", "d109", "d114")
SEEDS = (1, 2)


def run_name(arm: str, s: int) -> str:
    return f"d040_5b_s{s}_b20k" if arm == "d040_5b" else f"{arm}_s{s}"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def clip_sets(d143):
    """D-143's test clips and tipping grid exactly (the train draws are made, not rendered, to keep the RNG in step)."""
    d142 = d143._d142()
    rng = np.random.default_rng(1430)
    for _ in range(d143.N_TRAIN):
        d143.draw(rng, False)
    test = []
    for i in range(d143.N_TEST):
        p = d143.draw(rng, True)
        test.append((f"test{i}", d143.clip(p, 243000 + i), np.array(p["F"])))
    grid = [(f"{sname}|f{f0:g}|b{b1:g}|s{s}", d142._grid_audio(Fs, f0, b1, s), np.array(Fs))
            for sname, Fs in d142.SETS.items() for f0 in d142.F0S for b1 in d142.B1S for s in d143.GRID_SEEDS]
    return {"test": test, "grid": grid}


def fail_by_clip(model, dev, items):
    """D-142 / D-143's frame selection and failure (F2 nearer the true F3 than the true F2); per clip with frames."""
    from vocal_analysis.models.infer import analyse

    out = {}
    for j, (_, y, true) in enumerate(items):
        r = analyse(model, y.astype(np.float32), dev)
        n = len(r["voiced_p"])
        fl = []
        for i in range(int(0.2 * n), int(0.8 * n)):
            if r["voiced_p"][i] <= 0.5 or not np.isfinite(r["f0"][i]) or i * 160 < 512 or i * 160 + 512 > len(y):
                continue
            Fm, Bm = r["formants"][i, :4], r["bandwidths"][i, :4]
            if not (np.all(np.isfinite(Fm)) and np.all(np.isfinite(Bm))):
                continue
            fl.append(abs(np.log(Fm[1] / true[2])) < abs(np.log(Fm[1] / true[1])))
        if fl:
            out[j] = float(np.mean(fl))
    return out


def cmd_run() -> None:
    import torch

    from vocal_analysis.models.infer import load_teacher

    d143 = _load("d143", ROOT / "scripts/analysis/d143_probe.py")
    sets = clip_sets(d143)
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    REP.mkdir(parents=True, exist_ok=True)
    path = REP / "d147.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for arm in NULL + COUPLED + OTHER:
        for s in SEEDS:
            m = run_name(arm, s)
            if m in res:
                continue
            model = load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]
            res[m] = {k: fail_by_clip(model, dev, sets[k]) for k in ("test", "grid")}
            res[m]["grid_keys"] = [it[0] for it in sets["grid"]]
            path.write_text(json.dumps(res), encoding="utf-8")
            print(m, flush=True)
    cmd_report()  # from the saved json, whose clip keys are strings


def cmd_report(res=None) -> None:
    res = res or json.loads((REP / "d147.json").read_text(encoding="utf-8"))
    allm = [run_name(a, s) for a in NULL + COUPLED + OTHER for s in SEEDS]

    def primary(m):
        keys = res[m]["grid_keys"]
        g = res[m]["grid"]
        cell = [str(j) for j, k in enumerate(keys) if float(k.split("|")[1][1:]) <= 140
                and 250 <= float(k.split("|")[2][1:]) <= 500 and str(j) in g]
        return float(np.mean([g[j] for j in cell])), float(np.mean(list(g.values())))

    # D-145's fixed test-clip set: clips 5b fails (seed mean >= 0.5), among those every run scored
    common = sorted(set.intersection(*(set(res[m]["test"]) for m in allm)), key=int)
    f5 = np.mean([[res[run_name("d040_5b", s)]["test"][c] for c in common] for s in SEEDS], 0)
    fixed = [c for c, v in zip(common, f5) if v >= 0.5]
    tshare = {m: float(np.mean([res[m]["test"][c] >= 0.5 for c in fixed])) for m in allm}
    P = {m: primary(m) for m in allm}
    lines = ["D-147: own-head failure in 5b's decay branches (all from d040_5b_s{1,2}/ckpt_016000.pt, 4000 steps)",
             "  grid: the tipping grid (D-142, seeds 20-22), mean failure share in the primary cells (f0 110-140, B1 250-500) "
             "and over the whole grid",
             f"  test: share of D-145's fixed set ({len(fixed)} test clips 5b fails) still failing (>= 0.5) with the run's heads", ""]
    for title, arms in (("deciding: no data change, F1-F4 heads as 5b's", NULL), ("deciding: the coupled set (D-080)", COUPLED),
                        ("reported only: other branches without the coupled set", OTHER)):
        lines.append(f"  {title}")
        for a in arms:
            cells = "   ".join(f"s{s}: grid {P[run_name(a, s)][0]:.2f} (all {P[run_name(a, s)][1]:.2f}), "
                               f"test {tshare[run_name(a, s)]:.0%}" for s in SEEDS)
            lines.append(f"    {a:16s} {cells}")
    lines.append("")
    verdict = []
    for s in SEEDS:
        nul = [P[run_name(a, s)][0] for a in NULL]
        cp = [P[run_name(a, s)][0] for a in COUPLED]
        verdict.append(("below" if max(cp) < min(nul) else "within" if min(nul) <= min(cp) else "mixed"))
        lines.append(f"  seed {s}: primary-cell failure, null branches {min(nul):.2f}-{max(nul):.2f}, coupled {min(cp):.2f}-{max(cp):.2f}"
                     f" -> {verdict[-1]}")
    if all(v == "below" for v in verdict):
        out = "COUPLED SET: A0 and A1 fail less than every null branch on both seeds (chance under exchangeability 1 %)"
    elif any(v == "within" for v in verdict):
        out = "WITHIN BRANCH SPREAD: a null branch fails as little as the coupled arms on at least one seed"
    else:
        out = "MIXED: reported, Hannah decides"
    lines.append(f"Reading (registered): {out}")
    (REP / "d147.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
