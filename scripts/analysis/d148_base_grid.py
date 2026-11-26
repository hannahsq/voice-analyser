"""D-148: does D-098's base (coupled share in a from-scratch mix, width 192, long crops) fail less than 5b on the
tipping grid? D-147's clips, frames and measures; its runs' scores are read from reports/analysis/d147/d147.json.
CPU, inference only.

    python scripts/analysis/d148_base_grid.py run      -> reports/analysis/d148/d148.{json,txt}
    python scripts/analysis/d148_base_grid.py report   -> reports/analysis/d148/d148.txt
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d148"
DECIDING = "d098_base_b20k"
REPORTED = ("d098_heads_b20k", "d085_base_b20k", "d085_heads_b20k")
# bars, from D-147's scores: LOWER below every branch without the coupled set (either seed); NOT LOWER at or above
# seed 1's lowest deciding null branch (D-147's NULL)


def _d147():
    spec = importlib.util.spec_from_file_location("d147", ROOT / "scripts/analysis/d147_branch_spread.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_run() -> None:
    import torch

    from vocal_analysis.models.infer import load_teacher

    d147 = _d147()
    sets = d147.clip_sets(d147._load("d143", ROOT / "scripts/analysis/d143_probe.py"))
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    REP.mkdir(parents=True, exist_ok=True)
    path = REP / "d148.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for m in (DECIDING,) + REPORTED:
        if m in res:
            continue
        model = load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]
        res[m] = {k: d147.fail_by_clip(model, dev, sets[k]) for k in ("test", "grid")}
        res[m]["grid_keys"] = [it[0] for it in sets["grid"]]
        path.write_text(json.dumps(res), encoding="utf-8")
        print(m, flush=True)
    cmd_report()


def _primary(r):
    keys, g = r["grid_keys"], r["grid"]
    cell = [str(j) for j, k in enumerate(keys) if float(k.split("|")[1][1:]) <= 140
            and 250 <= float(k.split("|")[2][1:]) <= 500 and str(j) in g]
    return float(np.mean([g[j] for j in cell])), float(np.mean(list(g.values())))


def cmd_report() -> None:
    d147 = _d147()
    res = json.loads((REP / "d148.json").read_text(encoding="utf-8"))
    ref = json.loads((ROOT / "reports/analysis/d147/d147.json").read_text(encoding="utf-8"))
    allref = [d147.run_name(a, s) for a in d147.NULL + d147.COUPLED + d147.OTHER for s in d147.SEEDS]
    # D-145 / D-147's fixed test-clip set, as D-147 defined it (on D-147's runs alone, so the set is the same)
    common = sorted(set.intersection(*(set(ref[m]["test"]) for m in allref)), key=int)
    f5 = np.mean([[ref[d147.run_name("d040_5b", s)]["test"][c] for c in common] for s in d147.SEEDS], 0)
    fixed = [c for c, v in zip(common, f5) if v >= 0.5]

    def tshare(r):
        got = [c for c in fixed if c in r["test"]]
        return float(np.mean([r["test"][c] >= 0.5 for c in got])), len(got)

    noncoupled = [d147.run_name(a, s) for a in d147.NULL + d147.OTHER for s in d147.SEEDS]
    pn = [_primary(ref[m])[0] for m in noncoupled]
    lines = ["D-148: own-head failure of D-098's base (one seed) against 5b's decay branches (D-147)",
             "  grid: primary cells (f0 110-140, B1 250-500) and whole grid, tipping grid seeds 20-22; "
             f"test: share of D-145's fixed set ({len(fixed)} clips 5b fails) still failing (>= 0.5)", "",
             f"  D-147, branches without the coupled set (18 runs): grid {min(pn):.2f}-{max(pn):.2f}; "
             f"5b seed 1 {_primary(ref['d040_5b_s1_b20k'])[0]:.2f}, seed 2 {_primary(ref['d040_5b_s2_b20k'])[0]:.2f}",
             "  D-147, coupled: " + ", ".join(f"{m} {_primary(ref[m])[0]:.2f}" for m in
                                               [d147.run_name(a, s) for a in d147.COUPLED for s in d147.SEEDS]), ""]
    for title, ms in (("deciding", (DECIDING,)), ("reported only", REPORTED)):
        lines.append(f"  {title}")
        for m in ms:
            p, a = _primary(res[m])
            t, n = tshare(res[m])
            lines.append(f"    {m:18s} grid {p:.2f} (all {a:.2f}), test {t:.0%} of {n}")
    p = _primary(res[DECIDING])[0]
    LOWER = min(pn)
    NOT_LOWER = min(_primary(ref[d147.run_name(a, 1)])[0] for a in d147.NULL)
    out = ("LOWER: base fails less than every branch without the coupled set, of either 5b seed" if p < LOWER
           else "NOT LOWER: base is within seed 1's null branches or above" if p >= NOT_LOWER
           else "UNCLEAR: within the spread between 5b's two seeds")
    lines += ["", f"Reading (registered; base {p:.2f}, bars < {LOWER:.3f} and >= {NOT_LOWER:.3f}): {out}"]
    (REP / "d148.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
