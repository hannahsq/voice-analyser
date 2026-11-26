"""D-149: two cheap checks of D-098's heads arm (0.04 on the tipping grid's primary cells against base's 0.22, D-148)
before any GPU work: (1) its checkpoints' stability on the grid, (2) the dropped formant on a private recording's takes.
CPU, inference only. The private file names are passed as arguments; per-take numbers stay in OUT (outside the repo).

    python scripts/analysis/d149_heads_check.py grid                 -> reports/analysis/d149/grid.json
    python scripts/analysis/d149_heads_check.py private REC OUT FILES -> OUT/d149_private.json
    python scripts/analysis/d149_heads_check.py report [OUT]          -> reports/analysis/d149/d149.txt (counts only)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d149"
ARMS = ("d098_base_b20k", "d098_heads_b20k")
REF = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STEPS = (17000, 18000, 19000, 20000)
LOWER = 0.159  # D-148's LOWER bar: below every branch of 5b without the coupled set (D-147)


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(path, dev):
    from vocal_analysis.models.infer import load_teacher

    return load_teacher(str(path), dev)[0]


def _dev():
    import torch

    torch.set_num_threads(6)
    return torch.device("cpu")


def cmd_grid() -> None:
    d147 = _mod("d147", "scripts/analysis/d147_branch_spread.py")
    d148 = _mod("d148", "scripts/analysis/d148_base_grid.py")
    grid = d147.clip_sets(d147._load("d143", ROOT / "scripts/analysis/d143_probe.py"))["grid"]
    dev = _dev()
    REP.mkdir(parents=True, exist_ok=True)
    path = REP / "grid.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for m in ARMS + REF:
        for st in STEPS:
            key = f"{m}@{st}"
            if key in res:
                continue
            r = {"grid": {str(k): v for k, v in d147.fail_by_clip(_load(ROOT / f"runs/{m}/ckpt_{st:06d}.pt", dev), dev, grid).items()},
                 "grid_keys": [it[0] for it in grid]}
            res[key] = {"primary": d148._primary(r)[0], "all": d148._primary(r)[1]}
            path.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(key, f"{res[key]['primary']:.3f}", flush=True)


def cmd_private(rec: Path, out: Path, files: list[str]) -> None:
    d138 = _mod("d138", "scripts/analysis/d138_real_balance.py")
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    dev = _dev()
    models = {m: _load(ROOT / f"runs/{m}/ckpt_last.pt", dev) for m in ARMS}
    res = {}
    for name, _, y in d138._load(rec, files):
        if name not in lv:
            continue
        t = lv[name]
        rd, _ = d138._read(models, dev, y, np.array(t["F"], float))
        f5 = float(np.mean([t["read"][m]["fail"] for m in REF]))
        res[name] = {"group": "fail" if f5 >= d138.FAIL else "ok" if f5 < d138.OK else "mid", "5b": f5,
                     **{m: rd[m]["fail"] for m in ARMS}}
        print(".", end="", flush=True)
    (out / "d149_private.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print()


def cmd_report(out: Path | None) -> None:
    g = json.loads((REP / "grid.json").read_text(encoding="utf-8"))
    lines = ["D-149: D-098's heads arm, two checks before GPU work", "",
             "(1) Checkpoint stability: tipping grid (seeds 20-22), primary-cell failure (f0 110-140, B1 250-500); whole grid in ()"]
    for m in ARMS + REF:
        lines.append(f"  {m:18s} " + "  ".join(f"{st // 1000}k {g[f'{m}@{st}']['primary']:.2f} ({g[f'{m}@{st}']['all']:.2f})"
                                              for st in STEPS))
    h = [g[f"{ARMS[1]}@{st}"]["primary"] for st in STEPS]
    b = [g[f"{ARMS[0]}@{st}"]["primary"] for st in STEPS]
    if max(h) < min(min(b), LOWER):
        v1 = "STABLE: every heads checkpoint below every base checkpoint and below D-148's bar"
    elif any(hh >= bb for hh, bb in zip(h, b)):
        v1 = "UNSTABLE: a heads checkpoint fails as much as base's at the same step"
    else:
        v1 = "PARTLY: below base step by step, but not throughout below both bars"
    lines += [f"  Reading (registered): {v1}", ""]
    if out is not None and (out / "d149_private.json").exists():
        P = json.loads((out / "d149_private.json").read_text(encoding="utf-8"))
        fail = [v for v in P.values() if v["group"] == "fail"]
        ok = [v for v in P.values() if v["group"] == "ok"]
        lines.append(f"(2) A private recording: {len(fail)} takes 5b fails (seed mean >= 0.5), {len(ok)} it reads (< 0.2); "
                     "failure = F2 nearer Praat's F3 than its F2")
        cnt = {}
        for m in ARMS:
            fixed = sum(v[m] < 0.2 for v in fail)
            still = sum(v[m] >= 0.5 for v in fail)
            false = sum(v[m] >= 0.5 for v in ok)
            cnt[m] = (fixed, still, false)
            lines.append(f"  {m:18s} failing takes read (< 0.2) {fixed} of {len(fail)}, still failing {still}; "
                         f"newly failing among the read takes {false} of {len(ok)}")
        fh, _, xh = cnt[ARMS[1]]
        fb = cnt[ARMS[0]][0]
        if fh >= 3 and xh <= 1 and fh > fb:
            v2 = "HOLDS: the heads arm reads at least half the failing takes, more than base, with at most one new failure"
        elif fh <= 1:
            v2 = "DOES NOT HOLD: the heads arm reads at most one failing take; the grid effect is synthetic-only so far"
        else:
            v2 = "PARTLY: reported, Hannah decides"
        lines.append(f"  Reading (registered): {v2}")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d149.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "grid":
        cmd_grid()
    elif cmd == "private":
        cmd_private(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    elif cmd == "report":
        cmd_report(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
