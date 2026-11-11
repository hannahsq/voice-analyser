"""D-098's stop rule: for each run, wait until its training log reaches step 3000, then compare the median logged
gradient norm over steps 2001-3000 (the VTL gate open from step 1700) with its median over steps 501-1000 (before the
gate opens). Exit 1 if any run's ratio exceeds 2 (D-085: 33 in base; 5b about 1), else 0.

    python scripts/analysis/d098_watch.py runs/d098_base runs/d098_heads
"""
import json
import sys
import time
from pathlib import Path

LIMIT = 2.0


def rows(run: Path) -> list:
    p = run / "log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a line being written
            continue
        if "grad_norm" in r:
            out.append(r)
    return out


def median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


if __name__ == "__main__":
    runs = [Path(a) for a in sys.argv[1:]]
    while not all(any(r["step"] >= 3000 for r in rows(run)) for run in runs):
        time.sleep(120)
    bad = False
    for run in runs:
        rr = rows(run)
        before = median([r["grad_norm"] for r in rr if 500 < r["step"] <= 1000])
        after = median([r["grad_norm"] for r in rr if 2000 < r["step"] <= 3000])
        ratio = after / before
        bad |= not ratio <= LIMIT
        print(f"{run.name}: median gradient norm steps 501-1000 {before:.2f}, 2001-3000 {after:.2f}, ratio {ratio:.2f} "
              f"({'STOP' if not ratio <= LIMIT else 'ok'}, limit {LIMIT})", flush=True)
    sys.exit(1 if bad else 0)
