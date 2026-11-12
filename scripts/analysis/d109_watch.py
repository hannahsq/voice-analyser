"""D-109's stop rule: for each arm (a decay branch of d040_5b_sN with the reconstruction loss), wait until its log
reaches step 17000 (1000 steps into the branch), then compare, over steps 16501-17000, the median logged gradient norm
and the median formant term (`formants`) with 5b's own branch over the same steps (`d040_5b_sN_b20k`). Exit 1 if any
arm's gradient-norm ratio exceeds 2 or its formant-term ratio exceeds 1.10, or its reconstruction loss is not finite;
else 0.

    python scripts/analysis/d109_watch.py runs/d109_s1:runs/d040_5b_s1_b20k runs/d109_s2:runs/d040_5b_s2_b20k
"""
import json
import math
import sys
import time
from pathlib import Path

GRAD_LIMIT, FORMANT_LIMIT, AT = 2.0, 1.10, 17000


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


def window(rr, key):
    return median([r[key] for r in rr if AT - 500 < r["step"] <= AT and key in r])


if __name__ == "__main__":
    pairs = [tuple(Path(x) for x in a.split(":")) for a in sys.argv[1:]]
    while not all(any(r["step"] >= AT for r in rows(arm)) for arm, _ in pairs):
        bad = [arm for arm, _ in pairs if any(not math.isfinite(r.get("recon", 0.0)) for r in rows(arm))]
        if bad:
            print(f"STOP: non-finite reconstruction loss in {', '.join(a.name for a in bad)}", flush=True)
            sys.exit(1)
        time.sleep(120)
    stop = False
    for arm, ref in pairs:
        ra, rr = rows(arm), rows(ref)
        g = window(ra, "grad_norm") / window(rr, "grad_norm")
        f = window(ra, "formants") / window(rr, "formants")
        rec = window(ra, "recon")
        bad = not (g <= GRAD_LIMIT and f <= FORMANT_LIMIT and math.isfinite(rec))
        stop |= bad
        print(f"{arm.name} vs {ref.name}, steps {AT - 499}-{AT}: gradient norm ratio {g:.2f} (limit {GRAD_LIMIT}), formant term "
              f"ratio {f:.3f} (limit {FORMANT_LIMIT}), reconstruction loss {rec:.3f} -> {'STOP' if bad else 'ok'}", flush=True)
    sys.exit(1 if stop else 0)
