"""Does current code reproduce an earlier run's start? Compares the loss parts of the first logged step of two runs'
log.jsonl (same seed and settings). Prints the relative differences and "REPLICA: yes" when every part is within
the tolerance (default 0.5 %), else "REPLICA: no". Args: earlier run dir, replica run dir, optional tolerance."""
import json
import sys
from pathlib import Path

a, b = (Path(x) for x in sys.argv[1:3])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
first = lambda d: next(json.loads(x) for x in (d / "log.jsonl").read_text().splitlines() if "loss" in json.loads(x))  # noqa: E731
ra, rb = first(a), first(b)
assert ra["step"] == rb["step"], (ra["step"], rb["step"])
ok = True
for k in ("f0", "voicing", "sonorant", "formants", "bandwidth"):
    rel = abs(rb[k] - ra[k]) / max(abs(ra[k]), 1e-9)
    ok &= rel <= tol
    print(f"step {ra['step']} {k:10s} {ra[k]:+.5f} {rb[k]:+.5f} rel {rel:.2e}")
print(f"REPLICA: {'yes' if ok else 'no'}")
