"""Best of several arms by their decide_recipe.py files (D-030, rule fixed before running): highest decision rank
(better > same > grey > worse), ties broken by the lowest sum over F1-F4 of the mean difference in SE units.
Prints the arm's name and its decision. Args: recipe dir, arm names."""
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
RANK = {"better": 3, "same": 2, "grey": 1, "worse": 0}
best = None
for arm in sys.argv[2:]:
    t = (D / f"decide_{arm}_n3.txt").read_text(encoding="utf-8")
    dec = re.search(r"^DECISION: (\S+)", t, re.M).group(1)
    z = sum(float(re.search(rf"mean F{n}: \S+ \(SE \S+, (\S+) SE\)", t).group(1)) for n in range(1, 5))
    z = z if z == z else float("inf")  # NaN (diverged) sorts last
    key = (RANK[dec], -z)
    if best is None or key > best[0]:
        best = (key, arm, dec)
print(best[1], best[2])
