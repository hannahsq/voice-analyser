"""D-114 reported measure: D-110's shift signature (PTDB-TUG closed-phase frames, Hillenbrand) for d114 against 5b,
the same code as `d110_shift_signature.py` with the arm swapped.

    python scripts/analysis/d114_shift.py > reports/analysis/d114/shift_signature.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d110_shift_signature as S  # noqa: E402

if __name__ == "__main__":
    S.R = ("d114_s1", "d114_s2")
    rng = np.random.default_rng(S.SEED)
    lines = ["D-114 (D-110's measure): the shape of d114's F1 shift on real voices (S = d114 - 5b, seed means unless noted); "
             "F1 slope: per-speaker OLS on the reference F1 and log f0, Hz of S per 100 Hz of F1; medians, [95 % speaker bootstrap]", ""]
    S.ptdb(lines, rng)
    S.hillenbrand(lines, rng)
    print("\n".join(l.replace("d109 - 5b", "d114 - 5b").replace("d109's", "d114's") for l in lines))
