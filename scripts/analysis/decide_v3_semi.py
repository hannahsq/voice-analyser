"""Semi-private confirmation of the teacher v3 candidate (D-025, pre-registered 2026-10-17 before seed 2 ran and
before any semi-private use for it). Three full-size seed pairs, train_v2 (conformer_v2[_sK]_d16k) against train_v2 +
VocalSet + Zeroth (conformer_v2r2[_sK]_d16k), same seed. The two headline effects, means over the three pairs:
  E1 women's closed-phase F1 bias change on PTDB-TUG (median over speakers of the per-speaker median difference);
  E2 Hillenbrand F3 MAE change (all speakers).
Replicates when both are negative on dev and on semi-private each is at least half its dev size (E_semi <= E_dev / 2),
and no semi-private guard-rail fails: F1 MAE all <= +2 Hz, F3 and F4 detection >= -1 point, PTDB-TUG RPA all frames
>= -0.5 point. Semi-private confirms or not; it never decides anything else (D-024).
Args: scratch dir, dev paired dir, semi-private paired dir."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, PD, PS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
PAIRS = [("conformer_v2_d16k", "conformer_v2r2_d16k")] + [(f"conformer_v2_s{k}_d16k", f"conformer_v2r2_s{k}_d16k") for k in (1, 2)]
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
tier = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") for s in d["speaker"]])
every = [m for p in PAIRS for m in p]


def e1(a, b, t):
    m = (tier == t) & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.all([np.isfinite(d[x][:, 0]) for x in every], 0)
    return float(np.median([np.median(d[b][m & (d["speaker"] == s), 0] - d[a][m & (d["speaker"] == s), 0])
                            for s in np.unique(d["speaker"][m])]))


def paired(P, a, b, key):
    txt = (P / f"{a}_vs_{b}.txt").read_text(encoding="utf-8")
    return float(re.search(rf"{re.escape(key)}\s*:.*?diff\s+([+-][\d.]+)", txt).group(1))


res = {}
for t, P in (("dev", PD), ("semi", PS)):
    res[t] = {"E1": [e1(a, b, t) for a, b in PAIRS], "E2": [paired(P, a, b, "F3 MAE all") for a, b in PAIRS]}
    for key in ("F1 MAE all", "F3 DR  all", "F4 DR  all", "all frames"):
        res[t][key] = [paired(P, a, b, key) for a, b in PAIRS]
    print(f"{t}:")
    for k, v in res[t].items():
        print(f"  {k:11s} " + " ".join(f"{x:+6.1f}" for x in v) + f"   mean {np.mean(v):+6.1f}")
fails = []
for k in ("E1", "E2"):
    dv, sv = np.mean(res["dev"][k]), np.mean(res["semi"][k])
    if dv >= 0:
        fails.append(f"{k} not negative on dev")
    elif sv > dv / 2:
        fails.append(f"{k} semi {sv:+.1f} > half of dev {dv:+.1f}")
for key, op, lim in (("F1 MAE all", "<=", 2.0), ("F3 DR  all", ">=", -1.0), ("F4 DR  all", ">=", -1.0), ("all frames", ">=", -0.5)):
    v = np.mean(res["semi"][key])
    if not (v <= lim if op == "<=" else v >= lim):
        fails.append(f"semi {key.strip()} {v:+.1f}")
print(f"\nfailures: {fails or 'none'}")
print(f"DECISION: {'does not replicate' if fails else 'replicates'}")
