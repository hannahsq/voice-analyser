"""Pre-registered rule for the dose arm (D-026, 2026-10-14, before train_v2m was rendered): train_v2m = train_v2i
with p_interactive_coupled 0.5 (every tube clip has the interactive source, its feedback on for a random half; the
criteria were first written for train_v2h, p_interactive 0.5, replaced before rendering). Three S2 seeds (k35) against
mix_conformer_s0-2, and s0 / s1 continued to d16k, dev speakers. Carried forward ("v2m") when all hold:
  k35, means over the three same-seed pairs against mix_conformer:
    women's closed-phase F1 bias change <= -5 Hz;
    Hillenbrand F1 / F2 / F3 MAE <= +3 / +5 / +10 Hz, F3 detection >= -1 and F4 detection >= -2 points, RPA >= -1;
  d16k (the overshoot seen on v2i), mean of s0 / s1:
    women's closed-phase F1 bias within +-10 Hz and F3 bias (model minus closed phase) >= -10 Hz;
    identity resynthesis (resynth_identity_v2m_d16k.txt) within +-5 Hz overall.
Otherwise "fix". Also prints per-formant closed-phase errors (F1-F3, women and men) for mix, v2i, v2m.
Args: scratch dir, paired dir, identity report."""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S, P, IDENT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
K35 = {a: [f"{a}_conformer_s{k}" for k in range(3)] for a in ("mix", "v2i", "v2m")}
D16 = ["v2m_conformer_s0_d16k", "v2m_conformer_s1_d16k"]
every = [m for v in K35.values() for m in v] + D16


def ok(sex, j):
    return dev & (d["sex"] == sex) & np.isfinite(d["cp"][:, j]) & np.all([np.isfinite(d[m][:, j]) for m in every], 0)


def bias(m, sex="f", j=0):
    return float(np.median(d[m][ok(sex, j), j] - d["cp"][ok(sex, j), j]))


def change(b, a):  # as cp_dev.py: median over speakers of the per-speaker median difference
    m = ok("f", 0)
    return float(np.median([np.median(d[b][m & (d["speaker"] == s), 0] - d[a][m & (d["speaker"] == s), 0])
                            for s in np.unique(d["speaker"][m])]))


print("Closed phase (PTDB-TUG dev), model minus reference, Hz: F1 / F2 / F3")
for sex in ("f", "m"):
    for a, ms in (*K35.items(), ("v2m d16k", D16)):
        print(f"  {sex} {a:9s} " + " / ".join(f"{np.mean([bias(m, sex, j) for m in ms]):+6.1f}" for j in range(3)))
fails = []
ch = [change(f"v2m_conformer_s{k}", f"mix_conformer_s{k}") for k in range(3)]
print(f"\nk35 women's bias change v2m - mix: {', '.join(f'{x:+.1f}' for x in ch)}  mean {np.mean(ch):+.1f}  (limit <= -5)")
if np.mean(ch) > -5:
    fails.append("bias change")
KEYS = {"F1 MAE all": ("<=", 3.0), "F2 MAE all": ("<=", 5.0), "F3 MAE all": ("<=", 10.0), "F3 DR  all": (">=", -1.0),
        "F4 DR  all": (">=", -2.0), "all frames": (">=", -1.0)}
for key, (op, lim) in KEYS.items():
    v = [float(re.search(rf"{re.escape(key)}\s*:.*?diff\s+([+-][\d.]+)",
                         (P / f"mix_conformer_s{k}_vs_v2m_conformer_s{k}.txt").read_text(encoding="utf-8")).group(1))
         for k in range(3)]
    good = np.mean(v) <= lim if op == "<=" else np.mean(v) >= lim
    fails += [] if good else [key]
    print(f"  {key:12s} {', '.join(f'{x:+.1f}' for x in v)}  mean {np.mean(v):+.1f}  (limit {op} {lim:+.0f})  {'ok' if good else 'FAIL'}")
b1 = np.mean([bias(m) for m in D16]); b3 = np.mean([bias(m, "f", 2) for m in D16])
print(f"\nd16k women's closed-phase F1 bias {b1:+.1f} (limit +-10), F3 {b3:+.1f} (limit >= -10)")
fails += [] if abs(b1) <= 10 else ["d16k F1 bias"]
fails += [] if b3 >= -10 else ["d16k F3 bias"]
ident = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S+)\s+[+-][\d.]+\s+([+-][\d.]+)", IDENT.read_text(encoding="utf-8"), re.M)}
iv = np.mean([ident["v2m_s0_d16k"], ident["v2m_s1_d16k"]])
print(f"d16k identity resynthesis {iv:+.1f} (limit +-5)")
fails += [] if abs(iv) <= 5 else ["d16k identity"]
print(f"\nfailures: {fails or 'none'}")
print(f"DECISION: {'fix' if fails else 'v2m'}")
