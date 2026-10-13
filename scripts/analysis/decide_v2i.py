"""Pre-registered rule (D-026, 2026-10-13, written before any v2t result was read): six S2 seeds per arm (mix_conformer,
v2i_conformer, v2t_conformer, seeds 0-5), dev speakers. Per same-seed pair, the change in women's closed-phase F1 bias
(median over speakers of the per-speaker median difference, as cp_dev.py); pooled as the mean over the six pairs with a
t interval over seeds. Guard-rails from the paired files, means over the six pairs.

  Effect B (v2i - mix): <= -15 Hz large; -15 to -5 partial; above -5, or the 95 % interval reaching 0: not shown.
  Attribution A (v2i - v2t): <= -5 Hz credits the interaction (load feedback); otherwise the source shape and labels.
  Guard-rails per arm (vs mix): F1 / F2 / F3 MAE <= +3 / +5 / +10 Hz, RPA all frames >= -1 point; and, added after the
    first three v2i seeds (F3 / F4 losses on two of them), F3 detection >= -1 and F4 detection >= -2 points.
  Set carried forward: if B shown, v2i when A credits the interaction or v2i fails no more guard-rails than v2t, else
    v2t (when v2t - mix is itself shown). If the carried set fails an F3 / F4 guard-rail: "fix" (high-frequency damage
    is investigated before a full-size run). If B is not shown: "v2".

Prints the table and, last, "DECISION: v2i | v2t | fix | v2". Args: scratch dir, paired dir."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
S, P = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = range(6)
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
models = [f"{a}_conformer_s{k}" for a in ("mix", "v2i", "v2t") for k in SEEDS]
ok = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0]) & np.all([np.isfinite(d[m][:, 0]) for m in models], 0)
spk = np.unique(d["speaker"][ok])


def bias_change(b, a):
    return float(np.median([np.median(d[b][ok & (d["speaker"] == s), 0] - d[a][ok & (d["speaker"] == s), 0]) for s in spk]))


def pooled(v):
    v = np.array(v)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


print(f"PTDB-TUG dev women: {ok.sum()} frames, {len(spk)} speakers; change in closed-phase F1 bias per seed (Hz)")
res = {}
for b, a in (("v2i", "mix"), ("v2t", "mix"), ("v2i", "v2t")):
    v = [bias_change(f"{b}_conformer_s{k}", f"{a}_conformer_s{k}") for k in SEEDS]
    res[b, a] = pooled(v)
    print(f"  {b} - {a}: {', '.join(f'{x:+.1f}' for x in v)}  mean {res[b, a][0]:+.1f} [{res[b, a][1]:+.1f}, {res[b, a][2]:+.1f}]")

KEYS = {"F1 MAE all": ("<=", 3.0), "F2 MAE all": ("<=", 5.0), "F3 MAE all": ("<=", 10.0), "F3 DR  all": (">=", -1.0),
        "F4 DR  all": (">=", -2.0), "all frames": (">=", -1.0)}
HIGH = {"F3 MAE all", "F3 DR  all", "F4 DR  all"}
fails = {}
for arm in ("v2i", "v2t"):
    fails[arm] = []
    print(f"\nGuard-rails, {arm} - mix (paired files, mean over seeds)")
    for key, (op, lim) in KEYS.items():
        v = []
        for k in SEEDS:
            txt = (P / f"mix_conformer_s{k}_vs_{arm}_conformer_s{k}.txt").read_text(encoding="utf-8")
            v.append(float(re.search(rf"{re.escape(key)}\s*:.*?diff\s+([+-][\d.]+)", txt).group(1)))
        m = float(np.mean(v))
        good = m <= lim if op == "<=" else m >= lim
        if not good:
            fails[arm].append(key)
        print(f"  {key:12s} {', '.join(f'{x:+.1f}' for x in v)}  mean {m:+.1f}  (limit {op} {lim:+.0f})  {'ok' if good else 'FAIL'}")

shown = lambda r: r[0] <= -5 and r[2] < 0  # noqa: E731
B, A, T = res["v2i", "mix"], res["v2i", "v2t"], res["v2t", "mix"]
size = "large" if B[0] <= -15 else "partial" if shown(B) else "not shown"
print(f"\nEffect v2i - mix: {B[0]:+.1f} Hz ({size}); attribution v2i - v2t: {A[0]:+.1f} Hz "
      f"({'interaction' if A[0] <= -5 else 'source shape and labels'})")
if not shown(B):
    decision = "v2"
else:
    pick = "v2i" if A[0] <= -5 or len(fails["v2i"]) <= len(fails["v2t"]) or not shown(T) else "v2t"
    decision = "fix" if HIGH & set(fails[pick]) else pick
    print(f"Set carried forward: {pick}; guard-rail failures {fails[pick] or 'none'}")
print(f"DECISION: {decision}")
