"""D-034 stage 1b (pre-registered 2026-10-23): is student v3 (v2's blend x3, 36,000 steps) adopted over student v2?
Args: v2's fid_breakdown output (step 12,000), v3's (step 36,000), eval_heads.py output with S2#k and S3#k heads,
v2's and v3's training logs (log.jsonl: the synthetic held-out sets at the last checkpoint).
All three must hold:
  1. S3's six heads pass the stage-1 rule on seed means (eval_heads.py's "part-2 rule on means").
  2. Not worse than S2 on the bias measures: women's closed-phase |F1 bias| cut >= S2's - 2 Hz, men's rise <= S2's + 2 Hz,
     Hillenbrand F1 MAE cut >= S2's - 2 Hz (seed means, ensemble (c)).
  3. Fidelity: mean R^2 (per dimension) over the nine held-out speech populations >= v2's + 0.02, and no reported row
     (speech, singing, synthetic) below v2's by more than 0.01."""
import json
import re
import sys

SPEECH = ("English adults (LibriTTS", "English children", "English adults >=", "Icelandic children", "Icelandic teens",
          "Italian", "Portuguese", "Polish", "Korean")


def fidelity(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s{2}(\S.*?)\s{2,}(-?\d\.\d+) /", line)
        if m and not line.startswith("    "):
            rows.setdefault(m.group(1), float(m.group(2)))  # first checkpoint in the file
    return rows


def synthetic(path):
    last = [json.loads(line)["fidelity"] for line in open(path) if '"fidelity"' in line][-1]
    return {f"Synthetic {k[5:]}": v[0] for k, v in last.items() if k.startswith("held_synth")}


def seeds(path, group):
    for line in open(path, encoding="utf-8"):
        if line.startswith(f"  {group} ("):
            nums = [float(x) for x in re.findall(r"([+-]\d+\.\d) ±", line)]
            return nums[0], nums[1], nums[2], "worth a full arm" in line
    raise SystemExit(f"no seed-mean line for {group} in {path}")


f2, f3 = fidelity(sys.argv[1]) | synthetic(sys.argv[4]), fidelity(sys.argv[2]) | synthetic(sys.argv[5])
w2, m2, h2, _ = seeds(sys.argv[3], "S2")
w3, m3, h3, ok3 = seeds(sys.argv[3], "S3")
sp = [k for k in f2 if k.startswith(SPEECH)]
assert len(sp) == 9, sp
gain = sum(f3[k] - f2[k] for k in sp) / len(sp)
worst = min((f3[k] - f2[k], k) for k in f2)
c1 = ok3
c2 = (w3 >= w2 - 2) and (m3 <= m2 + 2) and (h3 >= h2 - 2)
c3 = gain >= 0.02 and worst[0] >= -0.01
print(f"1. S3 passes the stage-1 rule on seed means: {c1}")
print(f"2. bias against S2: women's cut {w3:+.1f} vs {w2:+.1f}; men's rise {m3:+.1f} vs {m2:+.1f}; "
      f"Hillenbrand F1 MAE cut {h3:+.1f} vs {h2:+.1f}: {c2}")
print(f"3. fidelity: speech mean R2 gain {gain:+.3f}; worst row {worst[1]} {worst[0]:+.3f}: {c3}")
for k in f2:
    print(f"   {k:44s} {f2[k]:.3f} -> {f3[k]:.3f}")
verdict = "student v3 adopted" if c1 and c2 and c3 else ("v2 stays (no gain from scale)" if c1 and c2 else "v2 stays")
print(f"DECISION: {verdict}")
