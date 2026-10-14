"""Pre-registered (D-026, 2026-10-14, before train_v2s was rendered): which set carries forward as the base for the
full-size run and M3 days 2-3, train_v2m or train_v2s (v2m with Rosenberg closure sharpness 1-3 per clip). From
decide_v2m.txt, decide_v2s.txt (the same rule for both) and sensitivity/shortcut_probe_sharp.txt:
  both pass: v2s if it improves the F3 it targets (F3NOFB at least 3 Hz smaller, or Hillenbrand F3 MAE at least 5 Hz
    smaller, means over the three k35 seeds) without giving back the rest (k35 women's closed-phase bias change no
    more than 3 Hz weaker, SHORTCUT no more than 2 Hz larger); otherwise v2m (fewer changes);
  one passes: that one; neither: "fix".
Prints the inputs and, last, "DECISION: v2m | v2s | fix". Arg: reports/analysis."""
import re
import sys
from pathlib import Path

A = Path(sys.argv[1])
dec, bias, f3 = {}, {}, {}
for arm in ("v2m", "v2s"):
    txt = (A / f"decide_{arm}.txt").read_text(encoding="utf-8")
    dec[arm] = re.search(r"^DECISION: (\S+)", txt, re.M).group(1)
    bias[arm] = float(re.search(r"k35 women's bias change .*?mean ([+-][\d.]+)", txt).group(1))
    f3[arm] = float(re.search(r"F3 MAE all .*?mean ([+-][\d.]+)", txt).group(1))
probe = (A / "sensitivity/shortcut_probe_sharp.txt").read_text(encoding="utf-8")
val = {(m.group(1), m.group(2)): float(m.group(3)) for m in re.finditer(r"^(SHORTCUT|F3NOFB|COUPLED) (\S+) ([\d.]+)", probe, re.M)}
for arm in ("v2m", "v2s"):
    print(f"{arm}: decision {dec[arm]}, k35 women's bias change {bias[arm]:+.1f}, Hillenbrand F3 MAE change {f3[arm]:+.1f}, "
          f"shortcut {val['SHORTCUT', arm]:.1f}, F3 without feedback {val['F3NOFB', arm]:.1f}, coupled {val['COUPLED', arm]:.1f} (Hz)")
passed = [a for a in ("v2m", "v2s") if dec[a] == a]
if len(passed) == 2:
    better = val["F3NOFB", "v2s"] <= val["F3NOFB", "v2m"] - 3.0 or f3["v2s"] <= f3["v2m"] - 5.0
    kept = bias["v2s"] <= bias["v2m"] + 3.0 and val["SHORTCUT", "v2s"] <= val["SHORTCUT", "v2m"] + 2.0
    pick = "v2s" if better and kept else "v2m"
elif passed:
    pick = passed[0]
else:
    pick = "fix"
print(f"DECISION: {pick}")
