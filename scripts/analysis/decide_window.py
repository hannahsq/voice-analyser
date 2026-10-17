"""Pre-registered rule (D-028, 2026-10-18, before running): should analyse() run files longer than 400 frames in
4 s windows (hop 2 s, each frame from its most central window) by default? Teacher v3 whole ("v3") against windowed
("v3w"), dev tier, the same model, so no seed noise:
  closed phase (cp_long.py, each PTDB-TUG utterance alone; every one is longer than 4 s): median |error| no more than
    2 % worse for any of F1-F3, either sex;
  PTDB-TUG f0 RPA (paired file, all frames): no more than 0.2 point worse;
  Hillenbrand (words of about 0.5 s): F1-F3 MAE unchanged (|diff| <= 0.1 Hz).
DECISION: "windows" if all hold, else "whole". Joined files and carry-over under windows are reported.
Args: paired file (v3 vs v3w), cp_long output (with v3 and v3w)."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
paired, cpl = (open(f, encoding="utf-8").read() for f in sys.argv[1:3])
ok = True
for key in ("F1 MAE all", "F2 MAE all", "F3 MAE all"):
    dv = float(re.search(rf"{key}\s*:.*?diff\s+([+-][\d.]+)", paired).group(1))
    ok &= abs(dv) <= 0.1
    print(f"Hillenbrand {key}: windowed minus whole {dv:+.1f} Hz")
rpa = float(re.search(r"all frames\s*:.*?diff\s+([+-][\d.]+)", paired).group(1))
ok &= rpa >= -0.2
print(f"PTDB-TUG RPA all frames: {rpa:+.2f} points")
alone = {(m, sx): list(map(float, v.split())) for m, sx, v in re.findall(r"^  (v3w?) (\S): \|err\| alone ([\d. ]+?)\s+joined", cpl, re.M)}
for sx in ("f", "m"):
    w, a = alone[("v3w", sx)], alone[("v3", sx)]
    rel = [100 * (x - y) / y for x, y in zip(w, a)]
    ok &= all(r <= 2 for r in rel)
    print(f"closed phase {sx}, utterance alone: median |error| whole {a} windowed {w} ({' '.join(f'{r:+.1f}' for r in rel)} %)")
for line in re.findall(r"^(?:JOINED|CARRY) v3w? .*$", cpl, re.M):
    print("  " + line)
print(f"DECISION: {'windows' if ok else 'whole'}")
