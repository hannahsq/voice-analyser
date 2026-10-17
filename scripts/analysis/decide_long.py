"""Pre-registered rule (D-028, 2026-10-17, before any model was measured): are long training clips worth a set?
Judged on teacher v3 (the name given as the first argument); other models in the files are reported only.
  harm, synthetic (long_test.py): whole-file MAE on frames 400-800 or 800+ at least 3 % above the 4 s windows' MAE,
    any of F1-F3;
  harm, real (cp_long.py JOINED): same-speaker joined files' closed-phase median |error| at least 3 % above alone,
    either sex, any of F1-F3;
  carry-over (cp_long.py CARRY): cross-sex, either order, |median shift| of speaker B's formants >= 10 cents, any
    of F1-F3.
DECISION: "long" (any of the three: a long-clip set with speaker switches and sustained larynx changes, roadmap 5),
"pooling" (none, but whole files at least 3 % better than 4 s windows on frames beyond 400, or joined at least 3 %
better: long clips worth trying for the evidence they pool, with the VTL head), "not now" otherwise.
Args: model name, long_test output, cp_long output."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
M = sys.argv[1]
lt, cl = (open(f, encoding="utf-8").read() for f in sys.argv[2:4])
mae = {(mode, band): list(map(float, v.split())) for mode, band, v in
       re.findall(rf"^MAE {re.escape(M)} (\S+)\s+(\S+) ([\d. ]+?)\s+\(n", lt, re.M)}
harm_s = help_s = False
for band in ("400-800", "800-end"):
    w, c = mae[("whole", band)], mae[("4s", band)]
    rel = [100 * (a - b) / b for a, b in zip(w, c)]
    harm_s |= any(r >= 3 for r in rel)
    help_s |= any(r <= -3 for r in rel)
    print(f"synthetic {band}: whole {w} 4s {c}  whole vs 4s % " + " ".join(f"{r:+.1f}" for r in rel))
joined = {sx: list(map(float, v.split())) for sx, v in re.findall(rf"^JOINED {re.escape(M)} (\S+) ([-+\d. ]+)$", cl, re.M)}
harm_r = any(x >= 3 for v in joined.values() for x in v)
help_r = any(x <= -3 for v in joined.values() for x in v)
for sx, v in joined.items():
    print(f"joined {sx}: % change in median |error| F1-F3 {v}")
carry = {(k, o): list(map(float, v.split())) for k, o, v in
         re.findall(rf"^CARRY {re.escape(M)} (\S+) (\S+) ([-+\d. ]+?)\s+\(", cl, re.M)}
co = any(abs(x) >= 10 for (k, _), v in carry.items() if k == "cross" for x in v)
for key, v in carry.items():
    print(f"carry {key[0]} {key[1]}: median shift toward A, cents F1-F3 {v}")
print(f"harm synthetic {harm_s}, harm real {harm_r}, carry-over {co}; whole better: synthetic {help_s}, real {help_r}")
print(f"DECISION: {'long' if harm_s or harm_r or co else 'pooling' if help_s or help_r else 'not now'}")
