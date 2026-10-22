"""D-033: tables from probe_results.json (probe.py): per evaluation set, every layer and teacher v3; then the best
layer per measure. Args: results json, out txt."""
import json
import sys

R = json.load(open(sys.argv[1]))
out = open(sys.argv[2], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out), print(*a))  # noqa: E731
HIGHER = ("rpa", "acc")  # higher is better; MAE lower
p("D-033: linear probe of Parakeet TDT 0.6B v3's encoder (L0 = subsampling output, L1-L24 = layers), fitted on teacher v3's"
  " training mixture only; teacher v3 (conformer_v2r2_s1_d16k) on the same frames. MAE in Hz, RPA and accuracy in %.")
for name, rows in R.items():
    keys = list(rows["teacher_v3"])
    fmt = lambda k, v: f"{100 * v:7.1f}" if any(h in k for h in HIGHER) else f"{v:7.1f}"  # noqa: E731
    short = [k.replace("_mae_hz", "").replace("_mae_", " ").replace("_acc", " acc").replace("f0_rpa", "f0 RPA") for k in keys]
    p(f"\n## {name}")
    p(f"{'':12s}" + "".join(f"{s[:7]:>8s}" for s in short))
    for who in [k for k in rows if k != "teacher_v3"] + ["teacher_v3"]:
        p(f"{who:12s}" + "".join(" " + fmt(k, rows[who][k]) for k in keys))
    best = []
    for k, s in zip(keys, short):
        layers = {w: r[k] for w, r in rows.items() if w != "teacher_v3"}
        b = (max if any(h in k for h in HIGHER) else min)(layers, key=layers.get)
        best.append(f"{s} {b} {fmt(k, layers[b]).strip()} (teacher {fmt(k, rows['teacher_v3'][k]).strip()})")
    p("best layer: " + "; ".join(best))
