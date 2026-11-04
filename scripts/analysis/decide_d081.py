"""D-081 decision: the oracle battery's verdicts (I from identify.txt, A from label_oracle.json, B from the heads'
JSONs) and what follows for the screening run.

    python scripts/analysis/decide_d081.py
"""

import json
import re
from pathlib import Path

O = Path("reports/analysis/d081")
TRUNKS = ("v31_s1", "v31_s2")


def spread(r):
    v = list(r["by_seed"].values())
    return abs(v[0] - v[1])


if __name__ == "__main__":
    ident = (O / "identify.txt").read_text()
    I = re.search(r"I \(without aspiration.*\): (.+)", ident).group(1).strip()
    A = json.loads((O / "label_oracle.json").read_text())
    B = {t: json.loads((O / f"heads_{t}.json").read_text()) for t in TRUNKS}
    L = ["D-081: the oracle battery", "", f"I  (is the offset audible, without aspiration): {I}",
         f"A1 (the glottal slope's check): {'valid' if A['A1']['valid'] else 'BROKEN'}",
         f"A2 (closed F1 from perfect acoustic-domain labels): {A['A2']['verdict']} (slope {A['A2']['full']['slope']:+.3f}; "
         f"without bandwidths {A['A2']['no bandwidths']['slope']:+.3f})",
         f"A3 (aVTL from iVTL context): {A['A3']['verdict']}", ""]
    # B1
    d = {t: B[t]["B1"]["h+<a>"]["mean"] - B[t]["B1"]["h"]["mean"] for t in TRUNKS}
    top = {t: B[t]["B1"]["h+<a>"]["mean"] for t in TRUNKS}
    b1 = ("usable" if all(top[t] >= 0.7 and d[t] >= 0.3 for t in TRUNKS) else "not usable" if all(x < 0.1 for x in d.values())
          else "partly")
    L.append("B1 glottal slope, closed F1 head (h / h+<a> / h+<a>+df1), by trunk: " + "; ".join(
        f"{t} {B[t]['B1']['h']['mean']:+.3f} / {B[t]['B1']['h+<a>']['mean']:+.3f} / {B[t]['B1']['h+<a>+df1']['mean']:+.3f}" for t in TRUNKS)
             + f" -> {b1}")
    d2 = {t: B[t]["B2"]["h+B1ac"]["mean"] - B[t]["B1"]["h"]["mean"] for t in TRUNKS}
    b2 = "helps" if all(x >= 0.2 for x in d2.values()) else "no help"
    L.append("B2 glottal slope with the true acoustic B1, change against h: " + ", ".join(f"{t} {x:+.3f}" for t, x in d2.items()) + f" -> {b2}")
    verdicts = {"B1": b1, "B2": b2}
    for test, name, unit in (("B3", "h+aperiodicity", "Hz, F1-F4 mean"), ("B4", "h+source", "Hz, B1-B4 mean"), ("B5", "h+iVTL", "cm")):
        parts, good = [], []
        for t in TRUNKS:
            h, o = B[t][test]["h"], B[t][test][name]
            rel = o["mean"] / h["mean"] - 1
            noise = 3 * max(spread(h), spread(o))
            good.append(rel <= -0.05 and h["mean"] - o["mean"] > noise)
            parts.append(f"{t} {h['mean']:.3f} -> {o['mean']:.3f} ({100 * rel:+.1f} %, 3x seed spread {noise:.3f})")
        verdicts[test] = "helps" if all(good) else "no help"
        L.append(f"{test} MAE ({unit}), h -> {name}: " + "; ".join(parts) + f" -> {verdicts[test]}")
    closed_in = I == "audible" and A["A1"]["valid"] and b1 == "usable"
    L += ["", "What follows for the screening run:",
          f"  closed-F1 question {'IN' if closed_in else 'OUT'} (needs I audible, A1 valid, B1 usable)"
          + ("; closed for the cascade's coupling (A2 not recoverable)" if A["A2"]["verdict"] == "not recoverable" else ""),
          f"  aperiodicity heads {verdicts['B3']}; source-shape heads (for bandwidths) {verdicts['B4']}; iVTL (for aVTL) {verdicts['B5']}; "
          f"long crops for the aVTL split: {A['A3']['verdict']}"]
    L.append("")
    L.append("DECISION: I " + I + "; A2 " + A["A2"]["verdict"] + "; " + "; ".join(f"{k} {v}" for k, v in verdicts.items()))
    (O / "decide_d081.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
