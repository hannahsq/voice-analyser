"""D-091 amendment 1: v3.1's F1 level on PTDB-TUG dev against each reference in the windowed closed-phase cache (the
closed phase, whole-cycle LPC, Praat), and the references against each other, by sex. Descriptive; no rule.
Median over frames, speaker bootstrap (2000). Frames: dev speakers with all three references and the model finite.

    python scripts/analysis/d091_ptdb_levels.py
"""

import json
from pathlib import Path

import numpy as np

CACHE = Path("cache/m3w/ptdb_egg_sub.npz")
OUT = Path("reports/analysis/d091")
V31 = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
SEED = 5091


def main():
    d = dict(np.load(CACHE, allow_pickle=False))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    dev = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    m = np.mean([d[k][:, 0] for k in V31], 0)
    cp, cyc, pr = d["cp"][:, 0], d["cycle"][:, 0], d["praat"][:, 0]
    pairs = {"v3.1 - closed phase": m - cp, "v3.1 - whole-cycle LPC": m - cyc, "v3.1 - Praat": m - pr,
             "whole-cycle LPC - closed phase": cyc - cp, "Praat - closed phase": pr - cp}
    ok = dev & np.isfinite(m) & np.isfinite(cp) & np.isfinite(cyc) & np.isfinite(pr)
    rng = np.random.default_rng(SEED)
    lines = ["D-091 amendment 1: F1 on PTDB-TUG dev, median over frames [95 % speaker bootstrap], Hz; v3.1 = seed mean of "
             + ", ".join(V31), ""]
    for sx, name in (("f", "women"), ("m", "men")):
        k = ok & (d["sex"] == sx)
        spk = np.unique(d["speaker"][k])
        lines.append(f"{name}: {len(spk)} speakers, {int(k.sum())} frames")
        for label, e in pairs.items():
            per = {s: e[k & (d["speaker"] == s)] for s in spk}
            bs = [np.median(np.concatenate([per[s] for s in rng.choice(spk, len(spk))])) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            lines.append(f"  {label:32s} {np.median(e[k]):+6.1f} [{lo:+.1f}, {hi:+.1f}]")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ptdb_levels.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
