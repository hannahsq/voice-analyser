"""Roadmap item 7's thresholds (D-050 amendment 2, as specified by D-126): one per probe measure, from the noise floors only.

threshold = max(seed spread, shuffle null), where
- seed spread: the largest absolute difference between any two of the five Conformer seeds (teacher v3's three, v3.1's
  two) on the measure (probe 14 has three of them scored; it uses those);
- shuffle null (slope measures whose ideal is 0 only): max(|null_low|, |null_high|), the stored 2.5 / 97.5 % points of
  the measure's shuffle null.
A model's measure is flagged when |seed mean - ideal| > threshold. Flagged, not failed (D-050).

Writes reports/probes/thresholds.json and prints a summary (with the flags of teacher v3.1's and v3's seed means).

    python scripts/analysis/probe_thresholds.py > reports/analysis/m3_close/probe_thresholds.txt
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

PROBES = Path("reports/probes")
OUT = PROBES / "thresholds.json"
V3 = ("conformer_v2r2_d16k", "conformer_v2r2_s1_d16k", "conformer_v2r2_s2_d16k")
V31 = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
SEEDS = V3 + V31
SKIP = {"bw_evidence"}  # probe 12 describes what the bandwidth heads rely on; it has no ideal, so no threshold


def _measures(probe: str, d: dict) -> dict[str, dict]:
    """measure -> {point, ideal, null (or None)} in one probe's json."""
    out = {}
    for k, v in d.items():
        if probe == "channel":
            # change against clean: cents and |error| change, ideal 0; sigma's change is meant to grow, so not here
            if k.endswith("/change_cents") or k.endswith("/abs_error_change_hz"):
                out[k] = {"point": float(v[0]), "ideal": 0.0, "null": None}
        elif probe == "reversal":
            if isinstance(v, dict) and "leak" in v:  # cross-speaker excess near the join; ideal 0
                out[f"{k}/leak"] = {"point": float(v["leak"]), "ideal": 0.0, "null": None}
        elif isinstance(v, dict) and "point" in v and "ideal" in v:
            null = None
            if "null_low" in v and v["ideal"] == 0.0:
                null = max(abs(v["null_low"]), abs(v["null_high"]))
            out[k] = {"point": float(v["point"]), "ideal": float(v["ideal"]), "null": null}
    return out


def main() -> None:
    table: dict[str, dict[str, dict]] = {}
    lines = []
    flags = {"v3.1": [], "v3": []}
    for pdir in sorted(p for p in PROBES.iterdir() if p.is_dir()):
        probe = pdir.name
        if probe in SKIP:
            lines.append(f"{probe}: skipped (descriptive; no ideal)")
            continue
        runs = {s: json.loads((pdir / f"{s}.json").read_text()) for s in SEEDS if (pdir / f"{s}.json").exists()}
        if len(runs) < 2:
            lines.append(f"{probe}: fewer than two seeds scored; no threshold")
            continue
        per = {s: _measures(probe, d) for s, d in runs.items()}
        keys = sorted(set.union(*(set(m) for m in per.values())))
        table[probe] = {}
        nflag = {"v3.1": 0, "v3": 0}
        for k in keys:
            # the seeds that have the measure (VTL and VTLi exist only in v3.1's two), at least two
            pts = {s: per[s][k]["point"] for s in per if k in per[s] and np.isfinite(per[s][k]["point"])}
            if len(pts) < 2:
                continue
            spread = max(abs(pts[a] - pts[b]) for a, b in combinations(pts, 2))
            nulls = [per[s][k]["null"] for s in pts if per[s][k]["null"] is not None]
            null = float(np.max(nulls)) if nulls else None
            thr = max(spread, null) if null is not None else spread
            ideal = per[next(iter(pts))][k]["ideal"]
            row = {"ideal": ideal, "threshold": thr, "seed_spread": spread, "null": null,
                   "seeds": sorted(pts)}
            for name, group in (("v3.1", V31), ("v3", V3)):
                have = [pts[s] for s in group if s in pts]
                if have:
                    dev = float(np.mean(have)) - ideal
                    row[f"{name}_deviation"] = dev
                    if abs(dev) > thr:
                        flags[name].append(f"{probe}:{k} {dev:+.4f} (threshold {thr:.4f})")
                        nflag[name] += 1
            table[probe][k] = row
        lines.append(f"{probe}: {len(table[probe])} measures, {len(runs)} seeds; flagged v3.1 {nflag['v3.1']}, v3 {nflag['v3']}")
    OUT.write_text(json.dumps({"rule": "D-050 amendment 2, as specified by D-126", "seeds": list(SEEDS),
                               "probes": table}, indent=1) + "\n", encoding="utf-8")
    print("Probe thresholds (D-050 amendment 2; D-126). Units: slopes as stored; changes in log2 (x 1200 = cents).")
    print("threshold = max(seed spread over the five Conformer seeds, shuffle null for ideal-0 slopes); flagged if |seed mean - ideal| > threshold\n")
    print("\n".join(lines))
    for name in ("v3.1", "v3"):
        print(f"\nFlagged, {name} seed mean ({len(flags[name])}):")
        print("\n".join(f"  {f}" for f in flags[name]) or "  none")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
