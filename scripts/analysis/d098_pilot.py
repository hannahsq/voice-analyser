"""D-098's width-control pilot (registered in D-098 before any pilot score): D-080's A0 (v3.1's width, the coupled share
in a 4000-step decay branch from 5b) on D-098's measures, beside 5b and D-098's base arm, and the registered rule:
the width control is ruled out only if A0 reaches base on every measure base was called better on against v3.1.

    VA_D085_TAG=d098 python scripts/analysis/d098_pilot.py <windowed scratch dir> > reports/analysis/d098/pilot_a0.txt

Needs cache/d098/{long,v2kn}_d080_a0_s{1,2}.* (d085_score.py) and the scratch dir's ptdb_egg_sub.npz.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("VA_D085_TAG", "d098")

REF = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
A0 = ("d080_a0_s1", "d080_a0_s2")
BASE = "d098_base_b20k"


if __name__ == "__main__":
    import numpy as np

    import decide_d085 as K

    S = Path(sys.argv[1])
    runs = (*REF, *A0, BASE)
    ev = {r: K.log_eval(r) for r in runs}
    lg = {r: K.jload(K.SC / f"long_{r}.json") for r in runs}
    V = {r: np.load(K.SC / f"v2kn_{r}.npz") for r in runs}
    rows = V[REF[0]]["rows"]
    for r in runs:
        if not np.array_equal(V[r]["rows"], rows):
            raise SystemExit(f"v2kn rows differ for {r}")
    G = K.Glottal(rows)
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([K.D.tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    women = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0])

    def cp_bias(r):
        m = women & np.isfinite(d[r][:, 0])
        return float(np.median(d[r][m, 0] - d["cp"][m, 0]))

    def cp_abs(r, ref):
        m = women & np.isfinite(d[r][:, 0]) & np.isfinite(d[ref][:, 0])
        return float(np.median(np.abs(d[r][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    syn = lambda e: np.array([e[f"F{n}_mae_hz"] for n in range(1, 5)])  # noqa: E731
    meas = {  # name -> per-run value getter
        "synthetic F1-F4 MAE, 400 frames": lambda r: syn(ev[r]),
        "synthetic F1-F4 MAE, 1600 frames": lambda r: syn(lg[r]["long1600"]),
        "synthetic F1-F4 MAE, long clips at 400": lambda r: syn(lg[r]["long400"]),
        "aVTL MAE %, 400 frames": lambda r: ev[r]["vtl_mae_pct"],
        "aVTL MAE %, 1600 frames": lambda r: lg[r]["long1600"]["vtl_mae_pct"],
        "aVTL MAE %, long clips at 400": lambda r: lg[r]["long400"]["vtl_mae_pct"],
        "resynth F1-F3 MAE": lambda r: np.asarray(K.D.resynth(r), float),
    }
    groups = {"5b (seed mean)": REF, "A0 (seed mean)": A0, "D-098 base": (BASE,)}
    gm = lambda f, rs: np.mean([f(r) for r in rs], 0)  # noqa: E731

    L = ["D-098 width-control pilot: D-080's A0 (v3.1 width, coupled share at strength 1, 4k-step branch from 5b; no long "
         "crops, no iVTL head) against 5b and D-098's base arm", ""]
    L.append("per group")
    for name, rs in groups.items():
        L.append(f"  {name}")
        L.append(f"    glottal slope {np.mean([G.slope(V[r]['f1']) for r in rs]):+.3f}  (per run: "
                 + ", ".join(f"{r} {G.slope(V[r]['f1']):+.3f}" for r in rs) + ")")
        for m, f in meas.items():
            v = gm(f, rs)
            L.append(f"    {m}: {np.round(v, 2) if np.ndim(v) else f'{v:.2f}'}")
        L.append(f"    women's cp F1 bias {np.mean([cp_bias(r) for r in rs]):+.1f} Hz")
    L.append("")

    L.append("A0 - D-098 base (A0's seed mean against base; negative = A0 better on errors, positive = A0 better on the slope)")
    sd, (lo, hi) = G.diff(V[BASE]["f1"], [V[r]["f1"] for r in A0])  # base - A0
    sd, lo, hi = -sd, -hi, -lo
    reach = {}
    reach["glottal slope"] = sd > -K.T["slope"]
    L.append(f"  glottal slope {sd:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'reaches' if reach['glottal slope'] else 'short'}")
    for m, f in meas.items():
        rc = K.rel(gm(f, A0), f(BASE))
        if np.ndim(rc):
            called = K.mae_call(rc)
            ok = not called or rc.mean() < 0
            L.append(f"  {m}: {', '.join(f'{100 * v:+.1f}' for v in rc)} % (mean {100 * rc.mean():+.1f} %) -> "
                     f"{'reaches' if ok else 'short'}")
        else:
            ok = float(rc) < K.T["avtl"]
            L.append(f"  {m}: {100 * float(rc):+.1f} % -> {'reaches' if ok else 'short'}")
        reach[m] = ok
    cpd = float(np.mean([cp_abs(r, BASE) for r in A0]))
    L.append(f"  women's cp F1 |error| {cpd:+.1f} Hz (reported)")
    L.append("")

    ruled = ["glottal slope", "aVTL MAE %, 1600 frames", "aVTL MAE %, long clips at 400", "synthetic F1-F4 MAE, long clips at 400"]
    short = [m for m in ruled if not reach[m]]
    L.append("rule (base's measures called better against v3.1): " + ", ".join(f"{m} {'reaches' if reach[m] else 'short'}" for m in ruled))
    L.append("PILOT: " + ("width control ruled out: A0 reaches base on every one" if not short
                          else "width control not ruled out (A0 short on " + "; ".join(short) + "): register it"))
    print("\n".join(L))
