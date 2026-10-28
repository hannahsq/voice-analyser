"""D-041 (pre-registered): f0-probe and single-formant-shift invariance of whole teachers' own formant heads, from
d037_cache.py's probe and shift parts. Per formant: |slope| against log2 f0 (f0 <= 250 Hz, cents per octave, per tract);
shift follow and drag (D-038). rA against the per-tract mean of teacher v3's three seeds, paired over tracts (2,000), and
whether rA lies beyond every v3 seed; rB and rC reported.
Usage: python d041_invariance.py <report .txt> name=<cache dir> ... (v3 seeds named v3_*, the candidate rA)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d038_score import B, F0_RISE, DRAG_RISE, FOLLOW_DROP, shift_slopes  # noqa: E402

from vocal_analysis.eval import invariance as inv  # noqa: E402


def f0_slopes(cache):
    """-> (tract keys, (k, 4) |slope| in cents per octave)"""
    log2F = np.load(cache / "probe" / "log2F.npy")
    meta = np.load(cache / "probe" / "meta.npz")
    est = np.nanmedian(log2F[:, 30:71], 1)
    keys, out = [], []
    for v, L in sorted({(a, b) for a, b in zip(meta["vowel"], meta["length_cm"])}):
        i = np.flatnonzero((meta["vowel"] == v) & (meta["length_cm"] == L) & (meta["f0"] <= inv.LOW_F0_MAX_HZ))
        x = np.log2(meta["f0"][i])
        keys.append(f"{v}|{L}")
        out.append([abs(np.polyfit(x, 1200 * est[i, n], 1)[0]) for n in range(4)])
    return np.array(keys), np.array(out)


def measures(cache):
    keys, f0 = f0_slopes(cache)
    sm = dict(np.load(cache / "shift" / "meta.npz"))
    sh = shift_slopes(np.load(cache / "shift" / "log2F.npy"), sm)
    return {"f0": (keys, f0), "follow": {n: (sh[n][0], sh[n][1]) for n in range(4)}, "drag": {n: (sh[n][0], sh[n][2]) for n in range(4)}}


def boot(a, b, keys, rng):
    us = np.unique(keys)
    idx = [np.flatnonzero(keys == u) for u in us]
    bs = [np.mean(np.concatenate([a[idx[k]] - b[idx[k]] for k in rng.integers(len(us), size=len(us))])) for _ in range(B)]
    return float(np.mean(a - b)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    out = Path(sys.argv[1])
    M = {a.split("=")[0]: measures(Path(a.split("=")[1])) for a in sys.argv[2:]}
    v3 = [k for k in M if k.startswith("v3_")]
    lines = [f"D-041: models {list(M)}; v3 seeds {v3}"]
    say = lines.append
    for k, m in M.items():
        say(f"\n{k}: f0 |slope| F1-F4 (c/oct, median over tracts) " + " / ".join(f"{np.median(m['f0'][1][:, n]):.1f}" for n in range(4))
            + "; follow " + " / ".join(f"{np.median(m['follow'][n][1]):.3f}" for n in range(4))
            + "; drag " + " / ".join(f"{np.median(m['drag'][n][1]):.3f}" for n in range(4)))
    rng = np.random.default_rng(0)
    res, flags = {}, []
    for cand in [k for k in M if not k.startswith("v3_")]:
        say(f"\n{cand} - mean(v3), paired over tracts; [beyond every v3 seed]")
        r = {}
        for n in range(4):
            for meas, sign, bound in (("f0", +1, F0_RISE), ("follow", -1, FOLLOW_DROP[n]), ("drag", +1, DRAG_RISE)):
                get = (lambda m: m["f0"][1][:, n]) if meas == "f0" else (lambda m, meas=meas: m[meas][n][1])
                keys = M[cand]["f0"][0] if meas == "f0" else M[cand][meas][n][0]
                ref = np.mean([get(M[s]) for s in v3], 0)
                d, lo, hi = boot(get(M[cand]), ref, keys, rng)
                per_seed = [float(np.mean(get(M[cand]) - get(M[s]))) for s in v3]  # candidate minus each seed
                beyond = all(sign * x > 0 for x in per_seed)
                worse = sign * d > bound and (lo > 0 if sign > 0 else hi < 0) and beyond
                r[f"F{n + 1}_{meas}"] = {"diff": d, "ci": [lo, hi], "minus_each_seed": per_seed, "beyond_all": beyond, "worse": worse}
                say(f"  F{n + 1} {meas:6s} {d:+8.3f} [{lo:+.3f}, {hi:+.3f}]  beyond all: {beyond}  bound {sign * bound:+.3f}" + ("  WORSE" if worse else ""))
        res[cand] = r
        if cand == "rA":
            flags = [k for k, v in r.items() if v["worse"]]
    say(f"\nDECISION: rA is {'LESS INVARIANT (' + ', '.join(flags) + ')' if flags else 'not less invariant'} than teacher v3's AdamW seeds")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
