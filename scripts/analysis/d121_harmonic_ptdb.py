"""D-121: around the second harmonic on PTDB-TUG dev, is v3.1's F1 pulled towards 2 f0 against the closed-phase reference,
and do the reconstruction arms (D-109, D-114) correct that or add a push of their own? CPU only, the closed-phase cache.

Frames: D-091's (closed-phase, whole-cycle and Praat F1 finite, f0 > 0), dev speakers, every listed model finite. Cells by
closed-phase F1 / f0 (D-119's bands) and sex. Cents; medians over frames; seed means; speaker bootstrap (2000).
E = 5b's F1 - closed phase; S = arm - 5b; dA = |arm - closed phase| - |5b - closed phase|; K(X) = X in the 1.5-2 cell
minus X in the 2-2.5 cell (women).

    python scripts/analysis/d121_harmonic_ptdb.py > reports/analysis/d121/d121.txt

Addendum (after the result, not registered): D121_BIN=praat or D121_BIN=model puts the frames into cells by Praat's F1 / f0
or by 5b's own F1 / f0 instead of the closed-phase F1 / f0. Cells chosen on the reference inflate any gradient of the
error across them (regression to the mean), and cells chosen on the model deflate it; Praat's is a third measurement.
    D121_BIN=praat python scripts/analysis/d121_harmonic_ptdb.py > reports/analysis/d121/d121_bin_praat.txt
"""
import json
import os
from pathlib import Path

import numpy as np

EGG = Path("cache/m3w/ptdb_egg_sub.npz")
B5 = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
ARMS = {"d109": ("d109_s1", "d109_s2"), "d114": ("d114_s1", "d114_s2"), "f5 (null arm)": ("d090_f5_s1", "d090_f5_s2")}
H_CUTS, H_NAMES = (1.5, 2.0, 2.5, 3.5), ("< 1.5", "1.5-2", "2-2.5", "2.5-3.5", ">= 3.5")
LO, HI = 1, 2  # the cells either side of 2 f0
MIN_FRAMES, MIN_SPK, N_BOOT, SEED = 200, 4, 2000, 5121
BIN = os.environ.get("D121_BIN", "cp")


def cents(a, b):
    with np.errstate(all="ignore"):
        return 1200 * np.log2(a / b)


if __name__ == "__main__":
    d = dict(np.load(EGG, allow_pickle=False))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    ok = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    ok &= np.isfinite(d["cp"][:, 0]) & np.isfinite(d["cycle"][:, 0]) & np.isfinite(d["praat"][:, 0]) & (d["f0"] > 0)
    for m in [*B5, *[x for v in ARMS.values() for x in v]]:
        ok &= np.isfinite(d[m][:, 0])
    cp, f0, spk, sex = d["cp"][ok, 0], d["f0"][ok], d["speaker"][ok], d["sex"][ok]
    F = {m: d[m][ok, 0] for m in [*B5, *[x for v in ARMS.values() for x in v]]}
    binvar = {"cp": cp, "praat": d["praat"][ok, 0], "model": np.mean([F[m] for m in B5], 0)}[BIN]
    cell = np.digitize(binvar / f0, H_CUTS)

    # per-frame quantities (cents), seed means of per-seed values
    Q = {"E: 5b - closed phase": np.mean([cents(F[m], cp) for m in B5], 0),
         "seed null: 5b s1 - s2": cents(F[B5[0]], F[B5[1]]),
         "ref: whole-cycle LPC - closed phase": cents(d["cycle"][ok, 0], cp),
         "ref: Praat - closed phase": cents(d["praat"][ok, 0], cp)}
    for a, ms in ARMS.items():
        Q[f"S: {a} - 5b"] = np.mean([cents(F[m], F[b]) for m, b in zip(ms, B5)], 0)
        Q[f"dA: {a}, change in |error|"] = np.mean([np.abs(cents(F[m], cp)) - np.abs(cents(F[b], cp)) for m, b in zip(ms, B5)], 0)
        for i in (0, 1):
            Q[f"  S: {a} - 5b, seed {i + 1}"] = cents(F[ms[i]], F[B5[i]])
            Q[f"  dA: {a}, seed {i + 1}"] = np.abs(cents(F[ms[i]], cp)) - np.abs(cents(F[B5[i]], cp))

    rng = np.random.default_rng(SEED)
    RULE = "READING (registered, women)" if BIN == "cp" else f"the registered rule on cells by {BIN} F1 / f0 (addendum, not the registered reading; women)"
    L = [("" if BIN == "cp" else f"ADDENDUM, cells by {BIN} F1 / f0 (after the result; not registered). ")
         + "D-121: v3.1's F1 against the closed-phase F1 around the harmonics, and the reconstruction arms' shifts there (PTDB-TUG dev)",
         f"cells: {dict(cp='closed-phase', praat='Praat', model='5b (seed mean)')[BIN]} F1 / f0; errors always against the closed-phase F1; cents; medians over frames; seed means; [95 % speaker bootstrap, 2000]",
         "K(X) = X in the 1.5-2 cell minus X in the 2-2.5 cell: above 0 for a pull towards 2 f0 (E), below 0 for a push away from it (S)", ""]
    verdict = {}
    for sx, name in (("f", "women"), ("m", "men")):
        g = sex == sx
        us = np.unique(spk[g])
        idx = {(s, c): np.where(g & (spk == s) & (cell == c))[0] for s in us for c in range(len(H_NAMES))}
        n_c = [sum(len(idx[s, c]) for s in us) for c in range(len(H_NAMES))]
        k_c = [sum(len(idx[s, c]) > 0 for s in us) for c in range(len(H_NAMES))]
        cells = [c for c in range(len(H_NAMES)) if n_c[c] >= MIN_FRAMES and k_c[c] >= MIN_SPK]
        draws = [rng.choice(us, len(us)) for _ in range(N_BOOT)]

        def med(x, c, draw):
            ii = [idx[s, c] for s in draw]
            ii = np.concatenate(ii) if ii else np.zeros(0, int)
            return np.median(x[ii]) if len(ii) else np.nan

        L.append(f"== {name}: {len(us)} speakers, {int(g.sum())} frames; median f0 {np.median(f0[g]):.0f} Hz, closed-phase F1 {np.median(cp[g]):.0f} Hz")
        L.append(f"  {'':42s} " + " | ".join(f"{H_NAMES[c] + f' (n {n_c[c]}, {k_c[c]} spk)':>24s}" for c in cells)
                 + (" |   K (1.5-2 minus 2-2.5)" if LO in cells and HI in cells else ""))
        for label, x in Q.items():
            row, B = [], {c: np.array([med(x, c, dr) for dr in draws]) for c in cells}
            for c in cells:
                lo, hi = np.nanpercentile(B[c], [2.5, 97.5])
                row.append(f"{med(x, c, us):+6.0f} [{lo:+5.0f}, {hi:+5.0f}]".rjust(24))
            ktxt = ""
            if LO in cells and HI in cells:
                kpt, kb = med(x, LO, us) - med(x, HI, us), B[LO] - B[HI]
                klo, khi = np.nanpercentile(kb, [2.5, 97.5])
                ktxt = f" | {kpt:+6.0f} [{klo:+5.0f}, {khi:+5.0f}]"
                tot = med(x, LO, us) + med(x, HI, us)
                tlo, thi = np.nanpercentile(B[LO] + B[HI], [2.5, 97.5])
                verdict[sx, label] = {"K": (kpt, klo, khi), "sum": (tot, tlo, thi)}
            L.append(f"  {label:42s} " + " | ".join(row) + ktxt)
        L.append("")
        if sx == "f":
            if LO not in cells or HI not in cells:
                L.append(f"{RULE}: UNREADABLE (a cell either side of 2 f0 lacks 200 frames or 4 speakers)")
            else:
                for a in ("d109", "d114"):
                    KE, KS = verdict[sx, "E: 5b - closed phase"]["K"], verdict[sx, f"S: {a} - 5b"]["K"]
                    KN = verdict[sx, "S: f5 (null arm) - 5b"]["K"]
                    dA = verdict[sx, f"dA: {a}, change in |error|"]["sum"]
                    if not (KN[1] <= 0 <= KN[2]) and abs(KN[0]) > 0.5 * abs(KS[0]):
                        v = "UNREADABLE (the null arm's K excludes 0 at more than half this arm's size)"
                    elif not KS[2] < 0:
                        v = "NOT LEARNED"
                    elif KE[1] > 0 and dA[2] < 0:
                        v = "CORRECTION"
                    elif not KE[1] > 0 and dA[1] > 0:
                        v = "ARTEFACT"
                    else:
                        v = "MIXED"
                    L.append(f"{RULE}, {a}{' (decides)' if a == 'd109' else ' (reported)'}: K(E) {KE[0]:+.0f} [{KE[1]:+.0f}, {KE[2]:+.0f}]; "
                             f"K(S) {KS[0]:+.0f} [{KS[1]:+.0f}, {KS[2]:+.0f}]; null arm's K(S) {KN[0]:+.0f} [{KN[1]:+.0f}, {KN[2]:+.0f}]; "
                             f"dA summed over the two cells {dA[0]:+.0f} [{dA[1]:+.0f}, {dA[2]:+.0f}] -> {v}")
            L.append("")
    print("\n".join(L))
