"""Closed-phase F1 for any saved models. Args: scratch dir, comma-separated models to show, then pairs b-a to
bootstrap (PTDB-TUG speakers). ARCTIC by voice and vowel group, PTDB-TUG by sex."""
import sys
import os
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
S, E, PAIRS = sys.argv[1], tuple(sys.argv[2].split(",")), [p.split("-") for p in sys.argv[3:]]
rng = np.random.default_rng(0)
RANGES = ((0, 450), (450, 650), (650, 5000))
GROUPS = {"close (iy ih uw uh)": ("iy", "ih", "uw", "uh"), "mid (eh ah er ao)": ("eh", "ah", "er", "ao"), "open (ae aa)": ("ae", "aa")}
for f, grp in (("arctic_egg", "voice"), ("ptdb_egg", "sex")):
    d = dict(np.load(f"{S}/{f}{os.environ.get('CP_SUFFIX', '')}.npz"))
    ok = np.isfinite(d["cp"][:, 0]) & np.all([np.isfinite(d[e][:, 0]) for e in E], 0)
    print(f"\n{f}: {ok.sum()} frames. F1 vs closed phase: median signed (median |err|); by cp F1 <450 / 450-650 / >650")
    for g in np.unique(d[grp]):
        m = ok & (d[grp] == g)
        r = d["cp"][m, 0]
        print(f"  {g} ({m.sum()} frames, cp F1 median {np.median(r):.0f})")
        for e in E:
            x = d[e][m, 0] - r
            rg = " / ".join(f"{np.median(x[(r >= lo) & (r < hi)]):+4.0f}" for lo, hi in RANGES)
            print(f"    {e:6s} {np.median(x):+5.1f} ({np.median(np.abs(x)):4.1f})   {rg}")
        if grp == "voice":
            for gname, ph in GROUPS.items():
                mm = m & np.isin(d["phone"], ph)
                print(f"      {gname:20s} n {mm.sum():5d}: " + "  ".join(f"{e} {np.median(d[e][mm, 0] - d['cp'][mm, 0]):+4.0f}" for e in E))
        else:
            spk = np.unique(d["speaker"][m])
            for stat, fn in (("bias", lambda e, mm: np.median(d[e][mm, 0] - d["cp"][mm, 0])),
                             ("|err|", lambda e, mm: np.median(np.abs(d[e][mm, 0] - d["cp"][mm, 0]))),
                             ("low-F1 bias", lambda e, mm: np.median((d[e][mm, 0] - d["cp"][mm, 0])[d["cp"][mm, 0] < 450])),
                             ("high-F1 bias", lambda e, mm: np.median((d[e][mm, 0] - d["cp"][mm, 0])[d["cp"][mm, 0] >= 650]))):
                for b_, a_ in PAIRS:
                    per = np.array([fn(b_, m & (d["speaker"] == s)) - fn(a_, m & (d["speaker"] == s)) for s in spk])
                    bs = [np.median(rng.choice(per, len(per))) for _ in range(2000)]
                    print(f"    {b_:5s} - {a_:5s} {stat:12s} {np.median(per):+5.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")
