"""D-151 post hoc (not registered): where flag B's flagged-run frames sit (near a voicing edge or not) and what
seed 2 reads there, against the other voiced frames. Reads cache/d151/; prints only.

    python scripts/analysis/d151_posthoc.py > reports/analysis/d151/d151_posthoc.txt
"""
import sys, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
spec = importlib.util.spec_from_file_location("d151", ROOT / "scripts/analysis/d151_ptdb_flag.py"); d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
if __name__ == "__main__":
    prior = d._d142().Prior(); m = d.MODELS[0]; thr = d.thresholds()[m]
    out = {s: {"edge_f": [], "edge_u": [], "F_f": [], "F_u": [], "L_f": [], "L_u": [], "Lrec": []} for s in "FM"}
    for p in sorted(d.CACHE.glob("*.npz")):
        z = np.load(p); s = str(z["sex"]); v = z[f"{m}|v"] > 0.5; F, L, utt = z[f"{m}|F"], z[f"{m}|L"], z["utt"]
        um = np.array([np.nanmedian(L[(utt == u) & v]) if ((utt == u) & v).any() else np.nan for u in np.unique(utt)])
        Lrec = np.array([np.nanmedian(np.delete(um, u)) for u in range(len(um))])
        ok = v & np.all(np.isfinite(F), 1); B = np.full(len(v), np.nan); B[ok] = prior.score(F[ok], Lrec[utt[ok]])
        fl = np.zeros(len(v), bool); edge = np.zeros(len(v), bool)
        for u in np.unique(utt):
            k = np.where(utt == u)[0]; fl[k] = d._runs(ok[k] & (B[k] > thr))
            vv = v[k].astype(int); dv = np.diff(np.r_[0, vv, 0]); b = np.r_[np.where(dv == 1)[0], np.where(dv == -1)[0]]
            dist = np.min(np.abs(np.arange(len(k))[:, None] - b[None, :]), 1) if len(b) else np.full(len(k), 99)
            edge[k] = dist < 5
        o = out[s]; o["Lrec"].append(np.nanmedian(Lrec))
        for t, sel in (("f", ok & fl), ("u", ok & ~fl)):
            o[f"edge_{t}"].append(edge[sel]); o[f"F_{t}"].append(F[sel]); o[f"L_{t}"].append(L[sel])
    for s, o in out.items():
        c = {k: np.concatenate(v) for k, v in o.items() if k != "Lrec"}
        print("women" if s == "F" else "men")
        print(f"  within 50 ms of a voicing edge: flagged {c['edge_f'].mean():.0%}, unflagged {c['edge_u'].mean():.0%}")
        for t in "fu":
            F = c[f"F_{t}"]; print(f"  {t}: median F1-F4 {np.round(np.median(F,0))}, F2/F1 {np.median(F[:,1]/F[:,0]):.2f}, frame L {np.median(c[f'L_{t}']):.1f}")
        Ff = c["F_f"]; print("  flagged F1 quartiles", np.round(np.percentile(Ff[:,0],[25,50,75])), "F2", np.round(np.percentile(Ff[:,1],[25,50,75])))
