"""Subset version (every 3rd PTDB-TUG file per speaker; runs/manifests_sub), on dev speakers only (D-024 tiers;
$VA_TIERS=dev,semi to include semi-private ones, a use to log in decisions.md). M2 criteria (D-014): paired speaker-bootstrap, chosen v2 branch vs v1, on Hillenbrand and PTDB-TUG."""
import sys, numpy as np
from pathlib import Path
from vocal_analysis.config import load_config
from vocal_analysis.eval.report import run_formant_baseline, run_f0_baseline, run_frame_labels
from vocal_analysis.eval.teacher_eval import teacher_baselines
from vocal_analysis.eval.metrics import cents_error
from vocal_analysis.eval import tiers as T
import os

def main():
    cache = Path(sys.argv[1]); models = dict(a.split("=", 1) for a in sys.argv[2:])
    cfg = load_config(None, ["data.manifests=runs/manifests_sub"]); log = lambda m: print(m, flush=True)
    fm, f0 = {}, {}
    for name, ck in models.items():
        fb, f0b = teacher_baselines(ck, name)
        fm[name] = run_formant_baseline(fb, cfg, cache, False, log)
        f0[name] = run_f0_baseline(f0b, "ptdb_tug", cfg, cache, False, log)
    labels = run_frame_labels("ptdb_tug", cfg, Path("runs/baselines_sub"), False, log)
    a, b = list(models)
    rng = np.random.default_rng(0)
    # Hillenbrand: per-speaker sums, then resample speakers within the subset
    allowed = set(os.environ.get("VA_TIERS", "dev").split(",")); tmap = T.load()
    in_tier = lambda sp: np.array([T.tier_of(str(x), tmap) in allowed for x in sp])  # noqa: E731
    print(f"tiers: {','.join(sorted(allowed))}")
    ref = fm[a]["ref"]; grp = fm[a]["group"]; spk = fm[a]["speaker"]
    assert np.array_equal(ref, fm[b]["ref"], equal_nan=True)
    ref = np.where(in_tier(spk)[:, None], ref, np.nan)  # speakers outside the tiers leave every table
    def table(mask, k, what):
        m = mask & np.isfinite(ref[:, k])
        sp = np.unique(spk[m]); idx = {s: np.flatnonzero(m & (spk == s)) for s in sp}
        def stat(est, sel):
            e = np.abs(est[sel, k] - ref[sel, k]); r = ref[sel, k]
            return np.mean(e) if what == "mae" else 100 * np.mean((e <= 0.2 * r) & (e <= 200))
        base = {n: stat(fm[n]["est"], np.flatnonzero(m)) for n in (a, b)}
        d = []
        for _ in range(1000):
            sel = np.concatenate([idx[s] for s in rng.choice(sp, len(sp))])
            d.append(stat(fm[b]["est"], sel) - stat(fm[a]["est"], sel))
        lo, hi = np.percentile(d, [2.5, 97.5])
        flag = "better" if (hi < 0 if what == "mae" else lo > 0) else "WORSE" if (lo > 0 if what == "mae" else hi < 0) else "within noise"
        return f"{base[a]:7.1f} {base[b]:7.1f}  diff {base[b] - base[a]:+6.1f} [{lo:+.1f}, {hi:+.1f}]  {flag}"
    print(f"\nHillenbrand, {a} vs {b} (paired speaker bootstrap, 95 %)")
    for gname, mask in [("women", grp == "women"), ("children", np.isin(grp, ["boys", "girls"])), ("men", grp == "men"), ("all", np.ones(len(grp), bool))]:
        print(f"  F1 MAE {gname:8s}: {table(mask, 0, 'mae')}")
    for k in (1, 2, 3):
        print(f"  F{k + 1} MAE all     : {table(np.ones(len(grp), bool), k, 'mae')}")
        print(f"  F{k + 1} DR  all     : {table(np.ones(len(grp), bool), k, 'dr')}")
    # PTDB-TUG f0 RPA
    r = f0[a]["ref"]; s2 = f0[a]["speaker"]; rv = np.isfinite(r) & (r > 0) & in_tier(s2)
    def rpa_hits(n):
        e = f0[n]["est"]; ok = np.isfinite(e) & (e > 0) & rv
        h = np.zeros(len(r), bool); h[ok] = np.abs(cents_error(e[ok], r[ok])) <= 50
        return h
    ha, hb = rpa_hits(a), rpa_hits(b)
    print(f"\nPTDB-TUG f0 RPA, {a} vs {b}")
    for nm, mask in [("all frames", rv), ("modal", rv & (labels == "modal")), ("modal < 85 Hz", rv & (labels == "modal") & (r < 85))]:
        sp = np.unique(s2[mask]); idx = {s: np.flatnonzero(mask & (s2 == s)) for s in sp}
        d = []
        for _ in range(1000):
            sel = np.concatenate([idx[s] for s in rng.choice(sp, len(sp))])
            d.append(100 * (hb[sel].mean() - ha[sel].mean()))
        lo, hi = np.percentile(d, [2.5, 97.5])
        A, B = 100 * ha[mask].mean(), 100 * hb[mask].mean()
        print(f"  {nm:14s}: {A:5.1f} {B:5.1f}  diff {B - A:+5.1f} [{lo:+.1f}, {hi:+.1f}]  {'WORSE' if hi < 0 else 'better' if lo > 0 else 'within noise'}  (n {mask.sum()})")

if __name__ == "__main__":
    main()
