"""D-094 (with amendment 1): does v3.1 break the f0-change invariance on real spectral envelopes (WORLD pairs)?

Renders `vocal_analysis.eval.probes.world` (data/probes/world_v1), scores teacher v3.1 (d040_5b_s2_b20k) with the
probe runner's per-item summaries, and reads the slope of log2 F1 (F2-F4, VTL, bandwidths reported) on log2 of the
f0 factor: group bootstrap and shuffle null (`probes.measures.summarise_slopes`); ideal 0.
Gate first (clean train_v2 validation clips, f0-band matched): |F1 slope| <= 0.03 overall and every band with >= 10
clips within +-0.05. Rule on the real items (Zeroth's 10 groups and LibriTTS-R dev-clean's 40; F1, all or women):
T2 has something to teach if slope >= +0.03 with its interval above 0; nothing if |slope| <= 0.015 with its interval
inside +-0.03 (all items and women both); otherwise unresolved.

    python scripts/analysis/d094_world_pairs.py > reports/analysis/d094/d094.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = "d040_5b_s2_b20k"  # teacher v3.1
O = Path("reports/analysis/d094")
OUTS = ("F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "VTL", "f0")

if __name__ == "__main__":
    from vocal_analysis.eval.probes import measures as M
    from vocal_analysis.eval.probes import run as R
    from vocal_analysis.eval.probes import world as W

    R.PROBES.setdefault("world", [{"arm": "ref"}] + [{"arm": "world", "r": r} for r in W.FACTORS if r != 1.0])
    if not (W.OUT / "meta.json").exists():
        W.render(log=lambda s: print(s, file=sys.stderr, flush=True))
    O.mkdir(parents=True, exist_ok=True)
    raw = O / f"raw_{MODEL}.npz"
    if not raw.exists():
        res = R.score_probe("world", f"runs/{MODEL}/ckpt_last.pt", log=lambda s: print(s, file=sys.stderr, flush=True))
        np.savez(raw, **{f"Y_{k}": v for k, v in res["Y"].items()})
    Y = {k[2:]: v for k, v in np.load(raw).items()}
    meta = json.loads((W.OUT / "meta.json").read_text(encoding="utf-8"))
    items = meta["items"]
    x = np.log2(meta["factors"])
    grp = np.array([it["group"] for it in items])
    kind = np.array([it["set"] for it in items])
    sex = np.array([it["sex"] for it in items])
    bnd = np.array([W.band(it["f0_median"]) if np.isfinite(it["f0_median"]) else -1 for it in items])

    def summ(o, keep):
        s = M.summarise_slopes(Y[o][keep], x, grp[keep])
        return s

    def fmt(s):
        return (f"{s.point:+.3f} [{s.low:+.3f}, {s.high:+.3f}]  null {s.null_median:+.3f} [{s.null_low:+.3f}, "
                f"{s.null_high:+.3f}]  n {s.n_items}/{s.n_groups}")

    L = [f"D-094: WORLD f0-change pairs, teacher v3.1 ({MODEL}); slope of log2 output on log2 f0 factor "
         f"({', '.join(map(str, W.FACTORS))}); point [95 % group bootstrap], shuffle null; ideal 0 (f0: 1)", ""]
    gate = kind == "gate"
    g1 = summ("F1", gate)
    L.append(f"gate (clean train_v2 validation, {gate.sum()} clips): F1 {fmt(g1)}")
    band_ok = True
    for b, (lo, hi) in enumerate(W.BANDS):
        k = gate & (bnd == b)
        if k.sum():
            s = summ("F1", k)
            ok = k.sum() < 10 or abs(s.point) <= 0.05
            band_ok &= ok
            L.append(f"  f0 {lo:.0f}-{hi if hi < 1e8 else 'up':} Hz: {fmt(s)}{'' if ok else '  OUTSIDE +-0.05'}")
    for o in OUTS[1:]:
        L.append(f"  {o}: {fmt(summ(o, gate))}")
    gate_ok = abs(g1.point) <= 0.03 and band_ok
    L.append(f"gate {'passes' if gate_ok else 'FAILS: the instrument is biased; the real-voice arm is not read'}")
    L.append("")

    real = ~gate
    a1, w1 = summ("F1", real), summ("F1", real & (sex == "f"))
    L.append(f"real items (Zeroth 10 groups + LibriTTS-R 40): F1 {fmt(a1)}")
    L.append(f"  women: F1 {fmt(w1)}")
    L.append(f"  men:   F1 {fmt(summ('F1', real & (sex == 'm')))}")
    L.append(f"  Zeroth only (the counterfactual probe's items): F1 {fmt(summ('F1', kind == 'zeroth'))}")
    L.append(f"  LibriTTS-R only: F1 {fmt(summ('F1', kind == 'libritts'))}")
    for b, (lo, hi) in enumerate(W.BANDS):
        k = real & (bnd == b)
        if k.sum() >= 5:
            L.append(f"  f0 {lo:.0f}-{hi if hi < 1e8 else 'up'} Hz: F1 {fmt(summ('F1', k))}")
    for o in OUTS[1:]:
        L.append(f"  {o}: {fmt(summ(o, real))}")
    L.append("")

    def teach(s):
        return s.point >= 0.03 and s.low > 0

    def nothing(s):
        return abs(s.point) <= 0.015 and s.low >= -0.03 and s.high <= 0.03

    if not gate_ok:
        verdict = "NOT READ (gate failed)"
    elif teach(a1) or teach(w1):
        verdict = "T2 HAS SOMETHING TO TEACH"
    elif nothing(a1) and nothing(w1):
        verdict = "NOTHING TO TEACH"
    else:
        verdict = "UNRESOLVED"
    L.append(f"DECISION: {verdict} (F1 all {a1.point:+.3f} [{a1.low:+.3f}, {a1.high:+.3f}]; women {w1.point:+.3f} "
             f"[{w1.low:+.3f}, {w1.high:+.3f}]; gate {g1.point:+.3f})")
    print("\n".join(L))
