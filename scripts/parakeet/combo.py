"""D-033 part 3: ridge probes on every combination of Parakeet layers 0-6, from the embedding cache (cache.py).
Joint statistics for all 7 layers (standardised, 7168 features + bias) per target, on the 3000 fitting crops (every 2nd
frame; first 2400 fit, last 600 choose the ridge strength); each combination is a sub-block, solved on the GPU, refitted
on all. Scored on every cached set; F1 and voicing winners chosen by the pre-registered rule (D-033 part 3): mean over the
training-pool validation sets of each set's MAE (accuracy for voicing), best, then the shallowest within 1 SE (paired
bootstrap over clips), then fewest layers. Then the part-2 bias analysis and rule for the F1 winner.
Args: cache dir, out dir."""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
C, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cudnn.enabled = False
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
NL, D = 7, 1024
F = NL * D
COMBOS = [c for r in range(1, NL + 1) for c in itertools.combinations(range(NL), r)]
NAMES = ["+".join(f"L{l}" for l in c) for c in COMBOS]
TARGETS = ["F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "f0", "voiced", "sonorant"]
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEL_SETS = ["synth_train_v2", "synth_train_v2s", "resynth_vocalset_val_d16k", "resynth_zeroth_korean_val_d16k"]
REP = open(OUT / "combo_report.txt", "w", encoding="utf-8")


def p(*a):
    print(*a, file=REP, flush=True)
    print(*a, flush=True)


def load(name):
    d = C / name
    return {f.stem: np.load(f, mmap_mode="r" if f.stem == "emb" else None) for f in d.glob("*.npy")}


def interp(emb: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """emb (E, F) for one item at encoder frames (frame e at mel 8e); mel (T,) -> (T, F), linear, clamped."""
    E = emb.shape[0]
    q = (mel / 8.0).clamp(0, E - 1)
    if E == 1:
        return emb[torch.zeros_like(q, dtype=torch.long)]
    i0 = q.floor().long().clamp(max=E - 2)
    a = (q - i0)[:, None]
    return emb[i0] * (1 - a) + emb[i0 + 1] * a


def frames(s, idx_items=None, every=1, chunk=40000):
    """yield (X (n, F) float32 unstandardised, frame slice) over a set's reference frames, items in order."""
    eo, ro, mel = s["emb_off"], s["ref_off"], s["mel"]
    items = range(len(eo) - 1) if idx_items is None else idx_items
    buf, start, pos = [], None, 0
    for i in items:
        e = torch.from_numpy(np.asarray(s["emb"][eo[i]:eo[i + 1]], np.float16)).to(dev).float().reshape(-1, F)
        m = torch.from_numpy(np.asarray(mel[ro[i]:ro[i + 1]], np.float32)[::every]).to(dev)
        buf.append(interp(e, m))
        start = ro[i] if start is None else start
        pos += len(m)
        if pos >= chunk:
            yield torch.cat(buf), (start, ro[i + 1])
            buf, start, pos = [], None, 0
    if buf:
        yield torch.cat(buf), (start, ro[items[-1] + 1])


def targets(s, sl, every=1):
    """labels of frames sl (crop sets) -> y (n, 11), w (n, 11), as the trainer's masks."""
    g = lambda k: np.asarray(s[k][sl[0]:sl[1]])[::every]  # noqa: E731
    Fm, B, f0, v, son = g("formants"), g("bandwidths"), g("f0"), g("voiced") > 0.5, g("sonorant") > 0.5
    wf0, wfm = g("w_f0"), g("w_formant")
    syn = wf0 > 0
    y, w = np.zeros((len(f0), 11)), np.zeros((len(f0), 11))
    for k in range(4):
        ok = son & np.isfinite(Fm[:, k]) & (Fm[:, k] > 0)
        y[ok, k], w[ok, k] = np.log2(Fm[ok, k]), wfm[ok]
        okb = son & np.isfinite(B[:, k]) & (B[:, k] > 0) & syn
        y[okb, 4 + k], w[okb, 4 + k] = np.log2(B[okb, k]), 1.0
    vv = v & np.isfinite(f0) & (f0 > 0) & syn
    y[vv, 8], w[vv, 8] = np.log2(f0[vv]), 1.0
    y[syn, 9], w[syn, 9] = v[syn], 1.0
    y[syn, 10], w[syn, 10] = son[syn], 1.0
    return y, w


# ------------------------------------------------------------------ 1. joint statistics on the fitting crops
t0 = time.time()
fit = load("fit")
n_fit = len(fit["emb_off"]) - 1
stats_file = OUT / "joint_stats.pt"
if stats_file.exists():
    st = torch.load(stats_file)
else:
    s1, s2, n = torch.zeros(F, device=dev, dtype=torch.float64), torch.zeros(F, device=dev, dtype=torch.float64), 0
    for X, _ in frames(fit, every=2):
        s1 += X.double().sum(0)
        s2 += (X.double() ** 2).sum(0)
        n += len(X)
    mu = (s1 / n).float()
    sd = ((s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt() + 1e-4).float()
    st = {"mu": mu.cpu(), "sd": sd.cpu()}
    for split, items in (("fit", range(int(0.8 * n_fit))), ("sel", range(int(0.8 * n_fit), n_fit))):
        xx = torch.zeros(11, F + 1, F + 1, device=dev)
        xy = torch.zeros(11, F + 1, device=dev)
        yy, ww = torch.zeros(11, dtype=torch.float64), torch.zeros(11, dtype=torch.float64)
        for X, sl in frames(fit, list(items), every=2, chunk=20000):
            y, w = targets(fit, sl, every=2)
            X = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=dev)], 1)
            Y, W = torch.from_numpy(y).float().to(dev), torch.from_numpy(w).float().to(dev)
            for t in range(11):
                m = W[:, t] > 0
                if m.any():
                    Xm = X[m]
                    xx[t] += (Xm * W[m, t, None]).T @ Xm
                    xy[t] += Xm.T @ (W[m, t] * Y[m, t])
                    yy[t] += float((W[m, t] * Y[m, t] ** 2).double().sum())
                    ww[t] += float(W[m, t].double().sum())
        st[split] = {"xx": xx.cpu(), "xy": xy.cpu(), "yy": yy, "w": ww}
        p(f"statistics: {split} done ({time.time() - t0:.0f} s)")
    torch.save(st, stats_file)
mu, sd = st["mu"].to(dev), st["sd"].to(dev)

# ------------------------------------------------------------------ 2. ridge per combination and target
coef_file = OUT / "combo_coef.pt"
if coef_file.exists():
    co = torch.load(coef_file)
    W_all, LAM = co["W"], co["lambda"]
else:
    W_all, LAM = [], np.zeros((len(COMBOS), 11))
    for t in range(11):
        fx, fy, fn = st["fit"]["xx"][t].to(dev), st["fit"]["xy"][t].to(dev), float(st["fit"]["w"][t])
        sx, sy, syy, sn = st["sel"]["xx"][t].to(dev), st["sel"]["xy"][t].to(dev), float(st["sel"]["yy"][t]), float(st["sel"]["w"][t])
        for ci, c in enumerate(COMBOS):
            idx = torch.cat([torch.arange(l * D, (l + 1) * D) for l in c] + [torch.tensor([F])]).to(dev)
            # Cholesky in fp32 (this GPU's fp64 is too slow for 127 x 11 x 8 solves up to 7169 wide; a failed
            # factorisation skips that strength); the held-out error in fp64, where the terms nearly cancel
            A, b, As, bs = fx[idx][:, idx], fy[idx], sx[idx][:, idx], sy[idx]
            As64, bs64 = As.double(), bs.double()
            eye = torch.eye(len(idx), device=dev)
            eye[-1, -1] = 0
            best = None
            for lam in LAMBDAS:
                Lc, info = torch.linalg.cholesky_ex(A + eye * lam * fn)
                if info.item():
                    continue
                wv = torch.cholesky_solve(b[:, None], Lc)[:, 0].double()
                mse = (syy - 2 * wv @ bs64 + wv @ (As64 @ wv)).item() / max(sn, 1e-9)
                if best is None or mse < best[0]:
                    best = (mse, lam)
            LAM[ci, t] = best[1]
            Lc = torch.linalg.cholesky(A + As + eye * best[1] * (fn + sn))
            wv = torch.cholesky_solve((b + bs)[:, None], Lc)[:, 0]
            if t == 0:
                W_all.append(torch.zeros(len(idx), 11, device=dev))
            W_all[ci][:, t] = wv
        p(f"ridge: target {TARGETS[t]} done ({time.time() - t0:.0f} s)")
    torch.save({"W": [w.cpu() for w in W_all], "lambda": LAM, "combos": COMBOS}, coef_file)
W_all = [w.to(dev) for w in W_all]


def predict_all(X):
    """X (n, F) raw -> (n_combos, n, 11)."""
    Xs = (X - mu) / sd
    out = torch.empty(len(COMBOS), len(X), 11, device=dev)
    for ci, c in enumerate(COMBOS):
        acc = W_all[ci][-1].expand(len(X), 11).clone()
        for j, l in enumerate(c):
            acc += Xs[:, l * D:(l + 1) * D] @ W_all[ci][j * D:(j + 1) * D]
        out[ci] = acc
    out[..., :9] = out[..., :9].clamp(4.0, 13.0)
    return out


# ------------------------------------------------------------------ 3. score: synthetic / resynthesised validation
R = {}
clip = {}  # set -> (per-clip F1 abs-error sums (combos, clips), counts (clips,), voicing correct (combos, clips), voicing n)
for name in SEL_SETS:
    s = load(name)
    n_items = len(s["emb_off"]) - 1
    sums = {k: np.zeros((len(COMBOS), 11)) for k in ("abs", "n", "hit")}
    tsum = {k: np.zeros(11) for k in ("abs", "n", "hit")}
    cF1, cN, vC, vN = np.zeros((len(COMBOS), n_items)), np.zeros(n_items), np.zeros((len(COMBOS), n_items)), np.zeros(n_items)
    for X, sl in frames(s, chunk=20000):
        y, w = targets(s, sl)
        P = predict_all(X).cpu().numpy()
        T = np.column_stack([s["t_log2F"][sl[0]:sl[1]], s["t_log2B"][sl[0]:sl[1]], s["t_log2f0"][sl[0]:sl[1]],
                             s["t_voiced"][sl[0]:sl[1]], s["t_sonorant"][sl[0]:sl[1]]])
        clip_of = (np.arange(sl[0], sl[1]) // 400)
        for t in range(11):
            m = w[:, t] > 0
            if not m.any():
                continue
            if t < 8:
                e = np.abs(2 ** P[:, m, t] - 2 ** y[m, t])
                sums["abs"][:, t] += e.sum(1)
                tsum["abs"][t] += np.abs(2 ** T[m, t] - 2 ** y[m, t]).sum()
                if t == 0:
                    for ci in range(len(COMBOS)):
                        cF1[ci] += np.bincount(clip_of[m], e[ci], n_items)
                    cN += np.bincount(clip_of[m], None, n_items)
            elif t == 8:
                sums["hit"][:, t] += (np.abs(1200 * (P[:, m, t] - y[m, t])) <= 50).sum(1)
                tsum["hit"][t] += (np.abs(1200 * (T[m, t] - y[m, t])) <= 50).sum()
            else:
                ok = (P[:, m, t] > 0.5) == (y[m, t] > 0.5)
                sums["hit"][:, t] += ok.sum(1)
                tsum["hit"][t] += ((T[m, t] > 0.5) == (y[m, t] > 0.5)).sum()
                if t == 9:
                    for ci in range(len(COMBOS)):
                        vC[ci] += np.bincount(clip_of[m], ok[ci], n_items)
                    vN += np.bincount(clip_of[m], None, n_items)
            sums["n"][:, t] += m.sum()
            tsum["n"][t] += m.sum()
    met = lambda a, nn, t: (a[t] / nn[t] if t < 8 else 100 * a[t] / nn[t]) if nn[t] else float("nan")  # noqa: E731
    R[name] = {NAMES[ci]: {TARGETS[t]: met(sums["abs" if t < 8 else "hit"][ci], sums["n"][ci], t) for t in range(11)}
               for ci in range(len(COMBOS))}
    R[name]["teacher_v3"] = {TARGETS[t]: met(tsum["abs" if t < 8 else "hit"], tsum["n"], t) for t in range(11)}
    clip[name] = (cF1, cN, vC, vN)
    p(f"[{name}] scored ({time.time() - t0:.0f} s); teacher F1 {R[name]['teacher_v3']['F1']:.1f}")

# ------------------------------------------------------------------ 4. selection (pre-registered)
rng = np.random.default_rng(0)


def select(kind):
    """kind 'F1' (lower better, all four sets) or 'voiced' (higher better, synthetic sets)."""
    sets = SEL_SETS if kind == "F1" else [s for s in SEL_SETS if not np.isnan(R[s]["L0"]["voiced"])]
    if kind == "F1":
        metric = np.mean([clip[s][0].sum(1) / clip[s][1].sum() for s in sets], 0)
    else:
        metric = -np.mean([clip[s][2].sum(1) / clip[s][3].sum() for s in sets], 0)  # negated: lower is better
    b = int(np.argmin(metric))
    boots = np.zeros((1000, len(COMBOS)))
    for s in sets:
        a, nn = (clip[s][0], clip[s][1]) if kind == "F1" else (clip[s][2], clip[s][3])
        for r in range(1000):
            ix = rng.integers(a.shape[1], size=a.shape[1])
            val = a[:, ix].sum(1) / nn[ix].sum()
            boots[r] += (val if kind == "F1" else -val) / len(sets)
    diff = boots - boots[:, [b]]
    se = diff.std(0)
    ok = [ci for ci in range(len(COMBOS)) if metric[ci] - metric[b] <= se[ci]]
    win = min(ok, key=lambda ci: (max(COMBOS[ci]), len(COMBOS[ci]), metric[ci]))
    sign = 1 if kind == "F1" else -100
    order = np.argsort(metric)
    p(f"\n## Selection: {kind} ({'mean MAE, Hz' if kind == 'F1' else 'mean accuracy, %'} over {', '.join(sets)})")
    p(f"best {NAMES[b]} {sign * metric[b]:.2f}; {len(ok)} within 1 SE; winner (shallowest, then fewest layers): "
      f"{NAMES[win]} {sign * metric[win]:.2f} (diff {sign * (metric[win] - metric[b]):+.2f}, SE {abs(sign) * se[win] if kind == 'F1' else 100 * se[win]:.2f})")
    p("top 15:")
    for ci in order[:15]:
        p(f"  {NAMES[ci]:28s} {sign * metric[ci]:7.2f}  (diff {sign * (metric[ci] - metric[b]):+.2f}, SE {se[ci] * (1 if kind == 'F1' else 100):.2f})")
    p("best by deepest layer used (Pareto):")
    for depth in range(NL):
        cands = [ci for ci in range(len(COMBOS)) if max(COMBOS[ci]) == depth]
        bi = min(cands, key=lambda ci: metric[ci])
        p(f"  deepest L{depth}: {NAMES[bi]:28s} {sign * metric[bi]:7.2f}")
    return win


win_f1, win_v = select("F1"), select("voiced")
json.dump(R, open(OUT / "combo_results.json", "w"), indent=1)


# ------------------------------------------------------------------ 5. gold sets (dev) for all combinations
def gold_formants(name, refkey, groups_fn):
    s = load(name)
    P, T, ref = [], [], np.asarray(s[refkey])
    for X, sl in frames(s, chunk=20000):
        P.append(predict_all(X)[..., :4].cpu().numpy())
    P = np.concatenate(P, 1)
    T = np.asarray(s["t_log2F"])
    return P, T, ref, groups_fn(s)


hb_P, hb_T, hb_ref, hb_g = gold_formants("gold_hillenbrand", "ref_formants",
                                         lambda s: {"all": np.ones(len(s["group"]), bool), "men": s["group"] == "m",
                                                    "women": s["group"] == "w", "children": np.isin(s["group"], ["b", "g"])})
cp_P, cp_T, cp_ref, cp_g = gold_formants("gold_ptdb_cp", "ref_cp", lambda s: {"women": s["sex"] == "f", "men": s["sex"] == "m"})


def mae_signed(pred_log2, ref, m):
    e = 2 ** pred_log2[m] - ref[m]
    return float(np.median(e)), float(np.mean(np.abs(e)))


G = {}
for gname, P, T, ref, groups, nf in (("hillenbrand", hb_P, hb_T, hb_ref, hb_g, 4), ("closed_phase", cp_P, cp_T, cp_ref, cp_g, 3)):
    G[gname] = {}
    for who, pred in [*((NAMES[ci], P[ci]) for ci in range(len(COMBOS))), ("teacher_v3", T)]:
        G[gname][who] = {f"F{k + 1}_{g}": mae_signed(pred[:, k], ref[:, k], gm & np.isfinite(ref[:, k]) & (ref[:, k] > 0))
                         for g, gm in groups.items() for k in range(nf)}
for name in ("gold_ptdb_tug", "gold_mir1k", "gold_mdb_stem_synth"):
    s = load(name)
    hit, vok, nv, nall = np.zeros(len(COMBOS)), np.zeros(len(COMBOS)), 0, 0
    for X, sl in frames(s, chunk=40000):
        P = predict_all(X)[..., 8:10].cpu().numpy()
        ref = np.asarray(s["ref_f0"][sl[0]:sl[1]])
        ok = np.isfinite(ref)
        v = ok & (ref > 0)
        hit += (np.abs(1200 * (P[:, v, 0] - np.log2(ref[v]))) <= 50).sum(1)
        vok += ((P[:, ok, 1] > 0.5) == (ref[ok] > 0)).sum(1)
        nv, nall = nv + v.sum(), nall + ok.sum()
    ref = np.asarray(s["ref_f0"])
    ok = np.isfinite(ref)
    v = ok & (ref > 0)
    tf0, tv = np.asarray(s["t_log2f0"]), np.asarray(s["t_voiced"])
    G[name] = {NAMES[ci]: {"f0_rpa": 100 * hit[ci] / nv, "voiced_acc": 100 * vok[ci] / nall} for ci in range(len(COMBOS))}
    G[name]["teacher_v3"] = {"f0_rpa": 100 * float(np.mean(np.abs(1200 * (tf0[v] - np.log2(ref[v]))) <= 50)),
                             "voiced_acc": 100 * float(np.mean((tv[ok] > 0.5) == (ref[ok] > 0)))}
    p(f"[{name}] scored ({time.time() - t0:.0f} s)")
json.dump({"validation": R, "gold": G, "winners": {"F1": NAMES[win_f1], "voiced": NAMES[win_v]}},
          open(OUT / "combo_results.json", "w"), indent=1)

# ------------------------------------------------------------------ 6. the winners on gold, and the part-2 rule for F1
for kind, w in (("F1", win_f1), ("voiced", win_v)):
    nm = NAMES[w]
    p(f"\n## Winner for {kind}: {nm} (teacher v3 alongside; layer 5 alone for reference)")
    for vs in SEL_SETS:
        p(f"  {vs:32s} " + "  ".join(f"{t} {R[vs][nm][t]:.1f}/{R[vs]['teacher_v3'][t]:.1f}/{R[vs]['L5'][t]:.1f}"
                                      for t in TARGETS if not np.isnan(R[vs][nm][t])))
    for g in ("gold_ptdb_tug", "gold_mir1k", "gold_mdb_stem_synth"):
        p(f"  {g:32s} f0 RPA {G[g][nm]['f0_rpa']:.1f}/{G[g]['teacher_v3']['f0_rpa']:.1f}/{G[g]['L5']['f0_rpa']:.1f}  "
          f"voicing {G[g][nm]['voiced_acc']:.1f}/{G[g]['teacher_v3']['voiced_acc']:.1f}/{G[g]['L5']['voiced_acc']:.1f}")
    for gname in ("hillenbrand", "closed_phase"):
        p(f"  {gname} (median signed / MAE, Hz): " + "; ".join(
            f"{k} {G[gname][nm][k][0]:+.1f}/{G[gname][nm][k][1]:.1f} (teacher {G[gname]['teacher_v3'][k][0]:+.1f}/{G[gname]['teacher_v3'][k][1]:.1f})"
            for k in G[gname][nm]))

c_hb = hb_T.copy()
c_hb[:, 0] = (hb_T[:, 0] + hb_P[win_f1][:, 0]) / 2
c_cp = cp_T.copy()
c_cp[:, 0] = (cp_T[:, 0] + cp_P[win_f1][:, 0]) / 2
fin = lambda r, m: m & np.isfinite(r) & (r > 0)  # noqa: E731
w0 = abs(mae_signed(cp_T[:, 0], cp_ref[:, 0], fin(cp_ref[:, 0], cp_g["women"]))[0])
w1 = abs(mae_signed(c_cp[:, 0], cp_ref[:, 0], fin(cp_ref[:, 0], cp_g["women"]))[0])
m0 = abs(mae_signed(cp_T[:, 0], cp_ref[:, 0], fin(cp_ref[:, 0], cp_g["men"]))[0])
m1 = abs(mae_signed(c_cp[:, 0], cp_ref[:, 0], fin(cp_ref[:, 0], cp_g["men"]))[0])
h0 = mae_signed(hb_T[:, 0], hb_ref[:, 0], fin(hb_ref[:, 0], hb_g["all"]))[1]
h1 = mae_signed(c_hb[:, 0], hb_ref[:, 0], fin(hb_ref[:, 0], hb_g["all"]))[1]
okr = (w0 - w1 >= 5) and (m1 - m0 <= 5) and (h1 - h0 <= 4)
p(f"\nPart-2 rule for the F1 winner {NAMES[win_f1]}, ensemble (c): women |F1 bias| {w0:.1f} -> {w1:.1f}; men {m0:.1f} -> {m1:.1f};"
  f" Hillenbrand F1 MAE {h0:.1f} -> {h1:.1f}  DECISION: {'worth a full arm' if okr else 'no'}")
for g in ("men", "women", "children"):
    p(f"  Hillenbrand {g}: teacher {mae_signed(hb_T[:, 0], hb_ref[:, 0], fin(hb_ref[:, 0], hb_g[g]))}, "
      f"(c) {mae_signed(c_hb[:, 0], hb_ref[:, 0], fin(hb_ref[:, 0], hb_g[g]))}")
p(f"done ({time.time() - t0:.0f} s)")
