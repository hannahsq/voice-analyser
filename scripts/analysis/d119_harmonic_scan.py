"""D-119: is the F1 shift D-109's reconstruction loss prefers on real audio locked to the harmonics or to frequency?

The branch-point models (d040_5b_s{1,2}/ckpt_016000.pt) on recon-store crops (VocalSet, Zeroth; 24 batches of
16 x 200 frames, the same crops for both models) and on train_v2 validation crops as the control. D-109's instrument
(model F1-F4 and B1-B4, higher-pole correction, harmonics to 4000 Hz, Rd searched per frame, level profiled).

  1. each frame's residual at its best Rd, fitted over all valid frame-harmonics as a(k) + b(f): a over harmonic number
     (1-12, 13+ pooled), b over frequency (250 Hz bins to 4000 Hz);
  2. the F1 shift the loss prefers (the model's F1 scaled by c cents, everything else fixed), by cells of F1 band x F1 / f0;
  3. the shift each part of the fit predicts on the same frames: the target replaced by the loss's own render at the
     frame's best Rd (zero misfit) plus a(k) alone (H), b(f) alone (F) or both (HF).

    VA_DEVICE=cuda python scripts/analysis/d119_harmonic_scan.py > reports/analysis/d119/d119.txt

The per-frame loss here renders once per frame and loops Rd over the residual (`recon_loss_fb` re-renders per Rd); it is
checked against `recon_loss_fb` on the first batch. Cell sums are saved in cache/d119/cells.npz.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d114_stretch_scan import argmin_parabola, synth_targets  # noqa: E402

C_GRID = np.arange(-200.0, 200.0 + 1e-9, 25.0)
N_BATCH, N_SYN, SEED = 24, 24, 1119  # the control at the real corpora's count (amendment 1)
F1_CUTS, F1_NAMES = (450.0, 600.0), ("F1 < 450", "F1 450-600", "F1 >= 600")
H_CUTS, H_NAMES = (1.5, 2.0, 2.5, 3.5), ("< 1.5", "1.5-2", "2-2.5", "2.5-3.5", ">= 3.5")
F0_CUTS, F0_NAMES = (150.0, 250.0), ("f0 < 150", "f0 150-250", "f0 >= 250")
NK, NF, BIN_HZ = 13, 16, 250.0
MIN_FRAMES, N_BOOT = 300, 500
VARS = ("obs", "H", "F", "HF")
SYNTH = "train_v2 validation (synthetic control)"
SMOKE = bool(os.environ.get("D119_SMOKE"))  # a code run on two batches; its output is not a result
if SMOKE:
    N_BATCH, N_SYN, MIN_FRAMES, N_BOOT = 2, 2, 20, 20


def shift(F, c):
    import torch

    return torch.cat([F[..., :1] * 2.0 ** (c / 1200.0), F[..., 1:]], -1)


def frame_terms(L):
    """D-109's gating: harmonics to 4000 Hz, frames with at least 4 of them"""
    import torch

    tgt, ok, f0, m = L["db"].float(), L["ok"].bool(), L["f0"].float(), L["mask"].bool()
    k = torch.arange(1, tgt.shape[-1] + 1, device=tgt.device, dtype=f0.dtype)
    f = f0[..., None] * k
    ok = ok & (f <= 4000.0)
    return tgt, ok, m & (ok.sum(-1) >= 4), f0, f


def per_frame(R, F, B, tgt, ok, m, f0, f, lf, full=False):
    """`recon_loss_fb(..., fmax=4000, hpc=True, channel=0, per_frame=True)` with ok / m already gated: (B, T), NaN on
    frames that carry no loss. full: also each frame's best Rd index, its residual there and the render."""
    import torch

    uf, ub = R.upper_poles(F[..., 3], f0)
    rend = R.render_db(torch.where(ok, f, torch.full_like(f, 1000.0)), torch.cat([F, uf], -1), torch.cat([B, ub], -1), None, None, False)
    w = ok.to(tgt.dtype)
    n = w.sum(-1)
    r = ((tgt - rend).unsqueeze(0) - lf[:, None, None, :]) * w  # (Rd, B, T, K)
    r = (r - r.sum(-1, keepdim=True) / n.clamp_min(1.0)[None, ..., None]) * w
    per = (torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=6.0) * w).sum(-1) / n.clamp_min(1.0)
    best, idx = per.min(0)
    out = torch.where(m & (n >= 4), best, torch.full_like(best, float("nan")))
    if not full:
        return out
    res = torch.gather(r, 0, idx[None, ..., None].expand(1, *r.shape[1:]))[0]
    return out, idx, res, rend


def fit_parts(N, S, Q):
    """least squares r = a(k) + b(f) from the counts N (NK, NF), sums S (NK, NF) and sum of squares Q of the residual.
    Returns a, b (a's weighted mean moved into b: a constant per frame is the level, which the loss profiles out) and
    the shares of the residual's sum of squares explained by a alone, b alone and both."""
    Nk, Nj, Sk, Sj = N.sum(1), N.sum(0), S.sum(1), S.sum(0)
    XtX = np.block([[np.diag(Nk), N], [N.T, np.diag(Nj)]])
    Xtr = np.concatenate([Sk, Sj])
    beta = np.linalg.lstsq(XtX, Xtr, rcond=None)[0]
    a, b = beta[:NK].copy(), beta[NK:].copy()
    c = (Nk * a).sum() / Nk.sum()
    a, b = a - c, b + c * (Nj > 0)
    return a, b, {"a": (Sk ** 2 / np.maximum(Nk, 1)).sum() / Q, "b": (Sj ** 2 / np.maximum(Nj, 1)).sum() / Q, "both": beta @ Xtr / Q,
                  "var": Q / N.sum(), "n": int(N.sum())}


def share_in(N, S, Q, a, b):
    """the share of a band's residual sum of squares that the pooled fit a(k) + b(f) explains"""
    p = a[:, None] + b[None, :]
    return 1.0 - (Q - 2 * (p * S).sum() + (p ** 2 * N).sum()) / Q


def pooled_cstar(rows):
    ok = np.all(np.isfinite(rows), 1)
    y = rows[ok].mean(0)
    return argmin_parabola(C_GRID, y), float(y[len(C_GRID) // 2])


def process(model, vq, batches, fixed_L, dev, sims, checks):
    """one model on one corpus: the residual fit, then the per-frame loss against the F1 shift for the observed target
    and (sims) each part's simulated target. Returns (rows {variant: (frames, n_c)}, meta (frames, 3: crop, F1, f0), fit)."""
    import torch

    from vocal_analysis.train import reconstruction as R

    K = 64
    lf = R.lf_source_db(torch.tensor(R.RD_SEARCH, dtype=torch.float32, device=dev), K)  # (Rd, K)
    kk = torch.arange(K, device=dev).clamp(max=NK - 1)
    N, S, Q = np.zeros((3, NK, NF)), np.zeros((3, NK, NF)), np.zeros(3)
    store = []
    with torch.no_grad():
        for bi, (au, lab) in enumerate(batches):
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                o = model(vq.log_power_with_context(au.to(dev), lab["f0"].shape[1]))
            L = {k: v.to(dev) for k, v in (fixed_L[bi] if fixed_L is not None else lab).items()}
            F = R.decode_formants(o["formant_logits"])[..., :4].float()
            B = torch.exp(o["logb_mean"].float())[..., :4]
            tgt, ok, m, f0, f = frame_terms(L)
            per, idx, res, rend = per_frame(R, F, B, tgt, ok, m, f0, f, lf, full=True)
            fj = (f / BIN_HZ).floor().long().clamp(0, NF - 1)
            sel = ok & torch.isfinite(per)[..., None]
            band = torch.bucketize(f0, torch.tensor(F0_CUTS, device=dev), right=True)[..., None].expand_as(f)
            code = ((band * NK + kk.expand_as(f)) * NF + fj)[sel].cpu().numpy()
            r = res[sel].double().cpu().numpy()
            N += np.bincount(code, minlength=3 * NK * NF).reshape(3, NK, NF)
            S += np.bincount(code, r, 3 * NK * NF).reshape(3, NK, NF)
            Q += np.bincount(code // (NK * NF), r ** 2, 3)
            store.append((F, B, tgt, ok, m, f0, f, idx, rend, fj, L))
        a, b, sh = fit_parts(N.sum(0), S.sum(0), Q.sum())
        sh["by_f0"] = [share_in(N[j], S[j], Q[j], a, b) if Q[j] > 0 else np.nan for j in range(3)]
        sh["n_f0"] = [int(N[j].sum()) for j in range(3)]
        a_t, b_t = torch.tensor(a, dtype=torch.float32, device=dev), torch.tensor(b, dtype=torch.float32, device=dev)

        rows, meta = {v: [] for v in (VARS if sims else VARS[:1])}, []
        for bi, (F, B, tgt, ok, m, f0, f, idx, rend, fj, L) in enumerate(store):
            base = rend + lf[idx]
            tg = {"obs": tgt}
            if sims:
                tg.update({"H": base + a_t[kk].expand_as(f), "F": base + b_t[fj], "HF": base + a_t[kk].expand_as(f) + b_t[fj]})
            for v, t in tg.items():
                rows[v].append(torch.stack([per_frame(R, shift(F, c), B, t, ok, m, f0, f, lf) for c in C_GRID], -1)
                               .reshape(-1, len(C_GRID)).cpu().numpy())
            nb, T = f0.shape
            meta.append(np.column_stack([np.repeat(np.arange(bi * nb, (bi + 1) * nb), T), F[..., 0].reshape(-1).cpu().numpy(),
                                         f0.reshape(-1).cpu().numpy()]))
            if checks is not None and bi == 0:
                d = 0.0
                for c in (0.0, 100.0):
                    ref = R.recon_loss_fb(shift(F, c), B, L, fmax=4000.0, hpc=True, channel=0, per_frame=True)
                    mine = per_frame(R, shift(F, c), B, tgt, ok, m, f0, f, lf)
                    same = bool((torch.isfinite(ref) == torch.isfinite(mine)).all())
                    d = max(d, float((ref - mine)[torch.isfinite(ref) & torch.isfinite(mine)].abs().max())) if same else float("inf")
                checks["fast loss against recon_loss_fb, largest difference"] = (d, d < 1e-3)
                z = torch.stack([per_frame(R, shift(F, c), B, base, ok, m, f0, f, lf) for c in C_GRID], -1).reshape(-1, len(C_GRID)).cpu().numpy()
                cz, l0 = pooled_cstar(z)
                checks["zero-misfit target: preferred shift (c), loss at 0"] = ((cz, l0), abs(cz) < 5 and l0 < 1e-3)
                rend50 = per_frame(R, shift(F, 50.0), B, tgt, ok, m, f0, f, lf, full=True)[3]
                z = torch.stack([per_frame(R, shift(F, c), B, rend50 + lf[idx], ok, m, f0, f, lf) for c in C_GRID], -1).reshape(-1, len(C_GRID)).cpu().numpy()
                c50 = pooled_cstar(z)[0]
                checks["target rendered with F1 +50 c: recovered shift (c)"] = (c50, abs(c50 - 50) < 5)
    return {v: np.concatenate(x) for v, x in rows.items()}, np.concatenate(meta), (a, b, sh)


def cell_sums(rows, meta, n_crop):
    """per cell (F1 band x F1 / f0 band) and crop: the summed per-frame loss against c for each variant, and the frame count"""
    crop, f1, f0 = meta[:, 0].astype(int), meta[:, 1], meta[:, 2]
    good = np.all([np.all(np.isfinite(r), 1) for r in rows.values()], 0) & (f0 > 0)
    cell = np.digitize(f1, F1_CUTS) * len(H_NAMES) + np.digitize(f1 / np.maximum(f0, 1e-6), H_CUTS)
    n_cell = len(F1_NAMES) * len(H_NAMES)
    cnt = np.zeros((n_cell, n_crop))
    np.add.at(cnt, (cell[good], crop[good]), 1.0)
    sums = {}
    for v, r in rows.items():
        s = np.zeros((n_cell, n_crop, len(C_GRID)))
        np.add.at(s, (cell[good], crop[good]), r[good])
        sums[v] = s
    return sums, cnt


def cstar(s, c, pick=None):
    """the minimising shift of a cell from its per-crop sums (optionally over a draw of crops); NaN if the draw is empty"""
    if pick is not None:
        s, c = s[pick], c[pick]
    return argmin_parabola(C_GRID, s.sum(0) / c.sum()) if c.sum() > 0 else np.nan


def wcorr(x, y, w):
    k = np.isfinite(x) & np.isfinite(y)
    if k.sum() < 3:
        return np.nan
    x, y, w = x[k], y[k], w[k] / w[k].sum()
    dx, dy = x - (w * x).sum(), y - (w * y).sum()
    den = np.sqrt((w * dx * dx).sum() * (w * dy * dy).sum())
    return float((w * dx * dy).sum() / den) if den > 0 else np.nan


def wslope(x, y, w):
    k = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[k], y[k], w[k] / w[k].sum()
    dx = x - (w * x).sum()
    return float((w * dx * (y - (w * y).sum())).sum() / (w * dx * dx).sum()) if (w * dx * dx).sum() > 0 else np.nan


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.data import SynthCrops, collate, split_ids
    from vocal_analysis.train.recon_data import ReconCrops, collate_recon

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    data = {name: list(DataLoader(ReconCrops([store], 200, seed=SEED, epoch_size=16 * N_BATCH), 16, num_workers=2, collate_fn=collate_recon))
            for name, store in (("VocalSet", "cache/recon/vocalset"), ("Zeroth-Korean", "cache/recon/zeroth"))}
    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    syn = SynthCrops("data/synth/train_v2/flat", split_ids(idx["n_clips"])[1], n_frames=200, seed=SEED, epoch_size=16 * N_SYN)
    data[SYNTH] = list(DataLoader(syn, 16, num_workers=4, collate_fn=collate))
    syn_L = [synth_targets(a, lab) for a, lab in data[SYNTH]]
    print("data ready", file=sys.stderr, flush=True)

    checks, RES = {}, {}
    for seed in (1, 2):
        model, _ = load_teacher(f"runs/d040_5b_s{seed}/ckpt_016000.pt", dev)
        for name, batches in data.items():
            first = seed == 1 and name == "VocalSet"
            rows, meta, fit = process(model, vq, batches, syn_L if name == SYNTH else None, dev, name != SYNTH, checks if first else None)
            sums, cnt = cell_sums(rows, meta, 16 * len(batches))
            RES[name, seed] = (sums, cnt, fit)
            print(f"seed {seed} {name}: done", file=sys.stderr, flush=True)
        del model
    Path("cache/d119").mkdir(parents=True, exist_ok=True)
    np.savez("cache/d119/cells_smoke.npz" if SMOKE else "cache/d119/cells.npz", c_grid=C_GRID, **{f"{n}|{s}|{v}": RES[n, s][0][v] for (n, s) in RES for v in RES[n, s][0]},
             **{f"{n}|{s}|cnt": RES[n, s][1] for (n, s) in RES})

    L = [("SMOKE RUN (two batches; not a result). " if SMOKE else "")
         + "D-119: is the F1 shift D-109's reconstruction loss prefers on real audio locked to the harmonics or to frequency?",
         f"branch-point models d040_5b_s{{1,2}}/ckpt_016000.pt; recon-store crops, {N_BATCH} batches of 16 x 200 frames per corpus (seed {SEED}); "
         f"control {N_SYN} batches", "shift: the model's F1 scaled by c cents, c on " + f"{C_GRID[0]:+.0f} .. {C_GRID[-1]:+.0f} in 25 c steps; "
         "the minimising c by a parabola through the lowest three grid points", ""]
    L.append("code checks (first VocalSet batch, seed 1):")
    for k, (v, ok) in checks.items():
        vs = ", ".join(f"{x:.2e}" if abs(x) < 1e-2 else f"{x:+.1f}" for x in np.atleast_1d(v))
        L.append(f"  {k}: {vs} -> {'ok' if ok else 'FAILS'}")
    checks_ok = all(ok for _, ok in checks.values())
    L.append("")

    L.append("1. the residual at each frame's best Rd, fitted as a(k) + b(f) (dB; shares of the residual's sum of squares)")
    for name in data:
        for seed in (1, 2):
            a, b, sh = RES[name, seed][2]
            L.append(f"  {name}, seed {seed}: {sh['n']} frame-harmonics, residual variance {sh['var']:.2f} dB^2; explained by a alone {sh['a']:.1%}, "
                     f"b alone {sh['b']:.1%}, both {sh['both']:.1%}; the joint fit within "
                     + ", ".join(f"{F0_NAMES[j]} {sh['by_f0'][j]:.1%} (n {sh['n_f0'][j]})" for j in range(3)))
            L.append("    a(k), k = 1..12, 13+: " + " ".join(f"{x:+.1f}" for x in a))
            L.append("    b(f), 250 Hz bins to 4000: " + " ".join(f"{x:+.1f}" for x in b))
    L.append("")

    L.append(f"2 and 3. the preferred F1 shift by cell (cents; cells with at least {MIN_FRAMES} frames in both seeds; seed means; 95 % crop bootstrap, {N_BOOT})")
    rng = np.random.default_rng(SEED)
    out, n_cells = {}, {}
    for name in data:
        sims = name != SYNTH
        vs = VARS if sims else VARS[:1]
        n_crop = RES[name, 1][1].shape[1]
        nfr = np.mean([RES[name, s][1].sum(1) for s in (1, 2)], 0)
        cells = [j for j in range(len(nfr)) if all(RES[name, s][1][j].sum() >= MIN_FRAMES for s in (1, 2))]
        n_cells[name] = len(cells)

        def table(pick=None):
            return {v: np.array([np.mean([cstar(RES[name, s][0][v][j], RES[name, s][1][j], pick) for s in (1, 2)]) for j in cells]) for v in vs}
        pt = table()
        per_seed = {s: np.array([cstar(RES[name, s][0]["obs"][j], RES[name, s][1][j]) for j in cells]) for s in (1, 2)}
        bs = [table(rng.integers(0, n_crop, n_crop)) for _ in range(N_BOOT)]
        w = nfr[cells]
        lo, hi = np.nanpercentile([b["obs"] for b in bs], [2.5, 97.5], 0)
        L.append(f"  {name}")
        L.append(f"    {'cell':28s} {'frames':>7s}  {'observed':>22s}  {'seeds':>13s}" + ("    H (a alone)  F (b alone)   HF (both)" if sims else ""))
        for i, j in enumerate(cells):
            nm = f"{F1_NAMES[j // len(H_NAMES)]}, F1/f0 {H_NAMES[j % len(H_NAMES)]}"
            edge = " (edge)" if abs(pt["obs"][i]) >= C_GRID[-1] - 1e-6 else ""
            L.append(f"    {nm:28s} {int(w[i]):7d}  {pt['obs'][i]:+6.0f} [{lo[i]:+5.0f}, {hi[i]:+5.0f}]  {per_seed[1][i]:+5.0f} / {per_seed[2][i]:+5.0f}"
                     + (f"   {pt['H'][i]:+10.0f}  {pt['F'][i]:+11.0f}  {pt['HF'][i]:+10.0f}" if sims else "") + edge)
        out[name] = {"obs": pt["obs"], "w": w, "cells": cells}
        if sims:
            stat = lambda t: (wcorr(t["obs"], t["H"], w), wcorr(t["obs"], t["F"], w), wcorr(t["obs"], t["HF"], w), wslope(t["HF"], t["obs"], w))  # noqa: E731
            p = stat(pt)
            b_ = np.array([stat(t) for t in bs])
            D, Dlo, Dhi = p[0] - p[1], *np.nanpercentile(b_[:, 0] - b_[:, 1], [2.5, 97.5])
            ci = lambda j: "[{:+.2f}, {:+.2f}]".format(*np.nanpercentile(b_[:, j], [2.5, 97.5]))  # noqa: E731
            L.append(f"    across cells (weighted by frames): r(observed, H) {p[0]:+.2f} {ci(0)}; r(observed, F) {p[1]:+.2f} {ci(1)}; "
                     f"r(observed, HF) {p[2]:+.2f} {ci(2)}; D = r_H - r_F {D:+.2f} [{Dlo:+.2f}, {Dhi:+.2f}]; slope of observed on HF {p[3]:+.2f} {ci(3)}")
            for s in (1, 2):
                t = {v: np.array([cstar(RES[name, s][0][v][j], RES[name, s][1][j]) for j in cells]) for v in vs}
                q = stat(t)
                L.append(f"      seed {s} alone: r_H {q[0]:+.2f}, r_F {q[1]:+.2f}, r_HF {q[2]:+.2f}, D {q[0] - q[1]:+.2f}, slope {q[3]:+.2f}")
            out[name].update(D=(D, Dlo, Dhi), r_hf=p[2])
            # the prediction beside the rule: within an F1 band, negative where F1 / f0 < 2 and positive where >= 2.5
            for fb in range(len(F1_NAMES)):
                lo_h = [pt["obs"][i] for i, j in enumerate(cells) if j // len(H_NAMES) == fb and j % len(H_NAMES) <= 1]
                hi_h = [pt["obs"][i] for i, j in enumerate(cells) if j // len(H_NAMES) == fb and j % len(H_NAMES) >= 3]
                verdict = "n/a (one side has no cell)" if not lo_h or not hi_h else ("held" if max(lo_h) < 0 < min(hi_h) else "not held")
                L.append(f"      prediction, {F1_NAMES[fb]}: F1/f0 < 2 " + (" ".join(f"{x:+.0f}" for x in lo_h) or "-") + "; >= 2.5 "
                         + (" ".join(f"{x:+.0f}" for x in hi_h) or "-") + f" -> {verdict}")
        # margins: by F1 / f0 alone and by F1 band alone (observed, seed means)
        for label, groups in (("by F1/f0 alone", [(H_NAMES[h], [j for j in range(len(nfr)) if j % len(H_NAMES) == h]) for h in range(len(H_NAMES))]),
                              ("by F1 band alone", [(F1_NAMES[f], [j for j in range(len(nfr)) if j // len(H_NAMES) == f]) for f in range(len(F1_NAMES))])):
            parts = []
            for nm, js in groups:
                if np.mean([RES[name, s][1][js].sum() for s in (1, 2)]) < MIN_FRAMES:
                    continue
                parts.append(f"{nm} {np.mean([cstar(RES[name, s][0]['obs'][js].sum(0), RES[name, s][1][js].sum(0)) for s in (1, 2)]):+.0f}")
            L.append(f"    observed {label}: " + "; ".join(parts))
        L.append("")

    real = [n for n in data if n != SYNTH]
    def wmedian(x, w):  # frame-weighted median over cells (amendment 1: the same statistic for the control and the real corpora)
        o = np.argsort(x)
        return float(x[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]) if len(x) else float("nan")

    med_real = wmedian(np.abs(np.concatenate([out[n]["obs"] for n in real])), np.concatenate([out[n]["w"] for n in real]))
    max_syn = wmedian(np.abs(out[SYNTH]["obs"]), out[SYNTH]["w"])
    L.append(f"READING (registered, amendment 1's gate): cells per corpus " + ", ".join(f"{n} {n_cells[n]}" for n in real)
             + f"; control's frame-weighted median |observed shift| {max_syn:.0f} c ({n_cells[SYNTH]} cells) against the real corpora's {med_real:.0f} c; "
             + "; ".join(f"{n}: r(observed, HF) {out[n]['r_hf']:+.2f}, D {out[n]['D'][0]:+.2f} [{out[n]['D'][1]:+.2f}, {out[n]['D'][2]:+.2f}]" for n in real))
    if not checks_ok or min(n_cells[n] for n in real) < 6 or not max_syn <= 0.5 * med_real:
        verdict = "UNREADABLE"
    elif min(out[n]["r_hf"] for n in real) < 0.5:
        verdict = "NOT IN THE MEAN RESIDUAL"
    elif all(out[n]["D"][1] > 0 for n in real):
        verdict = "HARMONIC-LOCKED"
    elif all(out[n]["D"][2] < 0 for n in real):
        verdict = "FREQUENCY-LOCKED"
    else:
        verdict = "MIXED"
    L.append(f"  -> {verdict}")
    print("\n".join(L))
