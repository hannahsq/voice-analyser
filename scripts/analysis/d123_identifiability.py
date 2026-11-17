"""D-123's pre-check (cited by the draft; no data, CPU only): can a source term in harmonic number, held per crop, take a
source misfit out of the reconstruction loss without taking F1 with it?

One made-up crop of 200 frames per case: a fixed tract (F1 as listed, F2-F4 1500 / 2500 / 3500 Hz, Hawks and Miller
bandwidths, D-109's higher-pole correction), an LF source at Rd 1.0 and an f0 contour. The target is the loss's own render
plus a misfit fixed in harmonic number, D-119's a(k) on Zeroth (+3.1, -1.6, -1.1, -0.5, -0.1 dB at k = 1..5). The loss is
D-109's (level per frame, Rd searched per frame), and the same with a per-crop offset per harmonic number profiled out
jointly with the level (k <= 8). F1 is scaled by c cents on every frame; reported: the minimising c (0 is right) and the
loss's curvature there (how much the loss still says about F1), for a flat f0, two vibrato depths and two glides.

    python scripts/analysis/d123_identifiability.py > reports/analysis/d123/identifiability.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d114_stretch_scan import argmin_parabola  # noqa: E402

C_GRID = np.arange(-200.0, 200.0 + 1e-9, 25.0)
T, K, KS = 200, 64, 8
A_K = (3.1, -1.6, -1.1, -0.5, -0.1)  # D-119, Zeroth, seed means of a(k), dB
CONTOURS = {"flat": lambda t: 0 * t, "vibrato +-0.5 st": lambda t: 0.5 * np.sin(2 * np.pi * 5.5 * t),
            "vibrato +-1 st": lambda t: 1.0 * np.sin(2 * np.pi * 5.5 * t), "glide 4 st": lambda t: 4.0 * (t / t[-1] - 0.5),
            "glide 8 st": lambda t: 8.0 * (t / t[-1] - 0.5)}


def crop_basis_out(resid, w, phi):
    """resid (T, K) minus, jointly, a level per frame and a combination of the basis phi (T, K, n) per crop; weighted least
    squares over the crop's valid harmonics (the algebra of `reconstruction.crop_channel_out` with any basis)"""
    import torch

    ws = w.sum(-1, keepdim=True).clamp_min(1e-9)
    rc = resid - (w * resid).sum(-1, keepdim=True) / ws
    pc = phi - (w.unsqueeze(-1) * phi).sum(-2, keepdim=True) / ws.unsqueeze(-1)
    G = torch.einsum("tkn,tk,tkm->nm", pc, w, pc)
    G = G + 1e-9 * G.diagonal().mean().clamp_min(1e-9) * torch.eye(phi.shape[-1], dtype=G.dtype)
    c = torch.linalg.solve(G, torch.einsum("tkn,tk,tk->n", pc, w, rc))
    return rc - torch.einsum("tkn,n->tk", pc, c)


def crop_loss(R, F, B, tgt, ok, f0, lf, per_crop_source):
    """D-109's per-frame loss on one crop (T, K), mean over frames; optionally with the per-crop harmonic-number term"""
    import torch

    f = f0[:, None] * torch.arange(1, K + 1, dtype=f0.dtype)
    uf, ub = R.upper_poles(F[:, 3], f0)
    rend = R.render_db(torch.where(ok, f, torch.full_like(f, 1000.0)), torch.cat([F, uf], -1), torch.cat([B, ub], -1), None, None, False)
    w = ok.to(tgt.dtype)
    n = w.sum(-1)
    r = ((tgt - rend).unsqueeze(0) - lf[:, None, :]) * w  # (Rd, T, K)
    r = (r - r.sum(-1, keepdim=True) / n.clamp_min(1.0)[None, :, None]) * w
    per = (torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction="none", delta=6.0) * w).sum(-1) / n.clamp_min(1.0)
    best, idx = per.min(0)
    if not per_crop_source:
        return float(best.mean())
    res = (tgt - rend - lf[idx]) * w  # each frame's Rd chosen as without the term (as `recon_loss_fb`'s channel branch)
    phi = torch.zeros(T, K, KS, dtype=tgt.dtype)
    for j in range(KS):
        phi[:, j, j] = 1.0
    res = crop_basis_out(res, w, phi) * w
    hub = torch.nn.functional.huber_loss(res, torch.zeros_like(res), reduction="none", delta=6.0)
    return float(((hub * w).sum(-1) / n.clamp_min(1.0)).mean())


if __name__ == "__main__":
    import torch

    from vocal_analysis.train import reconstruction as R

    torch.set_default_dtype(torch.float64)
    lf = R.lf_source_db(torch.tensor(R.RD_SEARCH), K)
    t = np.arange(T) * 0.01
    a = torch.zeros(K)
    a[:len(A_K)] = torch.tensor(A_K)
    print("D-123 pre-check: a per-crop source term in harmonic number, on made-up crops (no data). 0 c is the right answer.")
    print(f"misfit a(k) = {A_K} dB at k = 1..5; minimising F1 shift in cents and (curvature at 0, loss units per (100 c)^2)")
    print(f"{'f0':>5s} {'F1':>5s} {'F1/f0':>6s}  {'contour':18s}  {'D-109 loss':>22s}  {'+ per-crop source(k)':>22s}  curvature kept")
    for f0b in (120.0, 220.0, 330.0):
        for F1 in (300.0, 450.0, 600.0, 800.0):
            if F1 < 1.15 * f0b:
                continue
            for cname, fn in CONTOURS.items():
                f0 = torch.tensor(f0b * 2.0 ** (fn(t) / 12.0))
                F = torch.tensor([[F1, 1500.0, 2500.0, 3500.0]]).repeat(T, 1)
                B = R.hawks_miller_t(F, f0[:, None].clamp(min=50.0))
                f = f0[:, None] * torch.arange(1, K + 1, dtype=f0.dtype)
                ok = f <= 4000.0
                uf, ub = R.upper_poles(F[:, 3], f0)
                rend = R.render_db(torch.where(ok, f, torch.full_like(f, 1000.0)), torch.cat([F, uf], -1), torch.cat([B, ub], -1), None, None, False)
                tgt = rend + R.lf_source_db(torch.full_like(f0, 1.0), K) + a[None, :]
                cells = []
                for src in (False, True):
                    y = np.array([crop_loss(R, torch.cat([F[:, :1] * 2.0 ** (c / 1200.0), F[:, 1:]], -1), B, tgt, ok, f0, lf, src) for c in C_GRID])
                    i0 = len(C_GRID) // 2
                    curv = (y[i0 + 4] + y[i0 - 4] - 2 * y[i0])  # second difference over +-100 c
                    cells.append((argmin_parabola(C_GRID, y), curv))
                kept = cells[1][1] / cells[0][1] if cells[0][1] > 0 else float("nan")
                print(f"{f0b:5.0f} {F1:5.0f} {F1 / f0b:6.2f}  {cname:18s}  {cells[0][0]:+7.0f} c ({cells[0][1]:8.3f})    {cells[1][0]:+7.0f} c ({cells[1][1]:8.3f})    {kept:6.2f}")
