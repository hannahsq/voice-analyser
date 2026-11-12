"""D-106: D-105's check with SoFilNet's nuisance structure: two high poles (the frame's F4 label to 7800 Hz, 40-1000 Hz
wide), two broad poles (50-7800 Hz, 1000-4000 Hz wide) and two zeros (250-4000 Hz, 40-4000 Hz wide), plus Rd, fitted
per frame and F1 candidate by Adam (lr 0.05, 200 steps). Start: broad pole = zero pairs at 1000 and 3000 Hz, 3000 Hz
wide (cancelling), high poles at 7500 Hz, 500 Hz wide; Rd 1.0. Everything else as D-105. CPU, clips in parallel.

    python scripts/analysis/d106_sofilnet_check.py [n_clips] [--serial] > reports/analysis/d106/d106.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SNR, STEPS, LR = 10.0, 200, 0.05
COARSE = np.arange(-400, 401, 25)
FINE = np.arange(-25, 26, 5)
HP_HI, HP_B, BP_F, BP_B, Z_F, Z_B, RD_BAND = 7800.0, (40.0, 1000.0), (50.0, 7800.0), (1000.0, 4000.0), (250.0, 4000.0), (40.0, 4000.0), (0.3, 2.7)


def _logit_of(v, lo, hi):
    s = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(np.log(s / (1 - s)))


def fit(tgt, val, f0, poles, bws, free=True):
    """poles / bws (M, N, 8) label sections with F1 set per candidate; returns per-(M, N) loss and the fitted nuisance
    (M, N, 8): [broad pole 1, 2, zero 1, 2 frequencies, broad pole 1, 2, zero 1, 2 bandwidths]."""
    import torch

    from vocal_analysis.train import reconstruction as R

    M, N = poles.shape[:2]

    def sq(x, lo, hi):
        lo = torch.as_tensor(lo, dtype=x.dtype)
        return torch.exp(torch.log(lo) + torch.sigmoid(x) * (np.log(hi) - torch.log(lo)))

    def lg(v, lo, hi):
        s_ = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
        return np.log(s_ / (1 - s_))

    f4 = np.minimum(poles[..., 3], 7000.0)
    init = np.zeros((M, N, 12))
    init[..., 0:2] = lg(np.where(f4 < 7400.0, 7500.0, (f4 + HP_HI) / 2)[..., None], f4[..., None], HP_HI)  # high poles at 7500 Hz
    init[..., 2:4] = lg(500.0, *HP_B)
    init[..., 4], init[..., 5] = lg(1000.0, *BP_F), lg(3000.0, *BP_F)  # broad poles
    init[..., 6:8] = lg(3000.0, *BP_B)
    init[..., 8], init[..., 9] = lg(1000.0, *Z_F), lg(3000.0, *Z_F)  # zeros, cancelling the broad poles
    init[..., 10:12] = lg(3000.0, *Z_B)
    u = torch.tensor(init, dtype=torch.float64, requires_grad=free)
    s0 = (1.0 - RD_BAND[0]) / (RD_BAND[1] - RD_BAND[0])
    ur = torch.full((M, N), float(np.log(s0 / (1 - s0))), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([ur] + ([u] if free else []), lr=LR)
    P0, B0 = torch.tensor(poles), torch.tensor(bws)
    F4 = torch.tensor(f4)[..., None]

    def parts():
        hp = sq(u[..., 0:2], F4, HP_HI)
        return (torch.cat([P0, hp, sq(u[..., 4:6], *BP_F)], -1), torch.cat([B0, sq(u[..., 2:4], *HP_B), sq(u[..., 6:8], *BP_B)], -1),
                sq(u[..., 8:10], *Z_F), sq(u[..., 10:12], *Z_B))

    def render_loss():
        if not free:  # the comparison render: no nuisance sections at all (D-104's), Rd only
            rd = RD_BAND[0] + (RD_BAND[1] - RD_BAND[0]) * torch.sigmoid(ur)
            return R.reconstruction_loss(tgt, val, f0, P0, B0, source="rd", profile="offset", rd=rd)[1]
        pf, pb, zf, zb = parts()
        rd = RD_BAND[0] + (RD_BAND[1] - RD_BAND[0]) * torch.sigmoid(ur)
        _, per = R.reconstruction_loss(tgt, val, f0, pf, pb, zf, zb, source="rd", profile="offset", rd=rd)
        return per

    for _ in range(STEPS):
        opt.zero_grad()
        per = render_loss()
        torch.nan_to_num(per, nan=0.0).sum().backward()
        opt.step()
    with torch.no_grad():
        per = render_loss().numpy()
        pf, pb, zf, zb = parts()
        nz = torch.cat([pf[..., -2:], zf, pb[..., -2:], zb], -1).numpy()
    return per, nz


def one(job):
    import torch

    torch.set_num_threads(1)
    from d102_render_check import load_cfg
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.train import reconstruction as R

    root, coupled, i = job
    idx, cfg = load_cfg(root)
    clip = generate(int(i), cfg, idx["seed"])
    lab = clip.labels
    db, ok = R.harmonic_targets(clip.audio, lab["f0"], lab["voiced"].astype(bool), k_max=64, snr_min_db=SNR)
    fr = np.flatnonzero(lab["voiced"].astype(bool) & lab["sonorant"].astype(bool) & np.isfinite(lab["formants"][:, 0])
                        & (ok.sum(1) >= 4))[::5]
    fr = fr[fr < len(db)]
    if not len(fr):
        return []
    Fc = lab["formants"][fr]
    Fa = lab.get("formants_acoustic", lab["formants"])[fr] if coupled else Fc
    Ba = lab.get("bandwidths_acoustic", lab["bandwidths"])[fr] if coupled else lab["bandwidths"][fr]
    N = len(fr)

    def tensors(cents):  # cents (M, N)
        M = cents.shape[0]
        poles = np.repeat(Fa[None], M, 0).astype(np.float64)
        poles[:, :, 0] = Fc[None, :, 0] * 2 ** (cents / 1200)
        bws = np.repeat(Ba[None], M, 0).astype(np.float64)
        tgt = torch.tensor(np.repeat(db[fr][None], M, 0), dtype=torch.float64)
        val = torch.tensor(np.repeat(ok[fr][None], M, 0))
        f0 = torch.tensor(np.repeat(lab["f0"][fr][None], M, 0), dtype=torch.float64)
        return tgt, val, f0, poles, bws

    cc = np.repeat(COARSE[:, None], N, 1).astype(float)
    per_c, _ = fit(*tensors(cc))
    jc = np.argmin(np.where(np.isfinite(per_c), per_c, np.inf), 0)
    edge = (jc == 0) | (jc == len(COARSE) - 1)
    fc = np.clip(COARSE[jc][None, :] + FINE[:, None], -400, 400).astype(float)
    per_f, nz_f = fit(*tensors(fc))
    jf = np.argmin(np.where(np.isfinite(per_f), per_f, np.inf), 0)
    rec_c = fc[jf, np.arange(N)]
    nz = nz_f[jf, np.arange(N)]
    # at the label F1: the loss with the sections free and frozen (cancelled; Rd only)
    l_free, _ = fit(*tensors(np.zeros((1, N))))
    l_frozen, _ = fit(*tensors(np.zeros((1, N))), free=False)
    meta = clip.meta
    kind = "tube" if meta.get("tube") else ("cascade+extras" if (meta.get("n_extra_poles") or 0) > 0 else "cascade")
    rows = []
    for n in range(N):
        f1l, f1r = Fc[n, 0], Fc[n, 0] * 2 ** (rec_c[n] / 1200)
        rows.append({"f0": float(lab["f0"][fr[n]]), "kind": kind, "coupled": coupled, "closed": float(np.log2(Fc[n, 0])),
                     "acoustic": float(np.log2(Fa[n, 0])), "rec": float(np.log2(f1r)), "edge": bool(edge[n]),
                     "pole_at_label": bool(np.any(np.abs(nz[n, :2] / f1l - 1) <= 0.1)),
                     "pole_at_rec": bool(np.any(np.abs(nz[n, :2] / f1r - 1) <= 0.1)),
                     "zero_at_label": bool(np.any(np.abs(nz[n, 2:4] / f1l - 1) <= 0.1)),
                     "nz": nz[n].tolist(), "l_free": float(l_free[0, n]), "l_frozen": float(l_frozen[0, n])})
    return rows


if __name__ == "__main__":
    from d102_render_check import BANDS, load_cfg
    from vocal_analysis.train.data import split_ids

    n_clips = int(next((a for a in sys.argv[1:] if a.isdigit()), 200))
    jobs = []
    for root, coupled in (("data/synth/train_v2", False), ("data/synth/train_v2ku", True)):
        idx, _ = load_cfg(root)
        jobs += [(root, coupled, int(i)) for i in split_ids(idx["n_clips"])[1][:n_clips]]
    if "--serial" in sys.argv:
        got = [one(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(8) as ex:
            got = list(ex.map(one, jobs, chunksize=1))
    rows = [r for g in got for r in g]

    def med(v):
        return f"{np.median(v):+6.1f}" if len(v) else "   n/a"

    L = ["D-106: D-105's check with SoFilNet's nuisance structure (2 high poles above F4, 2 broad poles >= 1 kHz wide, 2 zeros "
         "250-4000 Hz; fitted per frame and F1 candidate), Rd continuous; "
         "SNR-gated targets 10 dB (cents; median over frames)", ""]
    gate = []
    for coupled in (False, True):
        sub = [r for r in rows if r["coupled"] == coupled]
        L.append(("coupled tube clips: recovered - closed | - acoustic" if coupled else "uncoupled clips: recovered - label")
                 + f"; {len(sub)} frames")
        for kd in ["all"] + sorted({r["kind"] for r in sub}):
            for lo, hi in BANDS:
                s = [r for r in sub if (kd == "all" or r["kind"] == kd) and lo <= r["f0"] < hi]
                if len(s) < 20:
                    continue
                band = f"f0 {lo:.0f}-{hi if hi < 1e8 else 'up'}"
                rc = 1200 * np.array([r["rec"] - r["closed"] for r in s])
                e = 100 * np.mean([r["edge"] for r in s])
                pl = 100 * np.mean([r["pole_at_label"] for r in s])
                pr = 100 * np.mean([r["pole_at_rec"] for r in s])
                zl = 100 * np.mean([r["zero_at_label"] for r in s])
                lf, lz = np.nanmedian([r["l_free"] for r in s]), np.nanmedian([r["l_frozen"] for r in s])
                extra = (f" | - acoustic {med(1200 * np.array([r['rec'] - r['acoustic'] for r in s]))}" if coupled else "")
                L.append(f"  {kd:15s} {band:12s} n {len(s):5d}: {med(rc)}{extra} (|.| {np.median(np.abs(rc)):5.1f}, edge {e:4.1f} %); "
                         f"broad pole within 10 % of label F1 {pl:4.1f} % (of recovered {pr:4.1f} %), zero within 10 % of label F1 {zl:4.1f} %; loss at label F1 with sections / without {lf:.2f} / {lz:.2f}")
                if not coupled and kd == "all" and len(s) >= 50 and hi <= 400:
                    gate.append((band, float(np.median(rc)), float(np.median(np.abs(rc))), e))
        L.append("")
    nzs = np.array([r["nz"] for r in rows if not r["coupled"]])
    if len(nzs):
        q = np.percentile(nzs, [25, 50, 75], 0)
        L.append("reported: fitted sections at the minimum, uncoupled, quartiles (Hz): "
                 + "; ".join(f"{nm} {q[0, k]:.0f} / {q[1, k]:.0f} / {q[2, k]:.0f}" for k, nm in enumerate(
                     ("broad pole 1 F", "broad pole 2 F", "zero 1 F", "zero 2 F", "broad pole 1 B", "broad pole 2 B", "zero 1 B", "zero 2 B"))))
        L.append("")
    ok = all(abs(m) <= 10 and a <= 10 and e <= 10 for _, m, a, e in gate)
    L.append("GATE: " + ("PASSES" if ok else "FAILS") + " (uncoupled, bands below 400 Hz: "
             + "; ".join(f"{b} median {m:+.1f} c, |.| {a:.1f} c, edge {e:.1f} %" for b, m, a, e in gate) + ")")
    print("\n".join(L))
