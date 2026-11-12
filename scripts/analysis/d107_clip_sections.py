"""D-107: extra sections held per clip. Four pole-zero pairs per clip (150-4000 Hz, 40-400 Hz wide, cancelling start at
400 / 900 / 1500 / 2300 Hz, 150 Hz wide), a pair-cancelling penalty 0.1 x sum((log2 pole f - log2 zero f)^2 +
(log2 pole b - log2 zero b)^2) added to the mean per-frame loss; four alternating rounds: (a) sections fixed, per frame
F1 on D-104's grid with Rd searched; (b) F1 fixed, sections (shared over the clip) and per-frame Rd by Adam (0.05, 200
steps). Recovered F1 = round 4's (a). Clips, frames and targets as D-104. CPU, clips in parallel.

    python scripts/analysis/d107_clip_sections.py [n_clips] [--serial] > reports/analysis/d107/d107.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SNR, ROUNDS, STEPS, LR, PEN = 10.0, 4, 200, 0.05, 0.1
F_BAND, B_BAND, RD_BAND = (150.0, 4000.0), (40.0, 400.0), (0.3, 2.7)
START_F, START_B = (400.0, 900.0, 1500.0, 2300.0), 150.0
GRID = np.arange(-400, 401, 5)


def lg(v, lo, hi):
    s = (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(np.log(s / (1 - s)))


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
    N, C = len(fr), len(GRID)
    T = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    tgt1, val1, f01 = T(db[fr]), torch.tensor(ok[fr]), T(lab["f0"][fr])
    sq = lambda x, lo, hi: torch.exp(np.log(lo) + torch.sigmoid(x) * (np.log(hi) - np.log(lo)))  # noqa: E731
    u = torch.tensor([[lg(f, *F_BAND) for f in START_F], [lg(START_B, *B_BAND)] * 4,
                      [lg(f, *F_BAND) for f in START_F], [lg(START_B, *B_BAND)] * 4], dtype=torch.float64)  # pole f, pole b, zero f, zero b

    def sections():
        return sq(u[0], *F_BAND), sq(u[1], *B_BAND), sq(u[2], *F_BAND), sq(u[3], *B_BAND)

    def grid_f1(pf, pb, zf, zb):
        """per frame: F1 (cents from the closed label) minimising the loss over the grid, Rd searched"""
        poles = np.repeat(Fa[None], C, 0).astype(np.float64)
        poles[:, :, 0] = Fc[None, :, 0] * 2 ** (GRID[:, None] / 1200)
        P = torch.cat([T(poles), pf.expand(C, N, 4)], -1)
        B = torch.cat([T(np.repeat(Ba[None], C, 0)), pb.expand(C, N, 4)], -1)
        tg, va, f0 = tgt1.expand(C, N, -1), val1.expand(C, N, -1), f01.expand(C, N)
        best = None
        for r in R.RD_SEARCH:
            _, per = R.reconstruction_loss(tg, va, f0, P, B, zf.expand(C, N, 4), zb.expand(C, N, 4), source="rd",
                                           profile="offset", rd=torch.full_like(f0, float(r)))
            per = torch.nan_to_num(per, nan=np.inf)
            best = per if best is None else torch.minimum(best, per)
        j = best.argmin(0).numpy()
        return GRID[j], (j == 0) | (j == C - 1), best.min(0).values.numpy()

    def loss_at(cents, pf, pb, zf, zb, rd):
        poles = Fa.astype(np.float64).copy()
        poles[:, 0] = Fc[:, 0] * 2 ** (cents / 1200)
        P = torch.cat([T(poles), pf.expand(N, 4)], -1)
        B = torch.cat([T(Ba), pb.expand(N, 4)], -1)
        return R.reconstruction_loss(tgt1, val1, f01, P, B, zf.expand(N, 4), zb.expand(N, 4), source="rd", profile="offset", rd=rd)[1]

    s0 = (1.0 - RD_BAND[0]) / (RD_BAND[1] - RD_BAND[0])
    with torch.no_grad():
        cents, edge, _ = grid_f1(*sections())
    for _ in range(ROUNDS):
        u.requires_grad_(True)
        ur = torch.full((N,), float(np.log(s0 / (1 - s0))), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([u, ur], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            pf, pb, zf, zb = sections()
            per = loss_at(cents, pf, pb, zf, zb, RD_BAND[0] + (RD_BAND[1] - RD_BAND[0]) * torch.sigmoid(ur))
            pen = PEN * ((torch.log2(pf) - torch.log2(zf)) ** 2 + (torch.log2(pb) - torch.log2(zb)) ** 2).sum()
            (torch.nanmean(per) + pen).backward()
            opt.step()
        u = u.detach()
        with torch.no_grad():
            cents, edge, _ = grid_f1(*sections())
    with torch.no_grad():
        pf, pb, zf, zb = sections()
        l_sec = grid_f1(pf, pb, zf, zb)[2]  # the grid minimum's loss, with the fitted sections
        poles = Fa.astype(np.float64)
        l_none = np.full(N, np.inf)
        for r in R.RD_SEARCH:  # at the label F1, without sections (D-104's render)
            _, per = R.reconstruction_loss(tgt1, val1, f01, T(poles), T(Ba), source="rd", profile="offset",
                                           rd=torch.full_like(f01, float(r)))
            l_none = np.minimum(l_none, torch.nan_to_num(per, nan=np.inf).numpy())
    ex = clip.meta.get("extras") or {}
    tp = np.asarray(ex.get("extra_pole_f", []), float)
    kind = {0: "none", 1: "nasal", 3: "subglottal", 4: "nasal+subglottal"}.get(len(tp), f"{len(tp)} pairs")
    near = (float(np.median([np.min(np.abs(1200 * np.log2(pf.numpy() / p))) for p in tp])) if len(tp) else np.nan)
    f1c = 1200 * np.log2(Fc[:, 0])
    steady = bool(np.percentile(f1c, 90) - np.percentile(f1c, 10) < 100)
    return [{"f0": float(lab["f0"][fr[n]]), "kind": kind, "engine": "tube" if clip.meta.get("tube") else "cascade",
             "coupled": coupled, "steady": steady, "closed": float(np.log2(Fc[n, 0])), "acoustic": float(np.log2(Fa[n, 0])),
             "rec": float(np.log2(Fc[n, 0] * 2 ** (cents[n] / 1200))), "edge": bool(edge[n]), "near": near,
             "l_sec": float(l_sec[n]), "l_none": float(l_none[n])} for n in range(N)]


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

    def line(s, coupled):
        rc = 1200 * np.array([r["rec"] - r["closed"] for r in s])
        e = 100 * np.mean([r["edge"] for r in s])
        extra = f" | - acoustic {med(1200 * np.array([r['rec'] - r['acoustic'] for r in s]))}" if coupled else ""
        return (f"{med(rc)}{extra} (|.| {np.median(np.abs(rc)):5.1f}, edge {e:4.1f} %); loss at minimum {np.median([r['l_sec'] for r in s]):.2f}, "
                f"at label without sections {np.median([r['l_none'] for r in s]):.2f}"), rc, e

    L = ["D-107: per-clip extra sections (4 pole-zero pairs, penalised towards cancelling), alternating fit; Rd per frame; "
         "SNR-gated targets (cents; median over frames)", ""]
    gate = []
    for coupled in (False, True):
        sub = [r for r in rows if r["coupled"] == coupled]
        L.append(("coupled tube clips: recovered - closed | - acoustic" if coupled else "uncoupled clips: recovered - label") + f"; {len(sub)} frames")
        groups = [("all", lambda r: True)] + [(f"extras {k}", (lambda k: lambda r: r["kind"] == k)(k)) for k in sorted({r["kind"] for r in sub})] \
            + [("steady vowel", lambda r: r["steady"]), ("moving vowel", lambda r: not r["steady"])]
        for gname, fn in groups:
            for lo, hi in BANDS:
                s = [r for r in sub if fn(r) and lo <= r["f0"] < hi]
                if len(s) < 20:
                    continue
                txt, rc, e = line(s, coupled)
                L.append(f"  {gname:24s} f0 {lo:.0f}-{hi if hi < 1e8 else 'up':<5} n {len(s):5d}: {txt}")
                if not coupled and gname == "all" and len(s) >= 50 and hi <= 400:
                    gate.append((f"f0 {lo:.0f}-{hi:.0f}", float(np.median(rc)), float(np.median(np.abs(rc))), e))
        L.append("")
    cl = {}
    for r in rows:
        if not r["coupled"] and np.isfinite(r["near"]):
            cl[(r["kind"], r["near"])] = r["kind"]
    if cl:
        L.append("reported: fitted sections against the true extras (uncoupled clips with extras): median over clips of the "
                 "median distance from each true pole to the nearest fitted pole (cents): "
                 + "; ".join(f"{k} {np.median([n for (kk, n) in cl if kk == k]):.0f}" for k in sorted(set(cl.values()))))
        L.append("")
    ok = all(abs(m) <= 10 and a <= 10 and e <= 10 for _, m, a, e in gate)
    L.append("GATE: " + ("PASSES" if ok else "FAILS") + " (uncoupled, bands below 400 Hz: "
             + "; ".join(f"{b} median {m:+.1f} c, |.| {a:.1f} c, edge {e:.1f} %" for b, m, a, e in gate) + ")")
    print("\n".join(L))
