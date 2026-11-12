"""D-103: D-102's render check with an LF source by Rd. The same clips, frames and F1 grid; for each F1 candidate the
frame's loss is minimised over Rd (0.3-2.7, 0.05 steps). Modes: rd (level profiled; primary), rd_tilt (level and
slope), tilt (D-102's). Reports each frame's best Rd at the label F1 against the label Rd. CPU, clips in parallel.

    python scripts/analysis/d103_rd_check.py [n_clips] [--serial] [--snr DB] [--poles N] [--fmax HZ] [--hpc] > reports/analysis/d103/d103.txt

--poles N renders only the first N poles (D-109's pre-check: the model predicts F1-F4); --fmax HZ caps the targets
(default 5000 Hz, as before); --hpc replaces F5-F8 with reconstruction.upper_poles from the label F4.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SNR = float(sys.argv[sys.argv.index("--snr") + 1]) if "--snr" in sys.argv else None  # D-104
NPOLES = int(sys.argv[sys.argv.index("--poles") + 1]) if "--poles" in sys.argv else None  # D-109 pre-check
HPC = "--hpc" in sys.argv  # D-109 pre-check: F5-F8 replaced by reconstruction.upper_poles from the label F4
FMAX = float(sys.argv[sys.argv.index("--fmax") + 1]) if "--fmax" in sys.argv else 5000.0
MODES = {"rd": ("rd", "offset"), "rd_tilt": ("rd", "tilt"), "tilt": ("profile", "tilt")}


def one(job, snr=None):
    import torch

    torch.set_num_threads(1)
    from d102_render_check import GRID, load_cfg
    from vocal_analysis.synth.sampler import generate
    from vocal_analysis.train import reconstruction as R

    root, coupled, i = job[:3]
    snr = job[3] if len(job) > 3 else snr
    npoles, fmax = (job[4], job[5]) if len(job) > 5 else (None, 5000.0)
    hpc = job[6] if len(job) > 6 else False
    idx, cfg = load_cfg(root)
    clip = generate(int(i), cfg, idx["seed"])
    lab = clip.labels
    db, ok = R.harmonic_targets(clip.audio, lab["f0"], lab["voiced"].astype(bool), fmax=fmax, k_max=64, snr_min_db=snr)
    ok0 = R.harmonic_targets(clip.audio, lab["f0"], lab["voiced"].astype(bool), fmax=fmax, k_max=64)[1] if snr is not None else ok
    fr = np.flatnonzero(lab["voiced"].astype(bool) & lab["sonorant"].astype(bool) & np.isfinite(lab["formants"][:, 0])
                        & (ok.sum(1) >= 4))[::5]
    fr = fr[fr < len(db)]
    if not len(fr):
        return []
    Fc = lab["formants"][fr]
    Fa = lab.get("formants_acoustic", lab["formants"])[fr] if coupled else Fc
    Ba = lab.get("bandwidths_acoustic", lab["bandwidths"])[fr] if coupled else lab["bandwidths"][fr]
    if npoles is not None:
        Fa, Ba = Fa[:, :npoles], Ba[:, :npoles]
    if hpc:
        uf, ub = R.upper_poles(torch.tensor(Fa[:, 3], dtype=torch.float64), torch.tensor(lab["f0"][fr], dtype=torch.float64))
        Fa, Ba = np.concatenate([Fa[:, :4], uf.numpy()], 1), np.concatenate([Ba[:, :4], ub.numpy()], 1)
    C, N = len(GRID), len(fr)
    poles = np.repeat(Fa[None], C, 0).astype(np.float64)
    poles[:, :, 0] = Fc[None, :, 0] * 2 ** (GRID[:, None] / 1200)
    P, B = torch.tensor(poles), torch.tensor(np.repeat(Ba[None], C, 0).astype(np.float64))
    tgt = torch.tensor(np.repeat(db[fr][None], C, 0), dtype=torch.float64)
    val = torch.tensor(np.repeat(ok[fr][None], C, 0))
    f0 = torch.tensor(np.repeat(lab["f0"][fr][None], C, 0), dtype=torch.float64)
    rec, edge = {}, {}
    best_rd = np.full(N, np.nan)
    for name, (source, prof) in MODES.items():
        if source == "rd":
            per_rd = []
            for r in R.RD_SEARCH:
                _, per = R.reconstruction_loss(tgt, val, f0, P, B, profile=prof, source="rd", rd=torch.full_like(f0, float(r)))
                per_rd.append(per.numpy())
            per_rd = np.array(per_rd)  # (n_rd, C, N)
            per = np.nanmin(np.where(np.isfinite(per_rd), per_rd, np.inf), 0)
            if name == "rd":
                best_rd = R.RD_SEARCH[np.argmin(np.where(np.isfinite(per_rd[:, C // 2]), per_rd[:, C // 2], np.inf), 0)]
        else:
            _, per = R.reconstruction_loss(tgt, val, f0, P, B, profile=prof)
            per = np.where(np.isfinite(per.numpy()), per.numpy(), np.inf)
        j = np.argmin(per, 0)
        rec[name], edge[name] = np.log2(poles[j, np.arange(N), 0]), (j == 0) | (j == C - 1)
    meta = clip.meta
    kind = "tube" if meta.get("tube") else ("cascade+extras" if (meta.get("n_extra_poles") or 0) > 0 else "cascade")
    rd_lab = lab["rd"][fr]
    return [{"f0": float(lab["f0"][fr[n]]), "kind": kind, "coupled": coupled, "closed": float(np.log2(Fc[n, 0])),
             "acoustic": float(np.log2(Fa[n, 0])), **{f"rec_{m}": float(rec[m][n]) for m in MODES},
             **{f"edge_{m}": bool(edge[m][n]) for m in MODES}, "rd_best": float(best_rd[n]), "rd_label": float(rd_lab[n]),
             "n_valid0": int(ok0[fr[n]].sum()), "n_valid": int(ok[fr[n]].sum()),
             "n_hi0": int(ok0[fr[n], int(2000 // lab["f0"][fr[n]]):].sum()), "n_hi": int(ok[fr[n], int(2000 // lab["f0"][fr[n]]):].sum())}
            for n in range(N)]


if __name__ == "__main__":
    from d102_render_check import BANDS, load_cfg
    from vocal_analysis.train.data import split_ids

    n_clips = int(next((a for a in sys.argv[1:] if a.isdigit()), 200))
    jobs = []
    for root, coupled in (("data/synth/train_v2", False), ("data/synth/train_v2ku", True)):
        idx, _ = load_cfg(root)
        jobs += [(root, coupled, int(i), SNR, NPOLES, FMAX, HPC) for i in split_ids(idx["n_clips"])[1][:n_clips]]
    if "--serial" in sys.argv:
        got = [one(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(8) as ex:
            got = list(ex.map(one, jobs, chunksize=2))
    rows = [r for g in got for r in g]

    def med(v):
        return f"{np.median(v):+6.1f}" if len(v) else "   n/a"

    L = [(f"D-109 pre-check: poles F1-F{NPOLES} only, targets to {FMAX:g} Hz. " if NPOLES or FMAX != 5000 else "")
         + ("F5-F8 by the higher-pole correction from the label F4. " if HPC else "")
         + ("D-104 (SNR-gated targets, %g dB): " % SNR if SNR is not None else "") + "D-103: the reconstruction loss's F1 minimum with every other parameter at its label, LF source by Rd "
         "(cents; median over frames; Rd minimised per F1 candidate)", ""]
    gate = []
    for coupled in (False, True):
        sub = [r for r in rows if r["coupled"] == coupled]
        L.append(("coupled tube clips (train_v2ku held-out): recovered - closed | - acoustic" if coupled
                  else "uncoupled clips (train_v2 validation): recovered - label") + f"; {len(sub)} frames")
        for kd in ["all"] + sorted({r["kind"] for r in sub}):
            for lo, hi in BANDS:
                s = [r for r in sub if (kd == "all" or r["kind"] == kd) and lo <= r["f0"] < hi]
                if len(s) < 20:
                    continue
                band = f"f0 {lo:.0f}-{hi if hi < 1e8 else 'up'}"
                cells = []
                for m in MODES:
                    rc = 1200 * np.array([r[f"rec_{m}"] - r["closed"] for r in s])
                    e = 100 * np.mean([r[f"edge_{m}"] for r in s])
                    if coupled:
                        ra = 1200 * np.array([r[f"rec_{m}"] - r["acoustic"] for r in s])
                        cells.append(f"{m} {med(rc)} | {med(ra)} (edge {e:4.1f} %)")
                    else:
                        cells.append(f"{m} {med(rc)} (|.| {np.median(np.abs(rc)):5.1f}, edge {e:4.1f} %)")
                    if not coupled and kd == "all" and m == "rd" and len(s) >= 50 and hi <= 400:
                        gate.append((band, float(np.median(rc)), float(np.median(np.abs(rc))), e))
                L.append(f"  {kd:15s} {band:12s} n {len(s):5d}: " + "; ".join(cells))
        L.append("")
    lf = [r for r in rows if not r["coupled"] and np.isfinite(r["rd_label"]) and np.isfinite(r["rd_best"])]
    if lf:
        d = np.array([r["rd_best"] - r["rd_label"] for r in lf])
        L.append(f"reported: best Rd at the label F1 (mode rd) - label Rd, uncoupled: median {np.median(d):+.2f}, "
                 f"median |.| {np.median(np.abs(d)):.2f} ({len(d)} frames)")
        L.append("")
    if SNR is not None:
        L.append(f"reported: harmonics removed by the SNR gate ({SNR:g} dB), uncoupled frames (all / above 2 kHz):")
        for lo, hi in BANDS:
            s = [r for r in rows if not r["coupled"] and lo <= r["f0"] < hi]
            if s:
                a0, a1 = sum(r["n_valid0"] for r in s), sum(r["n_valid"] for r in s)
                h0, h1 = sum(r["n_hi0"] for r in s), sum(r["n_hi"] for r in s)
                L.append(f"  f0 {lo:.0f}-{hi if hi < 1e8 else 'up'}: {100 * (1 - a1 / max(a0, 1)):.0f} % / {100 * (1 - h1 / max(h0, 1)):.0f} %")
        L.append("")
    ok = all(abs(m) <= 10 and a <= 10 and e <= 10 for _, m, a, e in gate)
    L.append("GATE: " + ("PASSES" if ok else "FAILS") + " (mode rd, uncoupled, bands below 400 Hz: "
             + "; ".join(f"{b} median {m:+.1f} c, |.| {a:.1f} c, edge {e:.1f} %" for b, m, a, e in gate) + ")")
    print("\n".join(L))
