"""D-123's registered check: which change to the reconstruction loss removes its jump across F1 = 2 f0, and is what is
left what the closed-phase reference asks for? No training; the branch-point models d040_5b_s{1,2}/ckpt_016000.pt.

Candidates (each one change to D-109's loss): base; A (an offset per harmonic number k = 1-8 per 200-frame crop); C (a
smooth curve in log frequency per recording); A+C (both per recording); B' (the offset per harmonic number per recording
and third-octave f0 bin, fitted on identifiable frames, held fixed on hard frames with F1 free); B'-fixed (B1 at the
prior and Rd at the recording's median on hard frames); D (the noise's power out of the targets; reported only).

The measure: the model's F1 scaled by c cents on one cell's frames at a time; the objective is each recording's total loss
over its loss-carrying frames, so a shared term's effect on the other frames counts (B''s hard-frame offset stays at its
unshifted value, as detached). Cells: F1 band x F1 / f0 by the scanned model's own F1. Data: 96 recordings per recon
store, each whole as five 200-frame crops; D-119's synthetic control; PTDB-TUG dev (whole utterances, targets at the
laryngograph f0, scored on D-121's closed-phase frames).

    VA_DEVICE=cuda python scripts/analysis/d123_check.py > reports/analysis/d123/check.txt
    D123_SMOKE=1: two recordings / batches / utterances; prints the code checks only (no tables).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d114_stretch_scan import argmin_parabola  # noqa: E402

C_GRID = np.arange(-200.0, 200.0 + 1e-9, 25.0)
C0 = int(np.argmin(np.abs(C_GRID)))
K, NK, FMAX, HOP_S = 64, 8, 4000.0, 0.010
CANDS = ("base", "A", "C", "A+C", "Bp", "Bp_fixed", "D")
NAMES = {"Bp": "B'", "Bp_fixed": "B'-fixed"}
F1_CUTS, F1_NAMES = (450.0, 600.0), ("F1 < 450", "F1 450-600", "F1 >= 600")
H_CUTS, H_NAMES = (1.5, 2.0, 2.5, 3.5), ("< 1.5", "1.5-2", "2-2.5", "2.5-3.5", ">= 3.5")
N_CELL = len(F1_NAMES) * len(H_NAMES)
F0_CUTS, F0_NAMES = (150.0, 250.0), ("f0 < 150", "f0 150-250", "f0 >= 250")
N_REC, REC_SEED, N_SYN, SYN_SEED = 96, 1123, 24, 1119
MIN_FRAMES, PT_MIN_FRAMES, PT_MIN_SPK, CURV_MIN = 300, 200, 4, 0.25
N_BOOT, PT_BOOT, BOOT_SEED = 500, 2000, 5123
IDENT_RATIO, SPAN_ST, MIN_IDENT, CHUNK_FRAMES = 2.5, 4.0, 20, 16000
SMOKE = bool(os.environ.get("D123_SMOKE"))
if SMOKE:
    N_REC, N_SYN, N_BOOT, PT_BOOT = 2, 2, 10, 10
SYNTH, PTDB = "train_v2 validation (synthetic control)", "PTDB-TUG dev"


def cname(c):
    return NAMES.get(c, c)


def shift(F, c, sel=None):
    import torch

    s = 2.0 ** (c / 1200.0)
    F1 = F[:, :1] * (s if sel is None else torch.where(sel[:, None], torch.full_like(F[:, :1], s), torch.ones_like(F[:, :1])))
    return torch.cat([F1, F[:, 1:]], -1)


# ------------------------------------------------------------------------------------------------------- targets (CPU)
def _targets_job(job):
    """harmonic targets (D-104's gate), with and without the noise's power taken out (D)"""
    from vocal_analysis.train import reconstruction as R

    x, f0, voiced = job
    fz = np.where(voiced, f0, np.nan)
    db, ok = R.harmonic_targets(x, fz, voiced, k_max=K, snr_min_db=10.0)
    dbD, okD = R.harmonic_targets(x, fz, voiced, k_max=K, snr_min_db=10.0, noise_subtract=True)
    return db.astype(np.float32), ok, dbD.astype(np.float32), okD


def run_jobs(jobs, workers=6):
    from concurrent.futures import ProcessPoolExecutor

    if SMOKE or len(jobs) < 4:
        return [_targets_job(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_targets_job, jobs, chunksize=1))


# ------------------------------------------------------------------------------------------------------- packs
def crop_audio(x, s, n):
    from vocal_analysis.features.vqt import N_MAX
    from vocal_analysis.train.reconstruction import HOP

    a0 = s * HOP - N_MAX // 2
    L = (n - 1) * HOP + N_MAX
    out = np.zeros(L, np.float32)
    lo, hi = max(0, a0), min(len(x), a0 + L)
    if hi > lo:
        out[lo - a0:hi - a0] = x[lo:hi]
    return out


def model_out(models, vq, audio, n_frames, dev):
    """{seed: (F (B, T, 4), B (B, T, 4))} for a batch of context-padded audio"""
    import torch

    from vocal_analysis.train import reconstruction as R

    res = {}
    with torch.no_grad():
        a = torch.from_numpy(np.asarray(audio)).to(dev)
        x = vq.log_power_with_context(a, n_frames)
        for s, m in models.items():
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                o = m(x)
            res[s] = (R.decode_formants(o["formant_logits"])[..., :4].float().cpu().numpy(),
                      torch.exp(o["logb_mean"].float())[..., :4].cpu().numpy())
    return res


def finish_pack(parts, models):
    """concatenate per-recording parts into one pack of frames"""
    P = {k: np.concatenate([p[k] for p in parts]) for k in ("db", "ok", "dbD", "okD", "f0", "mask", "maskD", "score", "crop_span")}
    P["rec"] = np.concatenate([np.full(len(p["f0"]), i) for i, p in enumerate(parts)])
    off, crops = 0, []
    for p in parts:
        crops.append(p["crop"] + off)
        off += p["crop"].max() + 1
    P["crop"] = np.concatenate(crops)
    P["F"] = {s: np.concatenate([p["F"][s] for p in parts]) for s in models}
    P["B"] = {s: np.concatenate([p["B"][s] for p in parts]) for s in models}
    for k in ("cp", "sex", "spk"):
        if k in parts[0]:
            P[k] = np.concatenate([p[k] for p in parts])
    P["n_rec"] = len(parts)
    P["rec_spk"] = [p.get("spk_name") for p in parts]
    return P


def f0_span_ok(f0, voiced, crop):
    """per frame: its crop's voiced f0 spans at least SPAN_ST semitones (5th to 95th percentile)"""
    out = np.zeros(len(f0), bool)
    for c in np.unique(crop):
        i = (crop == c) & voiced & (f0 > 0)
        if i.sum() >= 10:
            lo, hi = np.percentile(f0[i], [5, 95])
            out[crop == c] = 12 * np.log2(hi / lo) >= SPAN_ST
    return out


def store_pack(store, models, vq, dev):
    import torch  # noqa: F401

    files = sorted(Path(store).glob("*.npz"))
    pick = sorted(np.random.default_rng(REC_SEED).choice(len(files), N_REC, replace=False))
    raw = [dict(np.load(files[i])) for i in pick]
    jobs = [(z["audio"].astype(np.float64), z["f0"].astype(np.float64)[:len(z["mask"])], z["ok"].any(1)) for z in raw]
    tg = run_jobs(jobs)
    parts = []
    for z, (db, ok, dbD, okD) in zip(raw, tg):
        T = len(z["mask"])
        nc = max(1, min(5, T // 200))
        n = nc * 200
        aud = [crop_audio(z["audio"], j * 200, 200) for j in range(nc)]
        mo = model_out(models, vq, aud, 200, dev)

        def fit(v, fill=0):
            v = v[:n]
            return np.concatenate([v, np.full((n - len(v), *v.shape[1:]), fill, v.dtype)]) if len(v) < n else v
        f0 = fit(z["f0"].astype(np.float32))
        mask = fit(z["mask"], False)
        okD_ = fit(okD, False)
        p = {"db": fit(z["db"].astype(np.float32)), "ok": fit(z["ok"], False), "dbD": fit(np.nan_to_num(dbD)), "okD": okD_,
             "f0": f0, "mask": mask, "maskD": mask & (okD_.sum(1) >= 4), "score": mask.copy(), "crop": np.repeat(np.arange(nc), 200),
             "F": {s: mo[s][0].reshape(n, 4) for s in models}, "B": {s: mo[s][1].reshape(n, 4) for s in models}}
        p["crop_span"] = f0_span_ok(f0, mask, p["crop"])
        parts.append(p)
    return finish_pack(parts, models)


def synth_pack(models, vq, dev):
    from torch.utils.data import DataLoader

    from vocal_analysis.features.vqt import N_MAX
    from vocal_analysis.train.data import SynthCrops, collate, split_ids

    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    syn = SynthCrops("data/synth/train_v2/flat", split_ids(idx["n_clips"])[1], n_frames=200, seed=SYN_SEED, epoch_size=16 * N_SYN)
    parts = []
    for a, lab in DataLoader(syn, 16, num_workers=0 if SMOKE else 4, collate_fn=collate):
        mo = model_out(models, vq, a.numpy(), 200, dev)
        jobs = []
        for b in range(a.shape[0]):
            x = a[b].float().numpy()
            f0 = lab["f0"][b].numpy().astype(np.float64)
            v = lab["voiced"][b].numpy().astype(bool) & np.isfinite(f0)
            pad = np.concatenate([x[N_MAX // 2:], np.zeros(N_MAX // 2, np.float32)]).astype(np.float64)
            jobs.append((pad, np.nan_to_num(f0), v))
        tg = run_jobs(jobs)
        for b, (db, ok, dbD, okD) in enumerate(tg):
            T = lab["f0"].shape[1]
            son = (lab["voiced"][b] & lab["sonorant"][b]).numpy().astype(bool)
            f0 = np.nan_to_num(lab["f0"][b].numpy()).astype(np.float32)
            ok, okD = ok[:T], okD[:T]
            m = son & (ok.sum(1) >= 4)
            p = {"db": np.nan_to_num(db[:T]), "ok": ok, "dbD": np.nan_to_num(dbD[:T]), "okD": okD, "f0": f0, "mask": m,
                 "maskD": son & (okD.sum(1) >= 4), "score": m.copy(), "crop": np.zeros(T, int),
                 "F": {s: mo[s][0][b] for s in models}, "B": {s: mo[s][1][b] for s in models}}
            p["crop_span"] = f0_span_ok(f0, m, p["crop"])
            parts.append(p)
    return finish_pack(parts, models)


def ptdb_pack(models, vq, dev):
    import soundfile as sf
    from scipy.signal import resample_poly

    from vocal_analysis.train.reconstruction import HOP

    d = dict(np.load("cache/m3w/ptdb_egg_sub.npz", allow_pickle=False))
    tiers = json.load(open("vocal_analysis/eval/tiers.json"))["speakers"]
    keep = np.array([tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    keep &= np.isfinite(d["cp"][:, 0]) & np.isfinite(d["cycle"][:, 0]) & np.isfinite(d["praat"][:, 0]) & (d["f0"] > 0)
    utts = sorted({(d["speaker"][i], d["utt"][i]) for i in np.nonzero(keep)[0]})
    if SMOKE:
        utts = utts[:2]
    raw, jobs = [], []
    for spk, utt in utts:
        sexdir = "FEMALE" if spk[0] == "F" else "MALE"
        x, sr = sf.read(Path("data/ptdb_tug") / sexdir / "MIC" / spk / f"{utt}.wav", dtype="float64")
        if x.ndim > 1:
            x = x.mean(1)
        x16 = resample_poly(x, 1, sr // 16000)
        ref = np.loadtxt(Path("data/ptdb_tug") / sexdir / "REF" / spk / (utt.replace("mic_", "ref_", 1) + ".f0"), ndmin=2)[:, 0]
        nf = len(x16) // HOP + 1
        f0 = np.zeros(nf)
        j = np.arange(len(ref)) + 2  # reference frame i is centred at 20 ms + i * 10 ms
        ok_ = j < nf
        f0[j[ok_]] = ref[ok_]
        raw.append((spk, utt, x16, f0, nf))
        jobs.append((x16, f0, f0 > 0))
    tg = run_jobs(jobs)
    parts = []
    for (spk, utt, x16, f0, nf), (db, ok, dbD, okD) in zip(raw, tg):
        aud = crop_audio(x16.astype(np.float32), 0, nf)
        mo = model_out(models, vq, aud[None], nf, dev)
        sel = np.nonzero(keep & (d["speaker"] == spk) & (d["utt"] == utt))[0]
        fr = np.round(d["t"][sel] / HOP_S).astype(int)
        score = np.zeros(nf, bool)
        cp = np.full(nf, np.nan)
        inb = fr < nf
        score[fr[inb]] = True
        cp[fr[inb]] = d["cp"][sel[inb], 0]
        m = (f0 > 0) & (ok.sum(1) >= 4)
        p = {"db": np.nan_to_num(db), "ok": ok, "dbD": np.nan_to_num(dbD), "okD": okD, "f0": f0.astype(np.float32), "mask": m,
             "maskD": (f0 > 0) & (okD.sum(1) >= 4), "score": score & m, "crop": np.arange(nf) // 200,
             "F": {s: mo[s][0][0] for s in models}, "B": {s: mo[s][1][0] for s in models},
             "cp": cp, "sex": np.full(nf, spk[0]), "spk": np.full(nf, spk), "spk_name": spk}
        p["crop_span"] = f0_span_ok(f0, m, p["crop"])
        parts.append(p)
    return finish_pack(parts, models)


# ------------------------------------------------------------------------------------------------------- the loss
def hub_frame(out, w):
    import torch

    n = w.sum(-1)
    return (torch.nn.functional.huber_loss(out * w, torch.zeros_like(out), reduction="none", delta=6.0) * w).sum(-1) / n.clamp_min(1.0)


def level_out(r, w):
    n = w.sum(-1, keepdim=True).clamp_min(1.0)
    return (r - (r * w).sum(-1, keepdim=True) / n) * w


class Chunk:
    """the per-c renders, best Rd and residuals of one chunk of recordings for one seed and one target set"""

    def __init__(self, R, lf, P, sl, seed, dev, use_D=False, tgt_override=None, rdfix_override=None):
        import torch

        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt)  # noqa: E731
        self.rec_g = P["rec"][sl]
        self.rec = t(np.unique(self.rec_g, return_inverse=True)[1], torch.long)
        self.n_rec = int(self.rec.max()) + 1
        self.crop = t(np.unique(P["crop"][sl], return_inverse=True)[1], torch.long)
        self.n_crop = int(self.crop.max()) + 1
        self.f0 = t(P["f0"][sl])
        tg = t(P["dbD" if use_D else "db"][sl]) if tgt_override is None else tgt_override
        ok = t(P["okD" if use_D else "ok"][sl], torch.bool)
        m = t(P["maskD" if use_D else "mask"][sl], torch.bool)
        self.F, self.B = t(P["F"][seed][sl]), t(P["B"][seed][sl])
        kk = torch.arange(1, K + 1, device=dev, dtype=torch.float32)
        self.f = self.f0[:, None] * kk
        okk = ok & (self.f <= FMAX) & (self.f0[:, None] > 0)
        self.m2 = m & (okk.sum(-1) >= 4)
        self.w = (okk & self.m2[:, None]).float()
        self.tg, self.R, self.lf, self.dev = tg, R, lf, dev
        self.ident = (self.F[:, 0] >= IDENT_RATIO * self.f0) | t(P["crop_span"][sl], torch.bool)
        self.per, self.res, self.idx = [], [], []
        for c in C_GRID:
            p, r, i = self.render_rd(shift(self.F, c), self.B)
            self.per.append(p), self.res.append(r), self.idx.append(i)
        # B': groups recording x third-octave f0 bin; offsets fitted on identifiable frames of the unshifted model
        fb = torch.floor(3 * torch.log2(self.f0.clamp_min(50.0) / 50.0)).long()
        g = self.rec * 64 + fb.clamp(0, 63)
        _, self.grp = torch.unique(g, return_inverse=True)
        self.n_grp = int(self.grp.max()) + 1
        self.phiK = R.harmonic_number_basis((len(self.f0), K), NK, device=dev)
        self.phiF = R.log_freq_basis(self.f, 3)
        fitm = self.ident & self.m2
        _, self.coef0 = R.group_basis_out(self.res[C0], self.w, self.phiK, self.grp, self.n_grp, fit=fitm, return_coef=True)
        n_id = torch.zeros(self.n_grp, device=dev).index_add_(0, self.grp, fitm.float())
        self.covered = (~self.ident) & (n_id[self.grp] >= MIN_IDENT)
        # B'-fixed: Rd at the recording's median best Rd over its identifiable frames; B1 at the prior
        rd = torch.as_tensor(R.RD_SEARCH, device=dev, dtype=torch.float32)
        self.rdfix = torch.full((self.n_rec,), -1, dtype=torch.long, device=dev)
        for r_ in range(self.n_rec):
            sel = fitm & (self.rec == r_)
            if sel.any():
                med = rd[self.idx[C0][sel]].median()
                self.rdfix[r_] = torch.argmin((rd - med).abs())
        if rdfix_override is not None:
            self.rdfix = rdfix_override
        self.fixed_ok = (~self.ident) & (self.rdfix[self.rec] >= 0)
        self.res_fix = []
        rdf = self.rdfix[self.rec].clamp_min(0)
        for c in C_GRID:
            Fc = shift(self.F, c)
            Bf = torch.cat([R.hawks_miller_t(Fc[:, :1], self.f0[:, None].clamp_min(50.0)), self.B[:, 1:]], -1)
            self.res_fix.append(level_out((self.tg - self.render(Fc, Bf) - lf[rdf]) * self.w, self.w))

    def render(self, F, B):
        import torch

        uf, ub = self.R.upper_poles(F[:, 3], self.f0)
        okf = self.w > 0
        return self.R.render_db(torch.where(okf, self.f, torch.full_like(self.f, 1000.0)), torch.cat([F, uf], -1),
                                torch.cat([B, ub], -1), None, None, False)

    def render_rd(self, F, B):
        import torch

        rend = self.render(F, B)
        best, idx = None, None
        for j in range(self.lf.shape[0]):
            p = hub_frame(level_out((self.tg - rend - self.lf[j]) * self.w, self.w), self.w)
            if best is None:
                best, idx = p, torch.zeros_like(p, dtype=torch.long)
            else:
                better = p < best
                best, idx = torch.where(better, p, best), torch.where(better, torch.full_like(idx, j), idx)
        return best, (self.tg - rend - self.lf[idx]) * self.w, idx

    def frame_loss(self, cand, sel, ci):
        """(per-frame loss, frames carrying an F1 reading) with sel's frames at grid index ci, the rest unshifted"""
        import torch

        R, w, s1 = self.R, self.w, sel[:, None]
        res = torch.where(s1, self.res[ci], self.res[C0])
        per0 = torch.where(sel, self.per[ci], self.per[C0])
        if cand in ("base", "D"):
            return per0, self.m2
        if cand == "A":
            return hub_frame(R.group_basis_out(res, w, self.phiK, self.crop, self.n_crop) * w, w), self.m2
        if cand == "C":
            return hub_frame(R.group_basis_out(res, w, self.phiF, self.rec, self.n_rec) * w, w), self.m2
        if cand == "A+C":
            return hub_frame(R.group_basis_out(res, w, torch.cat([self.phiK, self.phiF], -1), self.rec, self.n_rec) * w, w), self.m2
        if cand == "Bp":
            out_i = R.group_basis_out(res, w, self.phiK, self.grp, self.n_grp, fit=self.ident & self.m2) * w
            out_h = R.group_basis_out(res, w, self.phiK, self.grp, self.n_grp, coef=self.coef0) * w
            per = torch.where(self.ident, hub_frame(out_i, w), hub_frame(out_h, w))
            return per, self.m2 & (self.ident | self.covered)
        if cand == "Bp_fixed":
            rf = torch.where(s1, self.res_fix[ci], self.res_fix[C0])
            return torch.where(self.ident, per0, hub_frame(rf, w)), self.m2 & (self.ident | self.fixed_ok)
        raise ValueError(cand)


def scan_chunk(ch, cells, cands, n_cells):
    """S[cand] (n_cells + 1, n_rec, n_c): each recording's total loss with cell X's frames shifted by c (X = n_cells: every
    cell frame); cnt[cand] (n_cells + 1, n_rec): the cell's frames carrying an F1 reading"""
    import torch

    S = {c: np.zeros((n_cells + 1, ch.n_rec, len(C_GRID))) for c in cands}
    cnt = {c: np.zeros((n_cells + 1, ch.n_rec)) for c in cands}
    cell_t = torch.as_tensor(cells, device=ch.dev, dtype=torch.long)
    for X in range(n_cells + 1):
        sel = (cell_t >= 0) if X == n_cells else (cell_t == X)
        sel = sel & ch.m2
        if not bool(sel.any()):
            continue
        with torch.no_grad():
            for cand in cands:
                for ci in range(len(C_GRID)):
                    per, wt = ch.frame_loss(cand, sel, ci)
                    obj = torch.zeros(ch.n_rec, device=ch.dev, dtype=torch.float64).index_add_(0, ch.rec, (per * wt).double())
                    S[cand][X, :, ci] = obj.cpu().numpy()
                cnt[cand][X] = torch.zeros(ch.n_rec, device=ch.dev).index_add_(0, ch.rec, (sel & wt).float()).cpu().numpy()
    return S, cnt


def frame_cells(P, seed):
    """cell per frame by the model's own F1 and the frame's f0 (-1 off the scored frames); PTDB-TUG: men offset by N_CELL"""
    F1, f0 = P["F"][seed][:, 0], P["f0"]
    c = np.digitize(F1, F1_CUTS) * len(H_NAMES) + np.digitize(F1 / np.maximum(f0, 1e-6), H_CUTS)
    if "sex" in P:
        c = c + N_CELL * (P["sex"] == "M")
    return np.where(P["score"] & (f0 > 0), c, -1)


def scan_pack(R, lf, P, seed, dev, cands, n_cells, use_D_for=("D",)):
    S = {c: np.zeros((n_cells + 1, P["n_rec"], len(C_GRID))) for c in cands}
    cnt = {c: np.zeros((n_cells + 1, P["n_rec"])) for c in cands}
    cells = frame_cells(P, seed)
    step = max(1, int(CHUNK_FRAMES / (len(P["rec"]) / P["n_rec"])))  # recordings per chunk, about CHUNK_FRAMES frames
    for r0 in range(0, P["n_rec"], step):
        sl = np.nonzero((P["rec"] >= r0) & (P["rec"] < r0 + step))[0]
        recs = np.unique(P["rec"][sl])
        for useD in (False, True):
            cs = [c for c in cands if (c in use_D_for) == useD]
            if not cs:
                continue
            ch = Chunk(R, lf, P, sl, seed, dev, use_D=useD)
            s, n = scan_chunk(ch, cells[sl], cs, n_cells)
            for c in cs:
                S[c][:, recs] = s[c]
                cnt[c][:, recs] = n[c]
            del ch
    return S, cnt


# ------------------------------------------------------------------------------------------------------- statistics
def cstar_curv(S, n):
    """minimising c (parabola) and curvature per frame (second difference over +-100 c) of summed objectives"""
    if n <= 0:
        return np.nan, np.nan
    y = S / n
    return argmin_parabola(C_GRID, y), float(y[C0 + 4] + y[C0 - 4] - 2 * y[C0])


def cell_stats(RES, cand, X, pick=None):
    """seed means of (c, curvature per frame, frames) for cell X, over a draw of recordings"""
    cs, cv, nn = [], [], []
    for s in (1, 2):
        S, n = RES[s][0][cand][X], RES[s][1][cand][X]
        if pick is not None:
            S, n = S[pick], n[pick]
        a, b = cstar_curv(S.sum(0), n.sum())
        cs.append(a), cv.append(b), nn.append(n.sum())
    return float(np.mean(cs)), float(np.mean(cv)), nn


def wmedian(x, w):
    o = np.argsort(x)
    return float(x[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]) if len(x) else float("nan")


# ------------------------------------------------------------------------------------------------------- checks
def code_checks(R, lf, P, dev, out):
    """the registered code checks, on the first recordings of the first store (seed 1)"""
    import torch

    sl = np.nonzero(P["rec"] < 3)[0]
    ch = Chunk(R, lf, P, sl, 1, dev)
    # (a) fast base loss against recon_loss_fb; (b) C with crop groups against recon_loss_fb(channel=3)
    nc = ch.n_crop
    T = len(sl) // nc
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=dt).reshape(nc, T, *np.shape(a)[1:])  # noqa: E731
    L = {"db": t(P["db"][sl]), "ok": t(P["ok"][sl], torch.bool), "f0": t(P["f0"][sl]), "mask": t(P["mask"][sl], torch.bool)}
    Fb, Bb = ch.F.reshape(nc, T, 4), ch.B.reshape(nc, T, 4)
    with torch.no_grad():
        ref = R.recon_loss_fb(Fb, Bb, L, fmax=FMAX, hpc=True, channel=0, per_frame=True).reshape(-1)
        mine = torch.where(ch.m2, ch.per[C0], torch.full_like(ch.per[C0], float("nan")))
        both = torch.isfinite(ref) & torch.isfinite(mine)
        da = float((ref - mine)[both].abs().max()) if bool((torch.isfinite(ref) == torch.isfinite(mine)).all()) else float("inf")
        out["base fast loss against recon_loss_fb, largest difference"] = (da, da < 1e-3)
        refc = R.recon_loss_fb(Fb, Bb, L, fmax=FMAX, hpc=True, channel=3, per_frame=True).reshape(-1)
        mc = hub_frame(R.group_basis_out(ch.res[C0], ch.w, ch.phiF, ch.crop, ch.n_crop) * ch.w, ch.w)
        mc = torch.where(ch.m2, mc, torch.full_like(mc, float("nan")))
        both = torch.isfinite(refc) & torch.isfinite(mc)
        db_ = float((refc - mc)[both].abs().max()) if bool((torch.isfinite(refc) == torch.isfinite(mc)).all()) else float("inf")
        out["C with one crop per group against recon_loss_fb(channel=3), largest difference"] = (db_, db_ < 1e-3)
        # (c) zero-misfit and +50 c targets, every candidate, all cell frames shifted together
        for label, c_true in (("zero-misfit target", 0.0), ("target rendered with F1 +50 c", 50.0)):
            ci = int(np.argmin(np.abs(C_GRID - c_true)))
            tz = torch.where(ch.w > 0, ch.tg - ch.res[ci], ch.tg)  # = render(F + c) + lf[best Rd] on the valid harmonics
            cells = np.where(P["mask"][sl], 0, -1)
            for cand in CANDS:
                if cand == "D":
                    continue
                tz_c = tz
                if cand == "Bp_fixed":  # the candidate's own render on hard frames: B1 at the prior, Rd fixed
                    rdf = ch.rdfix[ch.rec].clamp_min(0)
                    Fc = shift(ch.F, C_GRID[ci])
                    Bf = torch.cat([R.hawks_miller_t(Fc[:, :1], ch.f0[:, None].clamp_min(50.0)), ch.B[:, 1:]], -1)
                    hard = ch.fixed_ok[:, None] & (ch.w > 0)
                    tz_c = torch.where(hard, ch.render(Fc, Bf) + lf[rdf], tz)
                chz = Chunk(R, lf, P, sl, 1, dev, tgt_override=tz_c, rdfix_override=ch.rdfix if cand == "Bp_fixed" else None)
                S, n = scan_chunk(chz, cells, [cand], 1)
                c_hat = cstar_curv(S[cand][1].sum(0), n[cand][1].sum())[0]
                out[f"{label}: {cname(cand)} recovers (c)"] = (c_hat, abs(c_hat - c_true) <= 5.0)
                del chz
    del ch


def d_noise_free_check(out):
    """D's targets on noise-free renders (no aspiration, jitter, shimmer or channel) against the gated form's"""
    from dataclasses import replace

    from vocal_analysis.synth import sampler
    from vocal_analysis.train import reconstruction as R

    cfg = replace(sampler.SynthConfig(), hnr_db=None, jitter=(0.0, 0.0), shimmer_db=(0.0, 0.0), p_whisper_clip=0.0, p_h_onset=0.0,
                  gap_kinds={"silence": 1.0})
    diffs = []
    for i in range(2 if SMOKE else 6):
        clip = sampler.generate(i, cfg, seed=4123)
        f0 = np.asarray(clip.labels["f0"], np.float64)
        v = np.asarray(clip.labels["voiced"], bool) & np.isfinite(f0) & (f0 > 0)
        a, oka = R.harmonic_targets(clip.audio.astype(np.float64), np.where(v, f0, np.nan), v, k_max=K, snr_min_db=10.0)
        b, okb = R.harmonic_targets(clip.audio.astype(np.float64), np.where(v, f0, np.nan), v, k_max=K, snr_min_db=10.0, noise_subtract=True)
        k = oka & okb
        diffs.append(np.abs(a[k] - b[k]))
    d = np.concatenate(diffs)
    p95 = float(np.percentile(d, 95))
    # amendment 1: the sampler's renders move (f0 contours, vibrato, formant transitions), which leaks energy to the
    # half-integer sinusoids the noise estimate uses, so they are reported, not gated; the gate is a steady render
    out["(reported) D on the sampler's noise-free renders, which move: 95th percentile |change| (dB)"] = (p95, True)
    import torch

    diffs = []
    t = np.arange(16000) / 16000.0
    for f0b, F1 in ((120.0, 500.0), (220.0, 700.0), (330.0, 800.0)):
        F = torch.tensor([[F1, 1500.0, 2500.0, 3500.0]], dtype=torch.float64)
        f0t = torch.tensor([f0b], dtype=torch.float64)
        kk = np.arange(1, int(FMAX // f0b) + 1)
        f = torch.tensor(f0b * kk, dtype=torch.float64)[None]
        uf, ub = R.upper_poles(F[:, 3], f0t)
        Bw = R.hawks_miller_t(F, f0t[:, None])
        amp_db = (R.render_db(f, torch.cat([F, uf], -1), torch.cat([Bw, ub], -1), None, None, False)
                  + R.lf_source_db(torch.ones(1, dtype=torch.float64), len(kk)))[0].numpy()
        x = sum(10 ** (a / 20) * np.cos(2 * np.pi * k * f0b * t + 0.7 * k) for k, a in zip(kk, amp_db))
        T = len(x) // R.HOP + 1
        a, oka = R.harmonic_targets(x, np.full(T, f0b), np.ones(T, bool), k_max=K, snr_min_db=10.0)
        b, okb = R.harmonic_targets(x, np.full(T, f0b), np.ones(T, bool), k_max=K, snr_min_db=10.0, noise_subtract=True)
        k = oka & okb
        diffs.append(np.abs(a[k] - b[k]))
    p95 = float(np.percentile(np.concatenate(diffs), 95))
    out["D on steady noise-free renders (LF source, fixed tract, flat f0): 95th percentile |change| (dB)"] = (p95, p95 < 0.1)


def bp_precheck(R, lf, dev):
    """B''s pre-check on made-up crops: one recording of five crops at a shared f0 contour, easy crops (F1 above 2.5 f0)
    and hard crops alternating, the target the loss's own render plus D-119's a(k); hard crops' F1 shifted together"""
    import torch

    A_K = (3.1, -1.6, -1.1, -0.5, -0.1)
    a = torch.zeros(K, device=dev)
    a[:len(A_K)] = torch.tensor(A_K, device=dev)
    t = np.arange(200) * 0.01
    rows, ok_all = [], True
    for f0b in (120.0, 220.0, 330.0):
        for F1h in (300.0, 450.0, 600.0, 800.0):
            if not (1.5 * f0b <= F1h < IDENT_RATIO * f0b):
                continue
            for cont, st in (("flat", 0.0), ("glide 8 st", 8.0)):
                f0c = f0b * 2.0 ** (st * (t / t[-1] - 0.5) / 12.0)
                f0 = np.tile(f0c, 5)
                hard = np.repeat(np.arange(5) % 2 == 1, 200)
                F = np.column_stack([np.where(hard, F1h, 3.2 * f0b), np.full(1000, 1500.0), np.full(1000, 2500.0), np.full(1000, 3500.0)])
                Ft = torch.as_tensor(F, device=dev, dtype=torch.float32)
                f0t = torch.as_tensor(f0, device=dev, dtype=torch.float32)
                Bt = R.hawks_miller_t(Ft, f0t[:, None])
                f = f0t[:, None] * torch.arange(1, K + 1, device=dev, dtype=torch.float32)
                okk = f <= FMAX
                uf, ub = R.upper_poles(Ft[:, 3], f0t)
                rend = R.render_db(torch.where(okk, f, torch.full_like(f, 1000.0)), torch.cat([Ft, uf], -1), torch.cat([Bt, ub], -1), None, None, False)
                tg = rend + R.lf_source_db(torch.full_like(f0t, 1.0), K) + a
                crop = np.repeat(np.arange(5), 200)
                P = {"rec": np.zeros(1000, int), "crop": crop, "f0": f0.astype(np.float32), "db": tg.cpu().numpy(),
                     "ok": okk.cpu().numpy(), "mask": np.ones(1000, bool), "F": {1: F.astype(np.float32)}, "B": {1: Bt.cpu().numpy()},
                     "crop_span": f0_span_ok(f0, np.ones(1000, bool), crop), "n_rec": 1}
                ch = Chunk(R, lf, P, np.arange(1000), 1, dev)
                S, n = scan_chunk(ch, np.where(hard, 0, -1), ["base", "Bp"], 1)
                cb, vb = cstar_curv(S["base"][0].sum(0), n["base"][0].sum())
                cp, vp = cstar_curv(S["Bp"][0].sum(0), n["Bp"][0].sum())
                kept = vp / vb if vb > 0 else np.nan
                ok = abs(cp) <= 21.0 and kept >= 0.5
                ok_all &= ok
                rows.append(f"  f0 {f0b:3.0f}  F1 {F1h:3.0f} ({F1h / f0b:.2f} f0)  {cont:10s}  base {cb:+5.0f} c   B' {cp:+5.0f} c   curvature kept {kept:5.2f}  -> {'ok' if ok else 'FAILS'}")
                del ch
    return rows, ok_all


# ------------------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train import reconstruction as R

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    vq = TorchVQT(dev)
    lf = R.lf_source_db(torch.tensor(R.RD_SEARCH, dtype=torch.float32, device=dev), K)
    log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731

    pre_rows, pre_ok = bp_precheck(R, lf, dev)
    log("pre-check done")
    models = {s: load_teacher(f"runs/d040_5b_s{s}/ckpt_016000.pt", dev)[0].eval() for s in (1, 2)}
    PACKS = {"VocalSet": store_pack("cache/recon/vocalset", models, vq, dev)}
    log("VocalSet pack ready")
    checks = {}
    code_checks(R, lf, PACKS["VocalSet"], dev, checks)
    d_noise_free_check(checks)
    checks_ok = all(v[1] for v in checks.values())
    L = [("SMOKE RUN (code checks only; not a result). " if SMOKE else "") + "D-123's registered check: the reconstruction loss's candidates",
         "branch-point models d040_5b_s{1,2}/ckpt_016000.pt; D-109's instrument; F1 scaled by c cents on one cell's frames at a time, "
         "objective each recording's total loss; c by a parabola through the lowest three grid points; curvature per frame over +-100 c", "",
         "code checks:"]
    for k_, (v, ok) in checks.items():
        L.append(f"  {k_}: {v:.3g} -> {'ok' if ok else 'FAILS'}")
    L.append("")
    SMOKE_OUT = list(L)
    L.append("B''s pre-check (made-up crops; hard crops' preferred shift and curvature kept against base): " + ("passes" if pre_ok else "FAILS; B' is reported, not read"))
    L += pre_rows
    L.append("")

    PACKS["Zeroth-Korean"] = store_pack("cache/recon/zeroth", models, vq, dev)
    log("Zeroth pack ready")
    PACKS[SYNTH] = synth_pack(models, vq, dev)
    log("synthetic pack ready")
    PACKS[PTDB] = ptdb_pack(models, vq, dev)
    log("PTDB-TUG pack ready")
    del models

    RES = {}
    for name, P in PACKS.items():
        n_cells = 2 * N_CELL if name == PTDB else N_CELL
        RES[name] = {}
        for s in (1, 2):
            RES[name][s] = scan_pack(R, lf, P, s, dev, CANDS, n_cells)
            log(f"{name} seed {s} scanned")
    Path("cache/d123").mkdir(parents=True, exist_ok=True)
    np.savez("cache/d123/cells_smoke.npz" if SMOKE else "cache/d123/cells.npz", c_grid=C_GRID, **{f"{n}|{s}|{c}|S": RES[n][s][0][c] for n in RES for s in (1, 2) for c in CANDS},
             **{f"{n}|{s}|{c}|cnt": RES[n][s][1][c] for n in RES for s in (1, 2) for c in CANDS})

    rng = np.random.default_rng(BOOT_SEED)

    def readable(name, cand, X, base_cv):
        _, cv, nn = cell_stats(RES[name], cand, X)
        floor = PT_MIN_FRAMES if name == PTDB else MIN_FRAMES
        if min(nn) < floor:
            return False
        if name == PTDB:
            P = PACKS[name]
            spk = {P["rec_spk"][r] for r in range(P["n_rec"]) if any(RES[name][s][1][cand][X][r] > 0 for s in (1, 2))}
            if len(spk) < PT_MIN_SPK:
                return False
        return cand == "base" or (np.isfinite(cv) and base_cv > 0 and cv >= CURV_MIN * base_cv)

    def boot_picks(name, n):
        P = PACKS[name]
        if name != PTDB:
            return [rng.integers(0, P["n_rec"], P["n_rec"]) for _ in range(n)]
        spks = sorted(set(P["rec_spk"]))
        by = {s: [r for r in range(P["n_rec"]) if P["rec_spk"][r] == s] for s in spks}
        return [np.concatenate([by[spks[i]] for i in rng.integers(0, len(spks), len(spks))]) for _ in range(n)]

    TAB = {}
    for name in PACKS:
        n_cells = 2 * N_CELL if name == PTDB else N_CELL
        cells = range(N_CELL) if name != PTDB else range(2 * N_CELL)
        picks = boot_picks(name, PT_BOOT if name == PTDB else N_BOOT)
        TAB[name] = {}
        for cand in CANDS:
            for X in cells:
                c, cv, nn = cell_stats(RES[name], cand, X)
                if min(nn) <= 0:
                    continue
                bc = cell_stats(RES[name], "base", X)[1]
                rd = readable(name, cand, X, bc)
                bs = [cell_stats(RES[name], cand, X, p)[0] for p in picks] if rd else []
                lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
                TAB[name][cand, X] = dict(c=c, lo=lo, hi=hi, cv=cv, kept=cv / bc if bc > 0 else np.nan, n=float(np.mean(nn)), readable=rd)
            tot = cell_stats(RES[name], cand, n_cells)
            TAB[name][cand, "all"] = dict(c=tot[0], cv=tot[1], n=float(np.mean(tot[2])))

    def cell_name(X):
        sx = "" if X < N_CELL else "men, "
        X = X % N_CELL
        return f"{sx}{F1_NAMES[X // len(H_NAMES)]}, F1/f0 {H_NAMES[X % len(H_NAMES)]}"

    for name in PACKS:
        L.append(f"== {name}")
        hdr = f"  {'cell':34s} {'frames':>7s}  {'base c [95 %]':>20s}  " + "  ".join(f"{cname(c):>14s}" for c in CANDS[1:])
        L.append(hdr + "   (candidates: c, and curvature kept against base; * readable)")
        cells = sorted({X for (_, X) in TAB[name] if X != "all"})
        for X in cells:
            b = TAB[name].get(("base", X))
            if b is None:
                continue
            row = f"  {cell_name(X):34s} {int(b['n']):7d}  {b['c']:+6.0f} [{b['lo']:+5.0f}, {b['hi']:+5.0f}]{'*' if b['readable'] else ' '}"
            for cand in CANDS[1:]:
                t_ = TAB[name].get((cand, X))
                row += "  " + (f"{t_['c']:+5.0f} ({t_['kept']:4.2f}){'*' if t_['readable'] else ' '}" if t_ else f"{'-':>14s}")
            L.append(row)
        L.append("  all cell frames shifted together: " + "; ".join(f"{cname(c)} c {TAB[name][c, 'all']['c']:+.0f}, curvature {TAB[name][c, 'all']['cv']:.4f}" for c in CANDS))
        L.append("")

    # criterion 1: the jump
    L.append("CRITERION 1, the jump across 2 f0 on the stores: J = c(2-2.5) - c(1.5-2) per F1 band where base has both cells readable")
    crit = {c: {} for c in CANDS[1:]}
    stores = ("VocalSet", "Zeroth-Korean")
    unreadable = not checks_ok
    for cand in CANDS[1:]:
        ok_all, parts = True, []
        for name in stores:
            bands = [fb for fb in range(len(F1_NAMES)) if all(TAB[name].get(("base", fb * 5 + h), {}).get("readable") for h in (1, 2))]
            if cand == CANDS[1] and not bands:
                unreadable = True
            for fb in bands:
                Jb = TAB[name]["base", fb * 5 + 2]["c"] - TAB[name]["base", fb * 5 + 1]["c"]
                t1, t2 = TAB[name].get((cand, fb * 5 + 1)), TAB[name].get((cand, fb * 5 + 2))
                rd = bool(t1 and t2 and t1["readable"] and t2["readable"])
                J = t2["c"] - t1["c"] if rd else np.nan
                ok = rd and abs(J) <= abs(Jb) / 3
                ok_all &= ok
                parts.append(f"{name} {F1_NAMES[fb]}: base {Jb:+.0f}, {cname(cand)} " + (f"{J:+.0f}" if rd else "not readable") + (" ok" if ok else " no"))
        crit[cand][1] = ok_all
        L.append(f"  {cname(cand)}: " + "; ".join(parts) + f" -> {'PASSES' if ok_all else 'fails'}")
    L.append("")

    # criterion 2: the synthetic control
    L.append("CRITERION 2, synthetic control: frame-weighted median |c| over readable cells <= 10 c, and total F1 curvature >= 0.5 of base's")
    base_tot = TAB[SYNTH]["base", "all"]["cv"]
    for cand in CANDS[1:]:
        cs = [(abs(v["c"]), v["n"]) for (c_, X), v in TAB[SYNTH].items() if c_ == cand and X != "all" and v["readable"]]
        med = wmedian(np.array([x for x, _ in cs]), np.array([n for _, n in cs])) if cs else np.nan
        kept = TAB[SYNTH][cand, "all"]["cv"] / base_tot if base_tot > 0 else np.nan
        ok = bool(np.isfinite(med) and med <= 10.0 and kept >= 0.5)
        crit[cand][2] = ok
        L.append(f"  {cname(cand)}: median |c| {med:.0f} c over {len(cs)} cells; curvature kept {kept:.2f} -> {'PASSES' if ok else 'fails'}")
    L.append(f"  base for reference: median |c| {wmedian(np.array([abs(v['c']) for (c_, X), v in TAB[SYNTH].items() if c_ == 'base' and X != 'all' and v['readable']]), np.array([v['n'] for (c_, X), v in TAB[SYNTH].items() if c_ == 'base' and X != 'all' and v['readable']])):.0f} c")
    L.append("")

    # criterion 3: PTDB-TUG women
    L.append("CRITERION 3, PTDB-TUG dev women: where E's interval excludes 0, c opposite in sign to E; else |c| <= 25 c")
    P = PACKS[PTDB]
    women = P["sex"] == "F"
    spks = sorted({s for s in P["spk"][women & P["score"]]})
    E, CELLS = {}, {s: frame_cells(P, s) for s in (1, 2)}
    for X in range(N_CELL):
        def e_of(draw):
            vals = []
            for s in (1, 2):
                fr = women & (CELLS[s] == X) & np.isin(P["spk"], draw)
                if fr.sum() == 0:
                    return np.nan
                vals.append(np.median(1200 * np.log2(P["F"][s][fr, 0] / P["cp"][fr])))
            return float(np.mean(vals))
        e = e_of(spks)
        bs = [e_of([spks[i] for i in rng.integers(0, len(spks), len(spks))]) for _ in range(PT_BOOT)]
        E[X] = (e, *np.nanpercentile(bs, [2.5, 97.5]))
    for cand in CANDS:
        ok_all, parts, n_read = True, [], 0
        for X in range(N_CELL):
            t_ = TAB[PTDB].get((cand, X))
            if not t_ or not t_["readable"] or not np.isfinite(E[X][0]):
                continue
            n_read += 1
            e, elo, ehi = E[X]
            ok = (np.sign(t_["c"]) == -np.sign(e) and t_["c"] != 0) if (elo > 0 or ehi < 0) else abs(t_["c"]) <= 25.0
            ok_all &= ok
            parts.append(f"{cell_name(X)}: E {e:+.0f} [{elo:+.0f}, {ehi:+.0f}], c {t_['c']:+.0f} {'ok' if ok else 'no'}")
        ok_all &= n_read > 0
        if cand != "base":
            crit[cand][3] = ok_all
        L.append(f"  {cname(cand)} ({n_read} readable cells): " + "; ".join(parts) + f" -> {'PASSES' if ok_all else 'fails'}")
    L.append("")

    # reported beside
    L.append("REPORTED BESIDE")
    for name in (*stores, SYNTH, PTDB):
        P = PACKS[name]
        for s in (1,):
            F1, f0, m = P["F"][s][:, 0], P["f0"], P["mask"]
            band = np.digitize(f0, F0_CUTS)
            floor = [f"{F0_NAMES[b]} {np.mean(F1[m & (band == b)] < IDENT_RATIO * f0[m & (band == b)]):.0%}" for b in range(3) if (m & (band == b)).any()]
            L.append(f"  B's floor, {name} (seed 1): frames with model F1 below 2.5 f0: overall {np.mean(F1[m] < IDENT_RATIO * f0[m]):.0%}; " + ", ".join(floor))
        allX = 2 * N_CELL if name == PTDB else N_CELL
        for cand in ("Bp", "Bp_fixed"):
            have, tot = RES[name][1][1][cand][allX].sum(), RES[name][1][1]["base"][allX].sum()
            L.append(f"    {cname(cand)} coverage, {name} (seed 1): {have / max(tot, 1):.0%} of scored frames carry an F1 reading (base 100 %)")
    for name in (*stores, PTDB):
        for cand in CANDS:
            lo_ = [(TAB[name][cand, X]["c"], TAB[name][cand, X]["n"]) for X in range(N_CELL) if TAB[name].get((cand, X), {}).get("readable") and X % 5 <= 1]
            hi_ = [(TAB[name][cand, X]["c"], TAB[name][cand, X]["n"]) for X in range(N_CELL) if TAB[name].get((cand, X), {}).get("readable") and 2 <= X % 5 <= 3]
            f = lambda v: (f"{np.average([a for a, _ in v], weights=[n for _, n in v]):+.0f}" if v else "-")  # noqa: E731
            L.append(f"  {name}, {cname(cand)}: frame-weighted c below 2 f0 {f(lo_)}, at 2-3.5 f0 {f(hi_)}")
    for name in (*stores, SYNTH):
        diffs = [abs(TAB[name]["D", X]["c"] - TAB[name]["base", X]["c"]) for X in range(N_CELL) if TAB[name].get(("D", X), {}).get("readable") and TAB[name].get(("base", X), {}).get("readable")]
        P = PACKS[name]
        both = P["ok"] & P["okD"]
        L.append(f"  D against its bound, {name}: largest |c(D) - c(base)| over readable cells {max(diffs) if diffs else float('nan'):.0f} c; "
                 f"median target change {np.median(P['dbD'][both] - P['db'][both]):+.3f} dB, 1st percentile {np.percentile(P['dbD'][both] - P['db'][both], 1):+.3f} dB")
    for name in stores:
        for s in (1, 2):
            per = []
            for fb in range(len(F1_NAMES)):
                t1 = RES[name][s][0]["base"][fb * 5 + 1].sum(0), RES[name][s][1]["base"][fb * 5 + 1].sum()
                t2 = RES[name][s][0]["base"][fb * 5 + 2].sum(0), RES[name][s][1]["base"][fb * 5 + 2].sum()
                if t1[1] > 0 and t2[1] > 0:
                    per.append(f"{F1_NAMES[fb]} {cstar_curv(*t2)[0] - cstar_curv(*t1)[0]:+.0f}")
            L.append(f"  base jump, {name}, seed {s} alone: " + ", ".join(per))
    L.append("")

    # readings
    L.append("READINGS")
    if unreadable:
        L.append("  -> UNREADABLE (a code check failed or a store has no F1 band with base's two cells readable)")
    else:
        rem = {c: crit[c][1] for c in CANDS[1:] if c != "D" and (c != "Bp" or pre_ok)}
        for c in CANDS[1:]:
            L.append(f"  {cname(c)}: criterion 1 {'passes' if crit[c][1] else 'fails'}, 2 {'passes' if crit[c][2] else 'fails'}, 3 {'passes' if crit[c][3] else 'fails'}"
                     + (" -> removes the jump rightly" if all(crit[c].get(i) for i in (1, 2, 3)) else (" -> removes the jump" if crit[c][1] else ""))
                     + (" (reported only)" if c == "D" else "") + (" (pre-check failed: reported only)" if c == "Bp" and not pre_ok else ""))
        src = rem.get("Bp", False) or rem.get("A", False)
        chn = rem.get("C", False)
        if src and not chn:
            cause = "THE SOURCE'S SHAPE carries the jump on real recordings"
        elif chn and not src:
            cause = "THE RECORDING'S CURVE carries it"
        elif src and chn:
            cause = "NOT SEPARATED (both kinds remove it)"
        elif rem.get("A+C", False):
            cause = "BOTH TOGETHER, neither alone"
        else:
            cause = "NOT AMONG THESE (next: the model's bandwidths, nasal frames)"
        L.append(f"  -> the cause: {cause}")
    print("\n".join(SMOKE_OUT if SMOKE else L))
