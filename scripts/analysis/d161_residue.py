"""D-161, before training, items 2 and 4: the residue measure β refitted with D-160 and with length-set LPC, its
shuffle null, β* on the re-rendered resynthesised sets, and the monotonicity count of real sources.

    python scripts/analysis/d161_residue.py analyse MODEL CORPUS     (GPU) the corpus's recordings (train and val sets)
        -> cache/d161/analysis/MODEL/CORPUS/<recording>.npz: f0, voicing, sonorant, F1-F4, B1-B4, VTL (log2 cm)
    python scripts/analysis/d161_residue.py lengths                  -> cache/d161/lengths.json (per speaker, D-160's VTL)
    python scripts/analysis/d161_residue.py refit CORPUS [WORKERS]   (CPU) per speaker: the recovered source again, β
        against the stored tracks (a reproduction of tract_left), D-160 s1 and s2, and stable length-set Praat poles,
        and each basis's shuffle null -> cache/d161/residue/CORPUS/<speaker>.npz
    python scripts/analysis/d161_residue.py assemble SET             β*, β_model and β_LPC on the re-rendered set's
        frames -> data/resynth/SET_h40/flat/label_beta_star.npy (+ beta_model, beta_lpc), cache/d161/count/SET.npz
    python scripts/analysis/d161_residue.py flag                     (CPU) flag B's flagged VocalSet recordings
        (5b s2 at D-142's threshold, D-152's file median, singer length leave-one-out) -> cache/d161/flagged.json
    python scripts/analysis/d161_residue.py report                   -> reports/analysis/d161/residue.{txt,json}

β (`resynth.render.source_qc`'s fit): the recovered source's harmonics below 4 kHz = an LF spectrum (best Rd) + gain +
tilt + β x a tract response. β* = max(β_model, β_LPC), β_model = max over D-160's seeds of |β|; a basis that can't be
fitted on a frame (no poles, too few harmonics) is left out of the max; no basis at all leaves β* NaN.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache/d161"
REP = ROOT / "reports/analysis/d161"
D160 = {"s1": "d160_s1", "s2": "d160_s2"}
FLAG_MODEL = "d040_5b_s2_b20k"
SETS = {"vocalset": ("vocalset_train_d16k", "vocalset_val_d16k"),
        "zeroth_korean": ("zeroth_korean_train_d16k", "zeroth_korean_val_d16k")}
FS, HOP = 16000, 160
C_CM_S = 35300.0
N_H = 40
F_MAX = 4000.0
CLEAN = 0.1
POLE_TOL, POLE_BW_MAX = 0.05, 700.0
NULL_GAP, NULL_SEP = 100, 0.15  # frames (1 s), relative F1 or F2 difference


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _recordings(corpus: str):
    """[(row, set name, cfg, analysis dir, audio root)] for the corpus's train and val sets."""
    r40 = _mod("d161_resynth40")
    out = []
    for name in SETS[corpus]:
        d, cfg, rows, adir, root, _ = r40._setup(name)
        out += [(r, name, cfg, adir, root) for r in rows]
    return out


def _safe(s: str) -> str:
    return s.replace(":", "_").replace("/", "_")


# ---------------------------------------------------------------- analyse (GPU)

def cmd_analyse(model: str, corpus: str) -> None:
    import torch

    from vocal_analysis.device import setup_device
    from vocal_analysis.models.infer import analyse, load_teacher
    from vocal_analysis.resynth.corpus import _analysis_name, load_audio

    out = CACHE / f"analysis/{model}/{corpus}"
    out.mkdir(parents=True, exist_ok=True)
    recs = _recordings(corpus)
    todo = [x for x in recs if not (out / _analysis_name(x[0].path)).exists()]
    print(f"{model} {corpus}: {len(recs) - len(todo)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    dev = setup_device()
    md = load_teacher(str(ROOT / f"runs/{model}/ckpt_last.pt"), dev)[0]
    for k, (r, _, _, _, root) in enumerate(todo, 1):
        y = load_audio(root / r.path).astype(np.float32)
        n = len(y) // HOP + 1
        pad = int(np.ceil(len(y) / FS)) * FS  # whole seconds: few distinct shapes for MIOpen (D-063)
        res = analyse(md, np.pad(y, (0, pad - len(y))), dev)
        keep = {"f0": res["f0"], "voiced_p": res["voiced_p"], "sonorant_p": res["sonorant_p"],
                "formants": res["formants"][:, :4], "bandwidths": res["bandwidths"][:, :4]}
        if "vtl_log2cm" in res:
            keep["vtl_log2cm"] = res["vtl_log2cm"]
        tmp = out / (_analysis_name(r.path) + ".tmp.npz")
        np.savez(tmp, **{a: np.asarray(v)[:n].astype(np.float32) for a, v in keep.items()})
        tmp.replace(out / _analysis_name(r.path))
        if k % 200 == 0 or k == len(todo):
            print(f"  {k}/{len(todo)}", flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- lengths

def cmd_lengths() -> None:
    from vocal_analysis.resynth.corpus import _analysis_name

    out = {}
    for corpus in SETS:
        per: dict[str, list[float]] = {}
        for r, *_ in _recordings(corpus):
            zs = [np.load(CACHE / f"analysis/{m}/{corpus}/{_analysis_name(r.path)}") for m in D160.values()]
            v = np.all([z["voiced_p"] > 0.5 for z in zs], 0)
            lg = np.mean([z["vtl_log2cm"] for z in zs], 0)
            if v.any():
                per.setdefault(r.speaker_id, []).append(float(np.median(2 ** lg[v])))
        out.update({s: float(np.median(v)) for s, v in per.items()})
    (CACHE / "lengths.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    vals = np.array(list(out.values()))
    print(f"{len(out)} speakers, length {np.percentile(vals, [0, 50, 100]).round(1)} cm")


# ---------------------------------------------------------------- refit (CPU)

def _lf_table():
    from vocal_analysis.resynth.render import _RD_FIT
    from vocal_analysis.synth.glottal import source_h_db

    return source_h_db(_RD_FIT, N_H)


def _beta(h, f0, fn, bn, lf):
    """source_qc's fit per frame with given poles (T, P) and bandwidths (T, P), NaN poles absent. β NaN where the
    frame has fewer than 5 finite harmonics below 4 kHz or no finite pole."""
    from vocal_analysis.synth import filters

    k = np.arange(1, N_H + 1)
    beta = np.full(len(f0), np.nan)
    for i in np.flatnonzero(np.isfinite(f0)):
        ok = np.isfinite(h[i]) & (f0[i] * k < F_MAX)
        m = np.isfinite(fn[i]) & np.isfinite(bn[i])
        if ok.sum() < 5 or not m.any():
            continue
        fk = f0[i] * k[ok]
        g = 20 * np.log10(np.abs(filters.response(fk, fn[i, m][None, :], bn[i, m][None, :], FS)))
        basis = np.stack([np.ones(ok.sum()), np.log2(k[ok]), g], 1)
        d = h[i, ok][None, :] - lf[:, ok]
        coef, *_ = np.linalg.lstsq(basis, d.T, rcond=None)
        err = np.mean((d - (basis @ coef).T) ** 2, axis=1)
        beta[i] = coef[2, int(np.argmin(err))]
    return beta


def _with_upper(F, B, f0):
    """F1-F4 (T, 4) + uniform-tube poles above them (source_qc's tract); bandwidths B for F1-F4 (None: the prior)."""
    from vocal_analysis.backend.corrections import upper_poles
    from vocal_analysis.synth.tract import bandwidth_prior

    F = np.asarray(F, float)
    full = np.all(np.isfinite(F), 1)
    up = np.full((len(F), upper_poles(np.full((1, F.shape[1]), 1000.0), FS).shape[1]), np.nan)
    if full.any():
        up[full] = upper_poles(F[full], FS)
    fn = np.concatenate([F, up], 1)
    fn[~full] = np.nan  # source_qc fits only frames with every tracked formant
    bn = bandwidth_prior(fn, f0[:, None])
    if B is not None:
        bn[:, : F.shape[1]] = np.where(np.isfinite(B), B, bn[:, : F.shape[1]])
    return fn, bn


def _praat_poles(y, ceiling, n):
    """Praat Burg (5 formants, 10 ms) at the nearest Praat frame to each of our frames: (n, 5) poles and bandwidths."""
    import parselmouth

    fm = parselmouth.Sound(np.asarray(y, np.float64), FS).to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                                                         maximum_formant=float(ceiling))
    nf, t1, dt = fm.get_number_of_frames(), fm.t1, fm.dt
    F, B = np.full((nf, 5), np.nan), np.full((nf, 5), np.nan)
    for j in range(nf):
        t = t1 + j * dt
        for k in range(5):
            F[j, k] = fm.get_value_at_time(k + 1, t)
            B[j, k] = fm.get_bandwidth_at_time(k + 1, t)
    idx = np.clip(np.round((np.arange(n) * 0.01 - t1) / dt).astype(int), 0, nf - 1)
    return F[idx], B[idx]


def _stable(F, B):
    """Poles present within POLE_TOL in each of the ±2 neighbouring frames, bandwidth under POLE_BW_MAX; others NaN."""
    keep = np.isfinite(F) & np.isfinite(B) & (B < POLE_BW_MAX)
    n = len(F)
    for d in (-2, -1, 1, 2):
        nb = np.full_like(F, np.nan)
        if d > 0:
            nb[: n - d] = F[d:]
        else:
            nb[-d:] = F[: n + d]
        with np.errstate(invalid="ignore"):
            hit = np.any(np.abs(nb[:, None, :] - F[:, :, None]) <= POLE_TOL * F[:, :, None], axis=2)
        keep &= hit
    return np.where(keep, F, np.nan), np.where(keep, B, np.nan)


def _null(rec, pos, f12, rng):
    """Donor frames for the shuffle null: another voiced frame of the same speaker, in another recording or at least
    NULL_GAP frames away, whose F1 or F2 differs by more than NULL_SEP (-1: none found in 20 draws). rec, pos (frame
    in its recording), f12: per frame of the speaker."""
    n = len(rec)
    cand = np.flatnonzero(np.all(np.isfinite(f12), 1))
    donor = np.full(n, -1)
    if len(cand) < 2:
        return donor
    for i in cand:
        for _ in range(20):
            j = cand[rng.integers(len(cand))]
            far = rec[j] != rec[i] or abs(pos[j] - pos[i]) >= NULL_GAP
            sep = np.any(np.abs(f12[j] / f12[i] - 1) > NULL_SEP)
            if far and sep:
                donor[i] = j
                break
    return donor


def _speaker_job(args):
    from vocal_analysis.resynth.corpus import _analysis_name, load_audio
    from vocal_analysis.resynth.render import source_from_audio

    corpus, speaker, items, length = args
    lf = _lf_table()
    ceiling = 5 * C_CM_S / (2 * length)
    acc = {k: [] for k in ("rec", "frame", "f0", "voiced", "tl_stored", "b_old", "b_s1", "b_s2", "b_lpc",
                           "n_s1", "n_s2", "n_lpc", "tracks", "h")}
    bases = {"s1": [], "s2": [], "lpc": []}
    f12 = {"s1": [], "s2": [], "lpc": []}
    for ri, (r, name, cfg, adir, root) in enumerate(items):
        with np.load(adir / _analysis_name(r.path)) as z:
            analysis = {k: z[k] for k in z.files}
        audio = load_audio(root / r.path)
        src = source_from_audio(audio, analysis, cfg)
        n = len(src["f0"])
        f0 = np.where(src["voiced"], src["f0"], np.nan)
        h = src["source_h_db"]
        tr = np.asarray(src["tracks"], float)
        fn, bn = _with_upper(tr, None, src["f0"])
        acc["b_old"].append(_beta(h, f0, fn, bn, lf))
        for s, m in D160.items():
            with np.load(CACHE / f"analysis/{m}/{corpus}/{_analysis_name(r.path)}") as z:
                F, B = z["formants"][:n].astype(float), z["bandwidths"][:n].astype(float)
            fn, bn = _with_upper(F, B, src["f0"])
            bases[s].append((fn, bn))
            f12[s].append(F[:, :2])
            acc[f"b_{s}"].append(_beta(h, f0, fn, bn, lf))
        P, PB = _stable(*_praat_poles(audio, ceiling, n))
        order = np.argsort(np.where(np.isfinite(P), P, np.inf), 1)
        P, PB = np.take_along_axis(P, order, 1), np.take_along_axis(PB, order, 1)
        bases["lpc"].append((P, PB))
        f12["lpc"].append(P[:, :2])
        acc["b_lpc"].append(_beta(h, f0, P, PB, lf))
        acc["rec"].append(np.full(n, ri))
        acc["frame"].append(np.arange(n))
        acc["f0"].append(src["f0"])
        acc["voiced"].append(src["voiced"])
        acc["tl_stored"].append(src["tract_left"])
        acc["tracks"].append(tr[:, :4] if tr.shape[1] >= 4 else np.pad(tr, ((0, 0), (0, 4 - tr.shape[1])), constant_values=np.nan))
        acc["h"].append(h.astype(np.float32))
    rec = np.concatenate(acc["rec"])
    f0v = np.where(np.concatenate(acc["voiced"]), np.concatenate(acc["f0"]), np.nan)
    hh = np.concatenate(acc["h"])
    pos = np.concatenate(acc["frame"])
    rng = np.random.default_rng([161] + [ord(c) for c in speaker])
    for s in ("s1", "s2", "lpc"):
        fn = np.concatenate([b[0] for b in bases[s]])
        bn = np.concatenate([b[1] for b in bases[s]])
        ff = np.concatenate(f12[s])
        ff[~np.isfinite(f0v)] = np.nan
        donor = _null(rec, pos, ff, rng)
        ok = donor >= 0
        fd, bd = np.full_like(fn, np.nan), np.full_like(bn, np.nan)
        fd[ok], bd[ok] = fn[donor[ok]], bn[donor[ok]]
        acc[f"n_{s}"] = [_beta(hh, np.where(ok, f0v, np.nan), fd, bd, lf)]
    out = CACHE / f"residue/{corpus}/{_safe(speaker)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, paths=json.dumps([it[0].path for it in items]), length=length, ceiling=ceiling,
             **{k: np.concatenate(v) for k, v in acc.items()})
    return speaker, len(items)


def cmd_refit(corpus: str, workers: int = 20) -> None:
    lengths = json.loads((CACHE / "lengths.json").read_text(encoding="utf-8"))
    by: dict[str, list] = {}
    for x in _recordings(corpus):
        by.setdefault(x[0].speaker_id, []).append(x)
    jobs = [(corpus, s, items, lengths[s]) for s, items in sorted(by.items(), key=lambda kv: -len(kv[1]))
            if not (CACHE / f"residue/{corpus}/{_safe(s)}.npz").exists()]
    print(f"{corpus}: {len(by)} speakers, {len(jobs)} to fit", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, (s, n) in enumerate(ex.map(_speaker_job, jobs), 1):
            print(f"  {k}/{len(jobs)} {s} ({n} recordings)", flush=True)


# ---------------------------------------------------------------- assemble

def _star(b1, b2, bl):
    bm = np.fmax(np.abs(b1), np.abs(b2))
    return bm, np.abs(bl), np.fmax(bm, np.abs(bl))


def cmd_assemble(name: str) -> None:
    corpus = next(c for c, s in SETS.items() if name in s)
    res = {}
    for p in sorted((CACHE / f"residue/{corpus}").glob("*.npz")):
        z = np.load(p)
        paths = json.loads(str(z["paths"]))
        rec, frame = z["rec"], z["frame"]
        bm, bl, bs = _star(z["b_s1"], z["b_s2"], z["b_lpc"])
        for ri, path in enumerate(paths):
            k = rec == ri
            res[path] = {"bm": bm[k], "bl": bl[k], "bs": bs[k], "tracks": z["tracks"][k], "f0": z["f0"][k],
                         "null": np.stack([z["n_s1"][k], z["n_s2"][k], z["n_lpc"][k]], 1),
                         "b_old": z["b_old"][k], "tl": z["tl_stored"][k], "b1": z["b_s1"][k], "b2": z["b_s2"][k],
                         "blpc": z["b_lpc"][k]}
    sd = ROOT / f"data/resynth/{name}_h40"
    flat = sd / "flat"
    fo = np.load(flat / "frame_offsets.npy")
    n_frames = int(fo[-1])
    cols = {k: np.full(n_frames, np.nan, np.float32) for k in ("beta_star", "beta_model", "beta_lpc")}
    keep = {k: [] for k in ("frame_global", "rec_frame", "path_i", "b_old", "tl", "b1", "b2", "blpc", "null", "tracks", "f0")}
    paths = sorted(res)
    pidx = {p: i for i, p in enumerate(paths)}
    c = 0
    for s in sorted(sd.glob("shard_*.npz")):
        with np.load(s) as z:
            meta = json.loads(str(z["meta"]))
        for m in meta:
            a, b = int(fo[c]), int(fo[c + 1])
            r = res[m["source_path"]]
            st = int(round(m["qc"]["start_s"] * FS / HOP))
            j = np.arange(st, st + (b - a))
            cols["beta_star"][a:b], cols["beta_model"][a:b], cols["beta_lpc"][a:b] = r["bs"][j], r["bm"][j], r["bl"][j]
            if m["variant"] == 0:  # the count's per-frame extras, once per recording frame
                keep["frame_global"].append(np.arange(a, b))
                keep["rec_frame"].append(j)
                keep["path_i"].append(np.full(b - a, pidx[m["source_path"]]))
                for k2, src in (("b_old", "b_old"), ("tl", "tl"), ("b1", "b1"), ("b2", "b2"), ("blpc", "blpc"),
                                ("null", "null"), ("tracks", "tracks"), ("f0", "f0")):
                    keep[k2].append(r[src][j])
            c += 1
    assert c == len(fo) - 1, (c, len(fo) - 1)
    for k, v in cols.items():
        np.save(flat / f"label_{k}.npy", v)
    (CACHE / "count").mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"count/{name}.npz", paths=json.dumps(paths), **{k: np.concatenate(v) for k, v in keep.items()})
    print(name, "assembled", n_frames, "frames", flush=True)


# ---------------------------------------------------------------- flag B (VocalSet)

def cmd_flag() -> None:
    from vocal_analysis.resynth.corpus import _analysis_name

    d142, d151 = _mod("d142_flags"), _mod("d151_ptdb_flag")
    prior = d142.Prior()
    thr = d151.thresholds()[FLAG_MODEL]
    recs = _recordings("vocalset")
    by: dict[str, list] = {}
    for r, *_ in recs:
        by.setdefault(r.speaker_id, []).append(r)
    flagged, scores = [], {}
    for spk, rows in by.items():
        zs = [np.load(CACHE / f"analysis/{FLAG_MODEL}/vocalset/{_analysis_name(r.path)}") for r in rows]
        oks = [(z["voiced_p"] > 0.5) & np.all(np.isfinite(z["formants"]), 1) for z in zs]
        um = np.array([np.nanmedian(2 ** z["vtl_log2cm"][o]) if o.any() else np.nan for z, o in zip(zs, oks)])
        for u, (r, z, o) in enumerate(zip(rows, zs, oks)):
            if not o.any():
                continue
            L = float(np.nanmedian(np.delete(um, u)))
            sc = float(np.nanmedian(prior.score(z["formants"][o].astype(float), np.full(o.sum(), L))))
            scores[r.path] = sc
            if sc > thr:
                flagged.append(r.path)
    (CACHE / "flagged.json").write_text(json.dumps({"model": FLAG_MODEL, "threshold": thr, "flagged": sorted(flagged),
                                                    "n_scored": len(scores)}, indent=1), encoding="utf-8")
    print(f"flag B: {len(flagged)} of {len(scores)} VocalSet recordings flagged (threshold {thr:.2f})")


# ---------------------------------------------------------------- report

def _bands(f0):
    return {"f0 <= 150": f0 <= 150, "150-250": (f0 > 150) & (f0 <= 250), "> 250": f0 > 250}


def _count(h, f0, tracks, sel):
    """Rises after H2 on the frames sel: H3-H2 alone, and H4-H3 and up pooled; near = the higher harmonic within one
    prior bandwidth of an original tracked formant."""
    from vocal_analysis.synth.tract import bandwidth_prior

    h, f0, tracks = h[sel], f0[sel], tracks[sel]
    k = np.arange(1, N_H + 1)
    valid = np.isfinite(h) & (f0[:, None] * k[None, :] < F_MAX)
    d = h[:, 1:] - h[:, :-1]  # d[:, j] = H(j+2) - H(j+1)
    dv = valid[:, 1:] & valid[:, :-1]
    fk = f0[:, None] * k[None, 1:]  # the higher harmonic of each step
    bw = bandwidth_prior(tracks, f0[:, None])
    near = np.zeros_like(d, bool)
    for t in range(tracks.shape[1]):
        with np.errstate(invalid="ignore"):
            near |= np.abs(fk - tracks[:, t : t + 1]) <= bw[:, t : t + 1]
    out = {"frames": int(sel.sum())}
    s32 = dv[:, 1]
    if s32.any():
        r = d[s32, 1]
        out["h3_h2"] = {"n": int(s32.sum()), "p_rise_gt3": float(np.mean(r > 3)), "p_rise_gt0": float(np.mean(r > 0)),
                        "pct": np.percentile(r, [50, 90, 95, 99]).round(2).tolist(),
                        "rises_gt3_near_formant": float(np.mean(near[s32, 1][r > 3])) if (r > 3).any() else None}
    up = dv[:, 2:]
    if up.any():
        r = d[:, 2:][up]
        nr = near[:, 2:][up]
        out["upper"] = {"n_steps": int(up.sum()), "p_step_gt3": float(np.mean(r > 3)),
                        "p_frame_any_gt3": float(np.mean(np.any((d[:, 2:] > 3) & up, 1)[up.any(1)])),
                        "pct": np.percentile(r, [50, 90, 95, 99]).round(2).tolist(),
                        "rises_gt3_near_formant": float(np.mean(nr[r > 3])) if (r > 3).any() else None}
    return out


def cmd_report() -> None:
    flagged = set(json.loads((CACHE / "flagged.json").read_text(encoding="utf-8"))["flagged"])
    lines = ["D-161 before training: the residue measure (items 2 and 4)",
             f"  β* = max(β_model, β_LPC); clean = voiced label frame with β* < {CLEAN}; LPC: Praat Burg, 5 formants, ceiling"
             f" 5c / (2L), stable poles (±2 frames within {POLE_TOL:.0%}, bandwidth < {POLE_BW_MAX:.0f} Hz)"]
    out = {}
    null_all = []
    for corpus, names in SETS.items():
        for name in names:
            z = np.load(CACHE / f"count/{name}.npz")
            flat = ROOT / f"data/resynth/{name}_h40/flat"
            g = z["frame_global"]
            h = np.asarray(np.load(flat / "label_source_h_db.npy", mmap_mode="r")[g]).astype(float)
            v = np.asarray(np.load(flat / "label_voiced.npy", mmap_mode="r")[g]) & np.all(np.isfinite(h[:, :2]), 1)
            f0 = z["f0"]
            bm, bl, bs = _star(z["b1"], z["b2"], z["blpc"])
            res = {"voiced_frames": int(v.sum())}
            # reproduction of the stored tract_left
            m = v & np.isfinite(z["tl"]) & np.isfinite(z["b_old"])
            res["tract_left_refit_max_diff"] = float(np.max(np.abs(z["tl"][m] - z["b_old"][m]))) if m.any() else None
            # agreement
            from scipy.stats import spearmanr

            def rho(a, b):
                k = v & np.isfinite(a) & np.isfinite(b)
                return float(spearmanr(a[k], b[k])[0]) if k.sum() > 10 else None
            res["agreement"] = {"old_vs_model": rho(np.abs(z["tl"]), bm), "s1_vs_s2": rho(np.abs(z["b1"]), np.abs(z["b2"])),
                                "model_vs_lpc": rho(bm, bl)}
            res["measured"] = {"model": float(np.mean(np.isfinite(bm[v]))), "lpc": float(np.mean(np.isfinite(bl[v])))}
            # clean shares
            res["clean"] = {}
            for lab, b in (("beta_star", bs), ("beta_model", bm), ("beta_lpc", bl)):
                c = v & (b < CLEAN)
                res["clean"][lab] = {"n": int(c.sum()), "share": float(c.sum() / max(v.sum(), 1)),
                                     **{bn: float((c & sel).sum() / max((v & sel).sum(), 1)) for bn, sel in _bands(f0).items()}}
            # shuffle null rates
            nl = z["null"]
            res["null_rate"] = {}
            for j, lab in enumerate(("s1", "s2", "lpc")):
                k = v & np.isfinite(nl[:, j])
                res["null_rate"][lab] = {"n": int(k.sum()), "p_ge_0.1": float(np.mean(np.abs(nl[k, j]) >= CLEAN)) if k.any() else None}
                null_all.append((lab, k.sum(), np.sum(np.abs(nl[k, j]) >= CLEAN)))
            # the count, on clean frames and on all frames, by f0 band
            clean = v & (bs < CLEAN)
            res["count"] = {"clean": _count(h, f0, z["tracks"], clean), "all": _count(h, f0, z["tracks"], v),
                            "clean_by_f0": {bn: _count(h, f0, z["tracks"], clean & sel) for bn, sel in _bands(f0).items()}}
            if corpus == "vocalset":
                paths = json.loads(str(z["paths"]))
                fl = np.array([paths[i] in flagged for i in z["path_i"]])
                res["count"]["clean_flag_screened"] = _count(h, f0, z["tracks"], clean & ~fl)
                res["flag_screened_frames_removed"] = int((clean & fl).sum())
            out[name] = res
            cl = res["clean"]
            lines += ["", f"  {name}: {res['voiced_frames']} voiced label frames (variant 0)",
                      f"    stored tract_left reproduced to {res['tract_left_refit_max_diff']}",
                      f"    measured: model {res['measured']['model']:.1%}, LPC {res['measured']['lpc']:.1%};"
                      f" Spearman |old| vs model {res['agreement']['old_vs_model']}, s1 vs s2 {res['agreement']['s1_vs_s2']},"
                      f" model vs LPC {res['agreement']['model_vs_lpc']}",
                      "    clean frames: " + "; ".join(f"{k} {c['n']} ({c['share']:.1%}; by f0 " +
                                                      ", ".join(f"{bn} {c[bn]:.1%}" for bn in _bands(f0)) + ")" for k, c in cl.items()),
                      "    shuffle null, share |β| >= 0.1: " + ", ".join(f"{k} {r['p_ge_0.1']:.1%}" if r["p_ge_0.1"] is not None else f"{k} -"
                                                                     for k, r in res["null_rate"].items())]
            for lab in ("clean", "all", "clean_flag_screened"):
                cnt = res["count"].get(lab)
                if not cnt:
                    continue
                a, u = cnt.get("h3_h2"), cnt.get("upper")
                lines.append(f"    count ({lab}, {cnt['frames']} frames): " +
                             (f"H3-H2 > 3 dB {a['p_rise_gt3']:.2%} (> 0 {a['p_rise_gt0']:.1%}; pct 50/90/95/99 {a['pct']};"
                              f" of rises > 3 dB near a formant {a['rises_gt3_near_formant']})" if a else "H3-H2 -") + "; " +
                             (f"H4 and up: steps > 3 dB {u['p_step_gt3']:.2%}, frames with any {u['p_frame_any_gt3']:.1%}"
                              f" (near a formant {u['rises_gt3_near_formant']})" if u else "upper -"))
            for bn, cnt in res["count"]["clean_by_f0"].items():
                a = cnt.get("h3_h2")
                if a:
                    lines.append(f"      clean, {bn}: {cnt['frames']} frames, H3-H2 > 3 dB {a['p_rise_gt3']:.2%}"
                                 f" (near a formant {a['rises_gt3_near_formant']})")
    tot = {}
    for lab, n, k in null_all:
        a, b = tot.get(lab, (0, 0))
        tot[lab] = (a + n, b + k)
    rates = {lab: b / max(a, 1) for lab, (a, b) in tot.items()}
    out["null_rate_pooled"] = rates
    lpc_high = rates.get("lpc", 0) > 0.10
    out["rule"] = "LPC null above 10 %: Hannah decides the combination" if lpc_high else "beta* stands"
    lines += ["", "  shuffle null pooled over the sets: " + ", ".join(f"{k} {v:.1%}" for k, v in rates.items()),
              f"Rule (registered): {out['rule']}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "residue.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (REP / "residue.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the report prints β (a Windows console is cp1252)
    c = sys.argv[1]
    if c == "analyse":
        cmd_analyse(sys.argv[2], sys.argv[3])
    elif c == "lengths":
        cmd_lengths()
    elif c == "refit":
        cmd_refit(sys.argv[2], *(int(a) for a in sys.argv[3:4]))
    elif c == "assemble":
        cmd_assemble(sys.argv[2])
    elif c == "flag":
        cmd_flag()
    elif c == "report":
        cmd_report()
    else:
        raise SystemExit(__doc__)
