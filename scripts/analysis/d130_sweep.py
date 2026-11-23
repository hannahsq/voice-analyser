"""D-130: the vowel-plane sweep probe. Smooth paths through Story's (q1, q2) shape space (plus lip protrusion), exact
labels from `tube.poles`, rendered at fixed f0 through the cascade; each model's tracked formants against the truth.
The teacher (non-causal, windowed as `va analyse`) is the control for D-128's causal students (whole clip at once,
which equals their streaming).

    python scripts/analysis/d130_sweep.py render [--quick]          -> data/probes/vowelplane_v1/
    python scripts/analysis/d130_sweep.py score NAME CKPT [...]     -> reports/probes/vowelplane/NAME_raw.npz ($VA_DEVICE)
    python scripts/analysis/d130_sweep.py analyse TEACHERS STUDENT_GROUPS...
        TEACHERS: comma list of names (the control, seed-averaged); each STUDENT_GROUP: label=comma list of names
        -> reports/analysis/d130/ (tables, maps) and reports/probes/vowelplane/<name>.json

Implementation details fixed before anything was rendered (D-130 note): the 10 %-extended box closes the tract in
places, so each path point is pulled towards the box centre until the narrowest section is at least 0.05 cm^2;
labels are interpolated (log frequency) between exact solutions at about 600 points per path.
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("D130_CACHE", ROOT / "data/probes/vowelplane_v1"))  # overrides: quick tests only
RAW = Path(os.environ.get("D130_RAW", ROOT / "reports/probes/vowelplane"))
OUT = Path(os.environ.get("D130_OUT", ROOT / "reports/analysis/d130"))
FS, HOP = 16000, 160
LENGTHS = (17.5, 15.6, 13.0)
F0S = (100.0, 200.0, 300.0, 400.0)
LIPS = (0.0, 1.0)  # lip protrusion pm, cm
SEEDS = (0, 1)
SPEEDS = {"slow": 25.0, "fast": 75.0}  # combined F1/F2 movement, cents per 10 ms frame
CONDS = ("clean", "reverb_0.4", "reverb_0.8")
INTERLEAVED = {"a": (17.5, 100.0), "b": (15.6, 220.0)}
N_PATH = 600
MIN_AREA = 0.05
N_P = 6  # poles rendered


# ----------------------------------------------------------------------------- paths and labels


def raster(seed: int, n: int = N_PATH, rows: int = 6) -> np.ndarray:
    """(n, 2) unit-square points: a boustrophedon with semicircular turns, rows along x (seed 0) or y (seed 1)."""
    pts = []
    ys = np.linspace(0.05, 0.95, rows)
    r = (ys[1] - ys[0]) / 2
    for i, y in enumerate(ys):
        x = np.linspace(0.05, 0.95, 200)
        pts.append(np.column_stack([x if i % 2 == 0 else x[::-1], np.full(200, y)]))
        if i < rows - 1:
            a = np.linspace(-np.pi / 2, np.pi / 2, 40)
            side = 0.95 if i % 2 == 0 else 0.05
            sgn = 1 if i % 2 == 0 else -1
            pts.append(np.column_stack([side + sgn * r * np.cos(a) * 0.5, y + r + r * np.sin(a)]))
    p = np.concatenate(pts)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    u = np.linspace(0, seg[-1], n)
    p = np.column_stack([np.interp(u, seg, p[:, 0]), np.interp(u, seg, p[:, 1])])
    return p[:, ::-1] if seed == 1 else p


def shape_path(seed: int) -> np.ndarray:
    """(N_PATH, 2) (q1, q2) in Story's box extended by 10 %, each pulled towards the centre while the tract closes."""
    from vocal_analysis.synth import tube

    lo = np.array([tube.Q1_RANGE[0], tube.Q2_RANGE[0]])
    hi = np.array([tube.Q1_RANGE[1], tube.Q2_RANGE[1]])
    lo, hi = lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)
    q = lo + raster(seed) * (hi - lo)
    c = (lo + hi) / 2
    for i in range(len(q)):
        lam = 1.0
        while tube.vowel_areas(*(c + lam * (q[i] - c))).min() < MIN_AREA and lam > 0:
            lam -= 0.02
        q[i] = c + lam * (q[i] - c)
    return q


def solve(q: np.ndarray, length: float, pm: float) -> tuple[np.ndarray, np.ndarray]:
    from vocal_analysis.synth import tube

    F, Bw = np.full((len(q), N_P), np.nan), np.full((len(q), N_P), np.nan)
    lost = []
    for i, (a, b) in enumerate(q):
        try:
            f, bw = tube.poles(tube.build_shape(a, b, length, pm=pm), f_max_hz=7600.0)
        except RuntimeError:  # the continuation lost a root: filled from the neighbouring path points below
            lost.append(i)
            continue
        k = min(N_P, len(f))
        F[i, :k], Bw[i, :k] = f[:k], bw[:k]
    if lost:
        good = np.setdiff1d(np.arange(len(q)), lost)
        for k in range(N_P):
            for A in (F, Bw):
                ok = good[np.isfinite(A[good, k])]
                if len(ok) == len(good):
                    A[lost, k] = 2 ** np.interp(lost, ok, np.log2(A[ok, k]))
        print(f"  length {length}, pm {pm}: {len(lost)} of {len(q)} path points had no exact solution; interpolated", flush=True)
    return F, Bw


def timing(F: np.ndarray, speed: float) -> np.ndarray:
    """Seconds at each path point so the combined F1/F2 movement is `speed` cents per 10 ms."""
    c = 1200 * np.log2(F[:, :2])
    d = np.hypot(*np.diff(c, axis=0).T)
    return np.r_[0, np.cumsum(d / speed)] * 0.01


def render(F, Bw, t_path, f0, seed_key):
    """Audio of the path at fixed f0 (LF Rd 1.0, the cascade), and labels on 10 ms frames."""
    from vocal_analysis.synth import tract, voice

    dur = t_path[-1] + 0.1
    C = int(dur * FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / FS
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    tp = np.clip(tc, 0, t_path[-1])
    for k in range(N_P):
        ok = np.isfinite(F[:, k])
        if ok.sum() < len(F):  # a pole missing somewhere: render without it
            continue
        pf[:, k] = 2 ** np.interp(tp, t_path, np.log2(F[:, k]))
        pb[:, k] = 2 ** np.interp(tp, t_path, np.log2(Bw[:, k]))
    amp = np.clip(np.minimum(tc, dur - tc) / 0.02, 0, 1)  # 20 ms fades
    traj = voice.Trajectories(f0=np.full(C, f0), rd=np.full(C, 1.0), voice_amp=amp, whisper_amp=np.zeros(C),
                              fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(traj, np.random.default_rng(seed_key))["total"]
    y = 0.5 * y / (np.abs(y).max() + 1e-12)
    tf = np.arange(int(dur / 0.01) + 1) * 0.01
    lab = np.stack([2 ** np.interp(np.clip(tf, 0, t_path[-1]), t_path, np.log2(F[:, k])) for k in range(4)], 1)
    return y.astype(np.float32), tf, lab


def cmd_render(quick: bool = False) -> None:
    from scipy.signal import fftconvolve

    from vocal_analysis.synth import channel as ch

    CACHE.mkdir(parents=True, exist_ok=True)
    lengths, f0s, lips, seeds = (LENGTHS[:1], F0S[:2], LIPS[:1], SEEDS[:1]) if quick else (LENGTHS, F0S, LIPS, SEEDS)
    items = []
    paths = {}
    for seed in seeds:
        q = shape_path(seed)
        for L in lengths:
            for pm in lips:
                F, Bw = solve(q, L, pm)
                paths[(seed, L, pm)] = (q, F, Bw)
                print(f"path seed {seed}, length {L}, pm {pm}: F1 {np.nanmin(F[:, 0]):.0f}-{np.nanmax(F[:, 0]):.0f}, "
                      f"F2 {np.nanmin(F[:, 1]):.0f}-{np.nanmax(F[:, 1]):.0f} Hz", flush=True)
    u_path = np.linspace(0, 1, N_PATH)

    def save(name, y, tf, lab, extra):
        arrays = {"times": tf, "formants": lab, **extra}
        for ci, cond in enumerate(CONDS):
            if cond == "clean":
                z = y
            else:
                rng = np.random.default_rng([zlib.crc32(name.encode()), ci])
                z = fftconvolve(y, ch.synth_rir(rng, float(cond.split("_")[1])))[: len(y)]
            arrays[f"audio_{cond}"] = np.round(z / (np.abs(z).max() + 1e-12) * 16000).astype(np.int16)
        np.savez_compressed(CACHE / f"{name}.npz", **arrays)

    for (seed, L, pm), (q, F, Bw) in paths.items():
        for speed, v in SPEEDS.items():
            t_path = timing(F, v)
            for direction in ("fwd", "rev"):
                tp = t_path if direction == "fwd" else (t_path[-1] - t_path)[::-1]
                Fd, Bd, ud, qd = (F, Bw, u_path, q) if direction == "fwd" else (F[::-1], Bw[::-1], u_path[::-1], q[::-1])
                for f0 in f0s:
                    name = f"s{seed}_L{L:g}_pm{pm:g}_{speed}_{direction}_f{f0:g}"
                    y, tf, lab = render(Fd, Bd, tp, f0, zlib.crc32(name.encode()))
                    tt = np.clip(tf, 0, tp[-1])
                    extra = {"u": np.interp(tt, tp, ud), "q": np.stack([np.interp(tt, tp, qd[:, k]) for k in range(2)], 1)}
                    save(name, y, tf, lab, extra)
                    items.append({"name": name, "seed": seed, "length": L, "pm": pm, "speed": speed, "direction": direction,
                                  "f0": f0, "kind": "sweep"})
            print(f"rendered seed {seed}, length {L}, pm {pm}, {speed}", flush=True)
    # interleaved voices: alternating segments of two speakers' slow forward sweeps, gaps of 0-150 ms
    if not quick:
        for k in range(4):
            rng = np.random.default_rng([130, k])
            src = {}
            for key, (L, f0) in INTERLEAVED.items():
                q, F, Bw = paths[(k % 2, L, 0.0)] if (k % 2, L, 0.0) in paths else (None, *solve(shape_path(k % 2), L, 0.0))
                y, tf, lab = render(F, Bw, timing(F, SPEEDS["slow"]), f0, zlib.crc32(f"il{k}{key}".encode()))
                src[key] = (y, lab)
            pos = {"a": 0, "b": 0}
            ys, labs, spk, start = [], [], [], []
            key = "a"
            while all(pos[s] < len(src[s][1]) - 20 for s in pos):
                n = int(rng.uniform(40, 150))  # frames
                y, lab = src[key]
                seg = y[pos[key] * HOP:(pos[key] + n) * HOP].copy()
                fade = np.minimum(1, np.minimum(np.arange(len(seg)), np.arange(len(seg))[::-1]) / 80.0)
                ys.append(seg * fade)
                labs.append(lab[pos[key]:pos[key] + n][: len(seg) // HOP])
                spk += [key] * (len(seg) // HOP)
                start += [1] + [0] * (len(seg) // HOP - 1)
                pos[key] += n
                gap = int(rng.uniform(0, 15))
                ys.append(np.zeros(gap * HOP + (len(seg) - len(seg) // HOP * HOP)))
                labs.append(np.full((gap, 4), np.nan))
                spk += [""] * gap
                start += [0] * gap
                key = "b" if key == "a" else "a"
            y = np.concatenate(ys).astype(np.float32)
            lab = np.concatenate(labs)
            tf = np.arange(len(lab)) * 0.01
            name = f"interleaved_{k}"
            save(name, y, tf, lab, {"speaker": np.array(spk), "segment_start": np.array(start, bool)})
            items.append({"name": name, "kind": "interleaved"})
            print(f"rendered {name}: {len(y) / FS:.0f} s", flush=True)
    (CACHE / "meta.json").write_text(json.dumps({"items": items, "conditions": list(CONDS)}, indent=1))
    print(f"{len(items)} items")


# ----------------------------------------------------------------------------- scoring


def cmd_score(name: str, ckpt: str) -> None:
    import torch

    from vocal_analysis.models.infer import DEFAULT_WINDOW, analyse, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ckpt, dev)
    causal = bool(model.cfg.causal)
    meta = json.loads((CACHE / "meta.json").read_text())
    out = {}
    for it in meta["items"]:
        d = np.load(CACHE / f"{it['name']}.npz")
        for cond in meta["conditions"]:
            r = analyse(model, d[f"audio_{cond}"].astype(np.float32) / 32000, dev,  # stored at peak 16000: peak 0.5
                        window_frames=None if causal else DEFAULT_WINDOW)
            n = len(d["times"])
            pad = lambda a: np.concatenate([a[:n], np.full((max(0, n - len(a)), *a.shape[1:]), np.nan)])  # noqa: E731
            out[f"{it['name']}|{cond}|F"] = pad(r["formants"][:, :4])
            out[f"{it['name']}|{cond}|sig"] = pad(r["formant_sigma_cents"][:, :4])
            out[f"{it['name']}|{cond}|son"] = pad(r["sonorant_p"])
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / f"{name}_raw.npz", **out)
    print(f"{name}: {len(meta['items'])} items x {len(meta['conditions'])} conditions ({'causal, whole clip' if causal else 'windowed'})")


# ----------------------------------------------------------------------------- analysis

GRID_C = 25.0  # cents
DISP_C = 100.0


def _load(names):
    return [np.load(RAW / f"{n}_raw.npz") for n in names]


def _frames(meta, raws, cond, sel=lambda it: True):
    """Concatenated frames over selected sweep items: true F (n, 4), each model's F, sigma, sonorant, plus item keys."""
    T, Ms, Ss, Ps, keys = [], [[] for _ in raws], [[] for _ in raws], [[] for _ in raws], []
    for it in meta["items"]:
        if it["kind"] != "sweep" or not sel(it):
            continue
        d = np.load(CACHE / f"{it['name']}.npz")
        T.append(d["formants"])
        keys += [it["name"]] * len(d["formants"])
        for j, r in enumerate(raws):
            Ms[j].append(r[f"{it['name']}|{cond}|F"])
            Ss[j].append(r[f"{it['name']}|{cond}|sig"])
            Ps[j].append(r[f"{it['name']}|{cond}|son"])
    return (np.concatenate(T), [np.concatenate(m) for m in Ms], [np.concatenate(s) for s in Ss],
            [np.concatenate(p) for p in Ps], np.array(keys))


def _slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def measures(T, M, S, P, edge_frames=None) -> dict:
    """Measures 2-5 and 7 (1 and the maps are drawn separately) for one model on frames with p(sonorant) > 0.5."""
    ok = np.isfinite(T[:, :3]).all(1) & np.isfinite(M[:, :2]).all(1) & (P > 0.5)
    t, m, s = T[ok], M[ok], S[ok]
    out = {"frames": int(ok.sum()), "sonorant_share": float(np.mean(P[np.isfinite(T[:, 0])] > 0.5))}
    ct, cm = 1200 * np.log2(t[:, :2]), 1200 * np.log2(m[:, :2])
    for k, f in enumerate(("F1", "F2")):
        out[f"{f}_slope"] = _slope(ct[:, k], cm[:, k])
        out[f"{f}_median_abs_error_cents"] = float(np.median(np.abs(cm[:, k] - ct[:, k])))
        out[f"{f}_calibration"] = float(np.median(np.abs(cm[:, k] - ct[:, k])) / np.median(s[:, k]))
    out["fold_F2_to_F3"] = float(np.mean(np.abs(m[:, 1] - t[:, 2]) < np.abs(m[:, 1] - t[:, 1])))
    out["fold_F1_to_F2"] = float(np.mean(np.abs(m[:, 0] - t[:, 1]) < np.abs(m[:, 0] - t[:, 0])))
    # density ratio on a 25-cent grid, both smoothed by 50 cents
    from scipy.ndimage import gaussian_filter

    lo, hi = np.floor(ct.min(0) / GRID_C) * GRID_C - 200, np.ceil(ct.max(0) / GRID_C) * GRID_C + 200
    bins = [np.arange(lo[k], hi[k] + GRID_C, GRID_C) for k in range(2)]
    ht = gaussian_filter(np.histogram2d(ct[:, 0], ct[:, 1], bins)[0], 50 / GRID_C)
    hm = gaussian_filter(np.histogram2d(cm[:, 0], cm[:, 1], bins)[0], 50 / GRID_C)
    ht, hm = ht / ht.sum(), hm / hm.sum()
    mask = ht > 0.01 * ht.max()
    ratio = np.where(mask, hm / np.maximum(ht, 1e-12), np.nan)
    out["attractor_index_p99"] = float(np.nanpercentile(ratio, 99))
    out["hole_share"] = float(ht[mask & (ratio < 0.5)].sum() / ht[mask].sum())
    return out


def harmonic_stripes(T, M, P, f0) -> dict:
    ok = np.isfinite(T[:, :2]).all(1) & np.isfinite(M[:, :2]).all(1) & (P > 0.5)
    res = {}
    for k, f in enumerate(("F1", "F2")):
        r = lambda x: np.abs(np.mean(np.exp(2j * np.pi * x / f0)))  # noqa: E731
        res[f"{f}_stripe"] = float(r(M[ok, k]) - r(T[ok, k]))
    return res


def lag_ms(meta, raw, cond) -> dict:
    """Forward against reversed paths, matched by true path position u: lag = -(d_fwd - d_rev) . v / (2 |v|^2)."""
    out = {}
    for speed in SPEEDS:
        lags = []
        for it in meta["items"]:
            if it["kind"] != "sweep" or it["direction"] != "fwd" or it["speed"] != speed:
                continue
            rev = it["name"].replace("_fwd_", "_rev_")
            df, dr = np.load(CACHE / f"{it['name']}.npz"), np.load(CACHE / f"{rev}.npz")
            Mf, Mr = raw[f"{it['name']}|{cond}|F"], raw[f"{rev}|{cond}|F"]
            Pf, Pr = raw[f"{it['name']}|{cond}|son"], raw[f"{rev}|{cond}|son"]
            uf, ur = df["u"], dr["u"]
            order = np.argsort(ur)
            j = np.clip(np.searchsorted(ur[order], uf), 0, len(ur) - 1)
            jr = order[j]
            ct = 1200 * np.log2(df["formants"][:, :2])
            v = np.gradient(ct, axis=0) / 10.0  # cents per ms along the forward path
            d_f = 1200 * np.log2(Mf[:, :2]) - ct
            d_r = 1200 * np.log2(Mr[jr, :2]) - 1200 * np.log2(dr["formants"][jr, :2])
            sp2 = (v ** 2).sum(1)
            ok = np.isfinite(d_f).all(1) & np.isfinite(d_r).all(1) & (Pf > 0.5) & (Pr[jr] > 0.5) & (sp2 > 1e-4) \
                & (np.abs(uf - ur[jr]) < 2 / N_PATH)
            lags.append(-((d_f - d_r) * v).sum(1)[ok] / (2 * sp2[ok]))
        if lags:
            out[f"lag_ms_{speed}"] = float(np.median(np.concatenate(lags)))
    return out


def interleaved(meta, raw, cond) -> dict:
    """Median |error| (cents) F1, F2 in the first 30 frames after a speaker change, against the rest."""
    first, rest = [], []
    for it in meta["items"]:
        if it["kind"] != "interleaved":
            continue
        d = np.load(CACHE / f"{it['name']}.npz")
        T, M, P = d["formants"], raw[f"{it['name']}|{cond}|F"], raw[f"{it['name']}|{cond}|son"]
        since = np.zeros(len(T), int)
        c = 10 ** 6
        for i, s in enumerate(d["segment_start"]):
            c = 0 if s else c + 1
            since[i] = c
        ok = np.isfinite(T[:, :2]).all(1) & np.isfinite(M[:, :2]).all(1) & (P > 0.5)
        e = np.abs(1200 * np.log2(M[:, :2] / T[:, :2]))
        first.append(e[ok & (since < 30)])
        rest.append(e[ok & (since >= 30)])
    if not first:
        return {}
    f, r = np.concatenate(first), np.concatenate(rest)
    return {"F1_err_first300ms": float(np.median(f[:, 0])), "F1_err_rest": float(np.median(r[:, 0])),
            "F2_err_first300ms": float(np.median(f[:, 1])), "F2_err_rest": float(np.median(r[:, 1]))}


def error_map(ax, T, M, P, title):
    ok = np.isfinite(T[:, :2]).all(1) & np.isfinite(M[:, :2]).all(1) & (P > 0.5)
    ct, cm = 1200 * np.log2(T[ok, :2]), 1200 * np.log2(M[ok, :2])
    cell = np.floor(ct / DISP_C).astype(int)
    keys, inv = np.unique(cell, axis=0, return_inverse=True)
    X, Y, U, V = [], [], [], []
    for i, kk in enumerate(keys):
        m = inv.ravel() == i
        if m.sum() < 20:
            continue
        d = np.median(cm[m] - ct[m], 0)
        X.append(np.median(ct[m, 1])), Y.append(np.median(ct[m, 0])), U.append(d[1]), V.append(d[0])
    X, Y, U, V = map(np.array, (X, Y, U, V))
    ax.quiver(X / 1200, Y / 1200, U / 1200, V / 1200, angles="xy", scale_units="xy", scale=1, width=0.004, color="#2a6db0")
    ax.set_title(title, fontsize=9)
    _axes(ax)


def _axes(ax):
    hz = np.array([200, 300, 500, 800, 1200, 2000, 3000])
    ax.set_xticks(np.log2(hz)), ax.set_xticklabels(hz, fontsize=7)
    ax.set_yticks(np.log2(hz)), ax.set_yticklabels(hz, fontsize=7)
    ax.set_xlim(np.log2(3600), np.log2(500)), ax.set_ylim(np.log2(1400), np.log2(180))
    ax.set_xlabel("true F2 (Hz)", fontsize=8), ax.set_ylabel("true F1 (Hz)", fontsize=8)
    ax.grid(color="#dddddd", linewidth=0.5)


def ratio_map(ax, T, M, P, title):
    from scipy.ndimage import gaussian_filter

    ok = np.isfinite(T[:, :2]).all(1) & np.isfinite(M[:, :2]).all(1) & (P > 0.5)
    lt, lm = np.log2(T[ok, :2]), np.log2(M[ok, :2])
    b1, b2 = np.arange(np.log2(180), np.log2(1400), GRID_C / 1200), np.arange(np.log2(500), np.log2(3600), GRID_C / 1200)
    ht = gaussian_filter(np.histogram2d(lt[:, 0], lt[:, 1], [b1, b2])[0], 2)
    hm = gaussian_filter(np.histogram2d(lm[:, 0], lm[:, 1], [b1, b2])[0], 2)
    ht, hm = ht / ht.sum(), hm / hm.sum()
    lr = np.log2(np.maximum(hm, 1e-9) / np.maximum(ht, 1e-9))
    lr[(ht < 0.01 * ht.max()) & (hm < 0.01 * hm.max())] = np.nan
    im = ax.pcolormesh(b2, b1, np.clip(lr, -2, 2), cmap="RdBu_r", vmin=-2, vmax=2, shading="flat")
    ax.set_title(title, fontsize=9)
    _axes(ax)
    return im


def cmd_analyse(teachers: list[str], groups: list[tuple[str, list[str]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((CACHE / "meta.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    models = [("teacher", teachers)] + groups
    lines, js = [], {}
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731
    say("D-130: the vowel-plane sweep (registered 2026-11-23). Seed means; each model in brackets. Frames with p(sonorant) > 0.5.")
    for label, names in models:
        raws = _load(names)
        js[label] = {}
        say(f"\n== {label} ({', '.join(names)})")
        for cond in CONDS:
            for L in LENGTHS:
                for f0 in F0S:
                    T, Ms, Ss, Ps, _ = _frames(meta, raws, cond, lambda it: it["length"] == L and it["f0"] == f0)
                    per = [{**measures(T, M, S, P), **harmonic_stripes(T, M, P, f0)} for M, S, P in zip(Ms, Ss, Ps)]
                    mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
                    js[label][f"{cond}|L{L:g}|f{f0:g}"] = {"mean": mean, "each": per}
                    say(f"  {cond:10s} L {L:4.1f} f0 {f0:3.0f}: slope F1 {mean['F1_slope']:.3f} F2 {mean['F2_slope']:.3f}; "
                        f"|err| F1 {mean['F1_median_abs_error_cents']:.0f} F2 {mean['F2_median_abs_error_cents']:.0f} c; "
                        f"calib F1 {mean['F1_calibration']:.2f} F2 {mean['F2_calibration']:.2f}; folds F2>F3 {100 * mean['fold_F2_to_F3']:.1f} % "
                        f"F1>F2 {100 * mean['fold_F1_to_F2']:.1f} %; attractor {mean['attractor_index_p99']:.2f} holes {100 * mean['hole_share']:.1f} %; "
                        f"stripes F1 {mean['F1_stripe']:+.3f} F2 {mean['F2_stripe']:+.3f}; sonorant {100 * mean['sonorant_share']:.0f} %")
            lg = [lag_ms(meta, r, cond) for r in raws]
            il = [interleaved(meta, r, cond) for r in raws]
            js[label][f"{cond}|lag"] = lg
            js[label][f"{cond}|interleaved"] = il
            say(f"  {cond:10s} lag (ms, + = behind): " + "; ".join(f"{k} {np.mean([x[k] for x in lg]):+.1f}" for k in lg[0]))
            if il[0]:
                say(f"  {cond:10s} interleaved |err| first 300 ms vs rest: F1 {np.mean([x['F1_err_first300ms'] for x in il]):.0f} vs "
                    f"{np.mean([x['F1_err_rest'] for x in il]):.0f} c, F2 {np.mean([x['F2_err_first300ms'] for x in il]):.0f} vs "
                    f"{np.mean([x['F2_err_rest'] for x in il]):.0f} c")
        for n, r in zip(names, raws):
            (RAW / f"{n}.json").write_text(json.dumps({"label": label, **js[label]}, indent=1))
    # paired: each student group minus the teacher, slopes and errors, clean, per length (all f0 pooled)
    say("\nStudent minus teacher (seed means), clean:")
    for label, _ in groups:
        for L in LENGTHS:
            d = [js[label][f"clean|L{L:g}|f{f0:g}"]["mean"][k] - js["teacher"][f"clean|L{L:g}|f{f0:g}"]["mean"][k]
                 for f0 in F0S for k in ("F1_median_abs_error_cents", "F2_median_abs_error_cents")]
            say(f"  {label} L {L:4.1f}: |err| change F1 {np.mean(d[0::2]):+.1f} c, F2 {np.mean(d[1::2]):+.1f} c (mean over f0)")
    (OUT / "d130.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # maps: per model label, clean and reverb 0.8, all lengths and f0s pooled
    for label, names in models:
        raws = _load(names)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for col, cond in enumerate(("clean", "reverb_0.8")):
            T, Ms, Ss, Ps, _ = _frames(meta, raws, cond)
            error_map(axes[0, col], T, Ms[0], Ps[0], f"{label} ({names[0]}), {cond}: median error per cell")
            im = ratio_map(axes[1, col], T, Ms[0], Ps[0], f"{label} ({names[0]}), {cond}: log2 tracked / true density")
        fig.colorbar(im, ax=axes[1, :], shrink=0.7, label="log2 density ratio (red: piles up, blue: thins out)")
        fig.savefig(OUT / f"maps_{label}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"written {OUT}")


def cmd_maps_by_f0(models: list[tuple[str, str]], cond: str = "clean") -> None:
    """Extra (descriptive, added after the registered maps were seen): the error field per f0, one row per model
    (label=name), pooled over lengths, lips, directions and speeds; frames whose true F1 is below f0 (no harmonic
    can show it) drawn grey."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((CACHE / "meta.json").read_text())
    fig, axes = plt.subplots(len(models), len(F0S), figsize=(4.2 * len(F0S), 3.8 * len(models)), squeeze=False)
    for r, (label, name) in enumerate(models):
        raw = _load([name])
        for c, f0 in enumerate(F0S):
            T, Ms, _, Ps, _ = _frames(meta, raw, cond, lambda it: it["f0"] == f0)
            seen = T[:, 0] >= f0
            ax = axes[r, c]
            error_map(ax, T[seen], Ms[0][seen], Ps[0][seen], f"{label}, f0 {f0:g} Hz, {cond}")
            ax.axhline(np.log2(f0), color="#999999", linewidth=1, linestyle="--")
    fig.savefig(OUT / f"maps_by_f0_{cond}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"written {OUT / f'maps_by_f0_{cond}.png'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "maps_by_f0":
        cmd_maps_by_f0([(a.split("=")[0], a.split("=")[1]) for a in sys.argv[2:]])
        raise SystemExit
    if cmd == "render":
        cmd_render("--quick" in sys.argv)
    elif cmd == "score":
        args = sys.argv[2:]
        for i in range(0, len(args), 2):
            cmd_score(args[i], args[i + 1])
    elif cmd == "analyse":
        groups = [(g.split("=", 1)[0], g.split("=", 1)[1].split(",")) for g in sys.argv[3:]]
        cmd_analyse(sys.argv[2].split(","), groups)
    else:
        raise SystemExit(__doc__)
