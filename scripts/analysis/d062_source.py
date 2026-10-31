"""D-062: the area return phase x the aspiration's tilt, 2 x 2 on KL (Rosenberg-C only) and the interactive tube.
Probe, descriptive; readings in docs/decisions.md.

    python scripts/analysis/d062_source.py render <out>     (CPU, parallel: 8 arms x N clips, audio + labels)
    python scripts/analysis/d062_source.py measure <out>    (GPU in short bursts for teacher v3, then CPU)
    python scripts/analysis/d062_source.py report <out>     (CPU: D-045's discriminators refitted, readings)
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

N_CLIPS = int(os.environ.get("D062_N", 160))  # D062_N: a smaller run, to check the code path
SEED = 5062
FS = 16000
PER_CLIP = 60
ENGINES = ("kl", "tube")
ARMS = [f"{e}_{c}_{a}" for e, c, a in itertools.product(ENGINES, ("corner", "return"), ("asis", "d048"))]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip(args):
    arm, i = args
    from dataclasses import replace

    from vocal_analysis.synth import kl_set, sampler

    engine, closure, asp = arm.split("_")
    base = replace(_load("d059_realness").synth_config(), p_tube=1.0, asp_tilt_db_oct=None if asp == "asis" else -6.0)
    ret = closure == "return"
    if engine == "kl":
        c = kl_set.generate(i, replace(kl_set.KLSetConfig(), synth=base, p_lf_pulse=0.0, area_return=ret), SEED)
    else:
        c = sampler.generate(i, replace(base, p_interactive=1.0, interactive_return=ret), SEED)
    lab = c.labels
    keep = {k: lab[k] for k in ("t", "f0", "voiced", "whisper", "formants", "bandwidths", "rd")}
    keep["mixed"] = lab.get("formants_mixed", np.zeros_like(lab["formants"], bool))
    return arm, i, c.audio, keep


def render(out: Path):
    jobs = [(a, i) for a in ARMS for i in range(N_CLIPS) if not (out / "clips" / a / f"{i:04d}.npz").exists()]
    for a in ARMS:
        (out / "clips" / a).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(10) as ex:
        for k, (a, i, audio, lab) in enumerate(ex.map(_clip, jobs, chunksize=2), 1):
            np.savez(out / "clips" / a / f"{i:04d}.npz", audio=audio, **lab)
            if k % 80 == 0:
                print(f"rendered {k}/{len(jobs)}", flush=True)


def _channel(x0, voiced, i, channel):
    from vocal_analysis.synth.channel import augment

    mask = np.pad(np.repeat(voiced, 160), (0, max(0, len(x0) - 160 * len(voiced))))[: len(x0)]
    return augment(x0, np.random.default_rng([SEED, i]), channel, mask, codec_stream=False)[0]


def measure(out: Path):
    import torch

    from vocal_analysis.features.vqt import HOP
    from vocal_analysis.synth.channel import ChannelConfig

    D, d061 = _load("d045_discriminator"), _load("d061_source")
    T = D.Trunk(torch.device("cuda"))
    cfg3 = json.load(open(D.TEACHER_V3_CFG))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    for a in ARMS:
        path = out / f"{a}.npz"
        if path.exists():
            continue
        rows, real_rows = [], []
        for i in range(N_CLIPS):
            z = np.load(out / "clips" / a / f"{i:04d}.npz")
            x0 = z["audio"].astype(np.float64)
            centres = np.round(z["t"] * FS).astype(int)
            ok = (z["voiced"] & ~z["whisper"] & (z["f0"] > 0) & np.isfinite(z["formants"][:, :3]).all(1)
                  & np.isfinite(z["bandwidths"][:, :3]).all(1) & ~z["mixed"][:, :3].any(1)
                  & (centres > 0.02 * FS) & (centres < len(x0) - 0.02 * FS))
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            rng = np.random.default_rng([SEED, i])  # the same frames in every arm of an engine
            idx = np.sort(rng.choice(idx, min(PER_CLIP, len(idx)), replace=False))
            f0, F, B = z["f0"][idx], z["formants"][idx], z["bandwidths"][idx]
            xc = _channel(x0, z["voiced"], i, channel)
            for cond, x in (("clean", x0), ("channel", xc)):
                m = D.measures(x, centres[idx], f0, F[:, 0])
                r = T.file(x.astype(np.float32))
                ti = np.clip(np.round(centres[idx] / HOP).astype(int), 0, len(r["f0"]) - 1)
                rows.append({"cond": np.full(len(idx), cond), "clip": np.full(len(idx), i), "frame": idx, "f0": f0,
                             "rd": z["rd"][idx], "h1h2": m[:, 0], "tilt": m[:, 4], "exact": d061.corrected(m[:, 0], f0, F, B),
                             "teacher": d061.corrected(m[:, 0], f0, r["F"][ti], r["B"][ti])})
                if cond == "channel":  # D-059's realness frames: teacher-selected
                    sel = D.select(r, np.random.default_rng([SEED, i, 1]), PER_CLIP)
                    if len(sel):
                        real_rows.append(D.rows_from(r, sel, x, sel * HOP, {"speaker": f"{a}:{i}", "sex": "?", "age": -1}))
        np.savez(path, **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
        np.savez(out / f"{a}_disc.npz", **D.stack(real_rows))
        D.log(f"{a}: {sum(len(r['f0']) for r in rows)} label frames, {sum(len(r['f0']) for r in real_rows)} teacher frames")


def report(out: Path):
    D = _load("d045_discriminator")
    load = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}  # noqa: E731
    arms = {a: load(out / f"{a}.npz") for a in ARMS}
    disc = {a: load(out / f"{a}_disc.npz") for a in ARMS}
    syn, zer, so = load(Path("cache/d045/synth.npz")), load(Path("cache/d045/zeroth.npz")), load(Path("cache/d045/so.npz"))
    real = {"zeroth": zer, "so_adult": {k: v[so["group"] == "adult"] for k, v in so.items()}}
    rng = np.random.default_rng(62)
    NB = 1000
    clips = np.arange(N_CLIPS)
    draws = [rng.choice(clips, len(clips)) for _ in range(NB)]

    def per_clip(d, cond, stat):
        """stat(rows of one resample) over clip resamples: (point, lo, hi, boots)"""
        m = d["cond"] == cond
        by = {c: np.flatnonzero(m & (d["clip"] == c)) for c in clips}
        pt = stat(np.flatnonzero(m))
        b = np.array([stat(np.concatenate([by[c] for c in dr])) for dr in draws])
        return pt, b

    def fmt(pt, b):
        return f"{pt:+6.2f} [{np.nanpercentile(b, 2.5):+.2f}, {np.nanpercentile(b, 97.5):+.2f}]"

    L = [f"D-062: return phase x aspiration tilt, {N_CLIPS} clips per arm; medians [95 % paired bootstrap over clips]", ""]
    # Rd tertile edges per engine from its corner / as-is arm
    edges = {e: np.nanpercentile(arms[f"{e}_corner_asis"]["rd"][arms[f"{e}_corner_asis"]["cond"] == "clean"], [33.3, 66.7])
             for e in ENGINES}

    def tert(d, e, j):
        lo, hi = [(-np.inf, edges[e][0]), (edges[e][0], edges[e][1]), (edges[e][1], np.inf)][j]
        return (d["rd"] >= lo) & (d["rd"] < hi)

    def slope_stat(d, e, key="tilt"):
        t1, t3 = tert(d, e, 0), tert(d, e, 2)

        def s(ii):
            sel = np.zeros(len(d["rd"]), bool)
            sel[ii] = True
            return np.nanmedian(d[key][sel & t3]) - np.nanmedian(d[key][sel & t1])
        return s

    L.append("Clean, by Rd tertile (engine's edges): tilt dB/oct | H1-H2 | exact H1*-H2* dB (frames)")
    for a in ARMS:
        d, e = arms[a], a.split("_")[0]
        cl = d["cond"] == "clean"
        parts = []
        for j in range(3):
            m = cl & tert(d, e, j)
            parts.append(f"T{j + 1} {np.nanmedian(d['tilt'][m]):+6.2f} | {np.nanmedian(d['h1h2'][m]):+6.2f} | "
                         f"{np.nanmedian(d['exact'][m]):+6.2f} ({m.sum()})")
        L.append(f"  {a:20s} " + "   ".join(parts))
    L.append(f"  Rd tertile edges: " + ", ".join(f"{e} {edges[e][0]:.2f} / {edges[e][1]:.2f}" for e in ENGINES))
    L += ["", "1. Return ties tilt to Rd: tilt slope (T3 - T1, clean), return minus corner (rule <= -2 with interval below 0)"]
    for e in ENGINES:
        for asp in ("asis", "d048"):
            c, r = arms[f"{e}_corner_{asp}"], arms[f"{e}_return_{asp}"]
            pc, bc = per_clip(c, "clean", slope_stat(c, e))
            pr, br = per_clip(r, "clean", slope_stat(r, e))
            ok = (pr - pc) <= -2 and np.nanpercentile(br - bc, 97.5) < 0
            L.append(f"  {e:4s} {asp:4s}: corner {fmt(pc, bc)}  return {fmt(pr, br)}  difference {fmt(pr - pc, br - bc)}"
                     f" -> {'yes' if ok else 'no'}")
    L += ["", "2. Aspiration: D-048 minus as-is, clean tilt median (overall; T1 / T2 / T3)"]
    for e in ENGINES:
        for cl in ("corner", "return"):
            a0, a1 = arms[f"{e}_{cl}_asis"], arms[f"{e}_{cl}_d048"]
            med = lambda d, extra=None: (lambda ii: np.nanmedian(d["tilt"][ii] if extra is None else d["tilt"][ii][extra[ii]]))  # noqa: E731
            p0, b0 = per_clip(a0, "clean", med(a0))
            p1, b1 = per_clip(a1, "clean", med(a1))
            ts = []
            for j in range(3):
                q0 = np.nanmedian(a0["tilt"][(a0["cond"] == "clean") & tert(a0, e, j)])
                q1 = np.nanmedian(a1["tilt"][(a1["cond"] == "clean") & tert(a1, e, j)])
                ts.append(f"{q1 - q0:+.2f}")
            L.append(f"  {e:4s} {cl:6s}: {fmt(p1 - p0, b1 - b0)}   ({' / '.join(ts)})")
    L += ["", "3. Channel tilt against real voices (dB/oct; match if within 1.5 of both corpora)"]
    rt = {c: float(np.nanmedian(d["meas"][:, 4])) for c, d in real.items()}
    L.append("  real: " + ", ".join(f"{c} {v:+.2f} (f {np.nanmedian(d['meas'][d['sex'] == 'f', 4]):+.2f}, "
                                    f"m {np.nanmedian(d['meas'][d['sex'] == 'm', 4]):+.2f})" for c, d in real.items() for v in [rt[c]]))
    for a in ARMS:
        d = arms[a]
        p, b = per_clip(d, "channel", lambda ii, d=d: np.nanmedian(d["tilt"][ii]))
        ok = all(abs(p - v) <= 1.5 for v in rt.values())
        L.append(f"  {a:20s} {fmt(p, b)}  teacher H1*-H2* {np.nanmedian(d['teacher'][d['cond'] == 'channel']):+.2f}"
                 f" -> {'matches' if ok else 'no'}")
    L += ["", "4. Realness (D-045's discriminators refitted; log-odds real, f0-weighted median); minus the engine's corner_asis"]
    for kind in ("h", "a"):
        for cname, d in real.items():
            for sx in "fm":
                r = {k: v[d["sex"] == sx] for k, v in d.items()}
                ws = D.f0_weights(syn["f0"], r["f0"])
                model, auc = D.fit_one(D.design(syn, kind), ws, syn["speaker"], D.design(r, kind), r["speaker"])
                L.append(f"  D_{kind} {cname} {sx} (CV AUC {auc:.3f})")
                sc = {}
                for a in ARMS:
                    dd = disc[a]
                    v, w = model.decision_function(D.design(dd, kind)), D.f0_weights(dd["f0"], r["f0"])
                    clip = np.array([int(s.split(":")[1]) for s in dd["speaker"]])
                    by = {c: np.flatnonzero(clip == c) for c in clips}

                    def wmed(ii):
                        if not len(ii):
                            return np.nan
                        o = np.argsort(v[ii])
                        cw = np.cumsum(w[ii][o])
                        return float(v[ii][o][np.searchsorted(cw, 0.5 * cw[-1])])
                    sc[a] = (wmed(np.arange(len(v))), np.array([wmed(np.concatenate([by[c] for c in dr])) for dr in draws]))
                for a in ARMS:
                    e = a.split("_")[0]
                    p, b = sc[a]
                    p0, b0 = sc[f"{e}_corner_asis"]
                    flag = ""
                    if kind == "h" and sx == "f" and a != f"{e}_corner_asis":
                        flag = "  (>= 1, interval above 0)" if (p - p0) >= 1 and np.nanpercentile(b - b0, 2.5) > 0 else ""
                    L.append(f"    {a:20s} {fmt(p, b)}   minus baseline {fmt(p - p0, b - b0)}{flag}")
    text = "\n".join(L)
    print(text)
    (out / "d062.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"render": render, "measure": measure, "report": report}[sys.argv[1]](Path(sys.argv[2]))
