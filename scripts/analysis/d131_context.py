"""D-131: why do D-128's students carry less of the teacher's low-f0 F1 overread? Three arms against D-130's clips
(clean, slow, f0 100 and 200 Hz): W, the teacher in 100-frame windows; B, the sweep inside Story's box only (no
extension); J, D-130's paths with a speech-like source (jitter, shimmer, f0 wander).

    python scripts/analysis/d131_context.py render                 -> data/probes/vowelplane_d131/{box,jit}/
    python scripts/analysis/d131_context.py score NAME CKPT        -> reports/probes/vowelplane_d131/NAME_{box,jit,w100}_raw.npz
    python scripts/analysis/d131_context.py analyse                -> reports/analysis/d131/d131.txt
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d130_sweep", ROOT / "scripts/analysis/d130_sweep.py")
d130 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d130)

CACHE = ROOT / "data/probes/vowelplane_d131"
RAW = ROOT / "reports/probes/vowelplane_d131"
OUT = ROOT / "reports/analysis/d131"
F0S = (100.0, 200.0)
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")
JITTER, SHIMMER_DB, WANDER_MS, WANDER_ST = 0.01, 0.5, 150, 2.0
W_SHORT = 100
B = 1000


def box_path(seed: int) -> np.ndarray:
    """D-130's raster and closure rule in Story's box, without the 10 % extension."""
    from vocal_analysis.synth import tube

    lo = np.array([tube.Q1_RANGE[0], tube.Q2_RANGE[0]])
    hi = np.array([tube.Q1_RANGE[1], tube.Q2_RANGE[1]])
    q = lo + d130.raster(seed) * (hi - lo)
    c = (lo + hi) / 2
    for i in range(len(q)):
        lam = 1.0
        while tube.vowel_areas(*(c + lam * (q[i] - c))).min() < d130.MIN_AREA and lam > 0:
            lam -= 0.02
        q[i] = c + lam * (q[i] - c)
    return q


def render_jit(F, Bw, t_path, f0, seed_key):
    """d130.render with the speech-like source: f0 wander, jitter and shimmer (labels unchanged)."""
    from vocal_analysis.synth import sampler, tract, voice

    dur = t_path[-1] + 0.1
    C = int(dur * d130.FS / voice.CTRL) + 1
    tc = np.arange(C) * voice.CTRL / d130.FS
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    tp = np.clip(tc, 0, t_path[-1])
    for k in range(d130.N_P):
        if np.isfinite(F[:, k]).sum() < len(F):
            continue
        pf[:, k] = 2 ** np.interp(tp, t_path, np.log2(F[:, k]))
        pb[:, k] = 2 ** np.interp(tp, t_path, np.log2(Bw[:, k]))
    amp = np.clip(np.minimum(tc, dur - tc) / 0.02, 0, 1)
    rng = np.random.default_rng(seed_key)
    f0c = f0 * 2 ** sampler._smooth_noise(rng, C, WANDER_MS, WANDER_ST / 12.0)
    traj = voice.Trajectories(f0=f0c, rd=np.full(C, 1.0), voice_amp=amp, whisper_amp=np.zeros(C),
                              fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
    y = voice.synthesize(traj, rng, jitter=JITTER, shimmer_db=SHIMMER_DB)["total"]
    y = 0.5 * y / (np.abs(y).max() + 1e-12)
    tf = np.arange(int(dur / 0.01) + 1) * 0.01
    lab = np.stack([2 ** np.interp(np.clip(tf, 0, t_path[-1]), t_path, np.log2(F[:, k])) for k in range(4)], 1)
    return y.astype(np.float32), tf, lab


def cmd_render() -> None:
    for arm in ("box", "jit"):
        out = CACHE / arm
        if (out / "meta.json").exists():
            print(f"{arm}: exists")
            continue
        out.mkdir(parents=True, exist_ok=True)
        items = []
        for seed in d130.SEEDS:
            q = box_path(seed) if arm == "box" else d130.shape_path(seed)
            for L in d130.LENGTHS:
                for pm in d130.LIPS:
                    F, Bw = d130.solve(q, L, pm)
                    t_path = d130.timing(F, d130.SPEEDS["slow"])
                    for direction in ("fwd", "rev"):
                        tp = t_path if direction == "fwd" else (t_path[-1] - t_path)[::-1]
                        Fd, Bd, qd = (F, Bw, q) if direction == "fwd" else (F[::-1], Bw[::-1], q[::-1])
                        for f0 in F0S:
                            name = f"s{seed}_L{L:g}_pm{pm:g}_slow_{direction}_f{f0:g}"  # D-130's name for the J pairs
                            key = zlib.crc32(name.encode())
                            y, tf, lab = (d130.render if arm == "box" else render_jit)(Fd, Bd, tp, f0, key)
                            tt = np.clip(tf, 0, tp[-1])
                            np.savez_compressed(out / f"{name}.npz", times=tf, formants=lab,
                                                q=np.stack([np.interp(tt, tp, qd[:, k]) for k in range(2)], 1),
                                                audio_clean=np.round(y / (np.abs(y).max() + 1e-12) * 16000).astype(np.int16))
                            items.append({"name": name, "seed": seed, "length": L, "pm": pm, "direction": direction, "f0": f0})
                    print(f"{arm}: seed {seed}, length {L}, pm {pm}", flush=True)
        (out / "meta.json").write_text(json.dumps({"items": items}, indent=1))
        print(f"{arm}: {len(items)} items")


def _score(model, dev, files, window):
    from vocal_analysis.models.infer import analyse

    out = {}
    for name, path in files:
        d = np.load(path)
        r = analyse(model, d["audio_clean"].astype(np.float32) / 32000, dev, window_frames=window)
        n = len(d["times"])
        pad = lambda a: np.concatenate([a[:n], np.full((max(0, n - len(a)), *a.shape[1:]), np.nan)])  # noqa: E731
        out[f"{name}|F"], out[f"{name}|son"] = pad(r["formants"][:, :4]), pad(r["sonorant_p"])
    return out


def _d130_items():
    meta = json.loads((d130.CACHE / "meta.json").read_text())
    return [it for it in meta["items"] if it["kind"] == "sweep" and it["speed"] == "slow" and it["f0"] in F0S]


def cmd_score(name: str, ckpt: str) -> None:
    import torch

    from vocal_analysis.models.infer import DEFAULT_WINDOW, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ckpt, dev)
    causal = bool(model.cfg.causal)
    RAW.mkdir(parents=True, exist_ok=True)
    for arm in ("box", "jit"):
        dst = RAW / f"{name}_{arm}_raw.npz"
        if dst.exists():
            continue
        meta = json.loads((CACHE / arm / "meta.json").read_text())
        files = [(it["name"], CACHE / arm / f"{it['name']}.npz") for it in meta["items"]]
        np.savez_compressed(dst, **_score(model, dev, files, None if causal else DEFAULT_WINDOW))
        print(f"{name}: {arm} scored", flush=True)
    if not causal and not (RAW / f"{name}_w100_raw.npz").exists():
        files = [(it["name"], d130.CACHE / f"{it['name']}.npz") for it in _d130_items()]
        np.savez_compressed(RAW / f"{name}_w100_raw.npz", **_score(model, dev, files, W_SHORT))
        print(f"{name}: w100 scored", flush=True)


# ----------------------------------------------------------------------------- analysis


def _errors(items, cache, get):
    """Per clip: F1 errors (cents) on qualifying frames, from get(item) -> (F, son); None when the clip has no frames."""
    from vocal_analysis.synth import tube

    lo = np.array([tube.Q1_RANGE[0], tube.Q2_RANGE[0]])
    hi = np.array([tube.Q1_RANGE[1], tube.Q2_RANGE[1]])
    out = []
    for it in items:
        d = np.load(cache / f"{it['name']}.npz")
        T, q = d["formants"], d["q"]
        M, P = get(it)
        ok = np.isfinite(M[:, 0]) & (P > 0.5) & (T[:, 0] >= it["f0"]) & np.all((q >= lo) & (q <= hi), 1)
        out.append(1200 * np.log2(M[ok, 0] / T[ok, 0]))
    return out


def _levels(errs_t, errs_s, idx):
    """Teacher level (mean of seed medians), student level (mean of four), on the clips idx."""
    med = lambda E: np.median(np.concatenate([E[i] for i in idx]))  # noqa: E731
    t = np.mean([med(E) for E in errs_t])
    s = np.mean([med(E) for E in errs_s])
    return t, s


def cmd_analyse() -> None:
    base_items = _d130_items()
    base = {n: np.load(d130.RAW / f"{n}_raw.npz") for n in TEACHERS + STUDENTS}
    get_base = lambda n: lambda it: (base[n][f"{it['name']}|clean|F"], base[n][f"{it['name']}|clean|son"])  # noqa: E731
    arms = {}
    for arm in ("box", "jit"):
        items = json.loads((CACHE / arm / "meta.json").read_text())["items"]
        raw = {n: np.load(RAW / f"{n}_{arm}_raw.npz") for n in TEACHERS + STUDENTS}
        g = lambda n, raw=raw: lambda it: (raw[n][f"{it['name']}|F"], raw[n][f"{it['name']}|son"])  # noqa: E731
        arms[arm] = (items, CACHE / arm, {n: g(n) for n in TEACHERS + STUDENTS})
    raw_w = {n: np.load(RAW / f"{n}_w100_raw.npz") for n in TEACHERS}
    gw = {n: (lambda n: lambda it: (raw_w[n][f"{it['name']}|F"], raw_w[n][f"{it['name']}|son"]))(n) for n in TEACHERS}
    gw.update({n: get_base(n) for n in STUDENTS})  # students' context is unchanged
    arms["w100"] = (base_items, d130.CACHE, gw)
    rng = np.random.default_rng(131)
    lines = ["D-131: median signed F1 error (cents), frames inside Story's box, clean, slow; teacher = mean of the two "
             "seeds' medians, student = mean of four; G = teacher - student; [95 % bootstrap over clips]", ""]
    for f0 in F0S:
        bi = [it for it in base_items if it["f0"] == f0]
        eb_t = [_errors(bi, d130.CACHE, get_base(n)) for n in TEACHERS]
        eb_s = [_errors(bi, d130.CACHE, get_base(n)) for n in STUDENTS]
        n_b = len(bi)
        t0, s0 = _levels(eb_t, eb_s, range(n_b))
        boot0 = [(lambda ix: np.subtract(*_levels(eb_t, eb_s, ix)))(rng.integers(0, n_b, n_b)) for _ in range(B)]
        lines.append(f"f0 {f0:g} Hz ({n_b} clips per arm)")
        lines.append(f"  baseline (D-130 clips, 400-frame windows): teacher {t0:+5.1f}, students {s0:+5.1f}, "
                     f"G0 {t0 - s0:+5.1f} [{np.percentile(boot0, 2.5):+5.1f}, {np.percentile(boot0, 97.5):+5.1f}]")
        for arm, label in (("w100", "W: teacher in 100-frame windows"), ("box", "B: box only, no extension"),
                           ("jit", "J: jitter, shimmer, f0 wander")):
            items, cache, get = arms[arm]
            ai = [it for it in items if it["f0"] == f0]
            ea_t = [_errors(ai, cache, get[n]) for n in TEACHERS]
            ea_s = [_errors(ai, cache, get[n]) for n in STUDENTS]
            t1, s1 = _levels(ea_t, ea_s, range(len(ai)))
            paired = arm != "box"
            if paired:
                assert [it["name"] for it in ai] == [it["name"] for it in bi]
            d = []
            for _ in range(B):
                ix = rng.integers(0, len(ai), len(ai))
                jx = ix if paired else rng.integers(0, n_b, n_b)
                d.append(np.subtract(*_levels(ea_t, ea_s, ix)) - np.subtract(*_levels(eb_t, eb_s, jx)))
            lo, hi = np.percentile(d, [2.5, 97.5])
            lines.append(f"  {label:34s} teacher {t1:+5.1f} ({t1 - t0:+5.1f}), students {s1:+5.1f} ({s1 - s0:+5.1f}), "
                         f"G {t1 - s1:+5.1f}; change from G0 {t1 - s1 - (t0 - s0):+5.1f} [{lo:+5.1f}, {hi:+5.1f}]"
                         f"{'' if paired else ' (unpaired)'}")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d131.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        cmd_render()
    elif cmd == "score":
        cmd_score(sys.argv[2], sys.argv[3])
    elif cmd == "analyse":
        cmd_analyse()
