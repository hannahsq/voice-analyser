"""D-133: does the teacher call long noise-only stretches sonorant? Two synthetic vowels with a gap of background
noise between them (the noise runs under the vowels too), the gap 1-16 s; each model's sonorant probability on the
gap's frames by distance to the nearest vowel.

    python scripts/analysis/d133_noise_gaps.py render            -> data/probes/noise_gaps_v1/
    python scripts/analysis/d133_noise_gaps.py score NAME CKPT   -> reports/probes/noise_gaps/NAME[_w100].npz ($VA_DEVICE)
    python scripts/analysis/d133_noise_gaps.py analyse           -> reports/analysis/d133/d133.txt
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
CACHE = ROOT / "data/probes/noise_gaps_v1"
RAW = ROOT / "reports/probes/noise_gaps"
OUT = ROOT / "reports/analysis/d133"
FS, HOP = 16000, 160
GAPS = (1.0, 2.0, 4.0, 8.0, 16.0)
NOISES = (("white", -40.0), ("white", -60.0), ("pink", -40.0), ("pink", -60.0), ("brown", -40.0), ("brown", -60.0),
          ("dither", None))  # level: dB re the vowels' RMS; dither: white noise at -90 dBFS after normalising
SEEDS = (0, 1)
VOWEL_S = 2.0
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")
BINS = ((0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 99.0))


def _vowel(name, v, f0, key, d131):
    from vocal_analysis.synth import tube

    q1, q2 = tube.STORY_VOWELS[v]
    f, bw = tube.poles(tube.build_shape(q1, q2, 15.6), f_max_hz=7600.0)
    n = d131.d130.N_P
    F, Bw = np.tile(f[:n], (2, 1)), np.tile(bw[:n], (2, 1))
    y, _, _ = d131.render_jit(F, Bw, np.array([0.0, VOWEL_S - 0.1]), f0, key)
    return y.astype(np.float64)


def cmd_render() -> None:
    from vocal_analysis.synth import channel as ch

    spec = importlib.util.spec_from_file_location("d131_context", ROOT / "scripts/analysis/d131_context.py")
    d131 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d131)
    CACHE.mkdir(parents=True, exist_ok=True)
    items = []
    for seed in SEEDS:
        a = _vowel("a", "A", 110.0, zlib.crc32(f"a{seed}".encode()), d131)
        b = _vowel("i", "i", 110.0, zlib.crc32(f"i{seed}".encode()), d131)
        rms = np.sqrt(np.mean(np.r_[a, b] ** 2))
        for gap in GAPS:
            g = np.zeros(int(gap * FS))
            clean = np.r_[a, g, b]
            vowel_mask = np.r_[np.ones(len(a), bool), np.zeros(len(g), bool), np.ones(len(b), bool)]
            for kind, level in NOISES:
                name = f"s{seed}_gap{gap:g}_{kind}{'' if level is None else f'{-level:g}'}"
                rng = np.random.default_rng(zlib.crc32(name.encode()))
                if kind == "dither":
                    y = clean / np.abs(clean).max() * 0.5 + 10 ** (-90 / 20) * rng.standard_normal(len(clean))
                else:
                    z = ch._noise(rng, kind, len(clean))
                    z = z / np.sqrt(np.mean(z ** 2)) * rms * 10 ** (level / 20)
                    y = clean + z
                    y = y / np.abs(y).max() * 0.5
                fm = vowel_mask[::HOP]
                np.savez_compressed(CACHE / f"{name}.npz", audio=np.round(y * 32000).astype(np.int16), vowel=fm)
                items.append({"name": name, "seed": seed, "gap": gap, "noise": kind, "level": level})
        print(f"seed {seed} rendered", flush=True)
    (CACHE / "meta.json").write_text(json.dumps({"items": items}, indent=1))
    print(f"{len(items)} items")


def cmd_score(name: str, ckpt: str) -> None:
    import torch

    from vocal_analysis.models.infer import DEFAULT_WINDOW, analyse, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ckpt, dev)
    causal = bool(model.cfg.causal)
    items = json.loads((CACHE / "meta.json").read_text())["items"]
    RAW.mkdir(parents=True, exist_ok=True)
    for tag, window in ([("", None)] if causal else [("", DEFAULT_WINDOW), ("_w100", 100)]):
        out = {}
        for it in items:
            d = np.load(CACHE / f"{it['name']}.npz")
            r = analyse(model, d["audio"].astype(np.float32) / 32000, dev, window_frames=window)
            n = len(d["vowel"])
            out[f"{it['name']}|son"] = r["sonorant_p"][:n]
            out[f"{it['name']}|F"] = r["formants"][:n, :2]
            out[f"{it['name']}|sig"] = r["formant_sigma_cents"][:n, :2]
        np.savez_compressed(RAW / f"{name}{tag}.npz", **out)
        print(f"{name}{tag}: scored", flush=True)


def cmd_analyse() -> None:
    items = json.loads((CACHE / "meta.json").read_text())["items"]
    groups = (("teacher, 400-frame windows (va analyse)", [f"{t}" for t in TEACHERS]),
              ("teacher, 100-frame windows", [f"{t}_w100" for t in TEACHERS]),
              ("students", list(STUDENTS)))
    raw = {n: np.load(RAW / f"{n}.npz") for _, ns in groups for n in ns}
    lines = ["D-133: share of gap (noise-only) frames called sonorant (p > 0.5), by distance to the nearest vowel; "
             "each model, pooled over gaps 1-16 s and both seeds; then the frames called sonorant: median F2 and F1 sigma", ""]
    conds = [(k, lv) for k, lv in NOISES]
    for kind, level in conds:
        lab = kind if level is None else f"{kind} {level:g} dB"
        lines.append(f"{lab}:")
        sel = [it for it in items if it["noise"] == kind and it["level"] == level]
        for gname, names in groups:
            parts = []
            for n in names:
                hits = {b: [] for b in BINS}
                f2, s1 = [], []
                for it in sel:
                    d = np.load(CACHE / f"{it['name']}.npz")
                    v = d["vowel"]
                    idx = np.flatnonzero(v)
                    dist = np.array([np.min(np.abs(idx - i)) for i in range(len(v))]) / 100
                    son = raw[n][f"{it['name']}|son"] > 0.5
                    gap = ~v
                    for b in BINS:
                        m = gap & (dist >= b[0]) & (dist < b[1])
                        hits[b] += list(son[m])
                    f2 += list(raw[n][f"{it['name']}|F"][gap & son, 1])
                    s1 += list(raw[n][f"{it['name']}|sig"][gap & son, 0])
                share = ", ".join(f"{b[0]:g}-{b[1]:g} s {np.mean(hits[b]):4.0%}" if hits[b] else "-" for b in BINS)
                extra = f"; F2 {np.median(f2):.0f} Hz, F1 sigma {np.median(s1):.0f} c ({len(f2)} frames)" if f2 else "; none"
                parts.append(f"    {n:22s} {share}{extra}")
            lines.append(f"  {gname}:")
            lines += parts
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d133.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        cmd_render()
    elif cmd == "score":
        cmd_score(sys.argv[2], sys.argv[3])
    elif cmd == "analyse":
        cmd_analyse()
